import os
import contextlib
import time
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule as profiler_schedule,
    tensorboard_trace_handler,
)
from tqdm import tqdm
# from data_loader import DistanceDataset
from data_distance import DistanceDataset
from model import DistanceRegressor

# Argument parsing if needed, or simple implementation
USE_DEPTH = True
BACKBONE = 'convnext_base' # resnet50, convnext_tiny, etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '/home/jing/Desktop/DLCV_Final1'
fallback_data_root = '/home/jing/Desktop/PhysicalAI-Spatial-Intelligence-Warehouse/train'
dataset = DistanceDataset(data_dir + '/train', data_dir + '/train/train_dist_est_decoded.json', rgb=True, depth=USE_DEPTH)

ENABLE_PROFILING = False  # Flip to False after collecting a trace
PROFILE_STEPS = 150      # Number of iterations to capture
PROFILE_LOGDIR = "profiler_logs/distance_est"

LOG_USAGE_INTERVAL = 10

# TESTRUN_STEPS = 200

if ENABLE_PROFILING:
    os.makedirs(PROFILE_LOGDIR, exist_ok=True)

def get_gpu_stats(device=DEVICE):
    if not torch.cuda.is_available():
        return None, None

    device_index = None
    if isinstance(device, torch.device):
        if device.type != 'cuda':
            return None, None
        device_index = device.index
    elif isinstance(device, str):
        if not device.startswith('cuda'):
            return None, None
        if ':' in device:
            try:
                device_index = int(device.split(':')[1])
            except ValueError:
                device_index = None

    if device_index is None:
        try:
            device_index = torch.cuda.current_device()
        except Exception:
            device_index = 0

    gpu_util = None
    if hasattr(torch.cuda, 'utilization'):
        try:
            gpu_util = torch.cuda.utilization(device_index)
        except Exception:
            gpu_util = None

    gpu_mem = None
    try:
        gpu_mem = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    except Exception:
        gpu_mem = None

    return gpu_util, gpu_mem


def get_process_stats():
    # Get the ID of the current running process (your script)
    pid = os.getpid()
    process = psutil.Process(pid)
    
    # RSS (Resident Set Size) is the non-swapped physical memory your script is using
    memory_bytes = process.memory_info().rss
    memory_gb = memory_bytes / (1024 ** 3)
    
    # CPU usage of this specific process (can exceed 100% if using multiple cores)
    cpu_usage = process.cpu_percent(interval=None)
    
    return memory_gb, cpu_usage


def _worker_init_fn(_):
    """
    Torch tensor ops inside DistanceDataset (resize, interpolation, etc.)
    spawn CPU threads. When eight DataLoader workers each spin up their
    own pools the machine gets heavily oversubscribed, so cap every worker
    to a single thread to keep throughput consistent.
    """
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

# Optimization settings
BATCH_SIZE = 8
num_workers = 8 # Increased from 4
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=num_workers,
    worker_init_fn=_worker_init_fn,
    pin_memory=True,        # Faster transfer to GPU
    persistent_workers=True, # Avoids checking generic files usage
    prefetch_factor=1, # Buffers 4 batches per worker
    in_order=False
)

input_channels = 5 + int(USE_DEPTH) # 5 (RGB + 2 Masks) + Depth
print(f"Initializing model with backbone={BACKBONE}, input_channels={input_channels}")

model = DistanceRegressor(input_channels=input_channels, backbone=BACKBONE, pretrained=True).cuda()
# Commenting out load_state_dict for new training since architecture/channels changed
# model.load_state_dict(torch.load('ckpt/epoch_5_iter_6831.pth')) 
# Linear scaling rule explicitly requested by user: lr = 1e-4 * (batch_size / 36)
learning_rate = 1e-4 * (BATCH_SIZE / 36)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(loader)}")
print(next(iter(loader))[0].shape)
# Training loop
num_epochs = 5
# Limit batches for dry run if needed, but for now we'll just run it.
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    epoch_loss = 0
    num_batches = len(loader)
    log_interval = 50
    save_interval = 2000

    if ENABLE_PROFILING:
        profiler_ctx = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule(wait=5, warmup=5, active=10),
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(PROFILE_LOGDIR),
        with_stack=True,
        )
    else:
        profiler_ctx = contextlib.nullcontext()

    with profiler_ctx as prof:
        prof_enabled = ENABLE_PROFILING

        pbar = tqdm(loader)
        data_fetch_start = time.perf_counter()
        for i, (inputs, targets) in enumerate(pbar):
            if i == 0:
                print("inputs.shape:", inputs.shape)
            data_wait = time.perf_counter() - data_fetch_start
            iter_start = time.perf_counter()
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_duration = time.perf_counter() - iter_start
            data_fetch_start = time.perf_counter()

            batch_loss = loss.item()
            gpu_util, gpu_mem = get_gpu_stats()
            postfix = {
                'loss': f'{batch_loss:.4f}',
                'data_s': f'{data_wait:.3f}',
                'step_s': f'{iter_duration:.3f}',
            }
            if gpu_util is not None:
                postfix['GPU%'] = f'{gpu_util:3.0f}'
            if gpu_mem is not None:
                postfix['GPU_GB'] = f'{gpu_mem:.2f}'

            if (i + 1) % LOG_USAGE_INTERVAL == 0:
                mem, cpu = get_process_stats()
                postfix['RAM'] = f'{mem:.2f}GB'
                postfix['CPU'] = f'{cpu:.1f}%'

            pbar.set_postfix(postfix)
            epoch_loss += batch_loss

            if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
                print(f"[Epoch {epoch+1}/{num_epochs}] "
                      f"Iter {i+1}/{num_batches} "
                      f"Loss: {batch_loss:.4f}")


            if (epoch == 0 and (i in [50, 100, 150])) or ((i + 1) % save_interval == 0 and epoch > 3):
                torch.save(model.state_dict(), f"ckpt/epoch_{epoch+1}_iter_{i+1}.pth")
                print(f"Model saved at epoch {epoch+1}, iteration {i+1}")

            # if (i + 1) >= TESTRUN_STEPS:
            #     print(f"Stopping epoch early after {TESTRUN_STEPS} steps to write profiler trace.")
            #     break

    avg_loss = epoch_loss / num_batches
    print(f"[Epoch {epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"ckpt/epoch_{epoch+1}_iter_{i+1}.pth")
    print(f"Model saved at epoch {epoch+1}, iteration {i+1}")


