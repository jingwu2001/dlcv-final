import os
import argparse
import contextlib
import itertools
import random
import time
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train the distance regressor")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training from')
    parser.add_argument('--ckpt-dir', type=str, default='ckpt', help='Directory to store checkpoints')
    parser.add_argument('--save-interval', type=int, default=2000, help='Number of iterations between intra-epoch checkpoints')
    parser.add_argument('--log-interval', type=int, default=50, help='Number of iterations between loss logs')
    return parser.parse_args()


def _save_rng_state(state_dict):
    state_dict['python_rng_state'] = random.getstate()
    state_dict['numpy_rng_state'] = np.random.get_state()
    state_dict['torch_rng_state'] = torch.random.get_rng_state()
    if torch.cuda.is_available():
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state_all()


def save_training_state(path, model, optimizer, epoch, iteration, global_step, is_epoch_end=False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'global_step': global_step,
    }
    _save_rng_state(state)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    status = "(epoch end)" if is_epoch_end else ""
    print(f"Checkpoint saved to {path} {status}")


def load_training_state(path, model, optimizer):
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model.load_state_dict(checkpoint)
        print("Loaded weights only checkpoint; optimizer state not found")

    start_epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
    global_step = checkpoint.get('global_step', 0) if isinstance(checkpoint, dict) else 0
    iteration = checkpoint.get('iteration', 0) if isinstance(checkpoint, dict) else 0

    if isinstance(checkpoint, dict):
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'torch_rng_state' in checkpoint:
            torch.random.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    print(f"Resumed training from epoch {start_epoch}, iteration {iteration}, global step {global_step}")
    return start_epoch, global_step, iteration

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


def main():
    args = parse_args()
    num_epochs = args.epochs
    log_interval = args.log_interval
    save_interval = args.save_interval
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    resume_iter = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step, resume_iter = load_training_state(args.resume, model, optimizer)

    if start_epoch >= num_epochs:
        print(f"Start epoch {start_epoch} is >= configured num_epochs {num_epochs}, nothing to train.")
        return

    special_save_iters = [50, 100, 150]

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        num_batches = len(loader)
        current_start_iter = resume_iter if epoch == start_epoch else 0
        resume_iter = 0

        if current_start_iter >= num_batches:
            print(f"Resume iteration {current_start_iter} exceeds number of batches {num_batches}; skipping epoch.")
            continue

        epoch_iterable = loader
        if current_start_iter > 0:
            epoch_iterable = itertools.islice(loader, current_start_iter, None)

        effective_batches = num_batches - current_start_iter

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

        with profiler_ctx:
            data_fetch_start = time.perf_counter()
            pbar = tqdm(epoch_iterable, total=effective_batches if effective_batches > 0 else None)
            for batch_idx, (inputs, targets) in enumerate(pbar, start=current_start_iter):
                if batch_idx == current_start_iter:
                    print("inputs.shape:", inputs.shape)

                data_wait = time.perf_counter() - data_fetch_start
                iter_start = time.perf_counter()
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                preds = model(inputs)
                loss = criterion(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

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

                if (batch_idx + 1) % LOG_USAGE_INTERVAL == 0:
                    mem, cpu = get_process_stats()
                    postfix['RAM'] = f'{mem:.2f}GB'
                    postfix['CPU'] = f'{cpu:.1f}%'

                pbar.set_postfix(postfix)
                epoch_loss += batch_loss

                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                    print(f"[Epoch {epoch+1}/{num_epochs}] Iter {batch_idx+1}/{num_batches} Loss: {batch_loss:.4f}")

                should_save = False
                if epoch == 0 and batch_idx in special_save_iters:
                    should_save = True
                if save_interval > 0 and (batch_idx + 1) % save_interval == 0 and epoch > 3:
                    should_save = True

                if should_save:
                    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}_iter_{batch_idx + 1}.pth")
                    save_training_state(ckpt_path, model, optimizer, epoch, batch_idx + 1, global_step)

                # if (batch_idx + 1) >= TESTRUN_STEPS:
                #     print(f"Stopping epoch early after {TESTRUN_STEPS} steps to write profiler trace.")
                #     break

        processed_batches = num_batches - current_start_iter
        if processed_batches == 0:
            continue
        avg_loss = epoch_loss / processed_batches
        print(f"[Epoch {epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        final_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}_final.pth")
        save_training_state(final_path, model, optimizer, epoch + 1, 0, global_step, is_epoch_end=True)


if __name__ == "__main__":
    main()
