"""
Distance dataset loader mirroring the refactor style of inside_pred/data_inside.py.

Expected JSON item schema (per sample):
{
  "image": "xxxx.png",
  "rle": [
    {"size": [H, W], "counts": "..."},
    {"size": [H, W], "counts": "..."}
  ],
  "decoded_masks": [  # optional, created by utils/predecode_masks.py
    {"encoding": "packbits", "shape": [H, W], "data": "..."},
    {"encoding": "packbits", "shape": [H, W], "data": "..."}
  ],
  "normalized_answer": float
}
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import pycocotools.mask as mask_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _depth_name_from_image(image_name: str) -> str:
    """Dataset convention: xxxx.png -> xxxx_depth.png"""
    stem = image_name
    if stem.lower().endswith(".png"):
        stem = stem[:-4]
    return f"{stem}_depth.png"


def _decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode COCO-style RLE to a float mask in {0,1} with shape (H, W).

    The original loader expected `rle['counts']` to be a UTF-8 string.
    pycocotools accepts bytes, so we mirror that behaviour safely.
    """
    rle = dict(rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    m = mask_utils.decode(rle)
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    return m


@dataclass(frozen=True)
class DistanceSample:
    image_name: str
    rle: Sequence[Dict[str, Any]]
    normalized_answer: float
    decoded_masks: Optional[Sequence[Dict[str, Any]]] = None

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "DistanceSample":
        decoded_raw = obj.get("decoded_masks")
        decoded_masks = list(decoded_raw) if decoded_raw is not None else None
        return DistanceSample(
            image_name=str(obj["image"]),
            rle=list(obj.get("rle", [])),
            normalized_answer=float(obj["normalized_answer"]),
            decoded_masks=decoded_masks,
        )

    def has_predecoded_masks(self) -> bool:
        return self.decoded_masks is not None


class DistanceDataset(Dataset):
    """
    Returns:
      x: FloatTensor [C, H, W]
         - RGB(3) if rgb=True
         - depth(1) if depth=True
         - mask_a(1) + mask_b(1)
      y: FloatTensor [] (distance) or tuple with class index when cls_bin_center is given.

    Notes:
      - RGB/depth are resized with bilinear.
      - masks are resized with nearest to avoid boundary smoothing.
      - `distance_scale` converts the normalized answer to desired units (e.g. m, cm).
    """

    def __init__(
        self,
        data_dir: str,
        json_path: str,
        transform: Optional[Any] = None,
        rgb: bool = True,
        depth: bool = True,
        resize: Tuple[int, int] = (360, 640),
        cls_bin_center: Optional[Sequence[float]] = None,
        distance_scale: float = 1.0,
        min_distance: float = -1.0,
        max_distance: float = 100000.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "images"
        self.depth_dir = self.data_dir / "depths"
        self.use_rgb = bool(rgb)
        self.use_depth = bool(depth)
        self.resize_hw = tuple(int(x) for x in resize)
        self.transform = transform

        self.cls_bin_center = (
            torch.as_tensor(cls_bin_center, dtype=torch.float32)
            if cls_bin_center is not None
            else None
        )
        self.distance_scale = float(distance_scale)

        json_p = Path(json_path)
        raw = json.loads(json_p.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"JSON must be a list of samples, got: {type(raw)}")
        samples: List[DistanceSample] = [DistanceSample.from_json(x) for x in raw]

        print(f"Number of samples: {len(samples)}")

        print(f"Apply data filtering: Min distance: {min_distance}, Max distance: {max_distance}")
        self.samples: List[DistanceSample] = [
            s
            for s in samples
            if min_distance <= s.normalized_answer <= max_distance and s.image_name != "034000.png"
        ]
        print(f"Number of samples after filtering: {len(self.samples)}")

        # Resize pipelines
        self._rgb_tf = transforms.Compose(
            [
                transforms.Resize(self.resize_hw, interpolation=Image.BILINEAR),
                transforms.ToTensor(),  # [3,H,W] in [0,1]
            ]
        )
        self._depth_tf = transforms.Compose(
            [
                transforms.Resize(self.resize_hw, interpolation=Image.BILINEAR),
                transforms.ToTensor(),  # [1,H,W] in [0,1]
            ]
        )
        self._mask_tf = transforms.Compose(
            [
                transforms.Resize(self.resize_hw, interpolation=Image.NEAREST),
                transforms.ToTensor(),  # [1,H,W] in [0,1]
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    # --- helpers ---------------------------------------------------------

    def _load_rgb(self, image_name: str) -> Tensor:
        path = self.rgb_dir / image_name
        img = Image.open(path).convert("RGB")
        return self._rgb_tf(img)

    def _load_depth(self, image_name: str) -> Tensor:
        path = self.depth_dir / _depth_name_from_image(image_name)
        d = Image.open(path).convert("L")
        return self._depth_tf(d)

    def _decode_precomputed_mask(self, info: Dict[str, Any]) -> np.ndarray:
        shape = tuple(info["shape"])
        encoding = info.get("encoding", "flat")

        if encoding == "packbits":
            packed = np.frombuffer(base64.b64decode(info["data"]), dtype=np.uint8)
            total = int(np.prod(shape))
            unpacked = np.unpackbits(packed, count=total)
            mask = unpacked.reshape(shape)
        elif encoding == "flat":
            mask = np.array(info["data"], dtype=np.uint8).reshape(shape)
        else:
            raise ValueError(f"Unsupported decoded mask encoding: {encoding}")

        return mask.astype(np.float32)

    def _mask_array_to_tensor(self, mask: np.ndarray) -> Tensor:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        m_u8 = (m * 255.0).clip(0, 255).astype(np.uint8)
        m_img = Image.fromarray(m_u8, mode="L")
        return self._mask_tf(m_img)

    def _get_mask_arrays(self, sample: DistanceSample) -> Sequence[np.ndarray]:
        if sample.has_predecoded_masks():
            assert sample.decoded_masks is not None
            masks = [self._decode_precomputed_mask(m) for m in sample.decoded_masks]
        elif sample.rle:
            masks = [_decode_rle(m) for m in sample.rle]
        else:
            raise ValueError(
                f"Sample {sample.image_name} does not contain 'rle' or 'decoded_masks' entries."
            )

        if len(masks) != 2:
            raise ValueError(
                f"Expected exactly 2 masks, got {len(masks)} for sample {sample.image_name}"
            )
        return masks

    def _load_masks(self, sample: DistanceSample) -> List[Tensor]:
        mask_arrays = self._get_mask_arrays(sample)
        return [self._mask_array_to_tensor(m) for m in mask_arrays]

    # --- main API --------------------------------------------------------

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        components: List[Tensor] = []

        if self.use_rgb:
            components.append(self._load_rgb(s.image_name))

        if self.use_depth:
            components.append(self._load_depth(s.image_name))

        mask_a, mask_b = self._load_masks(s)
        components.extend([mask_a, mask_b])

        x = torch.cat(components, dim=0)

        if self.cls_bin_center is not None:
            # regression target is residual around the closest bin center
            centers = self.cls_bin_center
            target = torch.tensor(s.normalized_answer, dtype=torch.float32)
            distance_cls_idx = torch.argmin(torch.abs(centers - target))
            distance = target - centers[distance_cls_idx]
        else:
            distance_cls_idx = None
            distance = torch.tensor(
                s.normalized_answer * self.distance_scale,
                dtype=torch.float32,
            )

        if self.transform is not None:
            x = self.transform(x)

        if distance_cls_idx is not None:
            return x, distance, distance_cls_idx
        return x, distance
