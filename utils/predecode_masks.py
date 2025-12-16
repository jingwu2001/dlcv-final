import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_utils


def _decode_rle(rle_obj):
    """Decode a pycocotools RLE object into a uint8 numpy mask."""
    rle = dict(rle_obj)  # avoid mutating original entry
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    return mask_utils.decode(rle).astype(np.uint8)


def _maybe_resize(mask, resize_hw):
    """Resize a binary mask (if required) using nearest interpolation."""
    if resize_hw is None:
        return mask

    height, width = resize_hw
    pil_mask = Image.fromarray(mask * 255)
    pil_mask = pil_mask.resize((width, height), resample=Image.NEAREST)
    return (np.array(pil_mask) > 0).astype(np.uint8)


def _serialize_mask(mask, store_format):
    """
    Serialize a binary mask to JSON friendly format.

    We default to packbits+base64 to keep the output file size manageable,
    but an explicit raw/flat encoding is provided for debugging.
    """
    if store_format == "packbits":
        packed = np.packbits(mask.reshape(-1).astype(np.uint8))
        data = base64.b64encode(packed.tobytes()).decode("ascii")
        return {
            "encoding": "packbits",
            "shape": list(mask.shape),
            "data": data,
        }

    if store_format == "flat":
        return {
            "encoding": "flat",
            "shape": list(mask.shape),
            "data": mask.reshape(-1).astype(int).tolist(),
        }

    raise ValueError(f"Unsupported store format '{store_format}'")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Decode the COCO-style RLE masks contained in the distance "
            "estimation JSON file and persist the decoded representation "
            "back to disk so the training dataloader can skip on-the-fly "
            "mask decoding."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/jing/Desktop/PhysicalAI-Spatial-Intelligence-Warehouse/train/train_dist_est.json"),
        help="Path to the input JSON file that still contains COCO RLE masks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/jing/Desktop/PhysicalAI-Spatial-Intelligence-Warehouse/train/train_dist_est_decoded.json"),
        help="Where to store the JSON with decoded mask information.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional height to resize decoded masks to (e.g. 360).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional width to resize decoded masks to (e.g. 640).",
    )
    parser.add_argument(
        "--store-format",
        choices=("packbits", "flat"),
        default="packbits",
        help=(
            "How to serialize the decoded masks. 'packbits' keeps the "
            "file size manageable by packbit encoding+base64. "
            "'flat' writes the flattened 0/1 array directly (much larger)."
        ),
    )
    parser.add_argument(
        "--drop-rle",
        action="store_true",
        help="Drop the original 'rle' key from each sample to save space.",
    )
    args = parser.parse_args()

    resize = None
    if args.resize_height is not None and args.resize_width is not None:
        resize = (args.resize_height, args.resize_width)

    print(f"Loading samples from {args.input} ...")
    with args.input.open("r") as f:
        samples = json.load(f)

    decoded_samples = []
    for sample in tqdm(samples, desc="Decoding masks"):
        decoded_entry = dict(sample)
        decoded_masks = []
        for mask_rle in sample["rle"]:
            mask = _decode_rle(mask_rle)
            mask = _maybe_resize(mask, resize)
            decoded_masks.append(_serialize_mask(mask, args.store_format))
        decoded_entry["decoded_masks"] = decoded_masks
        if args.drop_rle:
            decoded_entry.pop("rle", None)
        decoded_samples.append(decoded_entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing decoded samples to {args.output} ...")
    with args.output.open("w") as f:
        json.dump(decoded_samples, f)
    print("Done.")


if __name__ == "__main__":
    main()
