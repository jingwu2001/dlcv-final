# Utils

## `predecode_masks.py`

Pre-decodes the COCO-style RLE masks stored in the PhysialAI Spatial Intelligence dataset and saves the decoded representation next to the dataset JSON. This lets the distance estimation dataloader skip on-the-fly mask decoding during training.

Example:

```bash
python3 predecode_masks.py \
  --input /home/jing/Desktop/PhysicalAI-Spatial-Intelligence-Warehouse/train/train_dist_est.json \
  --output /home/jing/Desktop/PhysicalAI-Spatial-Intelligence-Warehouse/train/train_dist_est_decoded.json \
  --resize-height 360 \
  --resize-width 640 \
  --store-format packbits \
  --drop-rle
```

The script adds a `decoded_masks` list to each sample entry. Each mask entry stores the shape of the decoded mask and a packbit+base64 payload that can be quickly unpacked to a float/bool tensor without pycocotools.
