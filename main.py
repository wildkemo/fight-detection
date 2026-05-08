#!/usr/bin/env python3
"""
Run the data-prep pipeline in order:

1. ``frame_extraction`` — videos → ``output/frames/...`` (JPEG folders)
2. ``human_detection`` — person crops → ``output/human_crops/...``
3. ``crop_roi`` — motion-focused crops → ``output/roi_crops/...``
4. ``optical_flow`` — dense flow stacks → ``output/optical_flow/...``

From the project root::

  python main.py
  python main.py --extract-resize --human-device cpu

Skip steps if you already ran them::

  python main.py --skip-extraction
  python main.py --skip-human
  python main.py --skip-roi
  python main.py --skip-flow
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _ensure_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def main() -> None:
    _ensure_root_on_path()
    os.chdir(ROOT)

    p = argparse.ArgumentParser(
        description="Run frame extraction → human detection → ROI → optical flow.",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip video → JPEG frame folders.",
    )
    p.add_argument(
        "--skip-human",
        action="store_true",
        help="Skip YOLO person cropping.",
    )
    p.add_argument(
        "--extract-resize",
        action="store_true",
        help="Resize extracted frames to FRAME_WIDTH×FRAME_HEIGHT (default: native resolution).",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Frames sampled per video (default: SEQUENCE_LENGTH from config).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Override dataset root (default: <project>/dataset from config).",
    )
    p.add_argument(
        "--frames-out",
        default=None,
        help="Override frames output root (default: output/frames).",
    )
    p.add_argument(
        "--human-out",
        default=None,
        help="Override human crops output root (default: output/human_crops).",
    )
    p.add_argument("--human-weights", default="yolov8n.pt")
    p.add_argument("--human-conf", type=float, default=0.25)
    p.add_argument("--human-pad-frac", type=float, default=0.15)
    p.add_argument("--human-device", default="cpu")
    p.add_argument(
        "--human-fallback",
        choices=("resize_full_frame", "black"),
        default="resize_full_frame",
    )
    p.add_argument(
        "--skip-roi",
        action="store_true",
        help="Skip motion-based ROI refinement on human crops.",
    )
    p.add_argument(
        "--roi-out",
        default=None,
        help="Override ROI output root (default: output/roi_crops).",
    )
    p.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip optical-flow encoding on ROI crops.",
    )
    p.add_argument(
        "--flow-out",
        default=None,
        help="Override optical-flow output root (default: output/optical_flow).",
    )
    p.add_argument(
        "--flow-encode",
        choices=("hsv", "vector_mag"),
        default="vector_mag",
        help="hsv: direction as hue | vector_mag: |v|, vx, vy in B,G,R",
    )
    args = p.parse_args()

    from src.utils.config import (
        DATASET_DIR,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        FRAMES_DIR,
        HUMAN_CROPS_DIR,
        OPTICAL_FLOW_DIR,
        ROI_CROPS_DIR,
        SEQUENCE_LENGTH,
    )

    dataset_dir = Path(args.dataset) if args.dataset else ROOT / DATASET_DIR
    frames_dir = Path(args.frames_out) if args.frames_out else ROOT / FRAMES_DIR
    human_out = Path(args.human_out) if args.human_out else ROOT / HUMAN_CROPS_DIR
    roi_out = Path(args.roi_out) if args.roi_out else ROOT / ROI_CROPS_DIR
    flow_out = Path(args.flow_out) if args.flow_out else ROOT / OPTICAL_FLOW_DIR
    seq_len = args.sequence_length if args.sequence_length is not None else SEQUENCE_LENGTH
    extract_resize_wh = (FRAME_WIDTH, FRAME_HEIGHT) if args.extract_resize else None

    if not args.skip_extraction:
        from src.data.frame_extraction import run_frame_extraction_pipeline

        print("=== Step 1/4: Frame extraction ===")
        run_frame_extraction_pipeline(
            dataset_dir=dataset_dir,
            frames_dir=frames_dir,
            sequence_length=seq_len,
            resize_wh=extract_resize_wh,
        )
    else:
        print("=== Step 1/4: skipped (--skip-extraction) ===")

    if not args.skip_human:
        from src.data.human_detection import run_human_crop_pipeline

        print("=== Step 2/4: Human detection (YOLO crops) ===")
        run_human_crop_pipeline(
            frames_root=frames_dir,
            out_root=human_out,
            model_weights=args.human_weights,
            conf=args.human_conf,
            pad_frac=args.human_pad_frac,
            target_wh=(FRAME_WIDTH, FRAME_HEIGHT),
            fallback_mode=args.human_fallback,
            device=args.human_device,
        )
    else:
        print("=== Step 2/4: skipped (--skip-human) ===")

    if not args.skip_roi:
        from src.data.crop_roi import run_roi_crop_pipeline

        print("=== Step 3/4: Motion ROI refinement ===")
        run_roi_crop_pipeline(
            input_root=human_out,
            output_root=roi_out,
            target_wh=(FRAME_WIDTH, FRAME_HEIGHT),
        )
    else:
        print("=== Step 3/4: skipped (--skip-roi) ===")

    if not args.skip_flow:
        from src.data.optical_flow import run_optical_flow_pipeline

        print("=== Step 4/4: Optical flow ===")
        run_optical_flow_pipeline(
            input_root=roi_out,
            output_root=flow_out,
            encode_mode=args.flow_encode,
            resize_wh=(FRAME_WIDTH, FRAME_HEIGHT),
        )
    else:
        print("=== Step 4/4: skipped (--skip-flow) ===")

    print("Done.")


if __name__ == "__main__":
    main()
