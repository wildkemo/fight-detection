from pathlib import Path
import cv2

from resize import resize
from denoise import denoise_guided
from contrast import contrast
from sharpen import sharpen
from color_space import color_space


# =========================
# CONFIG
# =========================

USE_DENOISE = False
USE_CONTRAST = False
USE_SHARPEN = False
USE_GRAYSCALE = False

FRAME_SIZE = (112, 112)

INPUT_ROOT = Path("output/frames")
OUTPUT_ROOT = Path("output/processed_frames")


# =========================
# PROCESS SINGLE FRAME
# =========================

def process_frame(frame):
    # Resize (ALWAYS)
    frame = cv2.resize(frame, FRAME_SIZE)

    if USE_GRAYSCALE:
        frame = color_space(frame)

    if USE_DENOISE:
        frame = denoise_guided(frame)

    if USE_CONTRAST:
        frame = contrast(frame)

    if USE_SHARPEN:
        frame = sharpen(frame)

    return frame


# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    if not INPUT_ROOT.exists():
        print(f"❌ Input folder not found: {INPUT_ROOT}")
        return

    # ✅ Create main output folder automatically
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for class_dir in INPUT_ROOT.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        print(f"\nProcessing class: {class_name}")

        for video_dir in class_dir.iterdir():
            if not video_dir.is_dir():
                continue

            # ✅ Create video output folder automatically
            output_video_dir = OUTPUT_ROOT / class_name / video_dir.name
            output_video_dir.mkdir(parents=True, exist_ok=True)

            frame_files = sorted(video_dir.glob("*.jpg"))

            print(f"  → {video_dir.name}: {len(frame_files)} frames")

            for i, frame_path in enumerate(frame_files):
                frame = cv2.imread(str(frame_path))

                if frame is None:
                    continue

                frame = process_frame(frame)

                output_path = output_video_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(output_path), frame)

                total_saved += 1

    print("\n" + "=" * 50)
    print("✅ PREPROCESSING COMPLETE")
    print(f"Total frames saved: {total_saved}")
    print(f"Saved in: {OUTPUT_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()