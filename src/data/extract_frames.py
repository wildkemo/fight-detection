from pathlib import Path
import cv2


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def extract_frames_from_video(
    video_path: str | Path,
    output_root: str | Path,
    class_name: str,
    target_fps: float = 5.0,
    image_extension: str = ".jpg",
) -> dict:
    """
    Extract frames from one video and save them inside:

    output_root/class_name/video_name/frame_000000.jpg

    Example:
    output/frames/Violence/video_001/frame_000000.jpg
    """

    video_path = Path(video_path)
    output_root = Path(output_root)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_output_dir = output_root / class_name / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps is None or original_fps <= 0:
        original_fps = target_fps

    frame_interval = max(1, round(original_fps / target_fps))

    frame_index = 0
    saved_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_interval == 0:
            frame_name = f"frame_{saved_index:06d}{image_extension}"
            frame_path = video_output_dir / frame_name

            cv2.imwrite(str(frame_path), frame)
            saved_index += 1

        frame_index += 1

    cap.release()

    duration_seconds = total_frames / original_fps if original_fps > 0 else 0

    return {
        "video_path": str(video_path),
        "output_dir": str(video_output_dir),
        "class_name": class_name,
        "original_fps": original_fps,
        "target_fps": target_fps,
        "frame_interval": frame_interval,
        "total_frames": total_frames,
        "saved_frames": saved_index,
        "duration_seconds": duration_seconds,
    }


def extract_frames_from_class_folder(
    input_class_dir: str | Path,
    output_root: str | Path,
    class_name: str,
    target_fps: float = 5.0,
) -> list[dict]:
    """
    Extract frames from all videos inside one class folder.

    Example input:
    dataset/Violence/

    Example output:
    output/frames/Violence/video_name/
    """

    input_class_dir = Path(input_class_dir)
    output_root = Path(output_root)

    if not input_class_dir.exists():
        raise FileNotFoundError(f"Input class folder not found: {input_class_dir}")

    results = []

    video_files = [
        file
        for file in input_class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]

    if len(video_files) == 0:
        print(f"Warning: No supported videos found in {input_class_dir}")
        return results

    for video_file in video_files:
        print(f"Extracting: {video_file}")

        metadata = extract_frames_from_video(
            video_path=video_file,
            output_root=output_root,
            class_name=class_name,
            target_fps=target_fps,
        )

        results.append(metadata)

    return results


def extract_frames_from_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    class_names: list[str],
    target_fps: float = 5.0,
) -> list[dict]:
    """
    Extract frames from all class folders.

    Expected input:
    dataset/
    ├── Violence/
    └── NonViolence/

    Expected output:
    output/frames/
    ├── Violence/
    │   └── video_name/
    └── NonViolence/
        └── video_name/
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for class_name in class_names:
        class_dir = dataset_root / class_name

        if not class_dir.exists():
            print(f"Skipping missing class folder: {class_dir}")
            continue

        print("\n" + "=" * 60)
        print(f"Processing class: {class_name}")
        print("=" * 60)

        class_results = extract_frames_from_class_folder(
            input_class_dir=class_dir,
            output_root=output_root,
            class_name=class_name,
            target_fps=target_fps,
        )

        all_results.extend(class_results)

    return all_results


def print_summary(results: list[dict]) -> None:
    """
    Print extraction summary.
    """

    total_videos = len(results)
    total_saved_frames = sum(item["saved_frames"] for item in results)

    print("\n" + "=" * 60)
    print("FRAME EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames saved:     {total_saved_frames}")
    print("-" * 60)

    for item in results:
        print(f"Video:          {item['video_path']}")
        print(f"Class:          {item['class_name']}")
        print(f"Original FPS:   {item['original_fps']:.2f}")
        print(f"Target FPS:     {item['target_fps']:.2f}")
        print(f"Frame interval: {item['frame_interval']}")
        print(f"Saved frames:   {item['saved_frames']}")
        print(f"Output folder:  {item['output_dir']}")
        print("-" * 60)


if __name__ == "__main__":
    dataset_root = Path("dataset")
    output_root = Path("output/frames")

    class_names = ["NonViolence", "Violence"]

    results = extract_frames_from_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        class_names=class_names,
        target_fps=5.0,
    )

    print_summary(results)