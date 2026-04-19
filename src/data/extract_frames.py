from pathlib import Path
import cv2


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 5.0,
    image_extension: str = ".jpg",
) -> dict:
    """
    Extract frames from a single video at a fixed target FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be saved.
        target_fps: Number of frames to save per second.
        image_extension: Output image format (e.g. '.jpg', '.png').

    Returns:
        A dictionary containing metadata about the extraction process.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback in case FPS metadata is missing or invalid
    if original_fps is None or original_fps <= 0:
        original_fps = target_fps

    # Determine how many original frames to skip between saved frames
    frame_interval = max(1, round(original_fps / target_fps))

    frame_index = 0
    saved_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            frame_name = f"frame_{saved_index:06d}{image_extension}"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_index += 1

        frame_index += 1

    cap.release()

    duration_seconds = total_frames / original_fps if original_fps > 0 else 0

    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "original_fps": original_fps,
        "target_fps": target_fps,
        "frame_interval": frame_interval,
        "total_frames": total_frames,
        "saved_frames": saved_index,
        "duration_seconds": duration_seconds,
    }


def extract_frames_from_class_folder(
    input_class_dir: str | Path,
    output_class_dir: str | Path,
    target_fps: float = 5.0,
) -> list[dict]:
    """
    Extract frames from all supported video files inside one class folder.

    Example:
        input_class_dir  -> src/data/dataset/Violence
        output_class_dir -> src/data/dataset/Violence

    Each video's frames are saved inside a subfolder named after the video.
    """
    input_class_dir = Path(input_class_dir)
    output_class_dir = Path(output_class_dir)

    if not input_class_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_class_dir}")

    results = []

    for video_file in input_class_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            video_output_dir = output_class_dir / video_file.stem

            metadata = extract_frames_from_video(
                video_path=video_file,
                output_dir=video_output_dir,
                target_fps=target_fps,
            )
            results.append(metadata)

    return results


def extract_frames_from_dataset(
    dataset_root: str | Path,
    class_names: list[str] | None = None,
    target_fps: float = 5.0,
) -> list[dict]:
    """
    Extract frames for all class folders inside the dataset root.

    Expected structure:
        dataset_root/
            Violence/
                video1.mp4
                video2.mp4
            NonViolence/
                video3.mp4
                video4.mp4

    Output structure:
        dataset_root/
            Violence/
                video1/
                    frame_000000.jpg
                    ...
                video2/
                    frame_000000.jpg
                    ...
            NonViolence/
                video3/
                    frame_000000.jpg
                    ...
    """
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if class_names is None:
        class_names = [item.name for item in dataset_root.iterdir() if item.is_dir()]

    all_results = []

    for class_name in class_names:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            print(f"Skipping missing class folder: {class_dir}")
            continue

        print(f"\nProcessing class: {class_name}")
        class_results = extract_frames_from_class_folder(
            input_class_dir=class_dir,
            output_class_dir=class_dir,
            target_fps=target_fps,
        )
        all_results.extend(class_results)

    return all_results


def print_summary(results: list[dict]) -> None:
    """
    Print a clean summary of extraction results.
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
        print(f"Video:         {item['video_path']}")
        print(f"Original FPS:  {item['original_fps']:.2f}")
        print(f"Target FPS:    {item['target_fps']:.2f}")
        print(f"Saved Frames:  {item['saved_frames']}")
        print(f"Output Folder: {item['output_dir']}")
        print("-" * 60)


if __name__ == "__main__":
    dataset_root = Path("src/data/dataset")

    results = extract_frames_from_dataset(
        dataset_root=dataset_root,
        class_names=["Violence", "NonViolence"],
        target_fps=5.0,
    )

    print_summary(results)