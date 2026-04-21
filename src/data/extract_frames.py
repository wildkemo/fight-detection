from pathlib import Path
import cv2


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def extract_frames(
    video_path: str | Path,
    target_fps: float = 5.0,
) -> list:
    """
    Extract frames from a single video and return them in memory as a list.

    Notes:
    - Does NOT save frames.
    - Does NOT create any folders.
    - Intended for preprocess.py, which applies processing then saves frames later.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps is None or original_fps <= 0:
        original_fps = target_fps

    frame_interval = max(1, round(original_fps / target_fps))

    frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            frames.append(frame)

        frame_index += 1

    cap.release()
    return frames


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    class_name: str,
    target_fps: float = 5.0,
    image_extension: str = ".jpg",
) -> dict:
    """
    Extract frames from a single video at a fixed target FPS and save them.

    Notes:
    - Assumes output_dir already exists.
    - Does NOT create any folders.
    - Saves all frames directly into one flat output folder.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}"
        )

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
    video_stem = video_path.stem

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            frame_name = (
                f"{class_name}_{video_stem}_frame_{saved_index:06d}{image_extension}"
            )
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
    output_dir: str | Path,
    class_name: str,
    target_fps: float = 5.0,
) -> list[dict]:
    """
    Extract frames from all supported video files inside one class folder.

    Notes:
    - Reads videos from the class folder.
    - Saves all frames directly into one shared output folder.
    - Does NOT create any subfolders.
    """
    input_class_dir = Path(input_class_dir)
    output_dir = Path(output_dir)

    if not input_class_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_class_dir}")

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}"
        )

    results = []

    for video_file in input_class_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            metadata = extract_frames_from_video(
                video_path=video_file,
                output_dir=output_dir,
                class_name=class_name,
                target_fps=target_fps,
            )
            results.append(metadata)

    return results


def extract_frames_from_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    class_names: list[str] | None = None,
    target_fps: float = 5.0,
) -> list[dict]:
    """
    Extract frames for all class folders inside the dataset root.

    Notes:
    - Assumes output_root already exists.
    - Saves all frames into one flat output folder.
    - Does NOT create class folders or video folders.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not output_root.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_root}"
        )

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
            output_dir=output_root,
            class_name=class_name,
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
    output_root = Path("src/data/output")

    results = extract_frames_from_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        class_names=["Violence", "NonViolence"],
        target_fps=5.0,
    )

    print_summary(results)