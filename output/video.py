from pathlib import Path
def verify_video_output(video_path: Path) -> bool:
    return video_path.exists() and video_path.stat().st_size > 0
