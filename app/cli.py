import argparse
from typing import Optional
from app.application import run_pipeline

def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-Principles Cinematic 2.5D Parallax Renderer (V0)")
    parser.add_argument("--input", type=str, required=True, help="Path to user-provided input image file.")
    parser.add_argument("--motion", type=str, default="Cinematic Push-In", choices=["Cinematic Push-In", "Dolly In", "Dolly Out", "Horizontal Pan", "Vertical Pan", "Orbit", "Micro Orbit"])
    parser.add_argument("--strength", type=str, default="Cinematic", choices=["Subtle", "Cinematic", "Strong"])
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--render-mode", type=str, default="auto", choices=["auto", "2.5d", "3d"])
    parser.add_argument("--quality", type=str, default="auto", choices=["auto", "fast", "balanced", "high", "ultra"])
    parser.add_argument("--resolution", type=str, default="auto", choices=["auto", "480p", "720p", "1080p", "1440p", "4k"])
    parser.add_argument("--render-video", action="store_true", default=False)
    parser.add_argument("--benchmark-hardware", action="store_true", default=False)
    parser.add_argument("--frames", type=int, default=48, choices=[48, 100])
    parser.add_argument("--benchmark-100", action="store_true", default=False)
    parser.add_argument("--reconstruction-quality", type=str, default="HIGH", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--motion-amplitude", type=str, default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"])
    return parser.parse_args(args)

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
