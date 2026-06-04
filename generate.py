import argparse
import sys
from pathlib import Path

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clothing variations using IP-Adapter + ControlNet")
    parser.add_argument("--input", required=True, type=Path, help="Source image (model wearing garment)")
    parser.add_argument("--mode", required=True, choices=["pose", "background", "color"], help="Variation type")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (required for background/color modes)")
    parser.add_argument("--pose", type=Path, default=None, help="Reference pose image (required for pose mode)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory (default: outputs/)")
    parser.add_argument("--resolution", type=int, default=1024, help="Generation resolution (default: 1024, use 768 for low VRAM)")
    return parser.parse_args()


def validate(args):
    if not args.input.exists():
        print(f"Error: input image not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.input.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: unsupported format {args.input.suffix!r}. Use: {', '.join(SUPPORTED_FORMATS)}", file=sys.stderr)
        sys.exit(1)
    if args.mode == "pose" and args.pose is None:
        print("Error: --pose <image> is required for pose mode", file=sys.stderr)
        sys.exit(1)
    if args.mode == "pose" and not args.pose.exists():
        print(f"Error: pose reference image not found: {args.pose}", file=sys.stderr)
        sys.exit(1)
    if args.mode in ("color", "background") and not args.prompt.strip():
        print("Warning: --prompt is empty. Color/background modes work best with a descriptive prompt.", file=sys.stderr)


def main():
    args = parse_args()
    validate(args)

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.input.stem}_{args.mode}.png"

    import torch
    from pipeline.loader import load_pipeline

    try:
        pipe = load_pipeline(args.mode)
    except torch.cuda.OutOfMemoryError:
        print("Error: CUDA out of memory. Try --resolution 768 to reduce VRAM usage.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "color":
        from pipeline.modes.color import generate_color_variation
        generate_color_variation(pipe, args.input, args.prompt, output_path, args.resolution)
    elif args.mode == "background":
        from pipeline.modes.background import generate_background_variation
        generate_background_variation(pipe, args.input, args.prompt, output_path, args.resolution)
    elif args.mode == "pose":
        from pipeline.modes.pose import generate_pose_variation
        generate_pose_variation(pipe, args.input, args.pose, output_path, args.resolution)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
