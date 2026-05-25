"""Convert a heybuddy .pt wake-word model into an ONNX model usable by prod/js.

Usage:
    cd model
    python tools/convert.py --input exports/heybuddy/ozwell_i_m_done_final.pt \\
        --output ../prod/js/models/ozwell-i\\'m-done.onnx

Default paths assume CWD is model/ and the .pt files live in exports/heybuddy/,
with ONNX output written to ../prod/js/models/ so the JS runtime picks them up
without any additional copy step.
"""
import argparse
from pathlib import Path

import onnx
from heybuddy.wakeword import WakeWordMLPModel, WakeWordTransformerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", "-i",
        default="exports/heybuddy/ozwell_i_m_done_final.pt",
        help="Path to the .pt file to convert (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        default="../prod/js/models/ozwell-i'm-done.onnx",
        help="Path to write the .onnx file (default: %(default)s)",
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Treat the input as a WakeWordTransformerModel instead of WakeWordMLPModel",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=19,
        help="ONNX opset version (default: %(default)s)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.transformer:
        model = WakeWordTransformerModel.from_file(str(input_path))
    else:
        model = WakeWordMLPModel.from_file(str(input_path))

    model.save_onnx(str(output_path), opset_version=args.opset_version, external_data=False)
    onnx.checker.check_model(str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
