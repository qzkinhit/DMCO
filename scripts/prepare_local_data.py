from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to dmco_code.zip")
    parser.add_argument("--output", default="data/raw", help="Directory for extracted data files")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        for member in zf.infolist():
            if not member.filename.startswith("data/") or member.is_dir():
                continue
            relative = Path(member.filename).relative_to("data")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    print(f"Extracted data to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
