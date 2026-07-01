from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAPER_TABLE2 = {
    ("adult", "DR"): [0.6306, 0.6356, 0.6380, 0.6385, 0.6352, 0.6374, 0.6321, 0.6250, 0.6289, 0.6342],
    ("adult", "DA"): [0.0000, 0.6641, 0.6210, 0.6132, 0.6132, 0.6146, 0.6144, 0.6269, 0.6248, 0.6248],
    ("adult", "CR"): [0.6322, 0.6391, 0.6415, 0.6370, 0.6408, 0.6338, 0.6393, 0.6414, 0.6403, 0.6408],
    ("adult", "CA"): [0.0000, 0.5302, 0.6527, 0.6094, 0.6286, 0.6286, 0.6139, 0.6188, 0.6139, 0.6155],
    ("adult", "DMCO"): [0.0000, 0.0000, 0.0000, 0.6641, 0.6641, 0.6641, 0.6641, 0.6081, 0.6170, 0.6049],
    ("cancer", "DR"): [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.7692, 0.8000, 0.8000, 0.8000],
    ("cancer", "DA"): [0.9375, 0.9474, 0.9247, 0.9362, 0.9474, 0.9362, 0.9362, 0.9474, 0.9474, 0.9474],
    ("cancer", "CR"): [0.8989, 0.8539, 0.8791, 0.8889, 0.8539, 0.9011, 0.9247, 0.9011, 0.9592, 0.9592],
    ("cancer", "CA"): [0.9592, 0.9495, 0.9495, 0.9495, 0.9495, 0.9495, 0.9495, 0.9495, 0.9592, 0.9495],
    ("cancer", "DMCO"): [0.9592, 0.9691, 0.9592, 0.9495, 0.9592, 0.9495, 0.9592, 0.9592, 0.9495, 0.9592],
    ("skin", "DR"): [0.8791, 0.8791, 0.8791, 0.9032, 0.8791, 0.8791, 0.8791, 0.8667, 0.9032, 0.8791],
    ("skin", "DA"): [0.9592, 0.9592, 0.9592, 0.9592, 0.9592, 0.9592, 0.9592, 0.9592, 0.9592, 0.9691],
    ("skin", "CR"): [0.9011, 0.8791, 0.8913, 0.8913, 0.8913, 0.8889, 0.8889, 0.8889, 0.8889, 0.9592],
    ("skin", "CA"): [0.9592, 0.9495, 0.9495, 0.9495, 0.9495, 0.9495, 0.9495, 0.9592, 0.9495, 0.9495],
    ("skin", "DMCO"): [0.9592, 0.9592, 0.9495, 0.9691, 0.9216, 0.9495, 0.9495, 0.9495, 0.9495, 0.9495],
    ("smartfactory", "DR"): [0.0222, 0.0238, 0.0270, 0.0286, 0.0221, 0.0221, 0.0221, 0.0237, 0.0221, 0.0221],
    ("smartfactory", "DA"): [0.0000, 0.7151, 0.7151, 0.7179, 0.7397, 0.7343, 0.7343, 0.7343, 0.7297, 0.7427],
    ("smartfactory", "CR"): [0.0221, 0.0237, 0.0254, 0.0205, 0.0222, 0.0154, 0.0239, 0.0154, 0.0154, 0.0154],
    ("smartfactory", "CA"): [0.0000, 0.0000, 0.0000, 0.9275, 0.8067, 0.9385, 0.9384, 0.9384, 0.9412, 0.9417],
    ("smartfactory", "DMCO"): [0.6784, 0.6784, 0.6768, 0.7178, 0.8131, 0.8803, 0.9134, 0.9306, 0.9454, 0.9488],
    ("nasa", "DR"): [34.6802] * 10,
    ("nasa", "DA"): [20.6762, 16.7141, 15.1782, 12.1341, 11.1648, 10.8069, 10.5492, 10.5492, 10.5492, 9.3171],
    ("nasa", "CR"): [29.9315, 28.5640, 24.1622, 22.9370, 20.3804, 20.1366, 18.5961, 17.5085, 15.6952, 13.3376],
    ("nasa", "CA"): [11.9683, 3.7141, 3.7320, 3.7320, 2.8069, 2.8069, 2.6210, 1.9214, 1.9639, 1.8630],
    ("nasa", "DMCO"): [18.8011, 11.9055, 11.6682, 10.4543, 9.1942, 9.0713, 10.1281, 8.2889, 6.3613, 7.6229],
    ("soilmoisture", "DR"): [2.7162] * 10,
    ("soilmoisture", "DA"): [0.4298, 0.2923, 0.4744, 0.4491, 0.4970, 0.4970, 0.4970, 0.4970, 0.4970, 0.4380],
    ("soilmoisture", "CR"): [2.6490, 2.6132, 2.4157, 2.3583, 2.2986, 2.1737, 2.1119, 1.9095, 1.8339, 1.6847],
    ("soilmoisture", "CA"): [0.2490, 0.1026, 0.0986, 0.0750, 0.1158, 0.1099, 0.1248, 0.1248, 0.1246, 0.1246],
    ("soilmoisture", "DMCO"): [0.4210, 0.4159, 0.3636, 0.2771, 0.2773, 0.2274, 0.2363, 0.1933, 0.0815, 0.1038],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reproduced Table 2 CSV with paper values.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for _, row in df.iterrows():
        key = (str(row["dataset"]).lower(), str(row["method"]).upper())
        values = PAPER_TABLE2.get(key)
        if not values:
            continue
        point = int(row["budget_point"])
        paper = values[point - 1]
        reproduced = float(row["paper_test_metric"])
        rows.append(
            {
                "dataset": key[0],
                "method": key[1],
                "budget_point": point,
                "paper": paper,
                "reproduced": reproduced,
                "delta": reproduced - paper,
                "abs_delta": abs(reproduced - paper),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        summary = out.groupby(["dataset", "method"])["abs_delta"].agg(["mean", "max"]).reset_index()
        print(summary.to_string(index=False))
    output = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).stem + "_comparison.csv")
    out.to_csv(output, index=False)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
