"""Optional CLI: train + evaluate for one city and print the accuracy table.

    python train.py "Mumbai"

Use the printed numbers in your README / resume (they are real, measured on unseen data).
"""
import json
import sys
from pathlib import Path

from model import train_city
from weather_utils import fetch_training_history, search_cities


def main(city: str) -> None:
    matches = search_cities(city, count=1)
    if not matches:
        sys.exit(f"City '{city}' not found.")
    place = matches[0]
    print(f"Training for {place['label']} ...")
    df, tz = fetch_training_history(place["lat"], place["lon"])
    bundle = train_city(df)

    print(f"\nHistory: {bundle['history'][0]} -> {bundle['history'][1]}  (timezone {tz})")
    print(f"Train rows: {bundle['n_train']:,} | Test rows: {bundle['n_test']:,} (forecasts made after {bundle['split']})")
    print("\nMean Absolute Error in °C by lead time (lower is better):\n")
    print(bundle["metrics"].round(2).to_string(index=False))

    print(f"\nFeature selection: kept {bundle['k']} of {len(bundle['feature_table'])} features (SelectKBest, F-test)\n")
    print(bundle["feature_table"].to_string(index=False))

    out = Path(__file__).parent / "reports"
    out.mkdir(exist_ok=True)
    fname = out / f"metrics_{place['label'].split(',')[0].lower().replace(' ', '_')}.json"
    fname.write_text(json.dumps(bundle["metrics"].round(3).to_dict(orient="records"), indent=2))
    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Bengaluru")
