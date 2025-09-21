# prepare_pairs.py
"""
Improved checker:
- Normalizes stems so mismatched sensor filenames still align.
- Prints raw vs normalized counts and a few example mappings.
- Writes a CSV report per class with raw->normalized mapping (reports/).
"""
from pathlib import Path
from collections import defaultdict
import csv

from config import DATA_ROOT, SENSORS, SENSOR_MODE_SUBDIR, CLASSES
from naming import normalize_stem

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_files(cdir: Path):
    return [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def stems_map(cdir: Path):
    """Return dict: normalized_key -> list[Path] (keep all paths that map to same key)"""
    m = defaultdict(list)
    for p in list_files(cdir):
        key = normalize_stem(p.stem)
        m[key].append(p)
    return m

def ensure_reports_dir(base: Path):
    rpt = base / "reports"
    rpt.mkdir(exist_ok=True)
    return rpt

def write_report_csv(rpt_dir: Path, cname: str, sensor_dirs, maps_per_sensor):
    out = rpt_dir / f"stem_report_{cname.replace(' ', '_')}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalized_key",
                    f"{SENSORS[0]}_raw_counts", f"{SENSORS[1]}_raw_counts", f"{SENSORS[2]}_raw_counts",
                    f"{SENSORS[0]}_examples", f"{SENSORS[1]}_examples", f"{SENSORS[2]}_examples"])
        all_keys = set().union(*[set(m.keys()) for m in maps_per_sensor])
        for k in sorted(all_keys):
            rows = []
            examples = []
            for m in maps_per_sensor:
                paths = m.get(k, [])
                rows.append(len(paths))
                examples.append("; ".join([p.name for p in paths[:3]]))
            w.writerow([k] + rows + examples)

def main():
    sensor_dirs = [DATA_ROOT / s / SENSOR_MODE_SUBDIR for s in SENSORS]
    for sd in sensor_dirs:
        if not sd.exists():
            raise FileNotFoundError(f"Missing: {sd}")

    rpt_dir = ensure_reports_dir(Path("."))

    print("Verifying class-wise triplets using normalized stems:\n")
    grand_total = 0
    for cname in CLASSES:
        maps_per_sensor = []
        counts_raw = []
        for sd in sensor_dirs:
            cdir = sd / cname
            if not cdir.exists():
                raise FileNotFoundError(f"Missing class dir: {cdir}")
            files = list_files(cdir)
            counts_raw.append(len(files))
            maps_per_sensor.append(stems_map(cdir))

        common_keys = set(maps_per_sensor[0].keys()) & set(maps_per_sensor[1].keys()) & set(maps_per_sensor[2].keys())
        aligned = 0
        # We count a key as aligned if each sensor has exactly 1 path for that key;
        # if >1 due to collisions, we still show as aligned but warn later in data.py.
        for k in common_keys:
            aligned += min(len(maps_per_sensor[0][k]), 1) * min(len(maps_per_sensor[1][k]), 1) * min(len(maps_per_sensor[2][k]), 1)

        print(f"{cname:>22}: raw counts A/P/H = {counts_raw[0]}/{counts_raw[1]}/{counts_raw[2]} | aligned keys = {len(common_keys)} | approx aligned samples = {aligned}")
        grand_total += aligned

        # CSV with detailed mapping to help diagnose
        write_report_csv(rpt_dir, cname, sensor_dirs, maps_per_sensor)

    print(f"\nApprox total aligned samples across all classes: {grand_total}")
    print("Details saved under ./reports/stem_report_<Class>.csv")
    print("\nIf aligned ~0, inspect a CSV to see which normalized keys disagree. "
          "If collisions (same key maps to many files), consider renaming or we will pick the first per key.")

if __name__ == "__main__":
    main()
