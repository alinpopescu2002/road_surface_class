"""Construiește data/raw/<clasa>/ din fișierele descărcate în data/downloads/."""

import argparse
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "data" / "downloads"
RAW = ROOT / "data" / "raw"

RDD_COUNTRIES = ["Czech", "India"]
CRACK_LABELS = {"D00", "D10", "D20"}
POTHOLE_LABEL = "D40"


def rdd_class(labels):
    if POTHOLE_LABEL in labels:
        return "potholes"
    if labels & CRACK_LABELS:
        return "cracks"
    if not labels:
        return "good_asphalt"
    return None


def collect_rdd():
    groups = {"good_asphalt": [], "cracks": [], "potholes": []}
    for country in RDD_COUNTRIES:
        zf = zipfile.ZipFile(DOWNLOADS / f"{country}.zip")
        names = set(zf.namelist())
        for xml in sorted(n for n in names if n.endswith(".xml") and "/train/" in n):
            labels = set(re.findall(r"<name>\s*(.*?)\s*</name>", zf.read(xml).decode("utf-8", "ignore")))
            cls = rdd_class(labels)
            img = xml.replace("/annotations/xmls/", "/images/").replace(".xml", ".jpg")
            if cls and img in names:
                groups[cls].append((zf, img))
    return groups


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if RAW.exists():
        shutil.rmtree(RAW)

    for cls, items in collect_rdd().items():
        items.sort(key=lambda t: t[1])
        rng.shuffle(items)
        items = items[: args.per_class]
        out = RAW / cls
        out.mkdir(parents=True)
        for zf, name in items:
            (out / Path(name).name).write_bytes(zf.read(name))
        print(f"{cls:15s}: {len(items)}")

    gravel = sorted((DOWNLOADS / "rscd_gravel").glob("*.jpg"))
    rng.shuffle(gravel)
    gravel = gravel[: args.per_class]
    out = RAW / "unpaved_gravel"
    out.mkdir(parents=True)
    for p in gravel:
        shutil.copy2(p, out / p.name)
    print(f"{'unpaved_gravel':15s}: {len(gravel)}")

    (RAW.parent / "splits.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
