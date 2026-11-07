"""Descarcă RDD2022 (Czech, India) și un eșantion de imagini cu pietriș din RSCD în data/downloads/."""

import json
import random
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import USER_AGENT, HTTPRangeFile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "downloads"

RDD_URL = "https://ndownloader.figshare.com/files/38030910"
RDD_COUNTRIES = ["Czech", "India"]

RSCD_URL = "https://ndownloader.figshare.com/files/36625041"
RSCD_N_GRAVEL = 1500
SEED = 42


def download_rdd():
    zf = None
    for country in RDD_COUNTRIES:
        dst = OUT / f"{country}.zip"
        if dst.exists() and zipfile.is_zipfile(dst):
            continue
        if zf is None:
            zf = zipfile.ZipFile(HTTPRangeFile(RDD_URL, block_size=32 << 20))
        print(f"[RDD] {country}.zip")
        with zf.open(f"RDD2022/{country}.zip") as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, 32 << 20)


def rscd_gravel_index():
    index_file = OUT / "rscd_gravel_index.json"
    if index_file.exists():
        return json.loads(index_file.read_text())
    print("[RSCD] citesc lista de fișiere")
    zf = zipfile.ZipFile(HTTPRangeFile(RSCD_URL, block_size=8 << 20))
    rows = [
        [i.filename, i.header_offset, i.compress_size, i.compress_type, i.file_size]
        for i in zf.infolist()
        if i.filename.lower().endswith(".jpg") and "gravel" in i.filename
    ]
    index_file.write_text(json.dumps(rows))
    return rows


def fetch_member(row, out_dir):
    name, offset, csize, ctype, size = row
    dst = out_dir / name.rsplit("/", 1)[1]
    if dst.exists():
        return True
    for attempt in range(12):
        try:
            end = offset + 30 + len(name.encode()) + 256 + csize
            req = urllib.request.Request(
                RSCD_URL, headers={"Range": f"bytes={offset}-{end}", "User-Agent": USER_AGENT}
            )
            buf = urllib.request.urlopen(req, timeout=60).read()
            n_len, e_len = struct.unpack("<HH", buf[26:30])
            start = 30 + n_len + e_len
            data = buf[start: start + csize]
            if ctype == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            assert len(data) == size
            dst.write_bytes(data)
            time.sleep(0.2)
            return True
        except urllib.error.HTTPError as ex:
            time.sleep(min(60, 5 * (attempt + 1)) if ex.code == 429 else 2)  # 429 = prea multe cereri
        except Exception:
            time.sleep(2)
    print(f"[RSCD] eșec: {name}")
    return False


def download_rscd_gravel():
    rows = [
        r for r in rscd_gravel_index()
        if "/train/" in r[0] and ("dry-gravel" in r[0] or "wet-gravel" in r[0])
    ]
    random.Random(SEED).shuffle(rows)
    rows = rows[:RSCD_N_GRAVEL]
    out_dir = OUT / "rscd_gravel"
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        ok = sum(ex.map(lambda r: fetch_member(r, out_dir), rows))
    print(f"[RSCD] {ok}/{len(rows)} imagini")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    download_rdd()
    download_rscd_gravel()
