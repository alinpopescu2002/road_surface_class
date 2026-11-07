"""Citirea unei arhive ZIP de pe un server HTTP, fără descărcarea ei completă."""

import io
import time
import urllib.request

USER_AGENT = "road-surface-class-dataset-prep"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block_size=1 << 20):
        self.url = url
        self.block_size = block_size
        self.pos = 0
        self._cache = {}
        self.size = self._fetch_size()

    def _get(self, start, end, retries=5):
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise IOError(f"Status neașteptat: {r.status}")
                    return r.read(), r.headers.get("Content-Range")
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def _fetch_size(self):
        _, content_range = self._get(0, 0)
        return int(content_range.split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _block(self, idx):
        if idx not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            start = idx * self.block_size
            end = min(start + self.block_size, self.size) - 1
            self._cache[idx], _ = self._get(start, end)
        return self._cache[idx]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block_size:
            data, _ = self._get(self.pos, self.pos + n - 1)
            self.pos += len(data)
            return data
        out = bytearray()
        while len(out) < n:
            idx, off = divmod(self.pos, self.block_size)
            chunk = self._block(idx)[off: off + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
