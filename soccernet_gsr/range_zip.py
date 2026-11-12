"""Leitura seletiva de um ZIP público por HTTP Range, sem baixar os frames."""
import io
import re

import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.cache_start, self.cache = 0, b""
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("Offset negativo.")
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        if size == 0 or self.pos >= self.size:
            return b""
        start, end = self.pos, min(self.pos + size, self.size)
        if not (self.cache_start <= start and end <= self.cache_start + len(self.cache)):
            fetch_end = min(max(end, start + 65536), self.size)
            # A URL de cada intervalo também distingue caches intermediários.
            sep = "&" if "?" in self.url else "?"
            url = f"{self.url}{sep}range_start={start}&range_end={fetch_end-1}"
            response = self.session.get(url, headers={"Range": f"bytes={start}-{fetch_end-1}"},
                                        stream=True, timeout=90)
            try:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status_code != 206 or not match or tuple(map(int, match.groups())) != (start, fetch_end-1, self.size):
                    raise IOError(f"Servidor não respeitou HTTP Range: {response.status_code}, {content_range}")
                chunks, total = [], 0
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > fetch_end - start:
                        raise IOError("Resposta excedeu o intervalo solicitado.")
                    chunks.append(chunk)
                data = b"".join(chunks)
                if len(data) != fetch_end - start:
                    raise IOError("Intervalo incompleto.")
                self.transferred += len(data)
                self.cache_start, self.cache = start, data
            finally:
                response.close()
        result = self.cache[start-self.cache_start:end-self.cache_start]
        self.pos = end
        return result

    def close(self):
        self.session.close()
        super().close()
