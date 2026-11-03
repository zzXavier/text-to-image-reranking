"""Fetch selected deflated members from the pinned public Pick-a-Pic embedding ZIP."""
import io
import pathlib
import sys
import time
import zipfile

import requests

SHA = "5b38c5808fac0bc9641ba13121ca8a6313bb81c0"
URL = ("https://huggingface.co/datasets/aniketr/pickapic-embeds/resolve/" + SHA
       + "/embeddings/pickapic_v1-pickscore-v1.zip")
SIZE = 7111259781


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, SIZE)[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = min(SIZE, self.pos+n) if n>=0 else SIZE
        if end<=self.pos:
            return b""
        start=self.pos
        for attempt in range(3):
            try:
                # Query makes range-specific signed CDN redirects cache-safe.
                r=requests.get(URL,params={"part":f"{start}-{end}"},headers={"Range":f"bytes={start}-{end-1}"},timeout=90)
                r.raise_for_status()
                if r.status_code!=206 or len(r.content)!=end-start:
                    raise ValueError(f"Unexpected range {r.status_code}, {r.headers.get('Content-Range')}")
                self.pos=end
                return r.content
            except requests.RequestException:
                if attempt==2:
                    raise
                time.sleep(2**attempt)


if __name__ == "__main__":
    with zipfile.ZipFile(RangeFile()) as z:
        if len(sys.argv)==1:
            for i in z.infolist():
                print(i.filename,i.compress_size,i.file_size)
        else:
            suffix=sys.argv[1]
            name=next(n for n in z.namelist() if n.endswith(suffix))
            pathlib.Path(sys.argv[2]).write_bytes(z.read(name))
