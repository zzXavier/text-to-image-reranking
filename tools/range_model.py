"""Resume a large fixed HF object in verifiable 16 MiB blocks if a single transfer stalls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import requests
from get_model import MODEL, REVISION


def download(out,workers):
    out.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors"
    r=requests.get(f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}",timeout=35)
    r.raise_for_status()
    entry=next(x for x in r.json() if x["path"]=="model.safetensors")
    length,expected=entry["size"],entry["lfs"]["oid"]
    block=16*1024*1024
    directory=out/"model-chunks"
    directory.mkdir(exist_ok=True)
    existing=out/"model.safetensors.part"
    def read_piece(i):
        start=i*block
        end=min(start+block,length)
        dest=directory/f"{i:04d}"
        if dest.exists() and dest.stat().st_size==end-start:
            return dest
        if existing.exists() and existing.stat().st_size>=end:
            with existing.open("rb") as f:
                f.seek(start)
                data=f.read(end-start)
        else:
            for attempt in range(5):
                try:
                    resp=requests.get(url,params={"block":i,"attempt":attempt},headers={"Range":f"bytes={start}-{end-1}"},timeout=(20,120))
                    resp.raise_for_status()
                    if resp.status_code!=206 or resp.headers.get("content-range","").split("/")[0]!=f"bytes {start}-{end-1}":
                        raise IOError("Range mismatch")
                    data=resp.content
                    break
                except (requests.RequestException,IOError):
                    if attempt==4:
                        raise
                    time.sleep(1+2**attempt)
        if len(data)!=end-start:
            raise IOError(f"short read at block {i}")
        tmp=dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return dest
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pieces=list(pool.map(read_piece,range((length+block-1)//block)))
    assembly=out/"model.verified-tmp"
    digest=hashlib.sha256()
    with assembly.open("wb") as dst:
        for p in pieces:
            data=p.read_bytes()
            dst.write(data)
            digest.update(data)
    if digest.hexdigest()!=expected:
        raise IOError("Model hash mismatch")
    assembly.replace(out/"model.safetensors")
    print("Verified",length,expected)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--workers",type=int,default=6)
    args=p.parse_args()
    download(args.out,args.workers)
