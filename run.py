import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch
from src.scoring import FrozenScorer, head_from, prediction_sets
from tools.get_model import fetch
from tools.get_pairs import prepare

ROOT=Path(__file__).resolve().parent


def train(rows,out):
    head=head_from(rows)
    (out/"head.json").write_text(json.dumps(head,indent=2))
    restored=json.loads((out/"head.json").read_text())
    predictions=[{"pair":r["id"],"winner":r["winner"],"predictions":prediction_sets(r,restored)}
                 for r in rows if r["split"]=="validation"]
    if not predictions:
        raise ValueError("No held-out candidates")
    (out/"predictions.json").write_text(json.dumps(predictions,indent=2))
    evaluate(out)


def evaluate(out):
    rows=json.loads((out/"predictions.json").read_text())
    table=[]
    for method in rows[0]["predictions"]:
        correct=sum(r["predictions"][method]==r["winner"] for r in rows)
        table.append(dict(method=method,correct=correct,pairs=len(rows),accuracy=correct/len(rows)))
    with (out/"summary.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,table[0].keys())
        w.writeheader()
        w.writerows(table)
    sheet=Image.new("RGB",(520,35+35*len(table)),"white")
    d=ImageDraw.Draw(sheet)
    d.text((10,5),"Held-out human preference agreement",fill="black")
    for i,r in enumerate(table):
        y=32+i*35
        d.text((10,y),r["method"],fill="black")
        d.rectangle((155,y,155+250*r["accuracy"],y+15),fill="#326b87")
        d.text((421,y),f"{r['correct']}/{r['pairs']}",fill="black")
    sheet.save(out/"comparison.png")
    print((out/"summary.csv").read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["demo","reproduce","evaluate"])
    p.add_argument("--cache",type=Path,default=ROOT/".cache")
    p.add_argument("--out",type=Path,default=ROOT/"results")
    p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    if args.mode=="evaluate":
        return evaluate(args.out)
    if args.mode=="demo":
        return train(json.loads((ROOT/"examples/features.json").read_text()),args.out)
    rows=prepare(args.cache/"pairs")
    torch.set_num_threads(2)
    scorer=FrozenScorer(fetch(args.cache/"PickScore"),args.device)
    features=[]
    for r in rows:
        imgs=[]
        for file,sha in zip(r["images"],r["sha256"]):
            path=args.cache/"pairs"/file
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                raise ValueError(f"Image changed: {file}")
            with Image.open(path) as im:
                imgs.append(im.convert("RGB"))
        values=scorer.features(r["prompt"],imgs)
        features.append(dict(id=r["id"],split=r["split"],winner=r["winner"],features=values.tolist()))
    (args.out/"features.json").write_text(json.dumps(features,indent=2))
    train(features,args.out)


if __name__=="__main__":
    main()
