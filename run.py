import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch
from src.scoring import FrozenScorer, attribute_queries, head_from, prediction_sets
from tools.get_model import fetch
from tools.pickscore_processor import fetch_processor
from tools.get_pairs import prepare
from src import probes

ROOT=Path(__file__).resolve().parent


def train(rows,out):
    head=head_from(rows)
    (out/"head.json").write_bytes(json.dumps(head,indent=2).encode("utf8"))
    restored=json.loads((out/"head.json").read_text())
    predictions=[{"pair":r["id"],"winner":r["winner"],"predictions":prediction_sets(r,restored),
                  "swapped_predictions":prediction_sets(r|{"features":list(reversed(r["features"]))},restored)}
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
    balanced=[]
    for method in rows[0]["predictions"]:
        total=sum(int(r["predictions"][method]==r["winner"])+int(r["swapped_predictions"][method]==1-r["winner"]) for r in rows)
        balanced.append(dict(method=method,correct_presentations=total,presentations=2*len(rows),
                             independent_pairs=len(rows),agreement=total/(2*len(rows))))
    with (out/"order-balanced.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,balanced[0].keys())
        writer.writeheader()
        writer.writerows(balanced)
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
    if (out/"probe-predictions.json").is_file():
        probes.report(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["demo","reproduce","evaluate","probes"])
    p.add_argument("--cache",type=Path,default=ROOT/".cache")
    p.add_argument("--out",type=Path)
    p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    args=p.parse_args()
    args.out=args.out or ROOT/("demo-output" if args.mode=="demo" else "results")
    args.out.mkdir(exist_ok=True,parents=True)
    if args.mode=="evaluate":
        return evaluate(args.out)
    if args.mode=="demo":
        train(json.loads((ROOT/"examples/features.json").read_text()),args.out)
        saved=ROOT/"results/probe-predictions.json"
        local=args.out/"probe-predictions.json"
        if saved.is_file() and saved.resolve()!=local.resolve():
            local.write_bytes(saved.read_bytes())
        if local.is_file():
            probes.report(args.out)
        return
    probe_prompts=[probes.prompt(case) for case in probes.CASES]
    if args.mode=="probes":
        torch.set_num_threads(2)
        judge=FrozenScorer(fetch(args.cache/"PickScore"),args.device,probe_prompts,fetch_processor(args.cache/"PickScore-processor"))
        return probes.score(judge,json.loads((args.out/"head.json").read_text()),args.out)
    rows=prepare(args.cache/"pairs")
    torch.set_num_threads(2)
    scorer=FrozenScorer(fetch(args.cache/"PickScore"),args.device,[r["prompt"] for r in rows]+probe_prompts,
                        fetch_processor(args.cache/"PickScore-processor"))
    # Persist each pair so an interrupted CPU extraction resumes at the next pair.
    cache_file=args.out/"features.partial.json"
    already=json.loads(cache_file.read_text()) if cache_file.exists() else []
    expected_ids=[r["id"] for r in rows]
    if [f["id"] for f in already]!=expected_ids[:len(already)]:
        raise ValueError("Partial extraction belongs to a different pair manifest")
    features=list(already)
    for r in rows[len(features):]:
        imgs=[]
        for file,sha in zip(r["images"],r["sha256"]):
            path=args.cache/"pairs"/file
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                raise ValueError(f"Image changed: {file}")
            with Image.open(path) as im:
                imgs.append(im.convert("RGB"))
        values=scorer.features(r["prompt"],imgs)
        features.append(dict(id=r["id"],split=r["split"],winner=r["winner"],attribute_kinds=[k for k,_ in attribute_queries(r["prompt"])],features=values.tolist()))
        tmp=cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(features))
        tmp.replace(cache_file)
        print(f"Scored {len(features)}/{len(rows)}",flush=True)
    (args.out/"features.json").write_text(json.dumps(features,indent=2))
    (args.out/"environment.json").write_text(json.dumps({"torch":str(torch.__version__),"device":args.device,
            "hardware":torch.cuda.get_device_name() if args.device=="cuda" else "cpu","precision":"float32","prompt_count":len(rows)},indent=2))
    train(features,args.out)
    probes.score(scorer,json.loads((args.out/"head.json").read_text()),args.out)
    cache_file.unlink(missing_ok=True)


if __name__=="__main__":
    main()
