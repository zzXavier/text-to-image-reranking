"""Known-answer probes separate from the natural Pick-a-Pic human preference dataset."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

COLORS={"red":(188,43,52),"green":(45,137,66),"blue":(49,96,183)}
CASES=[
    (2,"red","mug","left","blue","bottle"),
    (3,"green","bottle","right","red","mug"),
    (2,"blue","bottle","left","green","mug"),
    (3,"red","mug","right","blue","bottle"),
]
ERRORS=["correct","count","color","position","object"]


def prompt(case):
    n,color,subject,side,other_color,other=case
    return f"{['two','three'][n-2]} {color} {subject}s {side} of one {other_color} {other}"


def draw_item(d,x,y,color,subject):
    fill=COLORS[color]
    if subject=="mug":
        d.ellipse((x+7,y-10,x+23,y+9),outline=fill,width=5)
        d.rounded_rectangle((x-16,y-14,x+13,y+16),radius=5,fill=fill)
        d.line((x-13,y-10,x+10,y-10),fill=(241,239,236),width=2)
    else:
        d.rounded_rectangle((x-10,y-12,x+10,y+20),radius=5,fill=fill)
        d.rectangle((x-5,y-20,x+5,y-10),fill=fill)
        d.rectangle((x-5,y-23,x+5,y-20),fill=(65,65,65))


def render(case,error):
    n,color,subject,side,other_color,other=case
    if error=="count":
        n-=1
    elif error=="color":
        color=next(v for v in COLORS if v not in (color,other_color))
    elif error=="position":
        side="left" if side=="right" else "right"
    elif error=="object":
        subject=other
    canvas=Image.new("RGB",(224,224),(244,242,238))
    d=ImageDraw.Draw(canvas)
    d.line((12,154,212,154),fill=(208,204,200),width=2)
    center=69 if side=="left" else 155
    other_x=184 if side=="left" else 39
    for i in range(n):
        draw_item(d,round(center+(i-(n-1)/2)*38),131,color,subject)
    draw_item(d,other_x,131,other_color,other)
    return canvas


def score(scorer,head,out):
    rows=[]
    for i,case in enumerate(CASES):
        names=np.random.default_rng(4200+i).permutation(ERRORS).tolist()
        x=scorer.features(prompt(case),[render(case,j) for j in names])
        rows.append(dict(prompt=prompt(case),candidates=names,whole=x[:,0].tolist(),
                         attributes=x[:,1].tolist(),preference=(x@head["weights"]).tolist()))
    (out/"probe-predictions.json").write_text(json.dumps(rows,indent=2))
    return report(out)


def report(out):
    rows=json.loads((out/"probe-predictions.json").read_text())
    table=[]
    for method in ("whole","attributes","preference"):
        hits=sum(r["candidates"][int(np.argmax(r[method]))]=="correct" for r in rows)
        table.append(dict(method=method,exact_top1=hits,prompts=len(rows),candidates_per_prompt=5))
    with (out/"probe-summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,table[0].keys())
        w.writeheader()
        w.writerows(table)
    image=Image.new("RGB",(5*138,168),(255,255,255))
    d=ImageDraw.Draw(image)
    d.text((3,2),prompt(CASES[0]),fill=(10,10,10))
    for j,error in enumerate(ERRORS):
        image.paste(render(CASES[0],error).resize((136,112)),(j*138,24))
        d.text((j*138+3,143),error,fill=(15,15,15))
    image.save(out/"probe-example.png")
    print((out/"probe-summary.csv").read_text())
