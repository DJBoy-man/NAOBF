#!/usr/bin/env python3
"""Create a contact sheet of generated images and their semantic masks."""
import argparse, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data',required=True); ap.add_argument('--out',default=None); ap.add_argument('--count',type=int,default=12); ap.add_argument('--tile',type=int,default=320); a=ap.parse_args()
    root=Path(a.data); files=sorted(p for p in (root/'images').iterdir() if p.suffix.lower() in ('.png','.tif','.tiff'))[:a.count]
    if not files: raise SystemExit('No images found.')
    panels=[]
    for p in files:
        im=Image.open(p).convert('L').resize((a.tile,a.tile),Image.Resampling.LANCZOS)
        mask=Image.open(root/'masks'/'semantic'/(p.stem+'.png')).resize((a.tile,a.tile),Image.Resampling.NEAREST)
        # Encode class overlays: soma red, neurite green, blend over grayscale.
        gray=np.asarray(im).astype(np.float32); m=np.asarray(mask); rgb=np.stack([gray,gray,gray],axis=-1)
        rgb[...,0][m==1]=np.maximum(rgb[...,0][m==1],170); rgb[...,1][m==1]*=.45; rgb[...,2][m==1]*=.45
        rgb[...,1][m==2]=np.maximum(rgb[...,1][m==2],180); rgb[...,0][m==2]*=.45; rgb[...,2][m==2]*=.45
        panel=Image.fromarray(np.uint8(np.clip(rgb,0,255)),'RGB'); d=ImageDraw.Draw(panel); d.rectangle((0,0,a.tile,24),fill=(0,0,0)); d.text((7,5),p.name,fill='white'); panels.append(panel)
    cols=3; rows=math.ceil(len(panels)/cols); sheet=Image.new('RGB',(cols*a.tile,rows*a.tile),(24,24,24))
    for i,p in enumerate(panels): sheet.paste(p,((i%cols)*a.tile,(i//cols)*a.tile))
    out=Path(a.out) if a.out else root/'contact_sheet.png'; out.parent.mkdir(parents=True,exist_ok=True); sheet.save(out); print(out)
if __name__=='__main__': main()
