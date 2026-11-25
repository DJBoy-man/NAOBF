#!/usr/bin/env python3
"""Normalize externally annotated image/mask pairs to NAOBF's 0/1/2 semantic dataset schema.

Expected raw image files and single-channel semantic masks with matching stems.
Class values are supplied explicitly to avoid guessing annotation semantics.
"""
import argparse, csv, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
EXTS={'.png','.tif','.tiff','.jpg','.jpeg'}
def read_mask(p):
    a=tifffile.imread(p) if p.suffix.lower() in ('.tif','.tiff') else np.asarray(Image.open(p))
    if a.ndim==3: raise ValueError(f'{p} is RGB/multichannel; convert the label map to a single-channel class image first')
    return a

def parse_values(s): return {int(x.strip()) for x in s.split(',') if x.strip()}
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--images',required=True); ap.add_argument('--masks',required=True); ap.add_argument('--out',required=True); ap.add_argument('--background-values',default='0'); ap.add_argument('--soma-values',default='1'); ap.add_argument('--neurite-values',default='2'); ap.add_argument('--copy-images',action='store_true',help='Copy original image bytes instead of normalizing to PNG/TIFF')
    a=ap.parse_args(); idir=Path(a.images); mdir=Path(a.masks); out=Path(a.out)
    bg,soma,neur=parse_values(a.background_values),parse_values(a.soma_values),parse_values(a.neurite_values)
    if not bg or not soma or not neur: ap.error('Supply non-empty value sets for all three classes')
    if (bg&soma) or (bg&neur) or (soma&neur): ap.error('Class value sets must be disjoint')
    imgs=sorted(p for p in idir.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    if not imgs: raise SystemExit(f'No source images in {idir}')
    for sub in ('images','masks/semantic','masks/soma','masks/neurite','masks/union'): (out/sub).mkdir(parents=True,exist_ok=True)
    rows=[]; missing=[]
    for ix,p in enumerate(imgs):
        mp=next((q for q in (mdir/(p.stem+ext) for ext in ('.png','.tif','.tiff')) if q.exists()),None)
        if mp is None: missing.append(p.name); continue
        mask=read_mask(mp); raw=tifffile.imread(p) if p.suffix.lower() in ('.tif','.tiff') else np.asarray(Image.open(p))
        if raw.ndim==3: raw=raw[...,:3].mean(axis=2).astype(np.uint8)
        if raw.shape[:2]!=mask.shape: raise SystemExit(f'Shape mismatch: {p} {raw.shape} vs {mp} {mask.shape}')
        labels=np.zeros(mask.shape,dtype=np.uint8); unknown=set(int(v) for v in np.unique(mask))-bg-soma-neur
        if unknown: raise SystemExit(f'Unmapped annotation values in {mp.name}: {sorted(unknown)[:20]}; specify all labels explicitly')
        for v in soma: labels[mask==v]=1
        for v in neur: labels[mask==v]=2
        stem=f'real_{ix:05d}'
        ext=p.suffix.lower() if p.suffix.lower() in ('.tif','.tiff','.png') else '.png'
        dest=out/'images'/(stem+ext)
        if a.copy_images: shutil.copy2(p,dest)
        elif ext in ('.tif','.tiff'): tifffile.imwrite(dest,raw,photometric='minisblack')
        else: Image.fromarray(np.asarray(raw,dtype=np.uint8)).save(dest)
        Image.fromarray(labels,mode='L').save(out/'masks'/'semantic'/(stem+'.png'))
        Image.fromarray((labels==1).astype(np.uint8)*255,mode='L').save(out/'masks'/'soma'/(stem+'.png'))
        Image.fromarray((labels==2).astype(np.uint8)*255,mode='L').save(out/'masks'/'neurite'/(stem+'.png'))
        Image.fromarray((labels>0).astype(np.uint8)*255,mode='L').save(out/'masks'/'union'/(stem+'.png'))
        group= p.parent.relative_to(idir).as_posix() if p.parent!=idir else p.stem
        rows.append({'image':str(dest.relative_to(out)),'mask':f'masks/semantic/{stem}.png','source_image':str(p),'source_mask':str(mp),'group_id':group,'soma_pixels':int((labels==1).sum()),'neurite_pixels':int((labels==2).sum())})
        if (len(rows))%25==0: print(f'Prepared {len(rows)} labeled pairs')
    if not rows: raise SystemExit('No matching image/mask pairs found')
    with open(out/'manifest.csv','w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'Prepared {len(rows)} real labeled pairs under {out.resolve()}')
    if missing: print(f'Skipped {len(missing)} images without a matching mask; first few: {missing[:10]}')
    print('Class map: 0 background, 1 soma, 2 neurite. Confirm your source mask label mapping before training.')
if __name__=='__main__': main()
