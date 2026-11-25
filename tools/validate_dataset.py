#!/usr/bin/env python3
"""Check image/label alignment and report class balance for a NAOBF dataset."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile

EXTS={'.png','.tif','.tiff'}
def read_gray(path):
    if path.suffix.lower() in ('.tif','.tiff'): a=tifffile.imread(path)
    else: a=np.asarray(Image.open(path))
    if a.ndim==3: a=a[...,0]
    return a

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data',required=True); ap.add_argument('--out',default=None); a=ap.parse_args()
    root=Path(a.data); images=sorted(p for p in (root/'images').iterdir() if p.suffix.lower() in EXTS)
    if not images: raise SystemExit(f'No images found in {root / "images"}')
    rows=[]; errors=[]
    for p in images:
        mpath=root/'masks'/'semantic'/(p.stem+'.png')
        if not mpath.exists(): errors.append(f'Missing semantic mask for {p.name}'); continue
        im=read_gray(p); mask=np.asarray(Image.open(mpath))
        if im.shape[:2]!=mask.shape: errors.append(f'Shape mismatch for {p.name}: image={im.shape}, mask={mask.shape}'); continue
        vals,cnts=np.unique(mask,return_counts=True); label_counts={int(k):int(v) for k,v in zip(vals,cnts)}
        if not set(vals).issubset({0,1,2}): errors.append(f'Unexpected semantic values in {mpath.name}: {vals.tolist()}')
        if 1 not in vals: errors.append(f'No soma pixels in {mpath.name}')
        if 2 not in vals: errors.append(f'No neurite pixels in {mpath.name}')
        rows.append({'image':p.name,'height':im.shape[0],'width':im.shape[1],'dtype':str(im.dtype),'image_min':int(im.min()),'image_max':int(im.max()),'image_mean':float(im.mean()),'image_std':float(im.std()),'background_pixels':label_counts.get(0,0),'soma_pixels':label_counts.get(1,0),'neurite_pixels':label_counts.get(2,0),'soma_fraction':label_counts.get(1,0)/mask.size,'neurite_fraction':label_counts.get(2,0)/mask.size})
    out=Path(a.out) if a.out else root/'validation'; out.mkdir(parents=True,exist_ok=True)
    if rows:
        with open(out/'validation_metrics.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={'image_count':len(images),'valid_pairs':len(rows),'error_count':len(errors),'errors':errors,'overall_soma_fraction':float(sum(r['soma_pixels'] for r in rows)/max(sum(r['height']*r['width'] for r in rows),1)),'overall_neurite_fraction':float(sum(r['neurite_pixels'] for r in rows)/max(sum(r['height']*r['width'] for r in rows),1)),'warning':'Dataset checks confirm file/label consistency only; they do not validate biological or optical realism.'}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print('Metrics:',out/'validation_metrics.csv')
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
