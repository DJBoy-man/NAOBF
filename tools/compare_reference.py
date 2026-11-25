#!/usr/bin/env python3
"""Screen real reference and synthetic images using matched-size intensity/edge metrics.
Metrics are descriptive only and do not establish optical or biological validity.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import sobel, laplace
import tifffile
EXTS={'.tif','.tiff','.png','.jpg','.jpeg'}
def load(path,size):
    if path.suffix.lower() in ('.tif','.tiff'): a=tifffile.imread(path)
    else: a=np.asarray(Image.open(path))
    if a.ndim==3: a=a[...,:3].astype(np.float32).mean(axis=2)
    else: a=a.astype(np.float32)
    im=Image.fromarray(a.astype(np.float32),mode='F'); im.thumbnail((size,size),Image.Resampling.LANCZOS)
    can=Image.new('F',(size,size),color=float(np.mean(a))); can.paste(im,((size-im.width)//2,(size-im.height)//2)); raw=np.asarray(can,dtype=np.float32)
    lo,hi=np.percentile(raw,[1,99]); norm=np.clip((raw-lo)/max(float(hi-lo),1e-6),0,1)
    return raw,norm
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--real',required=True); ap.add_argument('--synthetic-dir',required=True); ap.add_argument('--out',default='comparison'); ap.add_argument('--size',type=int,default=512); ap.add_argument('--max-synthetic',type=int,default=12); a=ap.parse_args()
    real=Path(a.real); sd=Path(a.synthetic_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if not real.is_file(): raise SystemExit(f'Reference file not found: {real}')
    syn=sorted(p for p in sd.iterdir() if p.suffix.lower() in EXTS)[:a.max_synthetic]
    if not syn: raise SystemExit(f'No synthetic images found in {sd}')
    rows=[]; panels=[]
    for label,path in [('REAL_'+real.name,real)]+[(p.name,p) for p in syn]:
        raw,norm=load(path,a.size); rows.append({'image':label,'raw_mean':float(raw.mean()),'raw_std':float(raw.std()),'robust_contrast_std':float(norm.std()),'sobel_mean':float(np.hypot(sobel(norm,axis=0),sobel(norm,axis=1)).mean()),'laplacian_std':float(laplace(norm).std())})
        panel=Image.fromarray(np.uint8(norm*255),'L').convert('RGB'); d=ImageDraw.Draw(panel); d.rectangle((0,0,a.size,24),fill='black'); d.text((6,5),label[:75],fill='white'); panels.append(panel)
    with open(out/'metrics.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cols=2; nr=(len(panels)+cols-1)//cols; sheet=Image.new('RGB',(cols*a.size,nr*a.size),(30,30,30))
    for i,p in enumerate(panels): sheet.paste(p,((i%cols)*a.size,(i//cols)*a.size))
    sheet.save(out/'normalized_contact_sheet.png')
    (out/'LIMITATIONS.txt').write_text('All images are independently percentile-normalized and padded to the same size. Metrics help screen coarse contrast/edge statistics only; they do not validate realism. Only compare like modalities and comparable imaging conditions.\n')
    print(out/'metrics.csv'); print(out/'normalized_contact_sheet.png'); print('First comparison rows:')
    for row in rows[:4]: print(row)
if __name__=='__main__': main()
