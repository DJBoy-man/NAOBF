#!/usr/bin/env python3
"""Automated NAOBF dataset QA and optional real-reference appearance screening.

Internal checks validate file/label integrity. Reference screening compares coarse
image features only; it cannot certify biological or optical realism.
"""
from __future__ import annotations
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from scipy.ndimage import sobel, laplace
from scipy.stats import wasserstein_distance

IMAGE_EXTS={'.png','.tif','.tiff','.jpg','.jpeg'}

def read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {'.tif','.tiff'}:
        a=tifffile.imread(path)
    else:
        a=np.asarray(Image.open(path))
    # RGB/RGBA
    if a.ndim == 3 and a.shape[-1] in (3,4):
        a=a[...,:3].astype(np.float32).mean(axis=-1)
    # If a TIFF is a stack, sample the middle plane rather than silently flattening it.
    elif a.ndim == 3:
        a=a[a.shape[0]//2]
    if a.ndim != 2:
        raise ValueError(f'Expected 2-D grayscale image (or stack/RGB TIFF); got shape {a.shape}')
    return a.astype(np.float32)

def normalized(a: np.ndarray, size: int=256) -> np.ndarray:
    lo,hi=np.percentile(a,[1,99])
    n=np.clip((a-lo)/max(float(hi-lo),1e-6),0,1)
    im=Image.fromarray(n.astype(np.float32),mode='F')
    im.thumbnail((size,size),Image.Resampling.LANCZOS)
    canvas=Image.new('F',(size,size),color=float(np.mean(n)))
    canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
    return np.asarray(canvas,dtype=np.float32)

def features(path: Path) -> dict:
    a=read_gray(path); n=normalized(a)
    gy=sobel(n,axis=0); gx=sobel(n,axis=1)
    grad=np.hypot(gx,gy); lap=laplace(n)
    # Share of spectral power in higher spatial frequencies; coarse texture indicator only.
    power=np.abs(np.fft.rfft2(n-n.mean()))**2
    fy=np.fft.fftfreq(n.shape[0])[:,None]; fx=np.fft.rfftfreq(n.shape[1])[None,:]
    high=(np.sqrt(fx*fx+fy*fy) >= .18)
    hf=float(power[high].sum()/max(float(power.sum()),1e-12))
    return {
        'file':path.name,'height':int(a.shape[0]),'width':int(a.shape[1]),'dtype_range_max':float(np.max(a)),
        'raw_mean':float(a.mean()),'raw_std':float(a.std()),
        'norm_mean':float(n.mean()),'norm_std':float(n.std()),
        'norm_p05':float(np.percentile(n,5)),'norm_p50':float(np.percentile(n,50)),'norm_p95':float(np.percentile(n,95)),
        'gradient_mean':float(grad.mean()),'gradient_std':float(grad.std()),
        'laplacian_std':float(lap.std()),'high_frequency_power_fraction':hf
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,help='Generated NAOBF dataset folder')
    ap.add_argument('--reference-dir',default=None,help='Optional folder of same-modality real microscope images')
    ap.add_argument('--out',default=None,help='Report folder (default: DATA/self_validation)')
    ap.add_argument('--max-reference',type=int,default=200)
    ap.add_argument('--max-synthetic',type=int,default=200)
    args=ap.parse_args()
    root=Path(args.data); out=Path(args.out) if args.out else root/'self_validation'; out.mkdir(parents=True,exist_ok=True)
    imgs=sorted(p for p in (root/'images').glob('*') if p.suffix.lower() in IMAGE_EXTS)
    if not imgs: raise SystemExit(f'No images found in {root / "images"}')
    errors=[]; warnings=[]; rows=[]; hashes={}; dims=set()
    for p in imgs:
        try:
            im=read_gray(p); dims.add(tuple(im.shape))
            if not np.isfinite(im).all(): errors.append(f'{p.name}: image contains NaN/Inf')
            if float(im.max()) <= float(im.min()): errors.append(f'{p.name}: image is constant/blank')
            digest=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
            if digest in hashes: warnings.append(f'Duplicate image content: {hashes[digest]} and {p.name}')
            else: hashes[digest]=p.name
        except Exception as e:
            errors.append(f'{p.name}: cannot read image: {e}'); continue
        stem=p.stem
        mask_paths={
            'semantic':root/'masks'/'semantic'/f'{stem}.png',
            'soma':root/'masks'/'soma'/f'{stem}.png',
            'neurite':root/'masks'/'neurite'/f'{stem}.png',
            'union':root/'masks'/'union'/f'{stem}.png',
            'instances':root/'masks'/'instances'/f'{stem}.tif'
        }
        missing=[k for k,v in mask_paths.items() if not v.exists()]
        if missing:
            errors.append(f'{p.name}: missing mask(s): {", ".join(missing)}'); continue
        sem=np.asarray(Image.open(mask_paths['semantic']))
        soma=np.asarray(Image.open(mask_paths['soma']))>0
        neurite=np.asarray(Image.open(mask_paths['neurite']))>0
        union=np.asarray(Image.open(mask_paths['union']))>0
        inst=tifffile.imread(mask_paths['instances'])
        arrays={'semantic':sem,'soma':soma,'neurite':neurite,'union':union,'instances':inst}
        shape=im.shape
        for nm,ma in arrays.items():
            if ma.shape != shape: errors.append(f'{p.name}: {nm} shape {ma.shape} != image {shape}')
        if sem.shape != shape: continue
        vals=set(int(x) for x in np.unique(sem))
        if not vals.issubset({0,1,2}): errors.append(f'{p.name}: invalid semantic classes {sorted(vals)}')
        if 1 not in vals: errors.append(f'{p.name}: no soma pixels')
        if 2 not in vals: errors.append(f'{p.name}: no neurite pixels')
        if soma.shape==shape and not np.array_equal(soma,sem==1): errors.append(f'{p.name}: soma mask disagrees with semantic labels')
        if neurite.shape==shape and not np.array_equal(neurite,sem==2): errors.append(f'{p.name}: neurite mask disagrees with semantic labels')
        if union.shape==shape and not np.array_equal(union,sem>0): errors.append(f'{p.name}: union mask disagrees with semantic foreground')
        if inst.shape==shape and not np.array_equal(inst>0,sem>0): errors.append(f'{p.name}: instance foreground disagrees with semantic foreground')
        rows.append({'image':p.name,'height':shape[0],'width':shape[1],
                     'soma_fraction':float(np.mean(sem==1)),'neurite_fraction':float(np.mean(sem==2)),
                     'foreground_fraction':float(np.mean(sem>0)),'instance_count':int(len(np.unique(inst))-1) if inst.shape==shape else None})
    if len(dims)>1: warnings.append(f'Images have mixed dimensions: {sorted(dims)}')
    if len(imgs)<10: warnings.append('Fewer than 10 images: dataset coverage is too small to assess diversity robustly.')
    report={
        'dataset':str(root.resolve()),'image_count':len(imgs),'checked_mask_pairs':len(rows),
        'internal_integrity':'FAIL' if errors else 'PASS','errors':errors,'warnings':warnings,
        'mean_soma_fraction':float(np.mean([r['soma_fraction'] for r in rows])) if rows else None,
        'mean_neurite_fraction':float(np.mean([r['neurite_fraction'] for r in rows])) if rows else None,
        'reference_screen':None,
        'interpretation':'Internal PASS means files and labels are internally consistent. It does not establish biological/optical realism.'
    }
    if rows:
        with open(out/'label_quality.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Optional appearance screen versus external real images of the SAME imaging modality.
    if args.reference_dir:
        rd=Path(args.reference_dir)
        refs=sorted(p for p in rd.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[:args.max_reference]
        syn=imgs[:args.max_synthetic]
        if not refs:
            report['reference_screen']={'status':'NOT_RUN','reason':f'No readable candidate images found in {rd}'}
            warnings.append('Reference directory supplied but no images were found.')
        else:
            allfeat=[]
            for kind,files in [('real',refs),('synthetic',syn)]:
                for p in files:
                    try:
                        r=features(p); r['set']=kind; allfeat.append(r)
                    except Exception as e: warnings.append(f'Feature extraction skipped {p}: {e}')
            metric_cols=['norm_mean','norm_std','norm_p05','norm_p50','norm_p95','gradient_mean','gradient_std','laplacian_std','high_frequency_power_fraction']
            rr=[r for r in allfeat if r['set']=='real']; ss=[r for r in allfeat if r['set']=='synthetic']
            diffs={}; large=[]
            for col in metric_cols:
                rv=np.asarray([r[col] for r in rr]); sv=np.asarray([r[col] for r in ss])
                rs=float(np.std(rv,ddof=1)) if len(rv)>1 else 0.0
                scale=max(rs,0.02)
                delta=float(np.median(sv)-np.median(rv)) if len(rv) and len(sv) else None
                effect=abs(delta)/scale if delta is not None else None
                wd=float(wasserstein_distance(rv,sv)) if len(rv) and len(sv) else None
                diffs[col]={'real_median':float(np.median(rv)) if len(rv) else None,
                            'synthetic_median':float(np.median(sv)) if len(sv) else None,
                            'median_difference':delta,'scaled_abs_difference':effect,'wasserstein_distance':wd}
                if effect is not None and effect>1.5: large.append(col)
            status='REVIEW_MISMATCH' if large else 'NO_LARGE_COARSE_MISMATCH_DETECTED'
            report['reference_screen']={
                'status':status,'real_image_count':len(rr),'synthetic_image_count':len(ss),
                'features':diffs,'features_with_large_standardized_gap':large,
                'warning':'This is a coarse screening test, not a validated realism score. A low gap does not prove realism. Real images must match modality, sample type, magnification, pixel scale and acquisition conditions.'
            }
            if len(rr)<10: warnings.append('Fewer than 10 real-reference images; comparison is unstable.')
            if large: warnings.append('Large coarse feature gaps detected; inspect contact sheets and calibrate acquisition settings.')
            with open(out/'appearance_features.csv','w',newline='',encoding='utf-8') as f:
                cols=['set','file']+metric_cols+['height','width','raw_mean','raw_std','dtype_range_max']
                w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(allfeat)
    report['warnings']=warnings
    (out/'self_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(f'\nReport: {out / "self_validation_report.json"}')
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
