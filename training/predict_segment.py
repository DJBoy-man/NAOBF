#!/usr/bin/env python3
"""Run a trained NAOBF U-Net on one image or an image directory."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import tifffile
import torch
from torch import nn

# Reuse architecture from this directory without requiring package installation.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_unet import UNet, load_image


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--model',required=True); ap.add_argument('--input',required=True); ap.add_argument('--out',default='predicted_masks'); ap.add_argument('--bit-depth',type=int,default=8); a=ap.parse_args()
    ckpt=torch.load(a.model,map_location='cpu',weights_only=False); model=UNet(n_classes=3).eval(); model.load_state_dict(ckpt['model_state_dict'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    src=Path(a.input); files=[src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'))
    if not files: raise SystemExit(f'No supported input images found: {src}')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); max_code=float((1<<a.bit_depth)-1)
    for p in files:
        x=load_image(p).astype(np.float32)/max_code; h,w=x.shape; ph=(-h)%8; pw=(-w)%8
        if ph or pw: x=np.pad(x,((0,ph),(0,pw)),mode='reflect')
        tx=torch.from_numpy(np.ascontiguousarray(x[None,None])).to(device)
        with torch.no_grad(): pred=model(tx).argmax(1)[0].cpu().numpy()[:h,:w].astype(np.uint8)
        Image.fromarray(pred,mode='L').save(out/(p.stem+'_semantic.png'))
        Image.fromarray((pred==1).astype(np.uint8)*255,mode='L').save(out/(p.stem+'_soma.png'))
        Image.fromarray((pred==2).astype(np.uint8)*255,mode='L').save(out/(p.stem+'_neurite.png'))
        print('Saved masks for',p.name)
    print('Classes: 0 background, 1 soma, 2 neurite. These predictions need validation on real annotated images.')
if __name__=='__main__': main()
