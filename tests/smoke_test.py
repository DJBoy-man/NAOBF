#!/usr/bin/env python3
"""Small no-network smoke test for generation + annotations."""
from pathlib import Path
import json, subprocess, sys, tempfile
import numpy as np
from PIL import Image
import tifffile
ROOT=Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix='naobf_smoke_') as td:
        td=Path(td); out=td/'out'
        cmd=[sys.executable,str(ROOT/'src'/'naobf_simulator.py'),'--camera',str(ROOT/'config'/'camera_profile.example.json'),'--scene',str(ROOT/'config'/'scene_profile.example.json'),'--out',str(out),'--samples','3','--seed','1234']
        subprocess.run(cmd,check=True,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        files=sorted((out/'images').glob('*.png'))
        assert len(files)==3, f'expected 3 images, got {len(files)}'
        for p in files:
            image=np.asarray(Image.open(p)); mask=np.asarray(Image.open(out/'masks'/'semantic'/(p.stem+'.png')))
            assert image.shape==mask.shape, (p.name,image.shape,mask.shape)
            assert set(np.unique(mask)).issubset({0,1,2}), np.unique(mask)
            assert (mask==1).any(), f'no soma pixels: {p.name}'
            assert (mask==2).any(), f'no neurite pixels: {p.name}'
            soma=np.asarray(Image.open(out/'masks'/'soma'/(p.stem+'.png'))) > 0
            neur=np.asarray(Image.open(out/'masks'/'neurite'/(p.stem+'.png'))) > 0
            assert not np.any(soma&neur), f'soma/neurite masks overlap for {p.name}'
            inst=tifffile.imread(out/'masks'/'instances'/(p.stem+'.tif'))
            assert inst.shape==mask.shape
        meta=json.loads((out/'dataset_manifest.json').read_text())
        assert meta['sample_count']==3
    print('PASS: generated images, semantic labels, class masks, instance masks, and manifest are consistent.')
    print('NOTE: this verifies software plumbing only, not microscopy realism or biological validity.')
if __name__=='__main__': main()
