#!/usr/bin/env python3
"""Small PyTorch baseline for 3-class semantic segmentation on NAOBF datasets.

Classes: 0 background, 1 soma, 2 neurite. Synthetic-only results measure pipeline
plumbing, not expected performance on real camera frames.
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def load_image(path):
    if path.suffix.lower() in ('.tif','.tiff'):
        a=tifffile.imread(path)
    else:
        a=np.asarray(Image.open(path))
    if a.ndim==3: a=a[...,:3].mean(axis=2)
    return a

class SegDataset(Dataset):
    def __init__(self, root, max_code):
        self.root=Path(root); self.max_code=float(max_code)
        self.images=sorted(p for p in (self.root/'images').iterdir() if p.suffix.lower() in ('.png','.tif','.tiff'))
        self.pairs=[]
        manifest_rows={}
        manifest=self.root/'manifest.csv'
        if manifest.exists():
            with open(manifest,newline='',encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    val=row.get('image','')
                    manifest_rows[Path(val).name if val else '']=row
        self.groups=[]
        for p in self.images:
            m=self.root/'masks'/'semantic'/(p.stem+'.png')
            if m.exists():
                self.pairs.append((p,m))
                row=manifest_rows.get(p.name,{})
                group=next((row[k] for k in ('culture_id','recording_id','group_id','sample_id') if row.get(k) not in (None,'')),p.stem)
                self.groups.append(str(group))
        if not self.pairs: raise RuntimeError(f'No image/semantic-mask pairs found under {root}')
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip,mp=self.pairs[idx]
        x=load_image(ip).astype(np.float32)/self.max_code
        y=np.asarray(Image.open(mp),dtype=np.uint8)
        x=np.clip(x,0,1)
        # Augment with label-preserving flips/rotations only; real optical augmentation comes later.
        if self.training and random.random()<.5: x=x[:,::-1]; y=y[:,::-1]
        if self.training and random.random()<.5: x=x[::-1,:]; y=y[::-1,:]
        k=random.randrange(4) if self.training else 0
        if k: x=np.rot90(x,k); y=np.rot90(y,k)
        # Pad odd camera dimensions to a multiple of 8 for the U-Net encoder/decoder.
        ph=(-x.shape[0])%8; pw=(-x.shape[1])%8
        if ph or pw:
            x=np.pad(x,((0,ph),(0,pw)),mode='reflect')
            y=np.pad(y,((0,ph),(0,pw)),mode='constant',constant_values=0)
        x=np.ascontiguousarray(x[None,:,:],dtype=np.float32)
        y=np.ascontiguousarray(y,dtype=np.int64)
        return torch.from_numpy(x),torch.from_numpy(y)
    training=False

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self,n_classes=3,base=24):
        super().__init__(); self.enc1=Block(1,base); self.enc2=Block(base,base*2); self.enc3=Block(base*2,base*4); self.pool=nn.MaxPool2d(2)
        self.mid=Block(base*4,base*8); self.up3=nn.ConvTranspose2d(base*8,base*4,2,2); self.dec3=Block(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,2); self.dec2=Block(base*4,base*2); self.up1=nn.ConvTranspose2d(base*2,base,2,2); self.dec1=Block(base*2,base); self.out=nn.Conv2d(base,n_classes,1)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool(a)); c=self.enc3(self.pool(b)); z=self.mid(self.pool(c))
        z=self.dec3(torch.cat([self.up3(z),c],1)); z=self.dec2(torch.cat([self.up2(z),b],1)); z=self.dec1(torch.cat([self.up1(z),a],1)); return self.out(z)


def dice_loss(logits,target,n_classes=3):
    p=torch.softmax(logits,dim=1); t=nn.functional.one_hot(target,num_classes=n_classes).permute(0,3,1,2).float()
    losses=[]
    for c in (1,2):
        pc=p[:,c]; tc=t[:,c]; losses.append(1-(2*(pc*tc).sum()+1)/(pc.sum()+tc.sum()+1))
    return sum(losses)/len(losses)

def evaluate(model,loader,device):
    model.eval(); conf=torch.zeros((3,3),dtype=torch.float64,device=device)
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1); y=y.to(device)
            for t in range(3):
                for p in range(3): conf[t,p]+=((y==t)&(pred==p)).sum()
    result={}
    for c,name in ((1,'soma'),(2,'neurite')):
        tp=conf[c,c].item(); fp=conf[:,c].sum().item()-tp; fn=conf[c,:].sum().item()-tp
        result[f'{name}_dice']=2*tp/max(2*tp+fp+fn,1)
        result[f'{name}_iou']=tp/max(tp+fp+fn,1)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data',required=True); ap.add_argument('--out',default='training_output'); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--val-fraction',type=float,default=.2); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--bit-depth',type=int,default=8); ap.add_argument('--workers',type=int,default=0); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data=SegDataset(a.data,(1<<a.bit_depth)-1)
    if len(data)<3: raise SystemExit('Need at least 3 image/mask pairs for a train/validation split.')
    # Split by recording/culture group if manifest.csv contains one; never split frames
    # from one real recording across train/validation. Synthetic samples each form a group.
    group_to_indices=defaultdict(list)
    for i,g in enumerate(data.groups): group_to_indices[g].append(i)
    groups=list(group_to_indices.keys()); random.Random(a.seed).shuffle(groups)
    class EvalDataset(Dataset):
        def __init__(self,base,indices): self.base=base; self.indices=indices
        def __len__(self): return len(self.indices)
        def __getitem__(self,i):
            old=self.base.training; self.base.training=False
            try: return self.base[self.indices[i]]
            finally: self.base.training=old
    if len(groups)>=2:
        n_val_groups=max(1,min(len(groups)-1,int(round(len(groups)*a.val_fraction))))
        val_groups=set(groups[:n_val_groups])
        val_indices=[i for i,g in enumerate(data.groups) if g in val_groups]
        train_indices=[i for i,g in enumerate(data.groups) if g not in val_groups]
        print(f'Group-aware split: {len(train_indices)} train / {len(val_indices)} validation images across {len(groups)} groups')
    else:
        print('WARNING: only one group was found in manifest; falling back to an image-level split. For real movies, add distinct recording_id/culture_id values.')
        inds=list(range(len(data))); random.Random(a.seed).shuffle(inds); nval=max(1,min(len(inds)-1,int(round(len(inds)*a.val_fraction))))
        val_indices=inds[:nval]; train_indices=inds[nval:]
    train_set=torch.utils.data.Subset(data,train_indices)
    data.training=True
    val_loader=DataLoader(EvalDataset(data,val_indices),batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    train_loader=DataLoader(train_set,batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=UNet().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    ntrain=len(train_indices); nval=len(val_indices)
    class_weights=torch.tensor([.35,1.0,2.0],device=device)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=[]; best=float('inf')
    print(f'Dataset: {len(data)} pairs | train={ntrain} val={nval} | device={device}')
    for ep in range(1,a.epochs+1):
        model.train(); running=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); logits=model(x)
            loss=nn.functional.cross_entropy(logits,y,weight=class_weights)+.55*dice_loss(logits,y)
            loss.backward(); opt.step(); running+=float(loss.item())
        val=evaluate(model,val_loader,device); train_loss=running/max(len(train_loader),1)
        row={'epoch':ep,'train_loss':train_loss,**val}; rows.append(row)
        print(f"epoch {ep:03d}/{a.epochs} loss={train_loss:.4f} soma_dice={val['soma_dice']:.3f} neurite_dice={val['neurite_dice']:.3f}")
        score=1-(val['soma_dice']+val['neurite_dice'])/2
        if score<best:
            best=score; torch.save({'model_state_dict':model.state_dict(),'model':'NAOBF_UNet','classes':{'0':'background','1':'soma','2':'neurite'},'input_normalization_max_code':(1<<a.bit_depth)-1,'training_data':str(Path(a.data).resolve()),'seed':a.seed,'epoch':ep},out/'best_model.pt')
    with open(out/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out/'NOTICE.txt').write_text('Metrics are measured on held-out synthetic fields only. They are not estimates of performance on real camera images. Split and evaluate by independent biological culture/recording when real annotated data become available.\n')
    print(f'Best checkpoint: {out / "best_model.pt"}'); print(f'Metrics: {out / "metrics.csv"}')
if __name__=='__main__': main()
