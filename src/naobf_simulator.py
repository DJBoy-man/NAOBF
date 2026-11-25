#!/usr/bin/env python3
"""NAOBF: configurable, NAOMi-inspired synthetic brightfield training-data generator.

It generates procedural morphology + approximate transmitted-light rendering + exact
labels for the generated geometry. It is NOT a calibrated optical/neural-activity model.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter, gaussian_laplace, shift as ndi_shift
import tifffile


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_gray(path: Path, image: np.ndarray, bit_depth: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth <= 8:
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(path)
    else:
        tifffile.imwrite(path.with_suffix(".tif"), np.asarray(image, dtype=np.uint16), photometric="minisblack")


def bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return ((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)


def ellipse_points(cx, cy, rx, ry, angle, rng, count=40):
    points=[]; ca,sa=math.cos(angle),math.sin(angle)
    for th in np.linspace(0, 2*math.pi, count, endpoint=False):
        r=float(rng.normal(1.0, .055))
        x,y=rx*r*math.cos(th), ry*r*math.sin(th)
        points.append((cx+ca*x-sa*y, cy+sa*x+ca*y))
    return points


def draw_tapered_path(draw: ImageDraw.ImageDraw, pts: np.ndarray, w0: float, w1: float, scale: int):
    """Rasterize a tapered process with connected circular joins."""
    n=len(pts)
    for j in range(n-1):
        t=j/max(1,n-1)
        width=max(.55, (w0*(1-t)+w1*t)*scale)
        x0,y0=pts[j]*scale; x1,y1=pts[j+1]*scale
        iw=max(1, int(round(width)))
        draw.line((int(round(x0)),int(round(y0)),int(round(x1)),int(round(y1))), fill=255, width=iw)
        r=width/2
        draw.ellipse((x1-r,y1-r,x1+r,y1+r),fill=255)


def create_cell_masks(rng, width, height, ss, mpp, scene_cfg, cell_id):
    W,H=width*ss,height*ss
    soma_img=Image.new("L",(W,H),0); neurite_img=Image.new("L",(W,H),0); nucleus_img=Image.new("L",(W,H),0); granule_img=Image.new("L",(W,H),0)
    sd=ImageDraw.Draw(soma_img); nd=ImageDraw.Draw(neurite_img); nucd=ImageDraw.Draw(nucleus_img); gd=ImageDraw.Draw(granule_img)
    # Place somata away from exact borders but allow neurites to leave the frame.
    cx=float(rng.uniform(.04*width,.96*width)); cy=float(rng.uniform(.04*height,.96*height))
    d_um=float(rng.uniform(scene_cfg["soma_diameter_um_min"],scene_cfg["soma_diameter_um_max"]))
    rx=max(3.5,d_um/(2*mpp)*float(rng.uniform(.72,1.08)))
    ry=max(3.5,d_um/(2*mpp)*float(rng.uniform(.56,.90)))
    angle=float(rng.uniform(0,math.pi))
    poly=ellipse_points(cx,cy,rx,ry,angle,rng,count=int(rng.integers(28,46)))
    sd.polygon([(int(x*ss),int(y*ss)) for x,y in poly],fill=255)
    # Nucleus and small phase-dense inclusions are rendered optically but stay within the soma class.
    ncx=cx+float(rng.normal(0,rx*.12)); ncy=cy+float(rng.normal(0,ry*.12))
    nrx=rx*float(rng.uniform(.20,.34)); nry=ry*float(rng.uniform(.20,.36))
    nucpoly=ellipse_points(ncx,ncy,nrx,nry,angle+float(rng.normal(0,.2)),rng,count=28)
    nucd.polygon([(int(x*ss),int(y*ss)) for x,y in nucpoly],fill=int(rng.integers(135,220)))
    for _ in range(int(rng.integers(3,9))):
        gx=cx+float(rng.normal(0,rx*.42)); gy=cy+float(rng.normal(0,ry*.42)); gr=float(rng.uniform(.35,1.1))
        gd.ellipse(((gx-gr)*ss,(gy-gr)*ss,(gx+gr)*ss,(gy+gr)*ss),fill=int(rng.integers(90,180)))
    # Add low-amplitude cell-body inclusions to its optical texture (not to its label).
    n_roots=int(rng.integers(scene_cfg["neurite_count_min"],scene_cfg["neurite_count_max"]+1))
    for root in range(n_roots):
        th=float(rng.uniform(0,2*math.pi))
        radial=float(rng.uniform(.78,1.02))
        lx,ly=rx*radial*math.cos(th),ry*radial*math.sin(th)
        ca,sa=math.cos(angle),math.sin(angle)
        p0=np.array([cx+ca*lx-sa*ly,cy+sa*lx+ca*ly],dtype=np.float32)
        outward=np.array([math.cos(th+angle), math.sin(th+angle)],dtype=np.float32)
        # Longitudinal direction is biased outward from the soma; curvature is stochastic.
        direction_angle=math.atan2(outward[1],outward[0])+float(rng.normal(0,.38))
        direction=np.array([math.cos(direction_angle),math.sin(direction_angle)],dtype=np.float32)
        length_um=float(rng.uniform(scene_cfg["neurite_length_um_min"],scene_cfg["neurite_length_um_max"]))
        length_px=length_um/mpp*float(rng.uniform(.62,1.0))
        bend_vec=np.array([-direction[1],direction[0]],dtype=np.float32)*float(rng.normal(0,.31*length_px))
        p3=p0+direction*length_px+bend_vec
        p1=p0+direction*length_px*.24+bend_vec*float(rng.uniform(.08,.28))
        p2=p0+direction*length_px*.70+bend_vec*float(rng.uniform(.62,1.02))
        pts=bezier(p0,p1,p2,p3,n=max(35,int(length_px*1.15)))
        w0=float(rng.uniform(scene_cfg["neurite_width_um_start_min"],scene_cfg["neurite_width_um_start_max"]))/mpp
        wt=float(rng.uniform(scene_cfg["neurite_tip_width_um_min"],scene_cfg["neurite_tip_width_um_max"]))/mpp
        draw_tapered_path(nd,pts,w0,wt,ss)
        # One or two tapered side branches attached to the parent path.
        for _ in range(int(rng.integers(0,3))):
            if rng.random() > float(scene_cfg["branch_probability"]):
                continue
            frac=float(rng.uniform(.28,.78)); index=min(len(pts)-2,max(2,int(frac*(len(pts)-1))))
            start=pts[index]
            tangent=pts[index+1]-pts[index-1]
            norm=max(float(np.linalg.norm(tangent)),1e-6); tangent=tangent/norm
            theta=math.atan2(float(tangent[1]),float(tangent[0]))+float(rng.choice([-1,1]))*float(rng.uniform(.35,1.0))
            d2=np.array([math.cos(theta),math.sin(theta)],dtype=np.float32)
            blen=length_px*float(rng.uniform(.18,.48))
            bend2=np.array([-d2[1],d2[0]],dtype=np.float32)*float(rng.normal(0,.24*blen))
            q3=start+d2*blen+bend2
            q1=start+d2*blen*.22+bend2*.10
            q2=start+d2*blen*.68+bend2*.80
            qpts=bezier(start,q1,q2,q3,n=max(18,int(blen*.9)))
            draw_tapered_path(nd,qpts,max(.55,w0*.58),max(.45,wt*.7),ss)
    soma=np.asarray(soma_img,dtype=np.uint8)>0
    neurite=np.asarray(neurite_img,dtype=np.uint8)>0
    nucleus=np.asarray(nucleus_img,dtype=np.uint8)
    granules=np.asarray(granule_img,dtype=np.uint8)
    # No neurite target inside soma pixels: the semantic classes remain disjoint.
    neurite &= ~soma
    return soma,neurite,nucleus,granules,{"cell_id":cell_id,"center_x_px":cx,"center_y_px":cy,"soma_diameter_um":d_um,"soma_rx_px":rx,"soma_ry_px":ry,"base_neurites":n_roots}


def downsample_float(a: np.ndarray, size):
    im=Image.fromarray(np.asarray(a,dtype=np.float32), mode="F")
    return np.asarray(im.resize(size,Image.Resampling.LANCZOS),dtype=np.float32)



def read_swc(path: Path):
    """Read a standard SWC morphology: id type x y z radius parent."""
    nodes={}
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            bits=line.split()
            if len(bits)<7: continue
            try:
                nid=int(bits[0]); typ=int(bits[1]); x,y,z,r=float(bits[2]),float(bits[3]),float(bits[4]),abs(float(bits[5])); parent=int(bits[6])
            except ValueError: continue
            nodes[nid]={"id":nid,"type":typ,"xyz":np.array([x,y,z],dtype=np.float32),"radius":max(r,.15),"parent":parent}
    # Only retain nodes with valid parent links or an SWC root marker.
    nodes={k:v for k,v in nodes.items() if v["parent"]==-1 or v["parent"] in nodes}
    if len(nodes)<3: raise ValueError(f"{path} has fewer than 3 valid SWC nodes")
    return nodes


def create_cell_masks_from_swc(rng,width,height,ss,mpp,scene_cfg,cell_id,swc_path,projection="auto"):
    """Project a traced SWC tree into a 2-D mask and render a configurable soma.

    Coordinates are rescaled to a target projected process extent (documented as a
    morphology prior, not the original neuron's measured physical size).
    """
    nodes=read_swc(swc_path)
    type1=[n for n in nodes.values() if n["type"]==1]
    if type1: soma_xyz=np.mean([n["xyz"] for n in type1],axis=0)
    else:
        roots=[n for n in nodes.values() if n["parent"]==-1]
        soma_xyz=(roots[0] if roots else next(iter(nodes.values())))["xyz"]
    if projection not in ("xy","xz","yz","auto"): raise ValueError("projection must be xy, xz, yz or auto")
    axes={"xy":(0,1),"xz":(0,2),"yz":(1,2)}
    if projection=="auto":
        pairs=[("xy",(0,1)),("xz",(0,2)),("yz",(1,2))]
        def spread(pair):
            a,b=pair[1]
            coords=np.array([n["xyz"][[a,b]] for n in nodes.values()])
            return float(np.ptp(coords[:,0])*np.ptp(coords[:,1]))
        projection,max_axes=max(pairs,key=spread); ax=axes[projection]
    else: ax=axes[projection]
    coords=np.array([n["xyz"][list(ax)] for n in nodes.values()],dtype=np.float32)
    center3=soma_xyz[list(ax)].astype(np.float32)
    radial=np.sqrt(np.sum((coords-center3)**2,axis=1))
    extent=max(float(np.percentile(radial,98)),1.0)
    target_extent=float(rng.uniform(scene_cfg.get("swc_target_extent_um_min",75.0),scene_cfg.get("swc_target_extent_um_max",200.0)))
    scale_um=target_extent/extent
    # Each traced tree is placed and rotated like a new field-of-view realization.
    cx=float(rng.uniform(.04*width,.96*width)); cy=float(rng.uniform(.04*height,.96*height)); rot=float(rng.uniform(0,2*math.pi))
    ca,sa=math.cos(rot),math.sin(rot)
    xy_map={}
    for nid,node in nodes.items():
        delta=(node["xyz"][list(ax)]-center3)*scale_um/max(mpp,1e-6)
        x=delta[0]*ca-delta[1]*sa+cx; y=delta[0]*sa+delta[1]*ca+cy
        xy_map[nid]=np.array([x,y],dtype=np.float32)
    W,H=width*ss,height*ss
    soma_img=Image.new("L",(W,H),0); neurite_img=Image.new("L",(W,H),0); nucleus_img=Image.new("L",(W,H),0); granule_img=Image.new("L",(W,H),0)
    sd=ImageDraw.Draw(soma_img); nd=ImageDraw.Draw(neurite_img); nucd=ImageDraw.Draw(nucleus_img); gd=ImageDraw.Draw(granule_img)
    d_um=float(rng.uniform(scene_cfg["soma_diameter_um_min"],scene_cfg["soma_diameter_um_max"]))
    rx=max(4,d_um/(2*mpp)*float(rng.uniform(.75,1.05))); ry=max(4,rx*float(rng.uniform(.67,.96))); angle=float(rng.uniform(0,math.pi))
    poly=ellipse_points(cx,cy,rx,ry,angle,rng,count=42)
    sd.polygon([(int(x*ss),int(y*ss)) for x,y in poly],fill=255)
    # Simulated nucleus and granules add internal structure but remain soma-labeled.
    nrx=rx*float(rng.uniform(.23,.34)); nry=ry*float(rng.uniform(.22,.36)); ncx=cx+float(rng.normal(0,rx*.08)); ncy=cy+float(rng.normal(0,ry*.08))
    nucpoly=ellipse_points(ncx,ncy,nrx,nry,angle,rng,count=32)
    nucd.polygon([(int(x*ss),int(y*ss)) for x,y in nucpoly],fill=200)
    for _ in range(int(rng.integers(3,8))):
        gx=cx+float(rng.normal(0,rx*.35)); gy=cy+float(rng.normal(0,ry*.35)); gr=float(rng.uniform(.35,1.0))
        gd.ellipse(((gx-gr)*ss,(gy-gr)*ss,(gx+gr)*ss,(gy+gr)*ss),fill=140)
    # Render original parent-child topology; SWC radii become neurite widths.
    for nid,node in nodes.items():
        pid=node["parent"]
        if pid==-1 or pid not in nodes: continue
        parent=nodes[pid]; p0=xy_map[pid]; p1=xy_map[nid]
        # Skip edges internal to soma point cloud. Edges leaving the soma are clipped by soma later.
        if node["type"]==1 and parent["type"]==1: continue
        r_um=.5*(float(parent["radius"])+float(node["radius"]))*scale_um
        wpx=float(np.clip(2*r_um/max(mpp,1e-6),.55,3.8))
        draw_tapered_path(nd,np.stack([p0,p1]),wpx,max(.45,wpx*.68),ss)
    soma=np.asarray(soma_img,dtype=np.uint8)>0
    neurite=np.asarray(neurite_img,dtype=np.uint8)>0
    neurite &= ~soma
    nucleus=np.asarray(nucleus_img,dtype=np.uint8); granules=np.asarray(granule_img,dtype=np.uint8)
    row={"cell_id":cell_id,"center_x_px":cx,"center_y_px":cy,"soma_diameter_um":d_um,"soma_rx_px":rx,"soma_ry_px":ry,"base_neurites":sum(1 for n in nodes.values() if n["parent"] in {t["id"] for t in type1}),"morphology_source":swc_path.name,"swc_projection":projection}
    return soma,neurite,nucleus,granules,row

def synthesize_brightfield(soma_hi, neurite_hi, nucleus_hi, granules_hi, rng, camera_cfg, scene_cfg, width, height, ss):
    # Cell morphology is represented as a weak phase/absorption object, then passed
    # through an approximate edge-transfer/defocus model. It is not a full wave-optics solver.
    sigma_um=float(camera_cfg["optical_blur_sigma_um"])
    mpp=float(camera_cfg["pixel_size_um"])
    sigma_px=max(.35,sigma_um/mpp)
    soma_f=gaussian_filter(soma_hi.astype(np.float32), sigma=max(.5,.45*ss))
    neurite_f=gaussian_filter(neurite_hi.astype(np.float32), sigma=max(.35,.32*ss))
    texture=rng.normal(0,1,size=soma_hi.shape).astype(np.float32)
    texture=gaussian_filter(texture,sigma=1.5*ss)
    if texture.std()>1e-8: texture=texture/texture.std()
    soma_phase=soma_f*(1+float(scene_cfg["cell_texture_strength"])*texture)
    nucleus_f=gaussian_filter((nucleus_hi.astype(np.float32)/255.0),sigma=max(.45,.6*ss))
    granules_f=gaussian_filter((granules_hi.astype(np.float32)/255.0),sigma=max(.35,.4*ss))
    phase=np.clip(soma_phase*.92+neurite_f*.40+nucleus_f*.19+granules_f*.055,0,1.5)
    # Weak-object brightfield approximation: gentle absorption plus signed defocus edge contrast.
    phase_blur=gaussian_filter(phase,sigma=max(.45,sigma_px*ss))
    lap=gaussian_laplace(phase_blur,sigma=max(.6,.75*sigma_px*ss))
    edge_scale=float(np.percentile(np.abs(lap),99.4))
    if edge_scale>1e-9: lap=np.clip(lap/edge_scale,-1.6,1.6)
    absorption=float(scene_cfg["soma_absorption"])*soma_f+float(scene_cfg["neurite_absorption"])*neurite_f+0.055*nucleus_f+0.020*granules_f
    hi=float(camera_cfg["illumination_level"])-absorption+float(camera_cfg["defocus_halo_strength"])*lap
    hi=downsample_float(hi,(width,height))
    # Add smooth field illumination, vignette and background structure at sensor resolution.
    yy,xx=np.mgrid[0:height,0:width].astype(np.float32)
    xx=(xx-(width-1)/2)/max(width/2,1); yy=(yy-(height-1)/2)/max(height/2,1)
    radial=np.sqrt(xx*xx+yy*yy)
    vignette=1-float(camera_cfg["vignetting_strength"])*np.clip(radial**2,0,1.5)
    low=rng.normal(0,1,(height,width)).astype(np.float32)
    low=gaussian_filter(low,sigma=max(4,min(width,height)/10))
    if low.std()>1e-8: low/=low.std()
    fine=rng.normal(0,1,(height,width)).astype(np.float32)
    fine=gaussian_filter(fine,sigma=.6)
    if fine.std()>1e-8: fine/=fine.std()
    illumination=1+float(camera_cfg["illumination_nonuniformity"])*low
    image=hi*vignette*illumination+float(scene_cfg.get("background_texture_strength",0.0))*fine
    # Realistic sensor sampling approximation, parameterized by photons and read noise.
    reference_exposure=float(camera_cfg.get("reference_exposure_ms",camera_cfg["exposure_ms"]))
    exposure_ratio=float(camera_cfg["exposure_ms"])/max(reference_exposure,1e-6)
    photons=max(100,float(camera_cfg["shot_noise_photons"])*exposure_ratio)
    image=np.clip(image,0.015,.995)
    image=rng.poisson(image*photons).astype(np.float32)/photons
    image+=rng.normal(0,float(camera_cfg["read_noise_fraction"]),image.shape).astype(np.float32)
    image=np.clip(image,0,1)
    max_code=(1<<int(camera_cfg["bit_depth"]))-1
    return np.round(image*max_code).astype(np.uint8 if int(camera_cfg["bit_depth"])<=8 else np.uint16)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera",default="config/camera_profile.example.json",help="Camera profile JSON")
    ap.add_argument("--scene",default="config/scene_profile.example.json",help="Cell scene/morphology profile JSON")
    ap.add_argument("--out",default="output/naobf_dataset")
    ap.add_argument("--samples",type=int,default=100,help="Independent randomized fields for segmentation training")
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--cells",type=int,default=None,help="Override cells_per_frame in scene profile")
    ap.add_argument("--swc-dir",default=None,help="Optional folder of SWC traced neuronal morphologies; used instead of procedural branching")
    ap.add_argument("--swc-projection",choices=["auto","xy","xz","yz"],default="auto")
    ap.add_argument("--preview-only",action="store_true",help="Generate 1 sample even if --samples is larger")
    a=ap.parse_args()
    camera_path=Path(a.camera); scene_path=Path(a.scene)
    if not camera_path.is_file(): ap.error(f"Camera config not found: {camera_path}")
    if not scene_path.is_file(): ap.error(f"Scene config not found: {scene_path}")
    camera=read_json(camera_path); scene=read_json(scene_path)
    if camera.get("illumination_mode","transmitted_brightfield")!="transmitted_brightfield": ap.error("This prototype only implements transmitted brightfield. Do not use it as a phase-contrast/DIC renderer.")
    if float(camera["pixel_size_um"])<=0: ap.error("pixel_size_um must be the calibrated object-plane microns-per-pixel value")
    if int(camera["bit_depth"]) not in (8,10,12,14,16): ap.error("bit_depth must be one of 8, 10, 12, 14, 16")
    w,h=int(camera["image_width_px"]),int(camera["image_height_px"])
    ss=int(camera.get("supersampling",3)); ss=max(1,min(ss,5))
    if a.cells is not None: scene["cells_per_frame"]=a.cells
    n=1 if a.preview_only else int(a.samples)
    if n<1: ap.error("samples must be positive")
    out=Path(a.out); (out/"images").mkdir(parents=True,exist_ok=True)
    for sub in ("semantic","soma","neurite","union","instances"):
        (out/"masks"/sub).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed); records=[]; cells_csv=[]
    swc_files=sorted(Path(a.swc_dir).rglob("*.swc")) if a.swc_dir else []
    if a.swc_dir and not swc_files: ap.error(f"No .swc files found under {a.swc_dir}")
    print(f"Morphology source: {len(swc_files)} SWC traces" if swc_files else "Morphology source: procedural random trees (prototype)")
    for idx in range(n):
        # Each sample is a separate randomized culture field, avoiding dozens of near-identical frames.
        all_soma=np.zeros((h*ss,w*ss),dtype=bool); all_neurite=np.zeros_like(all_soma)
        all_nucleus=np.zeros((h*ss,w*ss),dtype=np.uint8); all_granules=np.zeros((h*ss,w*ss),dtype=np.uint8)
        inst_hi=np.zeros((h*ss,w*ss),dtype=np.uint16); cell_rows=[]
        cells_this_frame=int(scene["cells_per_frame"])
        if "cells_per_frame_min" in scene and "cells_per_frame_max" in scene:
            cells_this_frame=int(rng.integers(int(scene["cells_per_frame_min"]),int(scene["cells_per_frame_max"])+1)) if a.cells is None else int(a.cells)
        for cid in range(1,cells_this_frame+1):
            if swc_files:
                swc_path=swc_files[int(rng.integers(0,len(swc_files)))]
                try: soma,neurite,nucleus,granules,row=create_cell_masks_from_swc(rng,w,h,ss,float(camera["pixel_size_um"]),scene,cid,swc_path,a.swc_projection)
                except Exception as e: raise RuntimeError(f"Failed to use morphology {swc_path}: {e}") from e
            else:
                soma,neurite,nucleus,granules,row=create_cell_masks(rng,w,h,ss,float(camera["pixel_size_um"]),scene,cid)
            all_soma|=soma; all_neurite|=neurite; all_nucleus=np.maximum(all_nucleus,nucleus); all_granules=np.maximum(all_granules,granules)
            union=soma|neurite
            # Instance labels only fill previously unassigned pixels; semantic masks stay complete.
            inst_hi[(union)&(inst_hi==0)]=cid
            row.setdefault("morphology_source","procedural")
            row.setdefault("swc_projection","")
            row.update({"sample_id":idx,"instance_id":cid})
            cell_rows.append(row); cells_csv.append(row)
        sem_soma=Image.fromarray(all_soma.astype(np.uint8)*255).resize((w,h),Image.Resampling.NEAREST)
        sem_neur=Image.fromarray(all_neurite.astype(np.uint8)*255).resize((w,h),Image.Resampling.NEAREST)
        soma_m=np.asarray(sem_soma)>0; neur_m=np.asarray(sem_neur)>0
        neur_m &= ~soma_m
        union_m=soma_m|neur_m
        sem=np.zeros((h,w),dtype=np.uint8); sem[neur_m]=2; sem[soma_m]=1
        inst=Image.fromarray(inst_hi).resize((w,h),Image.Resampling.NEAREST)
        inst=np.asarray(inst,dtype=np.uint16)
        # Slight global focus/exposure drift across independent acquisition samples.
        local_camera=dict(camera)
        local_camera["illumination_level"]=float(np.clip(float(camera["illumination_level"])+rng.normal(0,.012),.55,.97))
        image=synthesize_brightfield(all_soma,all_neurite,all_nucleus,all_granules,rng,local_camera,scene,w,h,ss)
        stem=f"sample_{idx:05d}"
        ext=".png" if int(camera["bit_depth"])<=8 else ".tif"
        image_path=out/"images"/(stem+ext)
        if int(camera["bit_depth"])<=8: Image.fromarray(image,mode="L").save(image_path)
        else: tifffile.imwrite(image_path,image,photometric="minisblack")
        Image.fromarray(sem,mode="L").save(out/"masks"/"semantic"/(stem+".png"))
        Image.fromarray(soma_m.astype(np.uint8)*255,mode="L").save(out/"masks"/"soma"/(stem+".png"))
        Image.fromarray(neur_m.astype(np.uint8)*255,mode="L").save(out/"masks"/"neurite"/(stem+".png"))
        Image.fromarray(union_m.astype(np.uint8)*255,mode="L").save(out/"masks"/"union"/(stem+".png"))
        tifffile.imwrite(out/"masks"/"instances"/(stem+".tif"),inst,photometric="minisblack")
        records.append({"sample_id":idx,"image":str(image_path.relative_to(out)),"semantic_mask":f"masks/semantic/{stem}.png","soma_mask":f"masks/soma/{stem}.png","neurite_mask":f"masks/neurite/{stem}.png","union_mask":f"masks/union/{stem}.png","instances_mask":f"masks/instances/{stem}.tif","cells_requested":len(cell_rows),"soma_pixels":int(soma_m.sum()),"neurite_pixels":int(neur_m.sum()),"image_mean":float(image.astype(np.float32).mean()/((1<<int(camera["bit_depth"]))-1)),"image_std":float(image.astype(np.float32).std()/((1<<int(camera["bit_depth"]))-1))})
        if (idx+1)%10==0 or idx==n-1: print(f"Generated {idx+1}/{n} samples")
    with open(out/"manifest.csv","w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0].keys())); wr.writeheader(); wr.writerows(records)
    with open(out/"cell_ground_truth.csv","w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(cells_csv[0].keys())); wr.writeheader(); wr.writerows(cells_csv)
    (out/"camera_profile_used.json").write_text(json.dumps(camera,indent=2),encoding="utf-8")
    (out/"scene_profile_used.json").write_text(json.dumps(scene,indent=2),encoding="utf-8")
    meta={"generator":"NAOBF prototype v0.1","seed":a.seed,"sample_count":n,"image_dimensions_px":[w,h],"bit_depth":int(camera["bit_depth"]),"pixel_size_um":camera["pixel_size_um"],"camera_profile_is_example":bool(camera.get("profile_is_example",False)),"morphology_source":"swc" if swc_files else "procedural","swc_files_available":len(swc_files),"labels":{"semantic classes":{"0":"background","1":"soma","2":"neurite"}},"limitations":["procedural morphology; not extracted from real neuronal anatomy","image formation is an approximate weak-object brightfield rendering, not a full wave-optics model","no neural electrical activity-to-image coupling","synthetic labels are exact only for the generated procedural geometry","not validated against the user's camera or real cultures"]}
    (out/"dataset_manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Done. Dataset: {out.resolve()}")
    print("Semantic masks use 0=background, 1=soma, 2=neurite.")
    if camera.get("profile_is_example",False): print("WARNING: this is an example camera profile. Replace it with measured hardware settings before calibration.")

if __name__=="__main__": main()
