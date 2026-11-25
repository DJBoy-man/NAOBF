#!/usr/bin/env python3
"""Sanity-check a camera profile before generation."""
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--camera',default='config/camera_profile.example.json'); a=ap.parse_args(); p=Path(a.camera)
    try: d=json.loads(p.read_text())
    except Exception as e: raise SystemExit(f'Cannot read JSON {p}: {e}')
    required=['image_width_px','image_height_px','bit_depth','pixel_size_um','frame_rate_fps','exposure_ms','numerical_aperture','illumination_level','optical_blur_sigma_um','shot_noise_photons','read_noise_fraction']
    missing=[k for k in required if k not in d]
    if missing: raise SystemExit(f'Missing required fields: {missing}')
    issues=[]
    if d['image_width_px']<32 or d['image_height_px']<32: issues.append('Image dimensions should be >=32 px')
    if d['bit_depth'] not in (8,10,12,14,16): issues.append('bit_depth must be 8, 10, 12, 14 or 16')
    if d['pixel_size_um']<=0: issues.append('pixel_size_um must be a positive object-plane calibrated pixel size')
    if d['numerical_aperture']<=0: issues.append('numerical_aperture must be > 0')
    if d['exposure_ms']<=0 or d['frame_rate_fps']<=0: issues.append('Exposure and frame rate must be positive')
    if not 0<d['illumination_level']<=1: issues.append('illumination_level must be in (0,1]')
    if d.get('profile_is_example',False): issues.append('This profile is marked EXAMPLE ONLY; replace it with the real hardware calibration.')
    for issue in issues: print('WARNING:',issue)
    if issues: print(f'{len(issues)} warning(s); profile parsed but is not ready for calibrated simulation.')
    else: print('Camera profile passes basic checks. This does not prove calibration accuracy.')
if __name__=='__main__': main()
