# -*- coding: utf-8 -*-
"""
Build an S1-centric dataset for STReF-NDVI.

The temporal axis follows the Sentinel-1 dates in the matching table.
Missing Sentinel-2 labels do not remove Sentinel-1 dates.

Inputs:
    S1/S1_YYYYMMDD.tif
        Two bands: VV and VH, in dB by default.

    S2_NDVI_CloudMask/NDVI_CloudMask_YYYYMMDD_WGS84.tif
        Two bands: NDVI and cloud mask.

    WaterMask/WaterMask_YYYYMMDD_WGS84.tif
        One band: water mask.

    Matching/S1_S2_Label_Mapping.csv
        Sentinel-1 to Sentinel-2 label mapping.

Outputs:
    S1.npy              (T, 5, H, W)
    S2.npy              (T, H, W, 3): NDVI, cloud, water
    target_mask.npy     (T, H, W)
    S1_valid.npy        (T, H, W)
    good_pixel.npy      (H, W)
    dates.csv
    metadata.json
    normalization.pkl

Input rasters must already be spatially aligned.
Cloud and water masks use zero for clear and non-water pixels, respectively.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import rasterio
from numpy.lib.format import open_memmap
from tqdm import tqdm

LOGGER = logging.getLogger('stref.dataset')
CHANNELS = ['vh', 'rvi', 'vv_div', 'vv_diff', 'log_ratio']
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--s1-dir', type=Path, help='Default: DATA_ROOT/S1')
    parser.add_argument('--s2-dir', type=Path, help='Default: DATA_ROOT/S2_NDVI_CloudMask')
    parser.add_argument('--water-dir', type=Path, help='Default: DATA_ROOT/WaterMask')
    parser.add_argument('--mapping-file', type=Path, help='Default: DATA_ROOT/Matching/S1_S2_Label_Mapping.csv')
    parser.add_argument('--s1-unit', choices=['db', 'linear'], default='db')
    parser.add_argument('--vv-band', type=int, default=1, help='One-based band index')
    parser.add_argument('--vh-band', type=int, default=2)
    parser.add_argument('--ndvi-band', type=int, default=1)
    parser.add_argument('--cloud-band', type=int, default=2)
    parser.add_argument('--water-band', type=int, default=1)
    parser.add_argument('--ndvi-scale', type=float, default=1.)
    parser.add_argument('--ndvi-min', type=float, default=-.2)
    parser.add_argument('--ndvi-max', type=float, default=1.)
    parser.add_argument('--min-clear-frames', type=int, default=8)
    parser.add_argument('--percentile-low', type=float, default=1.)
    parser.add_argument('--percentile-high', type=float, default=99.)
    parser.add_argument('--normalization-file', type=Path, help='Reuse trusted normalization JSON; do not fit new parameters')
    parser.add_argument('--fit-mask', type=Path, help='Optional binary NPY (H,W) or (T,H,W) defining normalization fitting population')
    parser.add_argument('--missing-label', choices=['warn', 'error'], default='warn')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing generated files')
    args = parser.parse_args()
    if not 0 <= args.percentile_low < args.percentile_high <= 100:
        parser.error('Require 0 <= percentile-low < percentile-high <= 100.')
    if not -1 <= args.ndvi_min < args.ndvi_max <= 1:
        parser.error('NDVI bounds must lie within [-1,1].')
    if args.min_clear_frames < 1 or not np.isfinite(args.ndvi_scale) or args.ndvi_scale <= 0:
        parser.error('min-clear-frames and ndvi-scale must be positive.')
    if min(args.vv_band,args.vh_band,args.ndvi_band,args.cloud_band,args.water_band) < 1:
        parser.error('Band indices are one-based positive integers.')
    if args.normalization_file and args.fit_mask:
        parser.error('Use either normalization-file or fit-mask, not both.')
    args.s1_dir = args.s1_dir or args.data_root/'S1'
    args.s2_dir = args.s2_dir or args.data_root/'S2_NDVI_CloudMask'
    args.water_dir = args.water_dir or args.data_root/'WaterMask'
    args.mapping_file = args.mapping_file or args.data_root/'Matching'/'S1_S2_Label_Mapping.csv'
    return args


def date_string(value):
    value = str(value).strip()
    if not re.fullmatch(r'\d{8}',value):
        raise ValueError(f'Expected YYYYMMDD date, got {value!r}')
    datetime.strptime(value,'%Y%m%d')
    return value


def filename(value):
    """Accept portable filenames only; do not silently follow private absolute paths."""
    value = str(value).strip()
    if not value:
        return ''
    if '/' in value or '\\' in value or Path(value).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError('Mapping file columns must contain filenames only, not directory paths.')
    return value


def collect_inputs(args):
    files = {}
    for path in args.s1_dir.iterdir():
        match = re.fullmatch(r'S1_(\d{8})\.tiff?',path.name,re.IGNORECASE)
        if match:
            date = date_string(match.group(1))
            if date in files:
                raise ValueError(f'Duplicate S1 acquisition date: {date}')
            files[date] = path
    if not files:
        raise ValueError('No S1_YYYYMMDD.tif files found.')
    table = pd.read_csv(args.mapping_file,dtype=str,keep_default_na=False)
    required = {'S1_date','S2_label_date','S1_file','S2_label_file','matched'}
    if not required.issubset(table.columns):
        raise ValueError(f'Missing mapping columns: {sorted(required-set(table.columns))}')
    mapping = {}
    for row in table.to_dict('records'):
        date = date_string(row['S1_date'])
        if date in mapping:
            raise ValueError(f'Duplicate S1_date in mapping: {date}')
        if date not in files:
            raise ValueError(f'Mapping references absent S1 acquisition: {date}')
        s1_name = filename(row['S1_file'])
        if s1_name and s1_name != files[date].name:
            raise ValueError(f'S1 filename/date mismatch: {date}')
        token = row['matched'].strip().lower()
        if token not in {'true','1','yes','false','0','no',''}:
            raise ValueError(f'Invalid matched flag: {token!r}')
        matched = token in {'true','1','yes'}
        s2_name = filename(row['S2_label_file'])
        s2_date = date_string(row['S2_label_date']) if matched else ''
        if matched and not s2_name:
            raise ValueError(f'Matched row has no S2 filename: {date}')
        mapping[date] = dict(matched=matched,s2_file=s2_name,s2_date=s2_date)
    absent = len(set(files)-set(mapping))
    if absent:
        LOGGER.warning('%d S1 dates have no mapping row; retained without S2 labels.',absent)
    return files,mapping


def read_bands(path,bands,grid=None,mask_band=False):
    with rasterio.open(path) as src:
        if src.crs is None or max(bands)>src.count:
            raise ValueError(f'Missing CRS or required band: {path.name}')
        current=(src.height,src.width,src.crs,src.transform)
        if grid is not None and (current[:3]!=grid[:3] or not current[3].almost_equals(grid[3],precision=1e-12)):
            raise ValueError(f'Grid mismatch: {path.name}. Align inputs before building the dataset.')
        if abs(src.transform.determinant)<1e-30:
            raise ValueError(f'Invalid geographic transform: {path.name}')
        values=src.read(bands).astype(np.float32)
        valid=np.isfinite(values)
        if not mask_band:
            valid &= src.read_masks(bands)>0
        else:
            # Binary masks often declare 0 as nodata even though 0 is clear/non-water.
            # Treat explicit binary values as valid; unsupported codes are excluded later.
            valid &= np.isin(values,[0,1])
        values[~valid]=np.nan
    return values,current


def s1_features(raw,unit):
    valid=np.all(np.isfinite(raw),axis=0)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        power=np.power(10.,raw/10.) if unit=='db' else raw.copy()
        valid &= np.all(np.isfinite(power)&(power>=0),axis=0)
        vv,vh=np.where(np.isfinite(power)&(power>=0),power,0.)
        ratio=vv/(vh+EPS)
        features=np.stack([vh,4.*vh/(vv+vh+EPS),ratio,vv-vh,
                           np.log1p(np.clip(ratio,-.999,None))]).astype(np.float32)
        valid &= np.all(np.isfinite(features),axis=0)
    features[:,~valid]=0.
    return features,valid


def build(args,work):
    files,mapping=collect_inputs(args)
    dates=sorted(files)
    first,grid=read_bands(files[dates[0]],[args.vv_band,args.vh_band])
    h,w,crs,transform=grid
    t=len(dates)
    LOGGER.info('Building T=%d, C=%d, H=%d, W=%d',t,len(CHANNELS),h,w)
    if args.min_clear_frames>t:
        LOGGER.warning('min-clear-frames exceeds T; good_pixel will be empty.')
    arrays={}
    specs={'S1':((t,5,h,w),'float32'),'S2':((t,h,w,3),'float32'),
           'target_mask':((t,h,w),'bool'),'S1_valid':((t,h,w),'bool')}
    for name,(shape,dtype) in specs.items():
        arrays[name]=open_memmap(work/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
    records=[]
    for index,date in enumerate(tqdm(dates,desc='Building frames')):
        raw,_=read_bands(files[date],[args.vv_band,args.vh_band],grid)
        features,s1valid=s1_features(raw,args.s1_unit)
        arrays['S1'][index]=features
        arrays['S1_valid'][index]=s1valid
        s2=np.ones((h,w,3),dtype=np.float32)
        s2[:,:,0]=np.nan
        target=np.zeros((h,w),dtype=bool)
        info=mapping.get(date,{})
        loaded=False
        status='unmatched'
        if info.get('matched'):
            s2path=args.s2_dir/info['s2_file']
            waterpath=args.water_dir/f"WaterMask_{info['s2_date']}_WGS84.tif"
            if not s2path.exists() or not waterpath.exists():
                if args.missing_label=='error':
                    raise FileNotFoundError(f'Missing S2 or water mask for S1 date {date}')
                LOGGER.warning('%s: matched label/mask missing; retained without supervision.',date)
                status='missing_label_or_water'
            else:
                ndvi_data,_=read_bands(s2path,[args.ndvi_band],grid)
                cloud_data,_=read_bands(s2path,[args.cloud_band],grid,mask_band=True)
                water_data,_=read_bands(waterpath,[args.water_band],grid,mask_band=True)
                ndvi=ndvi_data[0]*args.ndvi_scale
                cloud,water=cloud_data[0],water_data[0]
                target=(cloud==0)&(water==0)&np.isfinite(ndvi)&(ndvi>=args.ndvi_min)&(ndvi<=args.ndvi_max)
                # Canonical binary output; invalid mask observations are conservative exclusions.
                s2=np.stack([ndvi,np.where(cloud==0,0.,1.),np.where(water==0,0.,1.)],axis=-1).astype(np.float32)
                loaded=True
                status='loaded'
        arrays['S2'][index]=s2
        arrays['target_mask'][index]=target
        records.append(dict(S1_date=date,S2_label_date=info['s2_date'] if loaded else '',
                            supervised=bool(target.any()),label_loaded=loaded,label_status=status))
    fit_mask=None
    if args.fit_mask:
        fit_mask=np.load(args.fit_mask,mmap_mode='r',allow_pickle=False)
        if fit_mask.shape not in [(h,w),(t,h,w)]:
            raise ValueError('fit-mask must have shape (H,W) or (T,H,W).')
        for chunk in fit_mask:
            if not np.all(np.isin(chunk,[0,1])):
                raise ValueError('fit-mask must be binary, without NaN.')
    if args.normalization_file:
        with args.normalization_file.open(encoding='utf-8') as f:
            norm=json.load(f)
        if set(norm)!=set(CHANNELS):
            raise ValueError('Normalization JSON must contain exactly the five feature channel keys.')
        scope='reused_parameters'
    else:
        norm={}
        scope='explicit_fit_mask' if fit_mask is not None else 'all_valid_s1_pixels_and_dates'
        LOGGER.warning('Normalization fitting scope: %s. For held-out evaluation, use training-only fit-mask or reuse training parameters.',scope)
        valid=arrays['S1_valid'].astype(bool)
        if fit_mask is not None:
            valid &= fit_mask.astype(bool)
        if not valid.any():
            raise ValueError('No valid S1 values available for normalization fitting.')
        for i,ch in enumerate(CHANNELS):
            values=arrays['S1'][:,i][valid]
            lo,hi=np.percentile(values,[args.percentile_low,args.percentile_high])
            norm[ch]=dict(p_low=float(lo),p_high=float(hi),pct_low=args.percentile_low,
                          pct_high=args.percentile_high)
        del valid
    for i,ch in enumerate(CHANNELS):
        lo,hi=float(norm[ch]['p_low']),float(norm[ch]['p_high'])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<lo:
            raise ValueError(f'Invalid normalization limits: {ch}')
        for index in range(t):
            frame=np.clip((arrays['S1'][index,i]-lo)/(hi-lo+EPS),0.,1.)
            frame[~arrays['S1_valid'][index]]=0.
            arrays['S1'][index,i]=frame
        LOGGER.info('%s: lower=%.6g upper=%.6g',ch,lo,hi)
    good=arrays['target_mask'].sum(axis=0)>=args.min_clear_frames
    np.save(work/'good_pixel.npy',good)
    pd.DataFrame(records).to_csv(work/'dates.csv',index=False,encoding='utf-8-sig')
    metadata=dict(schema_version='1.0',time_steps=t,height=h,width=w,dates_s1=dates,
                  n_supervised=sum(r['supervised'] for r in records),channels=CHANNELS,
                  s1_in_db=args.s1_unit=='db',min_clear_frames=args.min_clear_frames,
                  normalization='percentile_clip',normalization_fit_scope=scope,
                  normalization_parameters='normalization.json',s1_pct_low=norm[CHANNELS[0]].get('pct_low'),
                  s1_pct_high=norm[CHANNELS[0]].get('pct_high'),ndvi_min=args.ndvi_min,ndvi_max=args.ndvi_max,
                  ndvi_scale=args.ndvi_scale,crs=crs.to_string(),crs_wkt=crs.to_wkt(),
                  transform=list(transform)[:6],transform_order='Affine: a,b,c,d,e,f',
                  s2_channels=['ndvi','cloud','water'],mask_convention='0=clear/non-water, 1=cloud/water/invalid',
                  s1_invalid_fill=0.,supervision_requires_s1_valid=False,
                  layout=dict(S1='T,C,H,W',S2='T,H,W,C',target_mask='T,H,W',S1_valid='T,H,W',good_pixel='H,W'),
                  band_indices=dict(vv=args.vv_band,vh=args.vh_band,ndvi=args.ndvi_band,
                                    cloud=args.cloud_band,water=args.water_band))
    for name,obj in [('metadata.json',metadata),('normalization.json',norm)]:
        with (work/name).open('w',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,allow_nan=False)
    # Compatibility output for existing training scripts; JSON is preferred for interchange.
    with (work/'normalization.pkl').open('wb') as f:
        pickle.dump(norm,f,protocol=4)
    for array in arrays.values():
        array.flush()
    for array in arrays.values():
        array._mmap.close()
    arrays.clear()
    LOGGER.info('Supervised frames: %d/%d; good pixels: %d/%d',metadata['n_supervised'],t,good.sum(),h*w)


def main():
    args=parse_args()
    logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(message)s')
    names=['S1.npy','S2.npy','target_mask.npy','S1_valid.npy','good_pixel.npy',
           'dates.csv','metadata.json','normalization.json','normalization.pkl']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.overwrite and any((args.output_dir/name).exists() for name in names):
        raise FileExistsError('Generated outputs already exist. Use another directory or --overwrite.')
    with tempfile.TemporaryDirectory(prefix='stref_build_',dir=args.output_dir) as tmp:
        work=Path(tmp)
        build(args,work)
        for name in names:
            shutil.move(str(work/name),str(args.output_dir/name))
    LOGGER.info('Dataset construction completed: %s',args.output_dir)


if __name__=='__main__':
    main()
