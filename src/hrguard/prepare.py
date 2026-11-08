"""Materialize only the fixed DEV subjects' five required raw files from an existing GalaxyPPG ZIP.

Read-only with respect to the source ZIP; NEVER opens a sealed ZIP member or
performs fitting. The resulting DEV files are ordinary extracted copies.
This is NOT dataset scientific verification; run RUN_READONLY.ps1 afterward.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from .core import DEV, SEAL

NEEDED = (
    'GalaxyWatch/PPG.csv', 'GalaxyWatch/ACC.csv', 'GalaxyWatch/HR.csv',
    'PolarH10/HR.csv', 'Event.csv',
)
# Complete, rooted member suffix: do not accept traversal, ambiguous or unrelated files.
SUBJECT_MEMBER = re.compile(r'^(?:[^/]+/)*?(P(?:0[1-9]|1\d|2[0-4]))/(GalaxyWatch/(?:PPG|ACC|HR)\.csv|PolarH10/HR\.csv|Event\.csv)$')
MAX_SINGLE_FILE_BYTES = 1_500_000_000


def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def unpack_dev(zippath:Path, output:Path, manifest:Path, allow_overwrite=False):
    if not zippath.is_file():raise FileNotFoundError(f'Existing ZIP not found: {zippath}')
    output=output.resolve();manifest=manifest.resolve(); zippath=zippath.resolve()
    if output==zippath.parent or output==zippath: # avoid accidental top-level sprawl
        raise ValueError('Choose a dedicated folder for DEV extraction')
    if (output / 'P08').exists() or any((output / s).exists() for s in SEAL):
        raise PermissionError('Destination already contains SEALED subject folders: choose a clean DEV-only output')
    entries={}
    with zipfile.ZipFile(zippath,'r') as zf:
        for info in zf.infolist():
            if info.is_dir():continue
            name=info.filename.replace('\\','/')
            if '/./' in '/'+name or '/../' in '/'+name or name.startswith('/'):
                raise ValueError('Unsafe ZIP member path (metadata): refuse')
            mat=SUBJECT_MEMBER.fullmatch(name)
            if not mat:continue
            sid,rel=mat.groups()
            if sid not in DEV:continue # do not open sealed ZIP members
            if info.file_size > MAX_SINGLE_FILE_BYTES:
                raise ValueError(f'Unexpectedly oversized DEV file {sid}/{rel}')
            key=(sid,rel)
            if key in entries:raise ValueError(f'Ambiguous duplicate DEV member {sid}/{rel}')
            entries[key]=info
        results={}; count=0; cached=0; total_bytes=0
        for sid in sorted(DEV):
            found=[];missing=[]
            for rel in NEEDED:
                info=entries.get((sid,rel))
                if info is None:
                    missing.append(rel);continue
                dst=output/sid/rel
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    if not dst.is_file():raise ValueError(f'Existing path is not a file: {dst}')
                    if not allow_overwrite:
                        # Reuse existing DEV file only when byte count + CRC match
                        # the ZIP member. No sealed content is accessed.
                        import zlib
                        crc=0
                        with dst.open('rb') as fh:
                            for block in iter(lambda:fh.read(1024*1024),b''):
                                crc=zlib.crc32(block,crc)
                        if dst.stat().st_size!=info.file_size or (crc&0xffffffff)!=info.CRC:
                            raise ValueError(f'Existing DEV file differs from ZIP: {dst}; refuse overwrite')
                        cached+=1
                    else:
                        dst.unlink()
                if not dst.exists():
                    # ZipExtFile verifies CRC when read completely; only DEV members opened.
                    fd,tmp=tempfile.mkstemp(prefix='.dev-tmp-',dir=dst.parent)
                    try:
                        with os.fdopen(fd,'wb') as out, zf.open(info,'r') as src:
                            shutil.copyfileobj(src,out,1024*1024)
                        os.replace(tmp,dst)
                    finally:
                        if os.path.exists(tmp):os.unlink(tmp)
                    count+=1
                total_bytes+=info.file_size
                found.append({'path':rel,'bytes':info.file_size,'sha256':sha256(dst)})
            results[sid]={'file_count':len(found),'missing':missing,'files':found}
    report={'state':'DEV_ZIP_EXTRACTED_NO_FIT','source_zip':str(zippath),
            'source_zip_size':zippath.stat().st_size,
            'dev_count':len(DEV),'sealed_UNOPENED':sorted(SEAL),
            'new_files':count,'reused_verified_files':cached,'dev_bytes':total_bytes,
            'raw_dev_root':str(output),'participants':results,
            'note':'Only five requested modalities for fixed DEV IDs materialized; no SEALED member read; CRC and SHA for DEV only. No performance evidence.'}
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    args=p.parse_args()
    report=unpack_dev(args.zip,args.out,args.manifest)
    print(json.dumps({'state':report['state'],'new_files':report['new_files'],
          'reused_files':report['reused_verified_files'],
          'dev_subjects_complete':sum(v['file_count']==5 for v in report['participants'].values()),
          'missing_by_dev':{s:v['missing'] for s,v in report['participants'].items() if v['missing']},
          'manifest':str(args.manifest)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
