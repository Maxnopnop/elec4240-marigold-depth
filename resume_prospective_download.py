"""Transport-only recovery using pinned mirror, exact official ZIP member checks."""
from pathlib import Path
import shutil
import prepare_prospective as preparation
from remote_archive import HTTPRangeFile

MIRROR='https://huggingface.co/datasets/ZHAO11564/SUNRGBD.zip/resolve/4fafeae5f5bad5764cb8f3e89c38a5389ad2b633/SUNRGBD.zip'
OUT=Path('results/prospective_v4')

def mirrored_file(official_url):
    assert official_url=='https://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
    remote=HTTPRangeFile(MIRROR)
    assert remote.size==6885481608
    return remote

def main():
    original=OUT/'acquisition_audit.json';saved=OUT/'acquisition_attempt_official.json'
    if original.exists() and not saved.exists():shutil.copyfile(original,saved)
    preparation.write(OUT/'acquisition_transport.json',{
        'reason':'Official Princeton endpoints began closing connections after89 accepted groups; no external inference occurred.',
        'official_url':'https://rgbd.cs.princeton.edu/data/SUNRGBD.zip','mirror_url':MIRROR,
        'mirror_revision':'4fafeae5f5bad5764cb8f3e89c38a5389ad2b633','archive_bytes':6885481608,
        'verification':'Every mirror member must match official central-directory name, byte length, compression metadata and CRC32. Previously cached official bytes retained.',
        'selection_rules_changed':False,'wrapper_sha256':preparation.sha(__file__),
        'transport_source_sha256':preparation.sha('remote_archive.py')})
    preparation.HTTPRangeFile=mirrored_file
    preparation.main()

if __name__=='__main__':main()
