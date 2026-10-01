"""Metadata-only census of the official SUNRGBD archive, before outcome access."""
from pathlib import Path
import json, zipfile, collections, hashlib
from remote_archive import HTTPRangeFile

def main():
    out = Path('results/prospective_v4'); out.mkdir(parents=True, exist_ok=True)
    url = 'https://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
    remote = HTTPRangeFile(url)
    with zipfile.ZipFile(remote) as z:
        entries = [{'name': i.filename, 'bytes': i.file_size, 'compressed_bytes': i.compress_size,
                    'crc32': i.CRC, 'offset': i.header_offset} for i in z.infolist()]
    cache = Path(r'E:\Codex\2026-09-27\yo\work\marigold-local\assets\sunrgbd_metadata')
    cache.mkdir(parents=True,exist_ok=True)
    dest = cache/'archive_inventory.json'
    dest.write_text(json.dumps({'url': url, 'archive_bytes': remote.size, 'entries': entries}), encoding='utf-8')
    print('INVENTORY', len(entries), 'ARCHIVE_BYTES', remote.size, 'SHA256', hashlib.sha256(dest.read_bytes()).hexdigest(), flush=True)
    print(collections.Counter('/'.join(e['name'].split('/')[:3]) for e in entries).most_common(20), flush=True)
    paths = [e['name'] for e in entries if '/image/' in e['name'] and e['name'].lower().endswith('.jpg')]
    print('RGB_EXAMPLES', paths[:8], flush=True)
    print('SUN3D_EXAMPLES', [p for p in paths if 'sun3d' in p.lower()][:12], flush=True)
    selected=[e for e in entries if 'sun3d' in e['name'].lower() and (('/image/' in e['name'] and e['name'].endswith('.jpg')) or ('/depth/' in e['name'] and e['name'].endswith('.png')))]
    (out/'sunrgbd_sun3d_inventory.json').write_text(json.dumps({'url':url,'archive_bytes':remote.size,
         'full_inventory_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'entries':selected}),encoding='utf-8')

if __name__ == '__main__': main()
