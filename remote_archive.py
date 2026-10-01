"""Read selected official ZIP members through verified HTTP byte ranges."""
import io
import re
import time
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import requests

_connections=threading.BoundedSemaphore(3)
_local=threading.local()


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.cache = {}
        _, total = self.fetch(0,1)
        self.size = total

    def fetch(self, start, end):
        if end-start > 4*1024**2:
            spans=[(s,min(end,s+4*1024**2)) for s in range(start,end,4*1024**2)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                pieces=list(pool.map(lambda span:self.fetch(*span),spans))
            assert len({total for _,total in pieces})==1
            return b''.join(data for data,_ in pieces),pieces[0][1]
        # Reuse TLS connections for the many small ZIP members. The Windows curl
        # backend intermittently closed connections under high request counts.
        if end-start < 1024*1024:
            with _connections:
                if not hasattr(_local,'session'):_local.session=requests.Session()
                r=_local.session.get(self.url,headers={'Range':f'bytes={start}-{end-1}'},timeout=(15,45),stream=True)
                try:
                    r.raise_for_status();assert r.status_code==206
                    match=re.match(rf'bytes {start}-{end-1}/(\d+)',r.headers['Content-Range'])
                    assert match
                    data=r.raw.read(end-start+1)
                    assert len(data)==end-start
                    return data,int(match.group(1))
                finally:r.close()
        with tempfile.TemporaryDirectory(prefix='elec4240_range_') as temp:
            headers = Path(temp)/'headers.txt'; body = Path(temp)/'body.bin'
            subprocess.run(['curl.exe','--fail','--silent','--show-error','--max-time','90',
                            '--max-filesize',str(64*1024**2),'--range',f'{start}-{end-1}',
                            '--dump-header',str(headers),'--output',str(body),self.url],check=True,timeout=100)
            match = re.search(rf'(?im)^content-range:\s*bytes {start}-{end-1}/(\d+)',headers.read_text())
            assert match, 'Server did not return the requested byte range'
            data = body.read_bytes()
            assert len(data) == end-start
            if len(data)>1000000: print('ARCHIVE_RANGE_BYTES',start,end,len(data),flush=True)
            return data,int(match.group(1))

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('Negative seek')
        return self.pos

    def read(self, size=-1):
        if size < 0: size = self.size-self.pos
        end = min(self.pos+size, self.size)
        if end <= self.pos: return b''
        assert end-self.pos <= 64*1024**2, 'Refuse unexpected large transfer'
        key = (self.pos, end)
        if key not in self.cache:
            for attempt in range(3):
                try:
                    data,total = self.fetch(self.pos,end)
                    assert total == self.size
                    self.cache[key] = data
                    break
                except (subprocess.SubprocessError, AssertionError):
                    if attempt == 2: raise
                    time.sleep(1+attempt)
        self.pos = end
        return self.cache[key]
