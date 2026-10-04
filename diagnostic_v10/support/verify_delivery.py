import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from html.parser import HTMLParser
import torch
from PIL import Image
from diagnostic_v10.common import OUT,WORK,ROOT,read,write,sha
from diagnostic_v10.gradients_precision_v2 import verify_amendment

verify_amendment();status=read(OUT/'status.json');done=read(OUT/'complete.json')
assert status['state']=='complete'
assert sha(OUT/'analysis.json')==done['analysis_sha256']
assert sha(OUT/'RESULTS.html')==done['report_sha256']
observation=read(OUT/'observations_checks.json')
assert sha(OUT/'OBSERVATIONS.html')==observation['report_sha256']
assert sha(OUT/'interpretation.json')==observation['interpretation_sha256']
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value and '://' not in value:
                assert (OUT/value.split('#')[0]).is_file(),value
for name in ['RESULTS.html','OBSERVATIONS.html']:
    Links().feed((OUT/name).read_text(encoding='utf-8'))
for path in (OUT/'figures').glob('*.png'):
    with Image.open(path) as image:image.verify()
resumes=0
for mark in (OUT/'noise_runs').glob('*/complete.json'):
    local=WORK/'noise_runs'/mark.parent.name
    meta=read(local/'resume.json');assert meta['step']==64 and sha(local/'resume.pt')==meta['sha256']
    state=torch.load(local/'resume.pt',map_location='cpu',weights_only=True)
    adapter=torch.load(local/'adapter.pt',map_location='cpu',weights_only=True)
    assert state['protocol_sha256']==sha(OUT/'protocol.json') and len(state['history'])==64
    assert state['optimizer']['state'] and all(torch.equal(state['adapter'][name],value) for name,value in adapter.items())
    resumes+=1
assert resumes==24
source={str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in (ROOT/'diagnostic_v10').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
files={str(f.relative_to(OUT)).replace('\\','/'):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_checks.json'}
write(OUT/'delivery_checks.json',dict(verification=done['verification'],saved_optimizer_states_matching_final_adapters=resumes,
                                    source_sha256=source,result_sha256=files,all_local_report_links_exist=True,
                                    figures_decodable=True,raw_directory=str(WORK),shutdown=False,
                                    note='Visual review performed separately by the assistant. No cloud upload or claim of significant model superiority.'))
print(dict(saved_optimizer_states=resumes,result_files=len(files),source_files=len(source),shutdown=False))
