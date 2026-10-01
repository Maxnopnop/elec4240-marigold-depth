"""Check frozen provenance, historical artifacts and final external delivery."""
from pathlib import Path
import hashlib,json,subprocess,re
import argparse,zlib
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/'results/prospective_v4'
    frozen=read(out/'protocol.json')['fingerprint']
    for name,h in frozen['source_sha256'].items():assert sha(root/name)==h,name
    for name,key in [('manifest.json','manifest_sha256'),('sample_size_decision.json','decision_sha256'),
                     ('acquisition_transport.json','acquisition_transport_sha256'),('power/design_and_results.json','power_results_sha256')]:
        assert sha(out/name)==frozen[key],name
    before='da8afd9180ed794ec04cec25c79eb0e0a5fb66a6'
    tree=subprocess.check_output(['git','ls-tree','-r',before,'--','results'],cwd=root,text=True)
    paths=[];hashes=[]
    for line in tree.splitlines():
        meta,name=line.split('\t',1);paths.append(name);hashes.append(meta.split()[2])
    current=subprocess.check_output(['git','hash-object','--stdin-paths'],input='\n'.join(paths)+'\n',cwd=root,text=True).splitlines()
    assert current==hashes,'Historical result blobs changed'
    metadata=read(out/'manifest.json');rows=metadata['samples']
    assert len(rows)==200 and len({r['space'] for r in rows})==200 and len({r['rgb_pixel_sha256'] for r in rows})==200
    plan=read(out/'selection_plan.json')
    assert [r['space'] for r in rows]==plan['candidate_spaces'][:200]
    assert all(r['image_member']==plan['candidates'][r['space']][0] for r in rows)
    assert not read(out/'acquisition_audit.json')['exclusions']
    assert sha(root/'prepare_prospective.py')==plan['source_sha256']
    transport=read(out/'acquisition_transport.json')
    assert sha(root/'resume_prospective_download.py')==transport['wrapper_sha256']
    assert sha(root/'remote_archive.py')==transport['transport_source_sha256']
    for r in rows:
        local=a.data/r['file'];assert sha(local)==r['sha256']
        members={kind:a.data/'raw_members'/(hashlib.sha256(r[kind+'_member'].encode()).hexdigest()+'.bin') for kind in ['image','depth']}
        assert sha(members['image'])==r['rgb_sha256'] and sha(members['depth'])==r['depth_png_sha256']
        assert zlib.crc32(members['image'].read_bytes())==r['image_crc32']
        assert zlib.crc32(members['depth'].read_bytes())==r['depth_crc32']
        rgb=np.array(Image.open(members['image']).convert('RGB'))
        encoded=np.array(Image.open(members['depth'])).astype(np.uint32)
        decoded=((encoded>>3)+(encoded&7)*8192).astype(np.float32)/1000
        with np.load(local) as f:
            np.testing.assert_array_equal(rgb,f['image'])
            np.testing.assert_array_equal(decoded,f['depth'])
    audit=read(out/'verification.json')
    assert audit['status']=='passed' and audit['metrics_recomputed']==12600
    assert sha(out/'protocol.json')==audit['protocol_sha256']
    for name,h in audit['audited_artifact_sha256'].items():assert sha(out/name)==h
    # This checker writes its own report only after all assertions pass.
    generated_report=(out/'delivery_checks.json').resolve()
    for doc in [root/'README.md',out/'RESULTS.md',out/'power/RESULTS.md']:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not target.startswith(('http:','https:','#')):
                local=(doc.parent/target.split('#')[0]).resolve()
                assert local==generated_report or local.exists(),(doc,target)
    files=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    assert not any(Path(p).suffix.lower() in ['.npy','.npz','.pt','.pth','.safetensors','.mat'] for p in files)
    report={'status':'passed','historical_result_blobs_unchanged':len(paths),'frozen_source_hashes_verified':len(frozen['source_sha256']),
            'first_200_ranked_spaces_selected':True,'all_first_ranked_frames_selected':True,
            'prediction_metrics_verified':12600,'distinct_source_groups':200,'broader_location_proxies':len(metadata['broader_location_counts']),
            'source_image_depth_pairs_redecoded':200,'source_member_crcs_and_hashes_verified':400,
            'no_raw_arrays_or_weights_tracked':True,'report_relative_links_valid':True}
    (out/'delivery_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
