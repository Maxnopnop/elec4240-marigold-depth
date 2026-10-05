import tempfile,json
from pathlib import Path
from unittest.mock import patch
import finish_project_upload as f

def case(kind):
 with tempfile.TemporaryDirectory(prefix='elec4240-upload-guard-') as tmp:
  root=Path(tmp);out=root/'out';out.mkdir();repo=root/'repo';repo.mkdir();(repo/'archive_tools').mkdir();(repo/'EXPERIMENT_ARCHIVE.md').write_text('Archive')
  work=root/'work';work.mkdir();plan=work/'.planning/marigold-local';plan.mkdir(parents=True)
  for n in ['upload_experiment_archives.py','finish_project_upload.py','test_upload_guard.py']:(work/n).write_text('# fixture')
  completion={'status':'complete','all_github_digests_verified':True,'experiments':1,'files':1,'assets':2,'bytes':10}
  (out/'COMPLETION.json').write_text(json.dumps(completion));(out/'ALL_EXPERIMENTS_MANIFEST.json').write_text(json.dumps({'uploads':[{'name':'a.zip','bytes':10,'sha256':'abc'}]}))
  asset={'name':'a.zip','state':'uploaded','size':10,'digest':'sha256:'+('bad' if kind=='bad_digest' else 'abc')}
  assets=[asset,{'name':'ALL_EXPERIMENTS_MANIFEST.json','digest':'sha256:'+f.u.digest(out/'ALL_EXPERIMENTS_MANIFEST.json')}]
  if kind=='cancel':(out/'CANCEL_SHUTDOWN').touch()
  if kind=='repeat':(out/'SHUTDOWN.json').write_text('{}')
  calls=[]
  def git(*args):
   if args[:2]==('rev-parse','HEAD'):return 'abc123'
   if args[0]=='ls-remote':return 'abc123 refs/heads/main'
   if args[0]=='diff':return 'delivery'
   return ''
  def native():
   assert (out/'FINAL_VERIFICATION.json').exists();calls.append('native');return {'state':'native_shutdown_request_accepted'}
  def req(method,url,**kw):return assets if '/assets?' in url else {'id':1,'draft':False,'html_url':'https://example.test/release'}
  with patch.object(f.u,'ROOT',root),patch.object(f.u,'OUT',out),patch.object(f.u,'REPO',repo),patch.object(f.u,'req',req),patch.object(f,'git',git),patch.object(f,'native_shutdown',native):
   try:f.finalize(True)
   except RuntimeError:
    assert kind in ['bad_digest','repeat']
  assert len(calls)==(1 if kind=='success' else 0),(kind,calls)
  print('PASS',kind)
for k in ['success','bad_digest','cancel','repeat']:case(k)

