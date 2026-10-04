"""Exercise completion/shutdown ordering with fake jobs and no OS shutdown calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from . import run,report

class CompletionSafety(unittest.TestCase):
    def scenario(self,extra_complete,cancel=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'workspace/outputs/repo';out=root/'results';work=Path(tmp)/'work'
            out.mkdir(parents=True);work.mkdir();(root.parents[1]/'work/.planning/marigold-local').mkdir(parents=True)
            (out/'smoke.json').write_text('{"status":"passed"}')
            (out/'extra_plan.json').write_text('{"source_sha256":{},"module":"fake_extra"}')
            (out/'conditions').mkdir();(out/'conditions/complete.json').write_text(json.dumps({'status':'complete' if extra_complete else 'failed'}))
            if cancel:(work/'CANCEL_SHUTDOWN').touch()
            events=[]
            def subprocess_fake(command,**kwargs):
                events.append('shutdown' if '/s' in command else 'extra')
                if '/s' in command:assert command[-3:]==['/s','/t','0'] and '/f' not in command
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            def archive_fake():events.append('archive');return {'archive':'verified.zip'}
            def report_fake():events.append('report');return {'verified':True}
            with patch.multiple(run,ROOT=root,OUT=out,WORK=work,ARMS=[],SEEDS=[]),patch.object(run,'freeze',return_value={}),patch.object(run,'audit'),patch.object(run,'verify'),patch.object(run,'archive',side_effect=archive_fake),patch.object(report,'main',side_effect=report_fake),patch.object(run.subprocess,'run',side_effect=subprocess_fake),patch.object(run.ctypes.windll.kernel32,'SetThreadExecutionState'):
                if extra_complete:run.main(True)
                else:
                    with self.assertRaises(AssertionError):run.main(True)
            return events,json.loads((out/'status.json').read_text())['stage']
    def test_failed_extra_never_archives_or_shuts_down(self):
        events,status=self.scenario(False);self.assertEqual(events,['extra']);self.assertEqual(status,'failed_no_shutdown')
    def test_success_shutdown_only_after_report_and_archive(self):
        events,_=self.scenario(True);self.assertEqual(events,['extra','report','archive','shutdown'])
    def test_cancel_file_overrides_success_shutdown(self):
        events,status=self.scenario(True,True);self.assertEqual(events,['extra','report','archive']);self.assertEqual(status,'complete_saved_without_shutdown')

if __name__=='__main__':unittest.main()
