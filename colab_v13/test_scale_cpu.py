"""CPU-only failure-path tests; actual CUDA recovery is tested in Colab."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import scale_runtime as runtime
import scale_report as report
import scale_queue as queue
import numpy as np
from PIL import Image


class ProtocolTests(unittest.TestCase):
    def test_cache_receipt_publication_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'local';drive=Path(tmp)/'drive'
            with patch.object(queue,'ROOT',root),patch.object(queue,'DRIVE',drive):
                self.assertFalse(queue.cache_complete())
                receipt={'status':'complete','shards':{'a':'digest'}}
                runtime.write(root/'scale_cache/cache_audit.json',receipt)
                self.assertFalse(queue.cache_complete())
                runtime.write(drive/'scale_cache/cache_audit.json',receipt)
                self.assertTrue(queue.cache_complete())
                runtime.write(drive/'scale_cache/cache_audit.json',{'status':'complete','shards':{}})
                with self.assertRaises(AssertionError):queue.cache_complete()

    def test_exposure_schedule(self):
        small=runtime.schedule(range(2048))
        large=runtime.schedule(range(4096))
        self.assertEqual(len(small),4096)
        self.assertEqual(len(set(small[:2048])),2048)
        self.assertEqual(len(set(small[2048:])),2048)
        self.assertEqual(len(set(large)),4096)
        self.assertEqual(large,runtime.schedule(list(reversed(range(4096)))))
        with self.assertRaises(AssertionError):runtime.schedule([1,1])

    def test_holm_and_registered_family(self):
        self.assertEqual(len(report.contrasts()),13)
        tests=[{'p':.04},{'p':.01},{'p':.9}]
        report.holm(tests)
        self.assertEqual([t['holm_p'] for t in tests],[.08,.03,.9])

    def test_interrupted_budget_remains_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.json'
            with patch('threading.Timer'):
                b=runtime.Budget(path,10,cap=100)
                b.reserve(50,'interrupted')
                other=runtime.Budget(path,0,cap=100)
                self.assertEqual(other.state['charged_seconds'],60)
                with self.assertRaises(AssertionError):other.reserve(41,'over cap')
                self.assertEqual(runtime.read(path)['charged_seconds'],60)
                path.unlink() # Simulate a vanished mutable Drive summary.
                recovered=runtime.Budget(path,0,cap=100)
                self.assertEqual(recovered.state['charged_seconds'],60)

    def test_checkpoint_missing_pointer_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);p=dest/'staging.pt'
            runtime.torch.save({'protocol_sha256':'test','step':2,'history':[{},{}]},p)
            final=dest/f'resume_00002_{runtime.sha(p)[:16]}.pt';p.rename(final)
            state,step=runtime.latest_state(dest,'test')
            self.assertEqual((state,step),(final,2))
            with self.assertRaises(AssertionError):runtime.latest_state(dest,'wrong protocol')

    def test_missing_matrix_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runtime,'WORK',Path(tmp)):
            with self.assertRaises(FileNotFoundError):runtime.verify_training_barrier('test')

    def test_corrupt_checkpoint_blocks_barrier(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runtime,'WORK',Path(tmp)):
            for size in runtime.SIZES:
                for seed in runtime.SEEDS:
                    for arm in runtime.ARMS:
                        dest=Path(tmp)/'runs'/runtime.label(arm,size,seed)
                        dest.mkdir(parents=True)
                        hashes={}
                        for step in [2048,4096]:
                            p=dest/f'adapter_{step}.pt';p.write_bytes(b'checkpoint');hashes[str(step)]=runtime.sha(p)
                        runtime.write(dest/'trained.json',{'updates':4096,'protocol_sha256':'test','checkpoints':hashes})
            runtime.verify_training_barrier('test')
            (dest/'adapter_4096.pt').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):runtime.verify_training_barrier('test')

    def test_full_report_recomputes_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(report,'WORK',root),patch.object(report,'OUT',root),patch.object(report,'ROOT',root):
                runtime.write(root/'protocol.json',{'test_fixture':True})
                ph=runtime.sha(root/'protocol.json')
                runtime.write(root/'scale_plan.json',{'fresh_holdout_images':list(range(8))})
                Image.fromarray(np.array([[255,0],[0,0]],dtype=np.uint8)).save(root/'a.png')
                Image.fromarray(np.array([[0,0],[0,255]],dtype=np.uint8)).save(root/'b.png')
                Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(root/'empty.png')
                records=[]
                for iid in range(8):
                    for j,filename in enumerate(['a.png','b.png']):
                        records.append({'image_id':iid,'ann_id':iid*2+j,'iou':1.,'distractor_iou':0.,'target_selected':True,
                            'empty':False,'empty_prompt_iou':0.,'prediction':filename,'sha256':runtime.sha(root/filename),
                            'empty_prediction':'empty.png','empty_sha256':runtime.sha(root/'empty.png'),
                            'truth':str(root/filename),'truth_sha256':runtime.sha(root/filename)})
                for arm in runtime.ARMS:
                    for size in runtime.SIZES:
                        for seed in runtime.SEEDS:
                            dest=root/'runs'/runtime.label(arm,size,seed);dest.mkdir(parents=True)
                            runtime.write(dest/'trained.json',{'updates':4096})
                            for step in [2048,4096]:
                                cp=dest/f'adapter_{step}.pt';cp.write_bytes(b'test checkpoint')
                                runtime.write(dest/f'fresh_holdout_{step}'/'metrics.json',{'protocol_sha256':ph,
                                    'checkpoint_sha256':runtime.sha(cp),'records':records,'mean_iou':1.,'both_targets_selected_rate':1.})
                result=report.main(lambda *args,**kwargs:None)
                self.assertEqual(result['predictions_recomputed'],384)
                analysis=runtime.read(root/'analysis.json')
                self.assertEqual(len(analysis['contrasts']),26)
                self.assertTrue(all(t['difference']==0 and t['holm_p']==1 for t in analysis['contrasts']))


if __name__=='__main__':unittest.main()
