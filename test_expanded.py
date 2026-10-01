"""Checks that guard the expanded experiment's leakage and comparison design."""
import json
from pathlib import Path
import unittest
from run_expanded import matrix,PROTOCOL

class ExpandedDesignTest(unittest.TestCase):
    def test_matrix_and_training_seeds(self):
        runs=matrix();self.assertEqual(len(runs),11)
        self.assertEqual(len({r['run_id'] for r in runs}),11)
        for mode,n in [('lora',32),('lora',64),('head',64)]:
            group=[r for r in runs if r['mode']==mode and r['train_images']==n]
            self.assertEqual({r['train_seed'] for r in group},{17,29,43})
        self.assertEqual(PROTOCOL['inference_seed_offset'],17)
        self.assertEqual(PROTOCOL['denoising_steps'],[1,4])

    def test_frozen_selection_preserves_pilot_roles(self):
        root=Path(__file__).parent/'results'
        pilot=json.loads((root/'split_manifest.json').read_text())['samples']
        rows=json.loads((root/'expanded_v1'/'selection_plan.json').read_text())['samples']
        byid={r['id']:r for r in rows}
        self.assertEqual(len(rows),200);self.assertEqual(len(byid),200)
        self.assertEqual(len({r['scene'] for r in rows}),200)
        for r in pilot:
            self.assertEqual(byid[r['id']]['split'],r['split'])
            self.assertEqual(byid[r['id']]['scene'],r['scene'])
        train=[r['id'] for r in rows if r['split']=='train']
        self.assertEqual(train[:32],[r['id'] for r in pilot if r['split']=='train'])
        self.assertEqual(sum(r.get('cohort')=='fresh96' for r in rows),96)
        self.assertEqual(sum(r.get('cohort')=='pilot24' for r in rows),24)
        self.assertFalse({r['scene'] for r in pilot}&{r['scene'] for r in rows if r.get('cohort')=='fresh96'})

if __name__=='__main__':unittest.main()
