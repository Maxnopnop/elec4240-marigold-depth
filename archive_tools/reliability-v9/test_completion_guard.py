import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import finish_and_shutdown as guard


class CompletionGuardTests(unittest.TestCase):
    def sample(self):
        return ({'state': 'complete', 'completed_trainings': 21},
                {'verification': {**guard.EXPECTED, 'historical_files_unchanged': 1307},
                 'comparisons': list(range(6)), 'summary': {m: {} for m in guard.MODES}},
                {'passed': True})

    def test_complete_only(self):
        pipeline, analysis, gate = self.sample()
        guard.validate_summary(pipeline, analysis, gate)
        for field in guard.EXPECTED:
            bad = copy.deepcopy(analysis)
            bad['verification'][field] -= 1
            with self.assertRaises(AssertionError):
                guard.validate_summary(pipeline, bad, gate)
        for state in ['executing', 'failed', 'baseline_gate_failed']:
            with self.assertRaises(AssertionError):
                guard.validate_summary({**pipeline, 'state': state}, analysis, gate)
        with self.assertRaises(AssertionError):
            guard.validate_summary(pipeline, analysis, {'passed': False})

    def test_archive_roundtrip_and_change_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input'
            source.mkdir()
            (source / 'weights.bin').write_bytes(bytes(range(256)) * 1024)
            (source / 'notes.txt').write_text('Depth and normals\n', encoding='utf-8')
            result = guard.backup([('experiment', source)], root / 'backup.zip')
            self.assertEqual(result['files'], 2)
            self.assertEqual(result['sha256'], guard.sha(root / 'backup.zip'))
            original = guard.zipfile.ZipFile.write
            def changing_write(z, filename, *args, **kwargs):
                original(z, filename, *args, **kwargs)
                Path(filename).write_bytes(b'changed during backup')
            with patch.object(guard.zipfile.ZipFile, 'write', changing_write):
                with self.assertRaises(AssertionError):
                    guard.backup([('experiment', source)], root / 'rejected.zip')
            self.assertFalse((root / 'rejected.zip').exists())


if __name__ == '__main__':
    unittest.main()
