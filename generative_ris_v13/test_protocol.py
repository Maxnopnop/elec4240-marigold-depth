"""Small scientific integrity checks, not model effectiveness tests."""
from .common import *
from .state import schedule
from .report import contrasts, holm
import unittest
from collections import Counter


class ProtocolTests(unittest.TestCase):
    def test_exposure(self):
        selection=read(WORK/'selection.json')
        small=selection['small_train_images'];large=selection['large_train_images']
        self.assertTrue(set(small)<set(large))
        self.assertFalse(set(large)&set(selection['fresh_holdout_images']))
        a=schedule(small);b=schedule(large)
        self.assertEqual(set(Counter(a).values()),{4})
        self.assertEqual(set(Counter(b).values()),{1})
        self.assertEqual(len(set(b[:2048])),2048)
        self.assertEqual(a,schedule(small[::-1]))

    def test_family(self):
        c=contrasts();self.assertEqual(len(set(c)),13)
        self.assertEqual(sum(a[0]!=b[0] for a,b in c),4)
        self.assertEqual(sum(a[1]!=b[1] for a,b in c),3)
        self.assertEqual(sum(a[2]!=b[2] for a,b in c),6)

    def test_holm_known_values_and_ties(self):
        tests=[{'p':.01},{'p':.04},{'p':.02},{'p':1}]
        holm(tests)
        self.assertEqual([r['holm_p'] for r in tests],[.04,.08,.06,1])
        tests=[{'p':.01} for _ in range(26)];holm(tests)
        self.assertTrue(all(r['holm_p']==.26 for r in tests))


if __name__=='__main__':unittest.main()
