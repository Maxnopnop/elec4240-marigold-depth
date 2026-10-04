"""Semantic metric checks: a text-insensitive union must not pass the switching test."""
import unittest
import numpy as np
from .train_baseline import iou

class Checks(unittest.TestCase):
    def test_switching_and_empty_do_not_fake_success(self):
        a=np.zeros((8,8),bool);b=a.copy();a[:,:3]=True;b[:,5:]=True
        self.assertEqual(iou(a,a),1.)
        self.assertEqual(iou(a,b),0.)
        union=a|b
        self.assertEqual(iou(union,a),iou(union,b))
        self.assertFalse(iou(union,a)>iou(union,b))
        empty=np.zeros_like(a)
        self.assertFalse(iou(empty,a)>iou(empty,b))
        self.assertGreater(iou(a,a)-iou(a,b),0.)
        self.assertLess(iou(b,a)-iou(b,b),0.)

if __name__=='__main__':unittest.main()
