"""End-to-end tests for the precise_numpy Python package (0.2.0 API)."""

import math
import os
import tempfile
import unittest
import warnings

import precise_numpy as pnp


class TestCreation(unittest.TestCase):
    def test_array(self):
        a = pnp.array([1.0, 2.0, 3.0])
        self.assertEqual(len(a), 3)
        self.assertEqual(a.shape, (3,))
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.size, 3)
        self.assertEqual(a.dtype, "interval64")
        self.assertEqual(a.itemsize, 16)
        self.assertTrue(a.is_exact())
        self.assertEqual(a.max_relative_error(), 0.0)

    def test_array_with_error(self):
        a = pnp.array([1.0, 2.0, 3.0], error=0.1)
        self.assertFalse(a.is_exact())
        self.assertAlmostEqual(a.max_radius(), 0.1)
        m, r = a.get(0)
        self.assertAlmostEqual(m, 1.0)
        self.assertAlmostEqual(r, 0.1)

    def test_array_2d_and_tuples(self):
        a = pnp.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(a.shape, (2, 2))
        b = pnp.array([(1.0, 0.1), (2.0, 0.2)])
        self.assertAlmostEqual(b.get(1)[1], 0.2)
        self.assertEqual(pnp.asarray(5.0).shape, (1,))

    def test_array_inhomogeneous_raises(self):
        with self.assertRaises(TypeError):
            pnp.array([[1.0], [2.0, 3.0]])

    def test_zeros_ones_full_empty(self):
        z = pnp.zeros([2, 3])
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(z.size, 6)
        self.assertEqual(len(z), 2)
        self.assertEqual(z.get(0), (0.0, 0.0))
        o = pnp.ones([4])
        self.assertEqual(o.get(0), (1.0, 0.0))
        f = pnp.full([2], 5.0, error=0.2)
        m, r = f.get(0)
        self.assertAlmostEqual(m, 5.0)
        self.assertAlmostEqual(r, 0.2)
        e = pnp.empty([3])
        self.assertEqual(e.shape, (3,))

    def test_eye_identity_diag(self):
        e = pnp.eye(3)
        self.assertEqual(e.shape, (3, 3))
        self.assertEqual(e.values(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(pnp.identity(2).shape, (2, 2))
        d = pnp.diag([1.0, 2.0])
        self.assertEqual(d.shape, (2, 2))
        self.assertEqual(d.values(), [1.0, 0.0, 0.0, 2.0])
        self.assertEqual(pnp.diag(d).values(), [1.0, 2.0])

    def test_linspace_arange(self):
        ls = pnp.linspace(0.0, 1.0, 5)
        self.assertEqual(len(ls), 5)
        self.assertEqual(ls.get(0)[0], 0.0)
        self.assertEqual(ls.get(4)[0], 1.0)
        self.assertEqual(pnp.linspace(0.0, 1.0, 4, endpoint=False).get(3)[0], 0.75)
        # num=0 returns an empty array (numpy behavior), not an error.
        self.assertEqual(pnp.linspace(0.0, 1.0, 0).shape, (0,))
        ar = pnp.arange(0.0, 5.0, 1.0)
        self.assertEqual(len(ar), 5)
        self.assertEqual(ar.get(4)[0], 4.0)
        with self.assertRaises(ValueError):
            pnp.arange(0.0, 5.0, 0.0)

    def test_from_raw_parts(self):
        a = pnp.from_raw_parts([1.0, 2.0], [0.1, 0.2], [2])
        self.assertEqual(a.radii(), [0.1, 0.2])
        with self.assertRaises(ValueError):
            pnp.from_raw_parts([1.0], [0.1, 0.2], [2])


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.a = pnp.array([10.0, 20.0, 30.0, 40.0], error=0.5)

    def test_int_index(self):
        m, r = self.a[0]
        self.assertAlmostEqual(m, 10.0)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(self.a[-1][0], 40.0)
        with self.assertRaises(IndexError):
            _ = self.a[4]

    def test_slice(self):
        s = self.a[1:3]
        self.assertEqual(s.values(), [20.0, 30.0])
        self.assertEqual(self.a[::2].values(), [10.0, 30.0])
        self.assertEqual(self.a[::-1].values(), [40.0, 30.0, 20.0, 10.0])

    def test_fancy_index(self):
        f = self.a[[0, 2]]
        self.assertEqual(f.values(), [10.0, 30.0])

    def test_bool_mask(self):
        mask = self.a > 20.0
        self.assertEqual(mask.tolist(), [False, False, True, True])
        self.assertEqual(self.a[mask].values(), [30.0, 40.0])

    def test_setitem(self):
        b = self.a.copy()
        b[0] = 99.0
        self.assertEqual(b.get(0)[0], 99.0)
        b[1:3] = [1.0, 2.0]
        self.assertEqual(b.values(), [99.0, 1.0, 2.0, 40.0])

    def test_2d_indexing(self):
        m = pnp.arange(0.0, 6.0, 1.0).reshape(2, 3)
        self.assertEqual(m[1, 2][0], 5.0)
        self.assertEqual(m[0].values(), [0.0, 1.0, 2.0])
        self.assertEqual(m[:, 1].values(), [1.0, 4.0])

    def test_ellipsis_indexing(self):
        m = pnp.arange(0.0, 6.0, 1.0).reshape(2, 3)
        self.assertEqual(m[..., 1].values(), [1.0, 4.0])
        self.assertEqual(m[0, ...].values(), [0.0, 1.0, 2.0])
        self.assertEqual(m[..., 0:2].values(), [0.0, 1.0, 3.0, 4.0])
        # Single ellipsis on a 1D array is a no-op full slice.
        self.assertEqual(self.a[..., 1:3].values(), [20.0, 30.0])
        with self.assertRaises(IndexError):
            m[..., ...]

    def test_newaxis_indexing(self):
        m = pnp.arange(0.0, 6.0, 1.0).reshape(2, 3)
        self.assertEqual(m[None].shape, (1, 2, 3))
        self.assertEqual(m[None].values(), m.values())
        self.assertEqual(m[:, None].shape, (2, 1, 3))
        self.assertEqual(m[None, :, 1].shape, (1, 2))
        self.assertEqual(m[1, None, 2].shape, (1,))

    def test_pickle(self):
        import pickle

        for arr in (self.a, pnp.arange(0.0, 4.0, 1.0).reshape(2, 2)):
            s = pickle.dumps(arr)
            b = pickle.loads(s)
            self.assertEqual(b.shape, arr.shape)
            self.assertEqual(b.values(), arr.values())
            self.assertEqual(b.radii(), arr.radii())
        mask = self.a > 15.0
        s = pickle.dumps(mask)
        b = pickle.loads(s)
        self.assertEqual(b.tolist(), mask.tolist())

    def test_iteration(self):
        vals = [v[0] for v in self.a]
        self.assertEqual(vals, [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(list(self.a)[1][0], 20.0)


class TestArithmetic(unittest.TestCase):
    def test_basic(self):
        a = pnp.array([1.0, 2.0, 3.0])
        b = pnp.array([4.0, 5.0, 6.0])
        self.assertEqual((a + b).values(), [5.0, 7.0, 9.0])
        self.assertTrue((a + b).is_exact())
        self.assertEqual((b - a).values(), [3.0, 3.0, 3.0])
        self.assertEqual((a * b).values(), [4.0, 10.0, 18.0])
        self.assertEqual((b / a).values(), [4.0, 2.5, 2.0])

    def test_scalar(self):
        a = pnp.array([1.0, 2.0, 3.0])
        self.assertEqual((a + 10.0).values(), [11.0, 12.0, 13.0])
        self.assertEqual((10.0 + a).values(), [11.0, 12.0, 13.0])
        self.assertEqual((10.0 - a).values(), [9.0, 8.0, 7.0])
        self.assertEqual((2.0 * a).values(), [2.0, 4.0, 6.0])
        self.assertEqual((a / 2.0).values(), [0.5, 1.0, 1.5])
        self.assertEqual((2.0 / a).values(), [2.0, 1.0, 2.0 / 3.0])

    def test_inplace(self):
        a = pnp.array([1.0, 2.0])
        b = pnp.array([3.0, 4.0])
        c = a
        c += b
        self.assertEqual(c.values(), [4.0, 6.0])
        c -= 1.0
        self.assertEqual(c.values(), [3.0, 5.0])
        c *= 2.0
        self.assertEqual(c.values(), [6.0, 10.0])
        c /= 2.0
        self.assertEqual(c.values(), [3.0, 5.0])

    def test_error_propagation(self):
        a = pnp.array([1.0, 2.0], error=0.1)
        b = pnp.array([3.0, 4.0], error=0.1)
        c = a + b
        self.assertAlmostEqual(c.radii()[0], 0.2)
        self.assertAlmostEqual((a * 2.0).radii()[0], 0.2)
        self.assertAlmostEqual((2.0 * a).radii()[0], 0.2)

    def test_division_by_zero_warns_and_entire(self):
        a = pnp.array([1.0, 2.0])
        b = pnp.array([0.0, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = a / b
        self.assertGreaterEqual(len(w), 1)
        self.assertEqual(c.radii(), [float("inf"), 0.0])

    def test_broadcasting(self):
        m = pnp.ones([2, 3])
        r = m + pnp.array([1.0, 2.0, 3.0])
        self.assertEqual(r.shape, (2, 3))
        self.assertEqual(r.values()[3:], [2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            m + pnp.ones([2, 2])

    def test_unary(self):
        a = pnp.array([-1.0, 2.0, -3.0], error=0.1)
        self.assertEqual((-a).values(), [1.0, -2.0, 3.0])
        self.assertEqual(abs(a).values(), [1.0, 2.0, 3.0])
        self.assertEqual((+a).values(), [-1.0, 2.0, -3.0])
        for r in (-a).radii():
            self.assertAlmostEqual(r, 0.1)


class TestComparisons(unittest.TestCase):
    def test_ordering(self):
        a = pnp.array([1.0, 2.0, 3.0])
        self.assertEqual((a < 2.5).tolist(), [True, True, False])
        self.assertEqual((a > 2.5).tolist(), [False, False, True])
        self.assertEqual((a <= 2.0).tolist(), [True, True, False])
        self.assertEqual((a >= 2.0).tolist(), [False, True, True])

    def test_overlap_equality(self):
        x = pnp.array([1.0, 3.0])
        y = pnp.array([2.0, 3.0])
        self.assertEqual((x == y).tolist(), [False, True])
        self.assertEqual((x != y).tolist(), [True, False])
        self.assertEqual((x == 3.0).tolist(), [False, True])

    def test_bool_array_ops(self):
        a = pnp.array([1.0, 2.0, 3.0])
        b = pnp.array([0.0, 2.0, 5.0])
        m1 = a > 1.5
        m2 = b < 4.0
        self.assertTrue(isinstance(m1, pnp.BoolArray))
        self.assertEqual((m1 & m2).tolist(), [False, True, False])
        self.assertEqual((m1 | m2).tolist(), [True, True, True])
        self.assertEqual((m1 ^ m2).tolist(), [True, False, True])
        self.assertEqual((~m1).tolist(), [True, False, False])
        self.assertTrue(m1.any())
        self.assertFalse(m1.all())
        self.assertEqual(m1.sum(), 2)


class TestMathFunctions(unittest.TestCase):
    def test_trig_exp_log(self):
        a = pnp.array([0.0, math.pi / 2, math.pi])
        self.assertLess(abs(a.sin().midpoint(1) - 1.0), 1e-10)
        self.assertLess(abs(a.cos().midpoint(0) - 1.0), 1e-10)
        self.assertLess(abs(a.tan().midpoint(0) - 0.0), 1e-10)
        e = pnp.array([0.0, 1.0]).exp()
        self.assertLess(abs(e.midpoint(1) - math.e), 1e-10)
        self.assertEqual(pnp.array([1.0, math.e]).ln().midpoint(1), 1.0)
        self.assertEqual(pnp.array([1.0, math.e]).log().midpoint(1), 1.0)
        self.assertEqual(pnp.log(pnp.array([1.0, math.e])).midpoint(1), 1.0)
        self.assertEqual(pnp.array([2.0, 4.0]).log2().values(), [1.0, 2.0])
        self.assertEqual(pnp.array([10.0, 100.0]).log10().values(), [1.0, 2.0])

    def test_sqrt_rounding(self):
        self.assertEqual(pnp.array([4.0, 9.0]).sqrt().values(), [2.0, 3.0])
        a = pnp.array([1.1, 2.5, 3.7])
        self.assertEqual(a.floor().values(), [1.0, 2.0, 3.0])
        self.assertEqual(a.ceil().values(), [2.0, 3.0, 4.0])
        self.assertEqual(a.trunc().values(), [1.0, 2.0, 3.0])
        self.assertEqual(a.round().values(), [1.0, 2.0, 4.0])
        self.assertEqual(round(a).values(), [1.0, 2.0, 4.0])
        self.assertEqual(round(a, 1).values(), [1.1, 2.5, 3.7])
        self.assertEqual(pnp.array([1.2345, 2.6789]).round(2).values(), [1.23, 2.68])
        self.assertEqual(pnp.round_(pnp.array([1.2345])).values(), [1.0])

    def test_clip_sign_nan(self):
        a = pnp.array([-2.0, 0.5, 5.0])
        self.assertEqual(a.clip(-1.0, 1.0).values(), [-1.0, 0.5, 1.0])
        self.assertEqual(a.sign().values(), [-1.0, 1.0, 1.0])
        n = pnp.array([float("nan"), float("inf"), 1.0])
        self.assertTrue(n.isnan().tolist()[0])
        self.assertTrue(n.isinf().tolist()[1])
        self.assertEqual(n.nan_to_num().values()[0], 0.0)

    def test_power_maximum_minimum(self):
        a = pnp.array([2.0, 3.0])
        self.assertEqual(a.power(pnp.array([2.0, 3.0])).values(), [4.0, 27.0])
        self.assertEqual(pnp.power(a, 2.0).values(), [4.0, 9.0])
        self.assertEqual(pnp.maximum(a, 2.5).values(), [2.5, 3.0])
        self.assertEqual(pnp.minimum(a, 2.5).values(), [2.0, 2.5])
        self.assertEqual(a.maximum(pnp.array([0.0, 4.0])).values(), [2.0, 4.0])

    def test_module_wrappers(self):
        a = pnp.array([4.0, 9.0])
        self.assertEqual(pnp.sqrt(a).values(), [2.0, 3.0])
        self.assertEqual(pnp.abs_(pnp.array([-1.0])).values(), [1.0])
        self.assertEqual(pnp.exp(pnp.array([0.0])).values(), [1.0])
        self.assertEqual(pnp.sin(pnp.array([0.0])).values(), [0.0])
        self.assertEqual(pnp.sign(pnp.array([-3.0])).values(), [-1.0])
        self.assertEqual(pnp.floor(pnp.array([1.5])).values(), [1.0])
        self.assertEqual(pnp.ceil(pnp.array([1.5])).values(), [2.0])
        self.assertEqual(pnp.trunc(pnp.array([1.5])).values(), [1.0])
        self.assertEqual(pnp.round_(pnp.array([1.5])).values(), [2.0])
        self.assertEqual(pnp.clip(pnp.array([5.0]), 0.0, 1.0).values(), [1.0])


class TestReductions(unittest.TestCase):
    def test_scalar_reductions(self):
        a = pnp.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(a.sum(), (10.0, 0.0))
        self.assertEqual(a.mean(), (2.5, 0.0))
        self.assertEqual(a.prod(), (24.0, 0.0))
        self.assertLess(abs(a.var()[0] - 1.25), 1e-10)
        self.assertLess(abs(a.std()[0] - math.sqrt(1.25)), 1e-10)
        self.assertEqual(a.min_val(), (1.0, 0.0))
        self.assertEqual(a.max_val(), (4.0, 0.0))
        self.assertEqual(pnp.sum_(a), (10.0, 0.0))
        self.assertEqual(pnp.mean(a), (2.5, 0.0))
        self.assertEqual(pnp.prod(a), (24.0, 0.0))
        self.assertEqual(pnp.max(a), (4.0, 0.0))
        self.assertEqual(pnp.min(a), (1.0, 0.0))

    def test_axis_reductions(self):
        m = pnp.arange(0.0, 6.0, 1.0).reshape(2, 3)
        self.assertEqual(m.sum(axis=0).values(), [3.0, 5.0, 7.0])
        self.assertEqual(m.sum(axis=1).values(), [3.0, 12.0])
        self.assertEqual(m.mean(axis=0).values(), [1.5, 2.5, 3.5])
        self.assertEqual(m.max(axis=0).values(), [3.0, 4.0, 5.0])
        self.assertEqual(m.min(axis=1).values(), [0.0, 3.0])
        self.assertEqual(m.prod(axis=1).values(), [0.0, 60.0])
        with self.assertRaises(ValueError):
            m.sum(axis=2)

    def test_arg_and_logical(self):
        a = pnp.array([3.0, 1.0, 2.0])
        self.assertEqual(a.argmax(), 0)
        self.assertEqual(a.argmin(), 1)
        m = pnp.array([1.0, 2.0, 3.0, 0.0]).reshape(2, 2)
        self.assertEqual(m.argmax(axis=0), [1, 0])
        self.assertEqual(m.argmin(axis=1), [0, 1])
        self.assertEqual(pnp.all(pnp.array([1.0, 2.0])), True)
        self.assertEqual(pnp.any(pnp.array([0.0, 2.0])), True)
        self.assertEqual(pnp.any(pnp.array([0.0, 0.0])), False)

    def test_cumsum_norm(self):
        a = pnp.array([1.0, 2.0, 3.0])
        self.assertEqual(a.cumsum().values(), [1.0, 3.0, 6.0])
        self.assertAlmostEqual(a.norm()[0], math.sqrt(14.0))
        self.assertEqual(pnp.norm(a)[0], math.sqrt(14.0))

    def test_tolist_nested(self):
        a = pnp.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(a.tolist(), [[(1.0, 0.0), (2.0, 0.0)], [(3.0, 0.0), (4.0, 0.0)]])
        b = pnp.zeros([2, 2, 2])
        self.assertEqual(len(b.tolist()), 2)
        self.assertEqual(len(b.tolist()[0][0]), 2)
        c = pnp.arange(0.0, 3.0, 1.0)
        self.assertEqual(c.tolist(), [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        self.assertEqual(pnp.zeros([0]).tolist(), [])

    def test_norm_ord_axis(self):
        import precise_numpy.linalg as la

        a = pnp.array([[1.0, -2.0], [-3.0, 4.0]])
        self.assertAlmostEqual(la.norm(a)[0], math.sqrt(30.0))
        self.assertAlmostEqual(la.norm(a, ord=1)[0], 6.0)
        self.assertAlmostEqual(la.norm(a, ord=-1)[0], 4.0)
        self.assertAlmostEqual(la.norm(a, ord=float("inf"))[0], 7.0)
        self.assertAlmostEqual(la.norm(a, ord=float("-inf"))[0], 3.0)
        self.assertAlmostEqual(la.norm(a, ord=2)[0], math.sqrt(15.0 + math.sqrt(221.0)))
        self.assertAlmostEqual(
            la.norm(a, ord="nuc")[0],
            math.sqrt(15.0 + math.sqrt(221.0)) + math.sqrt(15.0 - math.sqrt(221.0)),
        )
        v = pnp.array([3.0, -4.0])
        self.assertAlmostEqual(la.norm(v)[0], 5.0)
        self.assertAlmostEqual(la.norm(v, ord=1)[0], 7.0)
        self.assertAlmostEqual(la.norm(v, ord=float("inf"))[0], 4.0)
        r = la.norm(a, axis=0)
        self.assertEqual(r.shape, (2,))
        self.assertAlmostEqual(r.values()[0], math.sqrt(10.0))
        self.assertAlmostEqual(r.values()[1], math.sqrt(20.0))
        self.assertAlmostEqual(la.norm(a, ord=1, axis=1).values()[0], 3.0)
        self.assertAlmostEqual(la.norm(a, ord=float("inf"), axis=1).values()[1], 4.0)

    def test_empty_array(self):
        z = pnp.zeros([0])
        self.assertTrue(z.is_empty())
        self.assertEqual(len(z), 0)
        self.assertEqual(z.sum(), (0.0, 0.0))
        self.assertEqual(z.prod(), (1.0, 0.0))
        with self.assertRaises(IndexError):
            z.get(0)


class TestShapeOps(unittest.TestCase):
    def test_reshape(self):
        a = pnp.arange(0.0, 6.0, 1.0)
        self.assertEqual(a.reshape(2, 3).shape, (2, 3))
        self.assertEqual(a.reshape((3, 2)).shape, (3, 2))
        self.assertEqual(a.reshape(-1, 2).shape, (3, 2))
        self.assertEqual(pnp.reshape(a, 3, 2).shape, (3, 2))
        with self.assertRaises(ValueError):
            a.reshape(2, 2)

    def test_transpose_flatten(self):
        m = pnp.arange(0.0, 6.0, 1.0).reshape(2, 3)
        self.assertEqual(m.t.shape, (3, 2))
        self.assertEqual(m.t.values(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
        self.assertEqual(m.flatten().shape, (6,))
        self.assertEqual(m.ravel().shape, (6,))
        self.assertEqual(pnp.transpose(m).shape, (3, 2))

    def test_copy_sort_argsort(self):
        a = pnp.array([3.0, 1.0, 2.0])
        b = a.copy()
        b[0] = 99.0
        self.assertEqual(a.get(0)[0], 3.0)
        self.assertEqual(a.sort().values(), [1.0, 2.0, 3.0])
        self.assertEqual(a.argsort(), [1, 2, 0])
        self.assertEqual(pnp.sort(a).values(), [1.0, 2.0, 3.0])
        self.assertEqual(pnp.argsort(a), [1, 2, 0])


class TestStackingSplit(unittest.TestCase):
    def test_concatenate(self):
        a = pnp.array([1.0, 2.0])
        b = pnp.array([3.0, 4.0, 5.0])
        c = pnp.concatenate([a, b])
        self.assertEqual(c.shape, (5,))
        self.assertEqual(c.values(), [1.0, 2.0, 3.0, 4.0, 5.0])
        m = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        n = pnp.array([5.0, 6.0]).reshape(2, 1)
        r = pnp.concatenate([m, n], axis=1)
        self.assertEqual(r.shape, (2, 3))
        self.assertEqual(r.values(), [1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
        with self.assertRaises(ValueError):
            pnp.concatenate([pnp.zeros([2]), pnp.zeros([3, 1])])

    def test_stack_vstack_hstack(self):
        a = pnp.array([1.0, 2.0])
        b = pnp.array([3.0, 4.0])
        s = pnp.stack([a, b])
        self.assertEqual(s.shape, (2, 2))
        self.assertEqual(pnp.vstack([a, b]).shape, (2, 2))
        self.assertEqual(pnp.hstack([a, b]).values(), [1.0, 2.0, 3.0, 4.0])
        m = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        n = pnp.array([5.0, 6.0]).reshape(2, 1)
        self.assertEqual(pnp.hstack([m, n]).shape, (2, 3))

    def test_split(self):
        a = pnp.array([1.0, 2.0, 3.0, 4.0])
        p1, p2 = pnp.split(a, 2)
        self.assertEqual(p1.values(), [1.0, 2.0])
        self.assertEqual(p2.values(), [3.0, 4.0])
        p1, p2, p3 = pnp.split(a, [1, 2])
        self.assertEqual(p3.values(), [3.0, 4.0])
        m = pnp.arange(0.0, 4.0, 1.0).reshape(2, 2)
        parts = pnp.split(m, 2, axis=1)
        self.assertEqual(parts[1].values(), [1.0, 3.0])
        with self.assertRaises(ValueError):
            pnp.split(a, 3)
        # Out-of-bounds axis must raise, not crash the interpreter.
        with self.assertRaises(ValueError):
            pnp.split(a, 2, axis=3)

    def test_where_nonzero(self):
        a = pnp.array([1.0, 2.0, 3.0])
        w = pnp.where(a > 1.5, a, 0.0)
        self.assertEqual(w.values(), [0.0, 2.0, 3.0])
        nz = pnp.nonzero(a)
        self.assertEqual(list(nz[0]), [0, 1, 2])
        m = pnp.array([0.0, 1.0, 0.0, 2.0]).reshape(2, 2)
        nz2 = pnp.nonzero(m)
        self.assertEqual(list(nz2[0]), [0, 1])
        self.assertEqual(list(nz2[1]), [1, 1])
        self.assertEqual(list(a.nonzero()[0]), [0, 1, 2])


class TestDotMatmul(unittest.TestCase):
    def test_dot_1d(self):
        a = pnp.array([1.0, 2.0, 3.0])
        b = pnp.array([4.0, 5.0, 6.0])
        self.assertEqual(a.dot(b), (32.0, 0.0))
        self.assertEqual(pnp.dot(a, b), (32.0, 0.0))

    def test_matmul_2d(self):
        m1 = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        m2 = pnp.array([5.0, 6.0, 7.0, 8.0]).reshape(2, 2)
        res = m1.matmul(m2)
        self.assertEqual(res.shape, (2, 2))
        self.assertEqual(res.values(), [19.0, 22.0, 43.0, 50.0])
        self.assertEqual((m1 @ m2).values(), [19.0, 22.0, 43.0, 50.0])
        self.assertEqual(pnp.matmul(m1, m2).values(), [19.0, 22.0, 43.0, 50.0])

    def test_matmul_mixed(self):
        m = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        v = pnp.array([1.0, 1.0])
        self.assertEqual((m @ v).values(), [3.0, 7.0])
        self.assertEqual((v @ m).values(), [4.0, 6.0])
        with self.assertRaises(ValueError):
            m @ pnp.array([1.0, 2.0, 3.0]).reshape(3, 1)

    def test_matmul_empty_columns(self):
        # Regression: zero columns previously hit a zero chunk size panic
        # in the parallel path for M >= 256.
        a = pnp.ones([300, 2])
        b = pnp.zeros([2, 0])
        r = a @ b
        self.assertEqual(r.shape, (300, 0))
        self.assertEqual(r.size, 0)
        # Zero rows on the left too.
        self.assertEqual((pnp.zeros([0, 3]) @ pnp.zeros([3, 4])).shape, (0, 4))
        # Zero inner dimension: result is all zeros.
        self.assertEqual((a @ pnp.zeros([2, 0]) @ pnp.zeros([0, 2])).shape, (300, 2))


class TestLinalg(unittest.TestCase):
    def test_det_inv_solve(self):
        import precise_numpy.linalg as la

        m = pnp.array([4.0, 7.0, 2.0, 6.0]).reshape(2, 2)
        self.assertAlmostEqual(la.det(m)[0], 10.0)
        i = la.inv(m)
        expected = [0.6, -0.7, -0.2, 0.4]
        for k in range(4):
            self.assertLess(abs(i.get(k)[0] - expected[k]), 1e-10)
        x = la.solve(pnp.array([2.0, 1.0, 1.0, 3.0]).reshape(2, 2), pnp.array([3.0, 5.0]))
        self.assertLess(abs(x.get(0)[0] - 0.8), 1e-10)
        self.assertLess(abs(x.get(1)[0] - 1.4), 1e-10)

    def test_lstsq(self):
        import precise_numpy.linalg as la

        a = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
        b = pnp.array([2.0, 4.0, 6.0, 8.0])
        x = la.lstsq(a, b)
        self.assertLess(abs(x.get(0)[0] - 2.0), 1e-8)

    def test_eig(self):
        import precise_numpy.linalg as la

        evals, evecs = la.eig(pnp.array([2.0, 1.0, 1.0, 2.0]).reshape(2, 2))
        self.assertLess(abs(evals.get(0)[0] - 3.0), 1e-8)
        self.assertLess(abs(evals.get(1)[0] - 1.0), 1e-8)

    def test_svd(self):
        import precise_numpy.linalg as la

        m = pnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3)
        u, s, vt = la.svd(m)
        self.assertEqual(u.shape, (2, 2))
        self.assertEqual(s.shape, (2,))
        self.assertEqual(vt.shape, (2, 3))
        s_diag = pnp.from_raw_parts([s.get(0)[0], 0.0, 0.0, s.get(1)[0]], [0.0] * 4, [2, 2])
        rec = (u @ s_diag) @ vt
        for k in range(6):
            self.assertLess(abs(rec.get(k)[0] - m.get(k)[0]), 1e-8)

    def test_pinv(self):
        import precise_numpy.linalg as la

        m = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        p = la.pinv(m)
        eye = m @ p
        self.assertLess(abs(eye.get(0)[0] - 1.0), 1e-8)
        self.assertLess(abs(eye.get(3)[0] - 1.0), 1e-8)

    def test_pinv_empty(self):
        import precise_numpy.linalg as la

        # Regression: empty input previously panicked on s_vals[0].
        p = la.pinv(pnp.zeros([3, 0]))
        self.assertEqual(p.shape, (0, 3))
        self.assertEqual(la.pinv(pnp.zeros([0, 3])).shape, (3, 0))
        # Reduced SVD conventions: m >= n gives U (m, n), VT (n, n).
        u, s, vt = la.svd(pnp.zeros([3, 0]))
        self.assertEqual(u.shape, (3, 0))
        self.assertEqual(s.shape, (0,))
        self.assertEqual(vt.shape, (0, 0))
        u, s, vt = la.svd(pnp.zeros([0, 3]))
        self.assertEqual(u.shape, (0, 0))
        self.assertEqual(s.shape, (0,))
        self.assertEqual(vt.shape, (0, 3))


class TestRandom(unittest.TestCase):
    def test_seed_reproducibility(self):
        import precise_numpy.random as pr

        pr.seed(42)
        a = pr.rand(3)
        pr.seed(42)
        b = pr.rand(3)
        self.assertEqual(a.values(), b.values())

    def test_shapes(self):
        import precise_numpy.random as pr

        pr.seed(1)
        self.assertEqual(pr.rand(2, 3).shape, (2, 3))
        self.assertEqual(pr.randn(4).shape, (4,))
        self.assertEqual(pr.randint(0, 10, 5).shape, (5,))
        self.assertEqual(pr.uniform(0.0, 1.0, 5).shape, (5,))
        self.assertEqual(pr.normal(0.0, 1.0, 5).shape, (5,))
        self.assertEqual(pr.random(5).shape, (5,))
        self.assertEqual(pr.random_sample(5).shape, (5,))

    def test_ranges(self):
        import precise_numpy.random as pr

        pr.seed(3)
        r = pr.rand(100)
        for v in r.values():
            self.assertTrue(0.0 <= v < 1.0)
        i = pr.randint(-5, 5, 100)
        for v in i.values():
            self.assertTrue(-5 <= v < 5)
        with self.assertRaises(ValueError):
            pr.randint(5, 5)

    def test_scalar_forms(self):
        import precise_numpy.random as pr

        pr.seed(7)
        self.assertTrue(isinstance(pr.rand(), float))
        self.assertTrue(isinstance(pr.randint(0, 10), int))
        v = pr.randint(-5, 5)
        self.assertTrue(-5 <= v < 5)


class TestFileIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        a = pnp.array([1.0, 2.5, 3.0], error=0.1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.pn")
            pnp.save(path, a)
            b = pnp.load(path)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.values(), b.values())
        self.assertEqual(a.radii(), b.radii())

    def test_savetxt_loadtxt_roundtrip(self):
        a = pnp.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.txt")
            pnp.savetxt(path, a)
            b = pnp.loadtxt(path)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.values(), b.values())

    def test_load_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.pn")
            with open(path, "wb") as f:
                f.write(b"garbage")
            with self.assertRaises(ValueError):
                pnp.load(path)


class TestNumpyInterop(unittest.TestCase):
    def test_array_protocol(self):
        import numpy as _np
        import precise_numpy as _pnp

        a = _pnp.array([[1.5, 2.5], [3.5, 4.5]], error=0.1)
        x = _np.asarray(a)
        expected = _np.array([[1.5, 2.5], [3.5, 4.5]])
        self.assertEqual(x.shape, (2, 2))
        self.assertTrue(
            _np.allclose(x, expected),
            "np.asarray should return midpoints with shape preserved",
        )

    def test_array_protocol_dtype(self):
        import numpy as _np
        import precise_numpy as _pnp

        a = _pnp.array([1.25])
        x = _np.asarray(a, dtype=_np.float32)
        self.assertEqual(x.dtype, _np.float32)
        self.assertAlmostEqual(float(x.flat[0]), 1.25)

    def test_to_numpy(self):
        import numpy as _np
        import precise_numpy as _pnp

        a = _pnp.zeros([3])
        x = _pnp.to_numpy(a)
        self.assertIsInstance(x, _np.ndarray)
        self.assertEqual(list(x.shape), [3])
        z = _pnp.to_numpy(_pnp.zeros([2, 2]))
        self.assertEqual(list(z.shape), [2, 2])

    def test_from_numpy_scalar_error(self):
        import numpy as _np
        import precise_numpy as _pnp

        x = _np.array([1.0, 2.0, 3.0])
        a = _pnp.from_numpy(x, error=0.01)
        self.assertEqual(a.shape, (3,))
        self.assertEqual(a.values(), [1.0, 2.0, 3.0])
        self.assertEqual(a.radii(), [0.01, 0.01, 0.01])

    def test_from_numpy_array_error(self):
        import numpy as _np
        import precise_numpy as _pnp

        x = _np.array([[1.0, 4.0], [9.0, 16.0]])
        rads = [0.1, 0.2, 0.3, 0.4]
        a = _pnp.from_numpy(x, error=rads)
        self.assertEqual(a.radii(), rads)

    def test_from_numpy_default(self):
        import precise_numpy as _pnp

        a = _pnp.from_numpy([7.0])
        self.assertEqual(a.radius(0), 0.0)


class TestExtraLinalg(unittest.TestCase):
    def test_cholesky_spd(self):
        a = pnp.array([[4.0, 2.0], [2.0, 3.0]])
        chol = pnp.cholesky(a)
        self.assertEqual(chol.shape, (2, 2))
        ll = pnp.matmul(chol, chol.transpose())
        for k in range(4):
            mid_ll = ll.values()[k]
            mid_a = a.values()[k]
            self.assertTrue(abs(mid_ll - mid_a) < 1e-10)
        self.assertAlmostEqual(chol.values()[0], 2.0)
        self.assertAlmostEqual(chol.values()[2], 1.0)

    def test_cholesky_not_spd(self):
        a = pnp.array([[-1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            pnp.cholesky(a)

    def test_matrix_power_n0(self):
        a = pnp.eye(3)
        e = pnp.matrix_power(a, 0)
        self.assertEqual(e.shape, (3, 3))
        for i in range(3):
            self.assertAlmostEqual(e.values()[i * 3 + i], 1.0)

    def test_matrix_power_n1(self):
        a = pnp.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(pnp.matrix_power(a, 1).values(), a.values())

    def test_matrix_power_n2(self):
        a = pnp.array([[1.0, 0.0], [0.0, 2.0]])
        e = pnp.matrix_power(a, 2)
        self.assertAlmostEqual(e.values()[3], 4.0)

    def test_matrix_power_negative(self):
        a = pnp.array([[2.0, 0.0], [0.0, 3.0]])
        e = pnp.matrix_power(a, -1)
        m = pnp.matmul(e, a)
        expected = [1.0, 0.0, 0.0, 1.0]
        for k in range(4):
            self.assertTrue(m.values()[k] > expected[k] - 1e-10)
            self.assertTrue(m.values()[k] < expected[k] + 1e-10)

    def test_matrix_rank_full(self):
        a = pnp.eye(3)
        self.assertEqual(pnp.matrix_rank(a), 3)

    def test_matrix_rank_defective(self):
        a = pnp.array([[1.0, 2.0], [2.0, 4.0]])
        self.assertEqual(pnp.matrix_rank(a), 1)

    def test_cond_identity(self):
        c = pnp.cond(pnp.eye(3))
        self.assertTrue(0.0 < c[0] < 100.0)

    def test_cond_singular(self):
        a = pnp.array([[1.0, 2.0], [2.0, 4.0]])
        c = pnp.cond(a)
        self.assertTrue(c[0] > 1e10)


class TestNumpyUfunc(unittest.TestCase):
    def test_add_ufunc(self):
        import numpy as _np

        a = pnp.array([1.0, 2.0], error=0.01)
        b = pnp.array([3.0, 4.0], error=0.01)
        r = _np.add(a, b)
        self.assertIsInstance(r, pnp.IntervalArray)
        self.assertEqual(r.shape, (2,))

    def test_multiply_ufunc(self):
        import numpy as _np

        a = pnp.array([2.0, 3.0], error=0.01)
        r = _np.multiply(a, 3.0)
        self.assertIsInstance(r, pnp.IntervalArray)
        self.assertAlmostEqual(r.values()[0], 6.0)
        self.assertAlmostEqual(r.values()[1], 9.0)

    def test_negative_ufunc(self):
        import numpy as _np

        a = pnp.array([-1.0, 2.0], error=0.01)
        r = _np.negative(a)
        self.assertIsInstance(r, pnp.IntervalArray)
        self.assertAlmostEqual(r.values()[0], 1.0)
        self.assertAlmostEqual(r.values()[1], -2.0)

    def test_abs_ufunc(self):
        import numpy as _np

        a = pnp.array([-1.0, 2.0, -3.0])
        r = _np.absolute(a)
        self.assertIsInstance(r, pnp.IntervalArray)
        self.assertAlmostEqual(r.values()[0], 1.0)
        self.assertAlmostEqual(r.values()[1], 2.0)
        self.assertAlmostEqual(r.values()[2], 3.0)

    def test_matmul_ufunc(self):
        import numpy as _np

        a = pnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = pnp.array([[5.0, 6.0], [7.0, 8.0]])
        r = _np.matmul(a, b)
        self.assertIsInstance(r, pnp.IntervalArray)
        self.assertAlmostEqual(r.values()[0], 19.0)

    def test_exp_ufunc(self):
        import numpy as _np

        a = pnp.array([0.0, 1.0])
        r = _np.exp(a)
        self.assertIsInstance(r, pnp.IntervalArray)
        self.assertAlmostEqual(r.values()[0], 1.0)
        self.assertAlmostEqual(r.values()[1], math.e)

    def test_sqrt_ufunc(self):
        import numpy as _np

        a = pnp.array([1.0, 4.0, 9.0])
        r = _np.sqrt(a)
        self.assertIsInstance(r, pnp.IntervalArray)
        # mid=1, rad=0 for sqrt(1); mid should be 1.0, 2.0, 3.0
        self.assertTrue(r.values()[0] > 0.99)
        self.assertTrue(r.values()[1] > 1.99)
        self.assertTrue(r.values()[2] > 2.99)


class TestNumpyFunction(unittest.TestCase):
    def test_np_linalg_norm(self):
        import numpy as _np

        a = pnp.array([[1.0, -2.0], [-3.0, 4.0]])
        r = _np.linalg.norm(a)
        self.assertAlmostEqual(r[0], math.sqrt(30.0))

    def test_np_sum(self):
        import numpy as _np

        a = pnp.array([1.0, 2.0, 3.0])
        r = _np.sum(a)
        self.assertAlmostEqual(r[0], 6.0)

    def test_np_expand_dims(self):
        import numpy as _np

        a = pnp.array([1.0, 2.0, 3.0])
        r0 = _np.expand_dims(a, axis=0)
        self.assertIsInstance(r0, pnp.IntervalArray)
        self.assertEqual(r0.shape, (1, 3))
        self.assertEqual(r0.values(), [1.0, 2.0, 3.0])

        r1 = _np.expand_dims(a, axis=1)
        self.assertIsInstance(r1, pnp.IntervalArray)
        self.assertEqual(r1.shape, (3, 1))

        rn1 = _np.expand_dims(a, axis=-1)
        self.assertIsInstance(rn1, pnp.IntervalArray)
        self.assertEqual(rn1.shape, (3, 1))

        rn2 = _np.expand_dims(a, axis=-2)
        self.assertIsInstance(rn2, pnp.IntervalArray)
        self.assertEqual(rn2.shape, (1, 3))

    def test_np_concatenate(self):
        import numpy as _np

        a = pnp.array([1.0, 2.0])
        b = pnp.array([3.0, 4.0])
        c = _np.concatenate([a, b])
        self.assertIsInstance(c, pnp.IntervalArray)
        self.assertEqual(c.values(), [1.0, 2.0, 3.0, 4.0])

    def test_np_stack(self):
        import numpy as _np

        a = pnp.array([1.0, 2.0])
        b = pnp.array([3.0, 4.0])
        c = _np.stack([a, b])
        self.assertIsInstance(c, pnp.IntervalArray)
        self.assertEqual(c.shape, (2, 2))

    def test_np_squeeze(self):
        import numpy as _np

        a = pnp.ones([1, 2, 1, 3])
        # Squeeze all 1-sized dimensions
        r_all = _np.squeeze(a)
        self.assertIsInstance(r_all, pnp.IntervalArray)
        self.assertEqual(r_all.shape, (2, 3))

        # Squeeze specific 1-sized dimensions
        r_axis_0 = _np.squeeze(a, axis=0)
        self.assertIsInstance(r_axis_0, pnp.IntervalArray)
        self.assertEqual(r_axis_0.shape, (2, 1, 3))

        r_axis_2 = _np.squeeze(a, axis=2)
        self.assertIsInstance(r_axis_2, pnp.IntervalArray)
        self.assertEqual(r_axis_2.shape, (1, 2, 3))


class TestNpyIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.npy")
            a = pnp.array([[1.5, 2.5], [3.5, 4.5]], error=0.1)
            pnp.save_npy(path, a)
            b = pnp.load_npy(path)
        self.assertEqual(a.shape, b.shape)
        for k in range(4):
            self.assertAlmostEqual(a.values()[k], b.values()[k])
            self.assertAlmostEqual(a.radii()[k], b.radii()[k])

    def test_npy_with_np_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.npy")
            a = pnp.array([10.0, 20.0], error=0.5)
            pnp.save_npy(path, a)
            import numpy as _np

            raw = _np.load(path)
        self.assertEqual(raw.shape, (2, 2))
        self.assertTrue(_np.allclose(raw[..., 0], [10.0, 20.0]))
        self.assertTrue(_np.allclose(raw[..., 1], [0.5, 0.5]))


class TestAstype(unittest.TestCase):
    def test_astype_none(self):
        a = pnp.array([1.0, 2.0], error=0.1)
        self.assertIs(pnp.astype(a, None), a)

    def test_astype_float(self):
        import numpy as _np

        a = pnp.array([1.5, 2.5], error=0.1)
        r = pnp.astype(a, float)
        self.assertIsInstance(r, _np.ndarray)
        self.assertTrue(_np.allclose(r, [1.5, 2.5]))

    def test_astype_int(self):
        import numpy as _np

        a = pnp.array([1.6, 2.4], error=0.1)
        r = pnp.astype(a, int)
        self.assertIsInstance(r, _np.ndarray)
        self.assertEqual(list(r), [1, 2])


class TestVersion(unittest.TestCase):
    def test_version(self):
        self.assertEqual(pnp.__version__, "1.0.0")


if __name__ == "__main__":
    unittest.main()
