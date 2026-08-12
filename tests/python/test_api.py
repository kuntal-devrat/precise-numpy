import unittest
import math
import precise_numpy as pnp

class TestPreciseNumPy(unittest.TestCase):
    def test_array_creation(self):
        a = pnp.array([1.0, 2.0, 3.0])
        self.assertEqual(len(a), 3)
        self.assertEqual(a.shape(), [3])
        self.assertEqual(a.ndim(), 1)
        self.assertTrue(a.is_exact())
        self.assertEqual(a.max_relative_error(), 0.0)

    def test_array_with_error(self):
        a = pnp.array([1.0, 2.0, 3.0], error=0.1)
        self.assertFalse(a.is_exact())
        self.assertAlmostEqual(a.max_radius(), 0.1)
        m, r = a.get(0)
        self.assertAlmostEqual(m, 1.0)
        self.assertAlmostEqual(r, 0.1)

    def test_zeros_ones_full(self):
        z = pnp.zeros([2, 3])
        self.assertEqual(z.shape(), [2, 3])
        self.assertEqual(len(z), 6)
        self.assertTrue(z.is_exact())
        self.assertEqual(z.get(0), (0.0, 0.0))

        o = pnp.ones([4])
        self.assertEqual(o.shape(), [4])
        self.assertEqual(o.get(0), (1.0, 0.0))

        f = pnp.full([2], 5.0, error=0.2)
        m, r = f.get(0)
        self.assertAlmostEqual(m, 5.0)
        self.assertAlmostEqual(r, 0.2)

    def test_linspace_arange(self):
        ls = pnp.linspace(0.0, 1.0, 5)
        self.assertEqual(len(ls), 5)
        self.assertEqual(ls.get(0)[0], 0.0)
        self.assertEqual(ls.get(4)[0], 1.0)

        ar = pnp.arange(0.0, 5.0, 1.0)
        self.assertEqual(len(ar), 5)
        self.assertEqual(ar.get(0)[0], 0.0)
        self.assertEqual(ar.get(4)[0], 4.0)

    def test_indexing(self):
        a = pnp.array([10.0, 20.0, 30.0], error=0.5)
        m0, r0 = a[0]
        self.assertAlmostEqual(m0, 10.0)
        self.assertAlmostEqual(r0, 0.5)
        m_last, r_last = a[-1]
        self.assertAlmostEqual(m_last, 30.0)
        self.assertAlmostEqual(r_last, 0.5)
        with self.assertRaises(IndexError):
            _ = a[3]

    def test_arithmetic(self):
        a = pnp.array([1.0, 2.0, 3.0])
        b = pnp.array([4.0, 5.0, 6.0])

        c = a + b
        self.assertEqual(c.values(), [5.0, 7.0, 9.0])
        self.assertTrue(c.is_exact())

        d = b - a
        self.assertEqual(d.values(), [3.0, 3.0, 3.0])

        e = a * b
        self.assertEqual(e.values(), [4.0, 10.0, 18.0])

        f = b / a
        self.assertEqual(f.values(), [4.0, 2.5, 2.0])

    def test_scalar_arithmetic(self):
        a = pnp.array([1.0, 2.0, 3.0])
        self.assertEqual((a + 10.0).values(), [11.0, 12.0, 13.0])
        self.assertEqual((10.0 + a).values(), [11.0, 12.0, 13.0])
        self.assertEqual((a - 1.0).values(), [0.0, 1.0, 2.0])
        self.assertEqual((10.0 - a).values(), [9.0, 8.0, 7.0])
        self.assertEqual((a * 2.0).values(), [2.0, 4.0, 6.0])
        self.assertEqual((2.0 * a).values(), [2.0, 4.0, 6.0])
        self.assertEqual((a / 2.0).values(), [0.5, 1.0, 1.5])

    def test_negation_and_abs(self):
        a = pnp.array([-1.0, 2.0, -3.0], error=0.1)
        neg_a = -a
        self.assertEqual(neg_a.values(), [1.0, -2.0, 3.0])
        for r in neg_a.radii():
            self.assertAlmostEqual(r, 0.1)

        abs_a = abs(a)
        self.assertEqual(abs_a.values(), [1.0, 2.0, 3.0])

    def test_math_functions(self):
        a = pnp.array([0.0, math.pi / 2, math.pi])
        sin_a = a.sin()
        self.assertLess(abs(sin_a.midpoint(0) - 0.0), 1e-10)
        self.assertLess(abs(sin_a.midpoint(1) - 1.0), 1e-10)

        exp_a = pnp.array([0.0, 1.0]).exp()
        self.assertLess(abs(exp_a.midpoint(0) - 1.0), 1e-10)
        self.assertLess(abs(exp_a.midpoint(1) - math.e), 1e-10)

        sqrt_a = pnp.array([4.0, 9.0]).sqrt()
        self.assertEqual(sqrt_a.values(), [2.0, 3.0])

    def test_reductions(self):
        a = pnp.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(a.sum(), (10.0, 0.0))
        self.assertEqual(a.mean(), (2.5, 0.0))
        self.assertLess(abs(a.var()[0] - 1.25), 1e-10)
        self.assertLess(abs(a.std()[0] - math.sqrt(1.25)), 1e-10)
        self.assertEqual(a.min_val(), (1.0, 0.0))
        self.assertEqual(a.max_val(), (4.0, 0.0))

    def test_dot_and_matmul(self):
        a = pnp.array([1.0, 2.0, 3.0])
        b = pnp.array([4.0, 5.0, 6.0])
        self.assertEqual(a.dot(b), (32.0, 0.0))

        m1 = pnp.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
        m2 = pnp.array([5.0, 6.0, 7.0, 8.0]).reshape([2, 2])
        res = m1.matmul(m2)
        self.assertEqual(res.shape(), [2, 2])
        self.assertEqual(res.values(), [19.0, 22.0, 43.0, 50.0])

    def test_empty_array(self):
        z = pnp.zeros([0])
        self.assertTrue(z.is_empty())
        self.assertEqual(len(z), 0)
        self.assertEqual(z.sum(), (0.0, 0.0))

if __name__ == '__main__':
    unittest.main()
