"""A high-precision sine, and a correctly-rounded comparison with the platform's.

The oracle computes the sine of an argument in Decimal arithmetic -- pi by Machin's
formula, the argument reduced modulo 2*pi at a precision wider than the argument's
magnitude, the Taylor series summed to the working precision -- and rounds the result
to the target binary format by exact rational arithmetic (Fraction), so that a float32
or a float16 result is rounded from the exact value and not twice through float64.

Use `sin_correctly_rounded(value, dtype)`: `value` is the operand, as it is stored in
`dtype` (a float16 operand is not the float64 number that produced it).  It returns the
exact sine of that operand rounded to `dtype` with roundTiesToEven, or None when the
exact value lies too close to a rounding midpoint for the working precision to decide
the last bit -- the caller then skips the operand rather than trusting the comparison.
"""

from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache
import math

TAYLOR = 30
MARGIN = Fraction(1, 10 ** 30)


@lru_cache(maxsize=64)
def pi_decimal(digits):
    """pi, by Machin's formula."""
    getcontext().prec = digits + 20

    def atan_inv(n):
        n = Decimal(n)
        total = term = Decimal(1) / n
        k = 0
        while True:
            k += 1
            term = -term / (n * n)
            add = term / (2 * k + 1)
            if add == 0:
                break
            total += add
        return total

    pi = 16 * atan_inv(5) - 4 * atan_inv(239)
    getcontext().prec = digits
    return +pi


@lru_cache(maxsize=64)
def _two_pi(digits):
    return 2 * pi_decimal(digits)


def sin_decimal(x, digits):
    """The sine of the Decimal x, to `digits` significant digits."""
    getcontext().prec = digits
    two_pi = _two_pi(digits)
    k = (x / two_pi).to_integral_value(rounding="ROUND_HALF_EVEN")
    getcontext().prec = digits + TAYLOR
    r = +(x - k * two_pi)                       # reduced to [-pi, pi]
    term = total = r
    n = 1
    r2 = r * r
    while True:
        term = -term * r2 / ((2 * n) * (2 * n + 1))
        n += 1
        if term == 0 or abs(term) <= abs(total) * Decimal(10) ** (-(digits + 5)):
            break
        total += term
    getcontext().prec = digits
    return +total


def _floor(fr):
    return fr.numerator // fr.denominator


def _exponent(frac):
    """e with 2**e <= frac < 2**(e+1), by integer arithmetic."""
    n, d = frac.numerator, frac.denominator
    e = n.bit_length() - d.bit_length()
    if frac < Fraction(2) ** e:
        e -= 1
    while frac >= Fraction(2) ** (e + 1):
        e += 1
    return e


def _round_half_even(frac):
    q = _floor(frac)
    r = frac - q
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and q % 2 == 1):
        q += 1
    return q


def _near_midpoint(frac):
    return abs(frac - _floor(frac) - Fraction(1, 2)) < MARGIN


def format_rounding(dtype):
    """(precision p, smallest-subnormal exponent, overflow exponent) of a numpy type."""
    import numpy as np

    f = np.finfo(dtype)
    return f.nmant + 1, f.minexp - f.nmant, f.maxexp


def sin_correctly_rounded(value, dtype):
    """The exact sine of `value` (as stored in `dtype`) rounded to `dtype`, or None."""
    import numpy as np

    x = float(np.asarray(value, dtype=dtype))     # the operand as the format holds it
    if math.isnan(x) or math.isinf(x):
        return dtype("nan")
    p, min_exp, max_exp = format_rounding(dtype)
    digits = 45 + max(0, (int(math.log10(abs(x))) + 2) if x else 0)
    s = sin_decimal(Decimal(x), digits)
    frac = Fraction(s)

    sign = -1 if frac < 0 else 1
    frac = abs(frac)
    if frac == 0:
        return dtype(math.copysign(0.0, x))
    e = _exponent(frac)
    shift = e - (p - 1)
    scaled = frac / Fraction(2) ** shift
    uncertain = _near_midpoint(scaled)
    m = _round_half_even(scaled)
    if m == 2 ** p:
        m, shift = m // 2, shift + 1
    if shift < min_exp:                           # into the subnormal range
        scaled = frac / Fraction(2) ** min_exp
        uncertain = uncertain or _near_midpoint(scaled)
        m, shift = _round_half_even(scaled), min_exp
    if uncertain:
        return None
    if m == 0:
        return dtype(math.copysign(0.0, x))
    if m.bit_length() + shift - 1 > max_exp - 1:
        return dtype(math.copysign(float("inf"), x))
    return dtype(sign * float(m * Fraction(2) ** shift))
