"""A wrong implementation, kept as a check on the driver: the null rule of the sum.

A reader coming from `ops/add.md` and `ops/mul.md` carries the rule of their float sections --
"a result whose exact value is null is +0" -- to the sine.  The rule is right there and wrong
here: the sine of -0.0 is -0.0, the document requiring that the sign bit of the operand pass
to the result.  The departure is one bit, on one value, and the driver has to see it.

Select the behaviour with the environment variable SIN_DETECTOR:
    zero   the null rule of the sum
    period the sine computed on the operand reduced modulo the value of the type nearest 2*pi
"""

import os

import numpy as np


def sin(A):
    a = np.asarray(A)
    if a.dtype.kind != "f":
        return np.sin(a)
    which = os.environ.get("SIN_DETECTOR", "zero")
    if which == "zero":
        return np.sin(a) + a.dtype.type(0.0)
    two_pi = a.dtype.type(6.283185307179586)
    return np.sin(a - two_pi * np.floor(a / two_pi))
