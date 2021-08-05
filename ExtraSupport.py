#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:48:46 2021

@author: bmondal
"""
from __future__ import division
import sys
import numpy as np      
from scipy import linalg as LA
from math import pi
import matplotlib.pyplot as plt
from matplotlib import animation

from sympy import *
from sympy.interactive.printing import init_printing
from sympy.matrices import Matrix

np.set_printoptions(precision=3,linewidth=100,suppress=True)
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.getH(), rtol=rtol, atol=atol)
#%%
a, c, f, k, vss, vs0p1, vs1p0, vxx, vxy = symbols("a, c, f, k, vss, vs0p1, vs1p0, vxx, vxy")
theta = Symbol('theta', real=True)
g0 = cos(pi/2*theta)
g1 = I*sin(pi/2*theta)
g5 = -g0 + g1
g8 = -g5
g6 = g0+g1

b, l, m, n, p, h, q, j = vss*g0, vs0p1*g8, vs0p1*g6, vs1p0*g8,  vs1p0*g6, vxx*(-g5), vxy*(-g6), vxy*(-g8)
bs, ls, ms = vss*conjugate(g0), vs0p1*conjugate(g8), vs0p1*conjugate(g6)
ns, ps, hs = vs1p0*conjugate(g8),  vs1p0*conjugate(g6), vxx*conjugate(-g5)
qs, js = vxy*conjugate(-g6), vxy*conjugate(-g8)


# H = Matrix([[a,  b,  0,  l,  0,    2*m],
#             [bs, c,  ns, 0,  2*ps, 0  ],
#             [0,  n,  f,  h,  0,    2*q],
#             [ls, 0,  hs, k,  2*qs, 0  ],
#             [0,  p,  0,  q,  f,    h+j],
#             [ms, 0,  qs, 0,  hs+js,k  ]])

H = Matrix([[a,  vss,  0,  3*vs0p1],
            [vss, c, 3*vs1p0, 0  ],
            [0, vs1p0, f, 2*vxy],
            [vs0p1, 0, 2*vxy, k]])

#H.is_diagonalizable(H)
evals = H.eigenvals(simplify=True, rational=True)

#%%
p1_=np.identity(4)
p1_[2,0] = p1_[3,1] = -1
p1 = LA.inv(p1_)
print(p1_)
print(p1)

#%%vxy
vss = 0; vxy =0
H = Matrix([[a,  vss,  0,  3*vs0p1],
            [vss, c, 3*vs1p0, 0  ],
            [0, vs1p0, f, 2*vxy],
            [vs0p1, 0, 2*vxy, k]])

#H.is_diagonalizable(H)
evals = H.eigenvals(simplify=True, rational=True)
