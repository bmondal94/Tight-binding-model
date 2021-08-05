#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:34:45 2020

@author: bmondal
"""

import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt

# Eigenvectors and eigenvalues of a 2x2 matrix

# l_1 = 0.5*(a+c) - (np.sqrt(((a-c)*0.5)**2)+b**2)
# l_2 = 0.5*(a+c) + (np.sqrt(((a-c)*0.5)**2)+b**2)



H = np.matrix([[-0.09409766+0j,-6.86691385+0j],[-6.86691385+0j,-2.52210234+0j]])
enarray, eigenvectors, eigenvectors2 = LA.eig(H,left=True, right=True)

egp = np.argsort(enarray) # Sort index of the energies
#*********** Using eigenvectors ******************************************
evectors = eigenvectors.T[egp]  # row1 is eigenvector of eigenvalue1
# COHP calculation
COHP={}; kpp=0
for cc in range(len(evectors)):
    C = evectors[cc]
    COHP[str(kpp)+'-'+str(cc)] = C.conjugate().reshape((len(C),1)) * C * H
    
#%%  
H = np.matrix([[-2.,  -5.   ,  1.   , -0.    ],
              [-5.,   0.041,   -0.   ,  1.    ],
              [-1,    -0   ,   0.041,  -0     ],
              [-0,    -1.   ,   -0.,     0.041 ]])

H = np.matrix([[-0.041,  -5.   ,  0. ],
              [-5.,   -2,   -5.   ],
              [-0,    -5   ,   0.041]])

LA.eig(H)

#%% 
k1 = np.linspace(0,1,41)
k11 = np.pi*0.5*k1
g0 = np.cos(k11)
g1 = np.sin(k11) * 1j

f = 1.0414; k = 3.6686; v_xy = 7.8
Q = v_xy * g1

data = []
for q in Q:
    qs = q.conjugate()
    H = np.matrix([[f,q,0,0,0,q],
                   [qs,k,0,0,0,0],
                   [0,0,f,q,0,0],
                   [0,0,qs,k,qs,0],
                   [0,0,0,q,f,0],
                   [qs,0,0,0,0,k]])
    enarray = LA.eigvals(H)
    data.append(np.sort(enarray.real))

#%%
plt.figure()
plt.gca().set_prop_cycle(None)
plt.plot(k1,data)
plt.xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
plt.ylabel("E (eV)")


#%%
H = np.matrix([[-8.343, -3.667,  0.   , -2.178,  0.   ,  0.   ,  0.   ,  0.   ],
        [-3.667,  2.911,  0.   , -1.794,  0.   ,  0.   ,  0.   ,  0.384+1],
        [ 0.   ,  0.   ,  1.041, -9.578,  0.   ,  0.   ,  0.   ,  2.178],
        [-2.178, -1.794, -9.578, -5.487,  0.   ,  0.   ,  0.   , -6.978],
        [ 0.   ,  0.   ,  0.   ,  0.   ,  1.041, -9.578,  0.   ,  2.178],
        [ 0.   ,  0.   ,  0.   ,  0.   , -9.578,  3.669,  9.578, -3.669],
        [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  9.578,  1.041, -7.4  ],
        [ 0.   ,  0.384+1,  2.178, -6.978,  2.178, -3.669, -7.4  , -1.131]])

print(LA.eigvals(H))

#%%
H = np.matrix([[-8.343, -6.451,  0.   , -2.4  ,  0.   , -2.4  ,  0.   , -2.4  ],
        [-6.451, -2.657, -6.784,  0.   , -6.784,  0.   , -6.784,  0.   ],
        [ 0.   , -6.784,  1.041,  0.955,  0.   ,  4.578,  0.   ,  4.578],
        [-2.4  ,  0.   ,  0.955,  3.669,  4.578,  0.   ,  4.578,  0.   ],
        [ 0.   , -6.784,  0.   ,  4.578,  1.041,  0.955,  0.   ,  4.578],
        [-2.4  ,  0.   ,  4.578,  0.   ,  0.955,  3.669,  4.578,  0.   ],
        [ 0.   , -6.784,  0.   ,  4.578,  0.   ,  4.578,  1.041,  0.955],
        [-2.4  ,  0.   ,  4.578,  0.   ,  4.578,  0.   ,  0.955,  3.669]])

H_org = np.copy(H)

H[7] -= H[5] 
H[6] -= H[4]
H[5] -= H[3]
H[4] -= H[2]

H[:,7] -= H[:,5]
H[:,6] -= H[:,4]

H[:,4] += H[:,2]
H[:,5] += H[:,3]
H[5] *= 2
H[4] *= 2

H[:,4] *= 2
H[:,5] *= 2

H[5] += H[7]
H[4] += H[6]

H[:,4] += H[:,6]
H[:,5] += H[:,7]

H[:,4] /= 2; H[:,5] /= 2; H[:,6] /= 2; H[:,7] /= 2
H[5] /= 2
H[4] /= 2

H[:,4] -= H[:,2]
H[:,5] -= H[:,3]

#
H11 = H[:6,0:6]
H22 = H[6:,6:]
print(LA.eigvals(H))
print(LA.eigvals(H11))
print(LA.eigvals(H22))
print(LA.eigvals(H_org))

#%%
P1 = np.identity(8)
P1[2,3] = P1[3,4] = P1[5,6] = P1[6,7] = -1
P1_ = LA.inv(P1)
print(P1)
print(P1_)
#%%
P2_ = np.identity(8)
P2_[4,-2] = P2_[5,-1] = -1
P2_[6,-2] = P2_[7,-1] = 2
P2 = LA.inv(P2_)
print(P2)
print(P2_)
#%%
P3_ = np.copy(P1)
P3_[6,4] = P3_[7,5] = 0
P3 = LA.inv(P3_)
print(P3)
print(P3_)
 #%%
P4_ = np.identity(8)
P4_[4,4] = P4_[5,5] = 0.5
P4 = LA.inv(P4_)
print(P4)
print(P4_)