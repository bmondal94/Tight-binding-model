#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:23:32 2021

@author: bmondal
"""

import os
import sys
import glob
import numpy as np
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import curve_fit

params = {'legend.fontsize': 18,
          'figure.figsize': (10,8),
         'axes.labelsize': 24,
         'axes.titlesize': 24,
         'xtick.labelsize': 24,
         'ytick.labelsize': 24,
         'errorbar.capsize':24}
plt.rcParams.update(params)
np.set_printoptions(precision=3,suppress=True)


'''
#%% optimization simultanously by hstack

# G-X valebce band and last band
vba = adf_band[3]
vba2 = adf_band[7]
av = (vba+vba2)*.5
plt.figure(1)
plt.plot(k1,vba,'.',c='r')
plt.plot(k1,vba2,'.',c='r',label='DFT')
plt.plot(k1,av, 'k--', label='Average (DFT)')

## Curve fit
f = lambda k1, shift, alpha: shift + alpha*k1*k1 #K^2 
f = lambda k1, shift, alpha: shift + alpha*np.cos(np.pi*0.5*k1) # a*cos(K)
f = lambda k1, shift, alpha, beta: shift + alpha*np.cos(np.pi*0.5*k1)+ beta**np.sin(np.pi*0.5*k1) # a*cos(K)+b*sin(K)
f = lambda k1, shift, alpha: shift + np.exp(alpha*k1)*np.cos(np.pi*0.5*k1)
popt, pcov = curve_fit(f, k1, av)
plt.plot(k1,f(k1, *popt), '.-', label='Average (DFT)-curve-fit')
## Polynomial fit
z = np.poly1d(np.polyfit(k1, av, 10))
plt.plot(k1,z(k1), '.-', label='Average (DFT)-poly-fit')

#plt.title(r"$\Gamma \to$X GaAs")
plt.ylabel("Energy (eV)")
plt.xlabel("k-point")
plt.xlim(0,1)
plt.xticks([0,1],["$\Gamma$",'X'])
plt.legend()

E1 = 1.0414; E2 = 2.16686
E3 =1.9546; E4 = 6.534
alpha = 0.66; beta =0.66

def func1(k1, E1,E2,alpha,beta,E3,E4):
    #F = E1+alpha*k1**2
    #K = E2+beta*k1**2
    #F = E1+alpha*np.cos(np.pi*0.5*k1)
    #K = E2+beta*np.cos(np.pi*0.5*k1)
    F = E1+alpha * z(k1)
    K = E2+beta*z(k1)
    
    H2 = E3*E3*(np.cos(np.pi*0.5*k1)**2)
    J2 = -E4*E4*(np.sin(np.pi*0.5*k1)**2)
    f1 = ((F+K)*0.5)+ np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f2 = ((F+K)*0.5)- np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f = np.hstack([f1,f2])
    return f
    
yd = np.hstack([vba2, vba])
xd = np.hstack([k1,k1])
popt, pcov = curve_fit(func1, k1, yd, p0=[E1,E2,alpha,beta,E3,E4], bounds=(0,[10,10,np.inf,np.inf,10,10]))

adata = func1(k1,*popt)
plt.plot(k1,adata[:len(k1)],'m')
plt.plot(k1,adata[len(k1):],'m',label='sp3 quadratic eqn. fit')
plt.legend()
print("Optimized parameters [p1, p2, alpha, beta, vxx, vxy]:\n", popt)


#%%
band1 = adf_band[0]
band2 = adf_band[1]
band3 = adf_band[4]
band4 = adf_band[5]

#band3_ = np.hstack([band3[:12],band4[12:]])
#band4_ = np.hstack([band4[:12],band3[12:]])

band = np.hstack([band1, band2, band3, band4])
k1_ = np.hstack([k1, k1, k1, k1])
y = np.zeros(len(k1_))

a=-10.3431; c=-0.09569; f=1.0414; k=2.16686
vss=-5.0513; vxx=1.9546
vsaspga=1.48; vsgapas=1.839 
initial_guess = [a,c,f,k,vxx,vss,vsgapas,vsaspga]

def f4by4(xx,a,c,f,k,vxx,vss,vsgapas,vsaspga):
    x,k1 = xx
    k11 = pi*0.5*k1
    ck1 = np.cos(k11)**2
    sk1 = np.sin(k11)**2
    ff=(a-x)*(c-x)*(f-x)*(k-x) - (a-x)*(c-x)*vxx*vxx*ck1 \
                            - (f-x)*(k-x)*vss*vss*ck1 \
                            - (a-x)*(k-x)*vsgapas*vsgapas*sk1 \
                            - (f-x)*(c-x)*vsaspga*vsaspga*sk1 \
        +(vss*vxx*ck1 - vsgapas*vsaspga*sk1)**2
        
    return ff

poptt,pconvv = curve_fit(f4by4, (band1,k1), np.zeros(len(k1)), p0=initial_guess)
#poptt,pconvv = curve_fit(f4by4, (band,k1_), y, p0=initial_guess)
f4by4((band1,k1),*poptt)

a,c,f,k,v_xx,v_ss,v_sGa_pAs,v_sAs_pGa = poptt


plt.figure(1)
plt.plot(k1,band1,'.')
plt.plot(k1,band2,'.')
plt.plot(k1,band3,'.')
plt.plot(k1,band4,'.')
'''
#===============================================================================
#%% Quartic equation solution
def f4by4root_v1(k1,a,c,f,k,vxx,vss,vsgapas,vsaspga, Usx_1, Usx_2, KEe1, KEe2, KEe3, KEe4, n_band):
    k11 = pi*0.5*k1
    #k1_2 = z(k1) # k1*k1
    ck1 = np.cos(k11)**2
    sk1 = np.sin(k11)**2
    
    k1_2 = np.cos(k11*2)
    #ckxi1 = k1_2 * k1_2 
    skxi1 = np.sin(k11*2)**2
    sintsinxi = np.sin(k11)*np.sin(k11*2)
    
    KE1, KE2, KE3, KE4 = KEe1*k1_2, KEe2*k1_2, KEe3*k1_2, KEe4*k1_2    
    a, c, f, k = a+KE1, c+KE2, f+KE3, k+KE4
    
    vxxck1 = vxx*vxx*ck1
    vssck1 = vss*vss*ck1
    vsgapassk1 = vsgapas*vsgapas*sk1
    vsaspgask1 = vsaspga*vsaspga*sk1
    
    B = -(a+c+f+k)
    C1 = (a*c+f*k)+(a+c)*(f+k)-(vxxck1+vssck1+vsgapassk1+vsaspgask1)
    C2 = -(Usx_1*Usx_1 + Usx_2*Usx_2)*skxi1
    C = C1+C2
    D1 = -(a*c*(f+k)+f*k*(a+c)) + (a+c)*vxxck1 + (f+k)*vssck1 + (a+k)*vsgapassk1\
        + (f+c)*vsaspgask1
    D2 = (a+f)*Usx_2*Usx_2*skxi1 + (c+k)*Usx_1*Usx_1*skxi1 \
        - 2.*(vss*(vsaspga*Usx_2+vsgapas*Usx_1)+vxx*(vsaspga*Usx_1+vsgapas*Usx_2))*sintsinxi
    D = D1+D2
    E1 = a*c*f*k - a*c*vxxck1 - f*k*vssck1 - a*k*vsgapassk1 - f*c*vsaspgask1 + \
        (vss*vxx*ck1 - vsgapas*vsaspga*sk1)**2
    E2 = -(a*f*Usx_2*Usx_2+c*k*Usx_1*Usx_1)*skxi1\
        -2.*vsgapas*vsaspga*Usx_1*Usx_2*sintsinxi*sintsinxi\
            -2.0*vss*vxx*Usx_1*Usx_2*ck1*skxi1+Usx_1*Usx_1*skxi1*Usx_2*Usx_2*skxi1\
                +2*(vss*(f*vsaspga*Usx_2+k*vsgapas*Usx_1)+vxx*(c*vsaspga*Usx_1+a*vsgapas*Usx_2))*sintsinxi*np.cos(k11)
    E = E1+E2
    p = (8*C - 3*B*B)/8.
    q = (B*B*B - 4*B*C + 8*D)/8.
    Dt = 256*E*E*E - 192*B*D*E*E - 128*C*C*E*E + 144*C*D*D*E - 27*D*D*D*D +\
        144*B*B*C*E*E -6*B*B*D*D*E - 80*B*C*C*D*E + 18*B*C*D*D*D +16*C*C*C*C*E+\
        -4*C*C*C*D*D -27*B*B*B*B*E*E +18*B*B*B*C*D*E -4*(B*D)**3 -4*B*B*C*C*C*E +B*B*C*C*D*D
    Dt_0 = C*C - 3*B*D + 12*E
    Dt_1 = 2*C*C*C - 9*B*C*D + 27*B*B*E + 27*D*D - 72*C*E
    
    if (any(Dt < 0)):
       print(r"$\Delta<0$ case is not allowed. This will give always 2 complex roots")
       sys.exit()
    elif(all(Dt >0)):
        phi = np.arccos(0.5*Dt_1/(Dt_0**(1.5)))
        S =0.5*np.sqrt(-(2/3.)*p + (2/3.)*np.sqrt(Dt_0)*np.cos(phi/3.))
    else:
        print(r"$\Delta=0$ multiple root")
        sys.exit()
    
    root_dic = {}
    root_dic[0] = -(B/4.) -S - 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[1] = -(B/4.) -S + 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[2] = -(B/4.) +S - 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    root_dic[3] = -(B/4.) +S + 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    
    root = []
    for I in n_band: # THis array should be same as 'no_of_band' below
        root = np.hstack([root, root_dic[I]])

    return root

def f4by4root(k1,a,c,f,k,vxx,vss,vsgapas,vsaspga, Usx_1, Usx_2, Uxx_1, Uxx_2, KEe1, KEe2, KEe3, KEe4, n_band):
    k11 = pi*0.5*k1
    ck1 = np.cos(k11)**2
    sk1 = np.sin(k11)**2
    
    k1_2 = np.cos(k11*2)
    skxi1 = np.sin(k11*2)**2
    sintsinxi = np.sin(k11)*np.sin(k11*2)
    
    KE1, KE2, KE3, KE4 = KEe1*k1_2, KEe2*k1_2, KEe3*k1_2, KEe4*k1_2    
    a, c, f, k = a+KEe1+2*KE1, c+KEe2+2*KE2, f+2*KE3+Uxx_1, k+2*KE4+Uxx_2
    
    vxxck1 = vxx*vxx*ck1
    vssck1 = vss*vss*ck1
    vsgapassk1 = vsgapas*vsgapas*sk1
    vsaspgask1 = vsaspga*vsaspga*sk1
    
    B = -(a+c+f+k)
    C1 = (a*c+f*k)+(a+c)*(f+k)-(vxxck1+vssck1+vsgapassk1+vsaspgask1)
    C2 = -4*(Usx_1*Usx_1 + Usx_2*Usx_2)*skxi1
    C = C1+C2
    D1 = -(a*c*(f+k)+f*k*(a+c)) + (a+c)*vxxck1 + (f+k)*vssck1 + (a+k)*vsgapassk1\
        + (f+c)*vsaspgask1
    D2 = (a+f)*4*Usx_2*Usx_2*skxi1 + (c+k)*4*Usx_1*Usx_1*skxi1 \
        - 4.*(vss*(vsaspga*Usx_2+vsgapas*Usx_1)+vxx*(vsaspga*Usx_1+vsgapas*Usx_2))*sintsinxi
    D = D1+D2
    E1 = a*c*f*k - a*c*vxxck1 - f*k*vssck1 - a*k*vsgapassk1 - f*c*vsaspgask1 + \
        (vss*vxx*ck1 - vsgapas*vsaspga*sk1)**2
    E2 = -4*(a*f*Usx_2*Usx_2+c*k*Usx_1*Usx_1)*skxi1\
        -8.*vsgapas*vsaspga*Usx_1*Usx_2*sintsinxi*sintsinxi\
            -8.0*vss*vxx*Usx_1*Usx_2*ck1*skxi1+16.*Usx_1*Usx_1*skxi1*Usx_2*Usx_2*skxi1\
                +2.*(vss*(f*vsaspga*Usx_2+k*vsgapas*Usx_1)+vxx*(c*vsaspga*Usx_1+a*vsgapas*Usx_2))*skxi1
    E = E1+E2
    p = (8*C - 3*B*B)/8.
    q = (B*B*B - 4*B*C + 8*D)/8.
    Dt = 256*E*E*E - 192*B*D*E*E - 128*C*C*E*E + 144*C*D*D*E - 27*D*D*D*D +\
        144*B*B*C*E*E -6*B*B*D*D*E - 80*B*C*C*D*E + 18*B*C*D*D*D +16*C*C*C*C*E+\
        -4*C*C*C*D*D -27*B*B*B*B*E*E +18*B*B*B*C*D*E -4*(B*D)**3 -4*B*B*C*C*C*E +B*B*C*C*D*D
    Dt_0 = C*C - 3*B*D + 12*E
    Dt_1 = 2*C*C*C - 9*B*C*D + 27*B*B*E + 27*D*D - 72*C*E
    
    if (any(Dt< 0)):
       print(r"$\Delta<0$ case is not allowed. This will give always 2 complex roots")
       sys.exit()
    elif(all(Dt >0)):
        phi = np.arccos(0.5*Dt_1/(Dt_0**(1.5)))
        S =0.5*np.sqrt(-(2/3.)*p + (2/3.)*np.sqrt(Dt_0)*np.cos(phi/3.))
    else:
        print(r"$\Delta=0$ multiple root")
        sys.exit()
    
    root_dic = {}
    root_dic[0] = -(B/4.) -S - 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[1] = -(B/4.) -S + 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[2] = -(B/4.) +S - 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    root_dic[3] = -(B/4.) +S + 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    
    root = []
    for I in n_band: # THis array should be same as 'no_of_band' below
        root = np.hstack([root, root_dic[I]])

    return root

def f4by4rootLG(k1_lg,Uxy_1,Uxy_2,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga,Usx_1,Usx_2,Uxx_1,Uxx_2,\
                KEe1, KEe2, KEe3, KEe4, n_band):
    k11 = pi*0.5*k1_lg
    ck1 = np.cos(k11)
    sk1 = np.sin(k11)
    g0 = ck1*ck1*ck1 - 1j*sk1*sk1*sk1
    g1 = (-sk1 + 1j*ck1)*ck1*sk1
    
    k1_2 = np.cos(k11*2)**2
    skxi1 = np.sin(k11*2)**2
    
    KE1, KE2, KE3, KE4 = 3*KEe1*k1_2, 3*KEe2*k1_2, (2*KEe3+Uxx_1)*k1_2, (2*KEe4+Uxx_2)*k1_2    
    a, c, f, k = a+KE1, c+KE2, f+KE3, k+KE4
    
    b = vss*g0; d = vsaspga*g1; e = -vsgapas*g1; jj = vxy*g1; h = vxx*g0
    t = 1j*Usx_1*np.sin(2*pi*k1_lg); r = 1j*Usx_2*np.sin(2*pi*k1_lg)
    X = Uxy_1*skxi1; Y = Uxy_2*skxi1
    bs,ds,es,jjs,hs,ts,rs = np.conjugate((b,d,e,jj,h,t,r))
    
    '''
    #------------------
    from scipy import linalg as LA
    for a8, c8, f8, k8,b8,d8,e8,jj8,h8,t8,r8,bs8,ds8,es8,jjs8,hs8,ts8,rs8,X8,Y8 in\
        zip(a, c, f, k,b,d,e,jj,h,t,r,bs,ds,es,jjs,hs,ts,rs,X,Y):
        H = np.matrix([[a8, b8, t8, d8],[bs8, c8, es8, r8],[3*ts8, 3*e8, f8+2*X8, h8+2*jj8],[3*ds8, 3*rs8, hs8+2*jjs8, k8+2*Y8]])
        print(LA.eigvals(H))
    #--------------------------
    '''
    
    ft1 = f+2*X; ft2 = k+2*Y; ft3 = h+2*jj; ft4 = hs+2*jjs
    B = -(a+c+ft1+ft2)
    C = (a*c-b*bs)+(a+c)*(ft1+ft2)+ft1*ft2-ft3*ft4-3*(t*ts+d*ds+e*es+r*rs)
    D = (b*bs-a*c)*(ft1+ft2)-(a+c)*ft1*ft2+(a+c)*ft3*ft4+3*(a*r*rs+a*e*es+c*d*ds+\
                                                      c*t*ts-b*es*ts-bs*e*t-b*ds*r-bs*d*rs)\
        -3*(ft3*(rs*es+t*ds)+ft4*(r*e+d*ts)-ft1*(d*ds+r*rs)-ft2*(e*es+t*ts))
    
    E = (a*c-b*bs)*(ft1*ft2-ft3*ft4)+9*(ts*rs-e*ds)*(t*r-d*es)+3*(ft1*(rs*bs*d+b*ds*r-d*ds*c-r*rs*a)\
                                                            +ft2*(t*bs*e+ts*es*b-t*ts*c-a*e*es)\
                                                            +ft3*(-b*ds*es-t*rs*bs+c*ds*t+a*es*rs)\
                                                            +ft4*(-b*ts*r-bs*d*e+ts*c*d+a*r*e))
    p = (8*C - 3*B*B)/8.
    q = (B*B*B - 4*B*C + 8*D)/8.
    Dt = 256*E*E*E - 192*B*D*E*E - 128*C*C*E*E + 144*C*D*D*E - 27*D*D*D*D \
        +144*B*B*C*E*E -6*B*B*D*D*E - 80*B*C*C*D*E + 18*B*C*D*D*D +16*C*C*C*C*E+\
        -4*C*C*C*D*D -27*B*B*B*B*E*E +18*B*B*B*C*D*E -4*(B*D)**3 -4*B*B*C*C*C*E +B*B*C*C*D*D
    Dt_0 = C*C - 3*B*D + 12*E
    Dt_1 = 2*C*C*C - 9*B*C*D + 27*B*B*E + 27*D*D - 72*C*E
    
    if (any(Dt<0)):
       print(r"$\Delta<0$ case is not allowed. This will give always 2 complex roots")
       sys.exit()
    elif(all(Dt >0)):
        phi = np.arccos(0.5*Dt_1/(Dt_0**(1.5)))
        S =0.5*np.sqrt(-(2/3.)*p + (2/3.)*np.sqrt(Dt_0)*np.cos(phi/3.))
    else:
        print(r"$\Delta=0$ multiple root")
        sys.exit()
    
    root_dic = {}
    root_dic[0] = -(B/4.) -S - 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[1] = -(B/4.) -S + 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[2] = -(B/4.) +S - 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    root_dic[3] = -(B/4.) +S + 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    
    # Double degenerate bands
    root_dic[4] = 0.5*(f+k-X-Y)-np.sqrt(((f-k-X+Y)*0.5)**2 + (hs-jjs)*(h-jj))
    root_dic[5] = 0.5*(f+k-X-Y)+np.sqrt(((f-k-X+Y)*0.5)**2 + (hs-jjs)*(h-jj))
    
    root = []
    for I in n_band: # THis array should be same as 'no_of_band' below
        root = np.hstack([root, root_dic[I]])

    return np.real(root)

def func_p(k1, E4, ep1, ep2, E1,E2,alpha,beta,E3):
    # k1, vxy, p1, p2, uxx1, uxx2, uxx_pi1, uxx_pi2, vxx
    k1_2 = np.cos(pi*k1)
    F = ep1 + E1+ (E1+alpha)*k1_2
    K = ep2 + E2+ (E2+beta)*k1_2
    H2 = E3*E3*(np.cos(np.pi*0.5*k1)**2)
    J2 = -E4*E4*(np.sin(np.pi*0.5*k1)**2)
    f1 = ((F+K)*0.5)+ np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f2 = ((F+K)*0.5)- np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f = np.hstack([f1,f2])
    return f

def together(k1,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga, Usx_1, Usx_2, Uxx_1, Uxx_2, KEe1, KEe2, KEe3, KEe4, n_band):
    roots1 = f4by4root(k1,a,c,f,k,vxx,vss,vsgapas,vsaspga, Usx_1, Usx_2, Uxx_1, Uxx_2, KEe1, KEe2, KEe3, KEe4, n_band)
    roots2 = func_p(k1, vxy, f, k, KEe3, KEe4, Uxx_1, Uxx_2, vxx)
    return np.hstack([roots1,roots2])

def alltogether(KKK,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga, Usx_1, Usx_2, \
                    Uxx_1, Uxx_2, Uxy_1,Uxy_2, KEe1, KEe2, KEe3, KEe4, n_band,n_band_lg,k1len):
    k1=KKK[:k1len]
    root1 = together(k1,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga, Usx_1, Usx_2, \
                    Uxx_1, Uxx_2, KEe1, KEe2, KEe3, KEe4, n_band)
    
    k1_lg = KKK[k1len:]
    root2 = f4by4rootLG(k1_lg,Uxy_1,Uxy_2,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga,Usx_1,Usx_2,Uxx_1,Uxx_2,\
                KEe1, KEe2, KEe3, KEe4, n_band_lg)
    return np.hstack([root1,root2])
'''
#%%
no_of_band =[0,1,2,3] 
band_d = {}
band_d[0] = adf_band[0]
band_d[1] = adf_band[1]
band_d[2] = adf_band[4]
band_d[3] = adf_band[5]
#band_d[2] = np.hstack([band3[:12],band4[12:]])
#band_d[3] = np.hstack([band4[:12],band3[12:]])
p_band = np.hstack([adf_band[7], adf_band[3]])

a=-10.3431; c=-0.09569; f=1.0414; k=2.16686
vss=-5.0513; vxx=1.9546; vxy = 6.534
vsaspga=1.48; vsgapas=1.839 
initial_guess = [a,c,f,k,vxx,vss,vsgapas,vsaspga]
kin=[0,0,0,0,0,0,0,0]
bound_all=([-12,-5,0,0,0,-10,0,0,-10,-10,-10,-10,-2,-2,-1,-1],[1,5,5,6,10,1,10,10,10,10,10,10,2,2,2,2])

band, k1_ = [], []
for I in no_of_band:
    band = np.hstack([band,band_d[I]])
    k1_ = np.hstack([k1_, k1])

# First fit the 4x4 matrix bands then use the optimized parameter from there to
# optimize 2x2 matrix bands
poptt,pconvv = curve_fit(lambda k1,a,c,f,k,vxx,vss,vsgapas,vsaspga,Usx_1,Usx_2,Uxx_1,Uxx_2,KE1,\
                         KE2,KE3,KE4: f4by4root(k1,a,c,f,k,vxx,vss,vsgapas,\
                        vsaspga,Usx_1,Usx_2,Uxx_1,Uxx_2,KE1,KE2,KE3,KE4,no_of_band), k1, band, \
                        p0=initial_guess+kin, bounds=bound_all)
popt, pcov = curve_fit(lambda k1, E4: func_p(k1,E4,poptt[2],poptt[3],poptt[14],\
                                                                      poptt[15],poptt[10],poptt[11],poptt[4]), \
                       k1, p_band, p0=vxy)

print("Best fit parameters:\n$s_{As}$=%6.3f, s_{Ga}=%6.3f, p_{As}=%6.3f, p_{Ga}=%6.3f\n\
V_{xx}=%6.3f, V_{ss}=%6.3f, V_{sGa-pAs}=%6.3f, V_{sAs-pGa}=%6.3f,U_{sxAs}=%6.3f, U_{sxGA}=%6.3f,\n\
U_{xxAs}^{pi}=%6.3f, U_{xxGA}^{pi}=%6.3f,U_{ssAs}=%6.3f, U_{ssGA}=%6.3f, U_{xxAs}=%6.3f, U_{xxGa}=%6.3f\n\
V_{xy}==%6.3f\n"%tuple(np.hstack([poptt,popt])))


#%%
plot_y = np.split(f4by4root(k1,*poptt,[0,1,2,3]), 4)
pdata = func_p(k1,*popt,poptt[2],poptt[3],poptt[14],poptt[15],poptt[10],poptt[11],poptt[4])
    
plt.figure()
plt.ylabel("E (eV)")
plt.xlabel("k-points")
plt.xlim(0,1)
plt.xticks([0,1],["$\Gamma$",'X'])
for I in range(4):
    plt.plot(k1,band_d[I],'.')
    plt.plot(k1,plot_y[I],'-')
    
# # 2 fold degenerate p-bands
for I in range(2):
    plt.plot(k1,p_band[I*len(k1):(I+1)*len(k1)],'o-')
    plt.plot(k1,pdata[I*len(k1):(I+1)*len(k1)],'--',color='gray')


#plt.legend()
#===============================================================================
'''
#===============================================================================
def getoptimizedparameters(structure='GaAs_sz'):
    if structure == 'GaAs_tz2p':
        foldername='GaAs_Tz2p_smallcore'
        vbindex=17
        mmin = 14; mmax=13 # starting and final bands coloumn index
    elif structure == 'GaAs_sz':
        foldername='GaAs_SZ'
        vbindex=7
        mmin = 4; mmax=0
    
    csv_file = '/home/bmondal/MyFolder/Adf/'+foldername+'/GaAs.results/banddata.csv'
    df = pd.read_csv(csv_file)
    #%%
    mykeys = df.keys()
    print("keys:", df.keys())
    kpoint_pos=np.argwhere(np.diff(df[mykeys[0]])>=1).flatten()
    Efermi=np.amax(df[mykeys[vbindex]]) # maxima of valence band
    #%%
    adf_band = {}; adf_band_LG={}
    J=0
    for I in range(mmin,len(mykeys)-mmax):
        adf_band[J] = (df[mykeys[I]][kpoint_pos[0]:kpoint_pos[1]] - Efermi) * 27.114
        adf_band_LG[J] = (df[mykeys[I]][:kpoint_pos[0]] - Efermi) * 27.114
        J+=1
        
    band_d = {}
    band_d[0] = adf_band[0]
    band_d[1] = adf_band[1]
    band_d[2] = adf_band[4]
    band_d[3] = adf_band[5]
    p_band = np.hstack([adf_band[7], adf_band[3]])
    band_d_LG = {}
    band_d_LG[0] = adf_band_LG[0]
    band_d_LG[1] = adf_band_LG[1]
    band_d_LG[2] = adf_band_LG[4]
    band_d_LG[3] = adf_band_LG[7]
    band_d_LG[4] = adf_band_LG[3]
    band_d_LG[5] = adf_band_LG[5]    
        
    k1 = np.linspace(0,1, len(adf_band[0]))
    k1_lg = np.linspace(0.5,0, len(adf_band_LG[0]))
    #%% Fitting along G-X direction only
    no_of_band =[0,1,2,3] 
    no_of_band_LG =[0,1,2,3,4,5] 
    
    a=-10.3431; c=-0.09569; f=1.0414; k=2.16686
    vss=-5.0513; vxx=1.9546; vxy = 6.534
    vsaspga=1.48; vsgapas=1.839 
    #Uxy_1,Uxy_2 =0.4,0.4
    initial_guess = [a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga]
    kin=[0,0,0,0,0,0,0,0]
    bound_all=([-12,-5,-2,-2,-10,-10,-10,0,0,-10,-10,-10,-10,-2,-2,-1,-1],[1,5,5,6,10,10,1,10,10,10,10,10,10,2,2,2,2])
    
    band = []
    for I in no_of_band:
        band = np.hstack([band,band_d[I]])
    band = np.hstack([band,p_band])
    
    band_LG = []
    for I in no_of_band_LG:
        band_LG = np.hstack([band_LG,band_d_LG[I]])


    # Fit the 8x8 matrix bands 
    poptt,pconvv = curve_fit(lambda k1,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga,Usx_1,Usx_2,Uxx_1,Uxx_2,KEe1,\
                             KEe2,KEe3,KEe4: together(k1,a,c,f,k,vxx,vxy,vss,vsgapas,vsaspga, Usx_1, Usx_2, \
                                                   Uxx_1, Uxx_2, KEe1, KEe2, KEe3, KEe4, no_of_band), k1, band, \
                            p0=initial_guess+kin, bounds=bound_all)
    
    popt,pconv = curve_fit(lambda k1_lg,Uxy_1,Uxy_2:f4by4rootLG(k1_lg,Uxy_1,Uxy_2,*poptt, no_of_band_LG),k1_lg,band_LG)
    
    optimizedparameterset = np.hstack([poptt,popt])
    print("Best fit parameters:\n$s_{As}$=%6.3f, s_{Ga}=%6.3f, p_{As}=%6.3f, p_{Ga}=%6.3f\n\
V_{xx}=%6.3f, V_{xy}==%6.3f, V_{ss}=%6.3f, V_{sGa-pAs}=%6.3f, V_{sAs-pGa}=%6.3f,U_{sxAs}=%6.3f, U_{sxGa}=%6.3f,\n\
U_{xxAs}^{pi}=%6.3f, U_{xxGA}^{pi}=%6.3f,U_{ssAs}=%6.3f, U_{ssGA}=%6.3f, U_{xxAs}=%6.3f, U_{xxGa}=%6.3f\n\
U_{xyAs}=%6.3f,U_{xyGa}=%6.3f\n"%tuple(optimizedparameterset))
    
    
    #%%
    plot_y = np.split(together(k1,*poptt,[0,1,2,3]), 6)
        
    plt.figure()
    plt.ylabel("E (eV)")
    plt.xlabel("k-points")
    plt.xlim(-0.5,1)
    plt.axvline(color='k',linestyle='--')
    plt.xticks([-0.5,0,1],['L',"$\Gamma$",'X'])
    for I in range(4):
        plt.plot(k1,band_d[I],'.')
        plt.plot(k1,plot_y[I],'-')
        
    # # 2 fold degenerate p-bands
    for I in range(2):
        plt.plot(k1,p_band[I*len(k1):(I+1)*len(k1)],'o-')
        plt.plot(k1,plot_y[4+I],'-')
    
    
    plot_y_LG = np.split(f4by4rootLG(k1_lg,*popt,*poptt,[0,1,2,3,4,5]), 6)
    # plt.figure()
    # plt.ylabel("E (eV)")
    # plt.xlabel("k-points")
    # plt.xlim(0,0.5)
    #plt.xticks([0,0.5],["$\Gamma$",'X'])
    for I in range(6):
        plt.plot(-k1_lg,band_d_LG[I],'.')
        plt.plot(-k1_lg,plot_y_LG[I],'-')
    
    #plt.legend()
    return optimizedparameterset

#getoptimizedparameters()