#!/usr/bin/env python

# chadicohen.py Richard P. Muller, 12/99

# This program is the tight-binding program for Diamond/Zincblende
# structures that is presented in Chadi and Cohen's paper
# "Tight-Binding Calculations of the Valence Bands of Diamond and
# Zincblende Crystals", Phys. Stat. Soli. (b) 68, 405 (1975).  This
# program is written for diamond and zincblende structures only.

# Copyright 1999, Richard P. Muller and William A. Goddard, III
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# Here are some sample band gaps (from Kittel) that may aid in fitting:
# C    i 5.4       GaAs d 1.52 
# Si   i 1.17      GaP  i 2.32 
# Ge   i 0.744     GaN    3.5  (not from Kittel) 
# Sn   d 0.00      InP  d 1.42 
#                  InAs d 0.43

import sys,getopt
import numpy as np      
from scipy import linalg as LA
from math import pi
import matplotlib.pyplot as plt
from matplotlib import animation
from DFTfit import getoptimizedparameters

params = {'legend.fontsize': 24,
          'figure.figsize': (10,8),
         'axes.labelsize': 24,
         'axes.titlesize': 24,
         'xtick.labelsize': 24,
         'ytick.labelsize': 24,
         'errorbar.capsize':2}
plt.rcParams.update(params)

#%%

def error_and_exit(line):
    print(line)
    sys.exit()
    return

def help_and_exit():
    help()
    sys.exit()
    return

def help():
    print ("chadicohen.py: Tight-binding band structure of II-VI, III-V,")
    print ("and IV semiconductors. Based on Chadi/Cohen's approach.")
    print ("")
    print ("usage: chadicohen.py [options]")
    print ("")
    print ("Options:")
    print ("-n #  The number of points in each Brillouin zone region ")
    print ("      (default=10)")
    print ("-h    Print this help screen and exit")
    print ("-s    Structure to compute; currently supported are:")
    print ("      C    Diamond")
    print ("      Si   Silicon")
    print ("      Ge   Germanium")
    print ("      GaAs Gallium Arsenide")
    print ("      ZnSe Zinc Selenide")
    print ("")
    print ("Caveats:")
    print ("(1) The parameters in the code are simply taken from Chadi/Cohen.")
    print ("    No checking is performed to make sure that they work for ")
    print ("    the case of interest")
    print ("(2) This program assumes that Gnuplot is installed, and is ")
    print ("    started by the command \"gnuplot\". If this isn't the ")
    print ("    case on your system edit path_to_gnuplot accordingly")
    print ("(3) This program assumes that /usr/bin/env python can find")
    print ("    python on your system. If not, edit the first line of this")
    print ("    file accordingly.")
    print ("(4) This program assumes that the Numeric Extensions to Python")
    print ("    (see ftp://ftp-icf.llnl.gov/pub/python) are installed,")
    print ("    and are in your $PYTHONPATH.")
    print ("")
    print ("References:")
    print ("D.J. Chadi and M.L. Cohen, \"Tight Binding Calculations")
    print ("of the Valence Bands of Diamond and Zincblende Crystals.\"")
    print ("Phys. Stat. Sol. (b) 68, 405 (1975)" )
    return
#%%
def get_kpoints(n):
    L = (0.5,0.5,0.5)
    G = (0,0,0)
    X = (1,0,0)
    U = (1,0.25,0.25)
    K = (0,0.75,0.75)
    LG = kinterpolate(L,G,n+1) 
    GX = kinterpolate(G,X,n+1)
    XU = kinterpolate(X,U,n+1)
    KG = kinterpolate(K,G,n+1)
    return LG+GX[1:]+XU[1:]+KG[1:]

def kinterpolate(k1,k2,n):
    return [[k2[0]*i+k1[0]*(1-i),k2[1]*i+k1[1]*(1-i),k2[2]*i+k1[2]*(1-i)]
            for i in np.linspace(0,1,n)]   

# ---------------Top of main program------------------
# program defaults:

n = 20
structure = 'GaAs_sz'
snn = 1 # include 2nd nearest neighbour

# Get command line options:
opts, args = getopt.getopt(sys.argv[1:],'nhs:')

for (key,value) in opts:
    if key == '-n':   n = eval(value)
    if key == '-h':   help_and_exit()
    if key == '-s':   structure = value


# K points (these must be multiplied by 2*pi/a)
kpoints = np.array(get_kpoints(n))
#print(kpoints)
# Tight binding parameters; these are in eV:
if structure == 'C':
    e_s_c = e_s_a = 0.0     # Arbitrary;
    e_p_c = e_p_a = 7.40 - e_s_c
    v_ss = -15.2
    v_sc_p = v_sa_p = 10.25
    v_xx = 3.0; v_xy = 8.30
elif structure == 'Si':
    e_s_c = e_s_a = 0.0     # Arbitrary
    e_p_c = e_p_a = 7.20 - e_s_c
    v_ss = -8.13
    v_sc_p = v_sa_p = 5.88
    v_xx = 3.17; v_xy = 7.51
elif structure == 'Ge':
    e_s_c = e_s_a = 0.0     # Arbitrary
    e_p_c = e_p_a = 8.41 - e_s_c
    v_ss = -6.78
    v_sc_p = v_sa_p = 5.31
    v_xx = 2.62; v_xy = 6.82
elif structure == 'GaAs':
    e_s_c = -6.01; e_s_a = -4.79; e_p_c = 0.19; e_p_a = 4.59
    v_ss = -7.00
    v_sc_p = 7.28; v_sa_p = 3.70
    v_xx = 0.93; v_xy = 4.72
elif structure == 'GaAs_sz':
    # e_s_c = -5.505; e_s_a = -5.000; e_p_c = 3.818; e_p_a = 2.198
    # U_ss_c = -0.131; U_ss_a = 0.214; U_sx_c = -1.2625; U_sx_a = 0.351
    # U_xx_c = -0.587; U_xx_a = 1.192; U_xx_c_pi = -2.046; U_xx_a_pi = -0.520
    # U_xy_c = .4; U_xy_a = .4
    # v_ss = -6.934
    # v_sc_p = 6.268; v_sa_p = 3.218
    # v_xx = 1.580; v_xy = 6.848
    e_s_c,e_s_a,e_p_c,e_p_a,v_xx,v_xy,v_ss,v_sa_p,v_sc_p,U_sx_c,U_sx_a,\
        U_xx_c_pi,U_xx_a_pi,U_ss_c,U_ss_a,U_xx_c,U_xx_a, \
            U_xy_c,U_xy_a = tuple(getoptimizedparameters(structure=structure))
elif structure == 'GaAs_tz2p':
    e_s_c,e_s_a,e_p_c,e_p_a,v_xx,v_xy,v_ss,v_sa_p,v_sc_p,U_sx_c,U_sx_a,\
        U_xx_c_pi,U_xx_a_pi,U_ss_c,U_ss_a,U_xx_c,U_xx_a, \
            U_xy_c,U_xy_a = tuple(getoptimizedparameters(structure=structure))
elif structure == 'GaAs_vogl':
    e_s_c = -8.3431; e_s_a = -2.6569; e_p_c = 1.0414; e_p_a = 3.6686
    v_ss = -6.4513
    v_sc_p = 4.48; v_sa_p = 5.7839
    v_xx = 1.9546; v_xy = 5.0779
elif structure == 'ZnSe':
    e_s_c = -8.92; e_s_a = -0.28; e_p_c = 0.12; e_p_a = 7.42
    v_ss = -6.14; v_sc_p = 5.47; v_sa_p = 4.73; v_xx = 0.96; v_xy = 4.38
else:
    error_and_exit('Program can\'t cope with structure %s' % structure)

#%%

vkpts =np.split(kpoints,[n+1,2*n+1,3*n+1])
x_ =[np.linalg.norm(np.diff(vkpt,axis=0),axis=1) for vkpt in vkpts]
idx = [0,x_[1][0],x_[2][0],x_[3][0]]
x = np.cumsum(np.concatenate([np.insert(x_[jj],0,idx[jj]) for jj in range(len(x_))]))

xticks = [0,x[n],x[2*n],x[3*n],x[4*n]]
#xticks = [0,n,2*n,3*n,4*n]
xtickslabel = ['L',r"$\Gamma$",'X','U,K',r"$\Gamma$"]

def definefigure():
    fig, ax = plt.subplots()
    print("*** Considering individual orbital energy and comparing with Vogl, here a==Ga, c==As.")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabel)
    ax.axhline(y=0,c='k',ls='--')
    ax.axvline(xticks[1],c='k',ls='--')
    ax.axvline(xticks[2],c='k',ls='--')
    ax.axvline(xticks[3],c='k',ls='--')
    ax.set_xlim(0,x[4*n])
    #ax.set_xlim(0,4*n)
    return fig,ax

orbitals = ["s_c","s_a","px_c","py_c","pz_c","px_a","py_a","pz_a","s","p_c","p_a","p"]
def orbital_contrib(dataev): 
   fig,axx = definefigure()
   print("*** Note: If you are done with either 'BAND' or 'ORBITAL' just press 'ENTER'. It will continue. ")
   while True:
        bandindex = input("Which band do you want to analyze? Band index starts from 1.  ")
        if not bandindex: break
        dataev1 = dataev[:,int(bandindex) -1]
        print(orbitals)
        while True:
            orb_contrib = input("Which orbital do you want to analyze? Choose 1 orbital from above.  ")
            if not orb_contrib: break
            axx.set_ylim(0,100)
            axx.set_ylabel("Orbital contribution (%)")
            axx.set_title("Orbital contributions (Chadi) for %s ([$V_{sa-pc},V_{sc-pa}$])" % structure)
            pos = orbitals.index(orb_contrib)
            if orb_contrib == 's':
                y = np.sum(dataev1[:,0:2], axis=1)
            elif orb_contrib == 'p_c':
                y = np.sum(dataev1[:,2:5], axis=1)
            elif orb_contrib == 'p_a':
                y = np.sum(dataev1[:,5:], axis=1)
            elif orb_contrib == 'p':
                y = np.sum(dataev1[:,2:], axis=1)
            else:
                y = dataev1[:,pos]
                
            iiv = axx.plot(y,"*-",label='band-'+bandindex+'-'+orb_contrib)
            plt.legend()
   return iiv
#%%
fig,ax = definefigure()
ax.set_ylabel("E (eV)")
ax.set_xlabel("k-points")
ax.set_ylim(-13,12)
#ax.set_title("Band structure (Chadi) for %s ([$V_{sa-pc},V_{sc-pa}$])" % structure)
#ax.set_title("Band structure (Chadi) for %s " % structure)

#%%
kpp = kpoints * pi* 0.5
ckpp = np.cos(kpp)
skpp = np.sin(kpp)
ckx, cky, ckz = ckpp[:,0], ckpp[:,1], ckpp[:,2]
skx, sky, skz = skpp[:,0], skpp[:,1], skpp[:,2]
g0 =  ckx*cky*ckz - 1j*skx*sky*skz
g1 = -ckx*sky*skz + 1j*skx*cky*ckz
g2 = -skx*cky*skz + 1j*ckx*sky*ckz
g3 = -skx*sky*ckz + 1j*ckx*cky*skz

g0s, g1s, g2s, g3s = np.conjugate(g0), np.conjugate(g1), np.conjugate(g2), np.conjugate(g3)


H = np.zeros((8,8),dtype = np.complex64)
diag = np.array([e_s_c,e_s_a,e_p_c,e_p_c,e_p_c,e_p_a,e_p_a,e_p_a],dtype=np.complex64) # Make the diagonal elements

if snn:
    kpxi = kpoints * pi
    ckpxi = np.cos(kpxi)
    skpxi = np.sin(kpxi)
    cxi, ceta, czeta = ckpxi[:,0], ckpxi[:,1], ckpxi[:,2]
    sxi, seta, szeta = skpxi[:,0], skpxi[:,1], skpxi[:,2]
    # diagonals
    ss = cxi*ceta + ceta*czeta + czeta*cxi
    xx1 = cxi*(ceta + czeta); xx2 = ceta*czeta
    yy1 = ceta*(cxi + czeta); yy2 = cxi*czeta
    zz1 = czeta*(cxi + ceta); zz2 = cxi*ceta
    # off-diagonals 
    ## s-p interactions
    sx = 1j*sxi  *(ceta + czeta)
    sy = 1j*seta *(cxi  + czeta)
    sz = 1j*szeta*(cxi  + ceta)
    sx_s, sy_s, sz_s = np.conjugate(sx), np.conjugate(sy), np.conjugate(sz)
    ## p-p interactions
    xy = sxi*seta; xz = sxi*szeta; yz = seta*szeta    
    
    
np.fill_diagonal(H, diag)

single = 1 # If only to plot single plot
if single:
    splis = [0]
else:
    splis = np.linspace(-10,10,51)
ims = []
for i in range(len(splis)):
    data = []
    dataev = []
    if not single:
        U_xy_c = -1.10
        U_xy_a = splis[i]
    for ii in range(len(kpoints)):
        # Make the off-diagonal parts
        H[1,0] =   v_ss*g0s[ii]; H[0,1] =     v_ss*g0[ii]
        H[2,1] = -v_sa_p*g1[ii]; H[1,2] = -v_sa_p*g1s[ii]
        H[3,1] = -v_sa_p*g2[ii]; H[1,3] = -v_sa_p*g2s[ii]
        H[4,1] = -v_sa_p*g3[ii]; H[1,4] = -v_sa_p*g3s[ii]
        H[5,0] = v_sc_p*g1s[ii]; H[0,5] =   v_sc_p*g1[ii]
        H[6,0] = v_sc_p*g2s[ii]; H[0,6] =   v_sc_p*g2[ii]
        H[7,0] = v_sc_p*g3s[ii]; H[0,7] =   v_sc_p*g3[ii]
        H[6,2] = H[5,3] = v_xy*g3s[ii];   H[2,6] = H[3,5] = v_xy*g3[ii]
        H[7,2] = H[5,4] = v_xy*g2s[ii];   H[2,7] = H[4,5] = v_xy*g2[ii]
        H[7,3] = H[6,4] = v_xy*g1s[ii];   H[3,7] = H[4,6] = v_xy*g1[ii]
        H[5,2] = H[6,3] = H[7,4] = v_xx*g0s[ii];   H[2,5] = H[3,6] = H[4,7] = v_xx*g0[ii]
        if snn:
            diag2 = np.array([U_ss_c*ss[ii], U_ss_a*ss[ii], U_xx_c*xx1[ii] + U_xx_c_pi*xx2[ii],\
                              U_xx_c*yy1[ii] + U_xx_c_pi*yy2[ii], U_xx_c*zz1[ii] + U_xx_c_pi*zz2[ii],\
                                 U_xx_a*xx1[ii] + U_xx_a_pi*xx2[ii], U_xx_a*yy1[ii] + U_xx_a_pi*yy2[ii],\
                                     U_xx_a*zz1[ii] + U_xx_a_pi*zz2[ii]])
            np.fill_diagonal(H, diag+diag2)
            
            H[2,0] = U_sx_c*sx_s[ii]; H[0,2] = U_sx_c*sx[ii]
            H[3,0] = U_sx_c*sy_s[ii]; H[0,3] = U_sx_c*sy[ii]
            H[4,0] = U_sx_c*sz_s[ii]; H[0,4] = U_sx_c*sz[ii]
            H[5,1] = U_sx_a*sx_s[ii]; H[1,5] = U_sx_a*sx[ii]
            H[6,1] = U_sx_a*sy_s[ii]; H[1,6] = U_sx_a*sy[ii]
            H[7,1] = U_sx_a*sz_s[ii]; H[1,7] = U_sx_a*sz[ii]
            H[3,2] = H[2,3] = U_xy_c*xy[ii]
            H[4,2] = H[2,4] = U_xy_c*xz[ii]
            H[4,3] = H[3,4] = U_xy_c*yz[ii]
            H[6,5] = H[5,6] = U_xy_a*xy[ii]
            H[7,5] = H[5,7] = U_xy_a*xz[ii]
            H[7,6] = H[6,7] = U_xy_a*yz[ii]
    
        #enarray = LA.eigvals(H)
        enarray, eigenvectors = LA.eig(H)
        eigenvalues = enarray.real
        egp = np.argsort(eigenvalues)
        evectors = eigenvectors.T[egp] # row1 is eigenvector of eigenvalue1
        
        dataev.append(abs(evectors)**2*100.) 
        data.append(eigenvalues[egp])
        
    plt.gca().set_prop_cycle(None)
    ii = ax.plot(x,data,c='k')
    p = ax.text(0.82, 0.95, '[%.2f,%.2f]'%(U_xy_c,U_xy_a),transform=ax.transAxes)
    ims.append(ii+[p])

#%%
# Adding DFT
import pandas as pd

if structure == 'GaAs_tz2p':
    foldername='GaAs_Tz2p_smallcore'
    vbindex=17
    mmin = 14; mmax=13
elif structure == 'GaAs_sz':
    foldername='GaAs_SZ'
    vbindex=7
    mmin = 4; mmax=0

csv_file = '/home/bmondal/MyFolder/Adf/'+foldername+'/GaAs.results/banddata.csv'
df = pd.read_csv(csv_file)
mykeys = df.keys()
print("keys:", df.keys())
kpoint_pos=np.argwhere(np.diff(df[mykeys[0]])>=1).flatten()
Efermi=np.amax(df[mykeys[vbindex]]) # maxima of valence band
adf_band = {}
J=0
for I in range(mmin,len(mykeys)-mmax):
    adf_band[J] = (df[mykeys[I]] - Efermi) * 27.114
    J+=1

numar = np.diff(kpoint_pos)
x_new = np.hstack([np.linspace(0,x[n],num=kpoint_pos[0]),np.linspace(x[n],x[2*n],num=numar[0]),\
                np.linspace(x[2*n],x[3*n],num=numar[1]),np.linspace(x[3*n],x[4*n],num=(len(adf_band[0])-kpoint_pos[-1]))])

#plt.figure()
for I in range(len(adf_band)):
    ax.plot(x_new,adf_band[I],'--') 
#%%
if not single:
    Ani = {}
    I = str(np.random.choice(10000,1)[0])
    Ani[I] = animation.ArtistAnimation(fig,ims,interval=500,repeat=True, blit=True,repeat_delay=1000)
    
    def onClick(event):
        global anim_running
        if anim_running:
            Ani[I].event_source.stop()
            anim_running = False
        else:
            Ani[I].event_source.start()
            anim_running = True
            
    anim_running = True   
    fig.canvas.mpl_connect('key_press_event', onClick)
    fig.canvas.mpl_connect('button_press_event', onClick)
    
    plt.show()
    
    #orbital_contrib(np.array(dataev))

