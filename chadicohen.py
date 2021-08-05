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
from math import pi,cos,sin
import matplotlib.pyplot as plt
from matplotlib import animation

params = {'legend.fontsize': 18,
          'figure.figsize': (10,8),
         'axes.labelsize': 18,
         'axes.titlesize': 18,
         'xtick.labelsize': 18,
         'ytick.labelsize': 18,
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
    return [(k2[0]*i+k1[0]*(1-i),k2[1]*i+k1[1]*(1-i),k2[2]*i+k1[2]*(1-i))
            for i in np.linspace(0,1,n)]   

# ---------------Top of main program------------------
# program defaults:

n = 20
structure = 'GaAs'

# Get command line options:
opts, args = getopt.getopt(sys.argv[1:],'nhs:')

for (key,value) in opts:
    if key == '-n':   n = eval(value)
    if key == '-h':   help_and_exit()
    if key == '-s':   structure = value


# K points (these must be multiplied by 2*pi/a)
kpoints = get_kpoints(n)
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
elif structure == 'GaAs_test':
    e_s_c = -6.01; e_s_a = -6.01; e_p_c = 0.19; e_p_a = 4.59
    v_ss = 7.00
    v_sc_p = 7.28; v_sa_p = 3.70
    v_xx = -0.93; v_xy = -4.72
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
x_ = np.array([np.linalg.norm(np.diff(vkpt,axis=0),axis=1) for vkpt in vkpts])
idx = [0,x_[1][0],x_[2][0],x_[3][0]]
x = np.cumsum(np.concatenate([np.insert(x_[jj],0,idx[jj]) for jj in range(len(x_))]))

H = np.zeros((8,8),dtype = np.complex64)
diag = np.array([e_s_c,e_s_a,e_p_c,e_p_c,e_p_c,e_p_a,e_p_a,e_p_a],dtype=np.complex64) # Make the diagonal elements
np.fill_diagonal(H, diag)

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
ax.set_ylabel("Energy (eV)")
ax.set_xlabel("k-point")
ax.set_ylim(-13,11)
#ax.set_title("Band structure (Chadi) for %s ([$V_{sa-pc},V_{sc-pa}$])" % structure)
#ax.set_title("Band structure (Chadi) for %s " % structure)
#%%
splis = np.linspace(-10,20,51)
ims = []
for i in range(len(splis)):
    data = []
    dataev = []
    #v_sa_p = 2.5
    #v_sc_p = splis[i]
    for (kx,ky,kz) in kpoints:
        kxp,kyp,kzp = kx*pi/2.,ky*pi/2.,kz*pi/2.# The a's cancel here
    
        g0_real = cos(kxp)*cos(kyp)*cos(kzp);  g0_imag = -sin(kxp)*sin(kyp)*sin(kzp)
        g1_real = -cos(kxp)*sin(kyp)*sin(kzp); g1_imag = sin(kxp)*cos(kyp)*cos(kzp)
        g2_real = -sin(kxp)*cos(kyp)*sin(kzp); g2_imag = cos(kxp)*sin(kyp)*cos(kzp)
        g3_real = -sin(kxp)*sin(kyp)*cos(kzp); g3_imag = cos(kxp)*cos(kyp)*sin(kzp)

        # "s" stands for "star": the complex conjugate
        g0,g0s = g0_real+g0_imag*1j,g0_real-g0_imag*1j
        g1,g1s = g1_real+g1_imag*1j,g1_real-g1_imag*1j
        g2,g2s = g2_real+g2_imag*1j,g2_real-g2_imag*1j
        g3,g3s = g3_real+g3_imag*1j,g3_real-g3_imag*1j

        # Make the off-diagonal parts
        H[1,0] = v_ss*g0s;   H[0,1] = v_ss*g0
        H[2,1] = -v_sa_p*g1; H[1,2] = -v_sa_p*g1s
        H[3,1] = -v_sa_p*g2; H[1,3] = -v_sa_p*g2s
        H[4,1] = -v_sa_p*g3; H[1,4] = -v_sa_p*g3s
        H[5,0] = v_sc_p*g1s; H[0,5] = v_sc_p*g1
        H[6,0] = v_sc_p*g2s; H[0,6] = v_sc_p*g2
        H[7,0] = v_sc_p*g3s; H[0,7] = v_sc_p*g3
        H[5,2] = v_xx*g0s;   H[2,5] = v_xx*g0
        H[6,2] = v_xy*g3s;   H[2,6] = v_xy*g3
        H[7,2] = v_xy*g2s;   H[2,7] = v_xy*g2
        H[5,3] = v_xy*g3s;   H[3,5] = v_xy*g3
        H[6,3] = v_xx*g0s;   H[3,6] = v_xx*g0
        H[7,3] = v_xy*g1s;   H[3,7] = v_xy*g1
        H[5,4] = v_xy*g2s;   H[4,5] = v_xy*g2 
        H[6,4] = v_xy*g1s;   H[4,6] = v_xy*g1
        H[7,4] = v_xx*g0s;   H[4,7] = v_xx*g0
    
        #enarray = LA.eigvals(H)
        enarray, eigenvectors = LA.eig(H)
        eigenvalues = enarray.real
        egp = np.argsort(eigenvalues)
        evectors = eigenvectors.T[egp] # row1 is eigenvector of eigenvalue1
        
        dataev.append(abs(evectors)**2*100.) 
        data.append(eigenvalues[egp])
        
    plt.gca().set_prop_cycle(None)
    ii = ax.plot(x,data,c='k')
    ###ax.plot(x,np.array(data)[:,3]) #VB
    ###ax.plot(x,np.array(data)[:,4]) #CB
    #ii = ax.plot(data)
    p = ax.text(0.82, 0.95, '[%.2f,%.2f]'%(v_sa_p,v_sc_p),transform=ax.transAxes)
    ims.append(ii+[p])
    
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
