#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:07:52 2020

@author: bmondal
"""

import numpy as np      
from scipy import linalg as LA
from math import pi,cos,sin
import matplotlib.pyplot as plt
from matplotlib import animation

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
########################################################## Only G-X line
structure = 'GaAs_vogl'
e_s_c = -8.3431; e_s_a = -2.6569; e_p_c = 1.0414; e_p_a = 3.6686
v_ss = -6.4513
v_sc_p = 4.48; v_sa_p = 5.7839
v_xx =1.9546; v_xy = 5.0779

H = np.zeros((8,8),dtype = np.complex64)
diag = np.array([e_s_c,e_s_a,e_p_c,e_p_c,e_p_c,e_p_a,e_p_a,e_p_a],dtype=np.complex64) # Make the diagonal elements
np.fill_diagonal(H, diag)

n = 20
xticks = [0,n]
xtickslabel = [r"$\Gamma$",'X']

def definefigure():
    fig, ax = plt.subplots()
    print("*** Considering individual orbital energy and comparing with Vogl, here a==Ga, c==As.")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabel)
    ax.axhline(y=0,c='k',ls='--')
    ax.set_xlim(0,n)
    return fig,ax

fig,ax = definefigure()
ax.set_ylabel("Energy (eV)")
ax.set_title("Band structure (Chadi) for %s ([$V_{sa-pc},V_{sc-pa}$])" % structure)

splis = 1*np.linspace(0,10,11)
ims = []
for i in range(len(splis)):
    data = []
    dataev = []
    #v_sa_p = 3.5
    #v_sc_p = splis[i]
    for kx in np.linspace(0,1,n+1):
        g0 = g0s = cos(kx*pi/2.)
        g1 = sin(kx*pi/2.)*1j; g1s = -g1
        # Make the off-diagonal parts
        H[1,0] = v_ss*g0s;   H[0,1] = v_ss*g0
        H[2,1] = -v_sa_p*g1; H[1,2] = -v_sa_p*g1s
        H[5,0] = v_sc_p*g1s; H[0,5] = v_sc_p*g1
        H[5,2] = v_xx*g0s;   H[2,5] = v_xx*g0
        H[6,3] = v_xx*g0s;   H[3,6] = v_xx*g0
        H[7,4] = v_xx*g0s;   H[4,7] = v_xx*g0
        H[7,3] = v_xy*g1s;   H[3,7] = v_xy*g1
        H[6,4] = v_xy*g1s;   H[4,6] = v_xy*g1

    
        enarray, eigenvectors = LA.eig(H)
        eigenvalues = enarray.real
        egp = np.argsort(eigenvalues)
        evectors = eigenvectors.T[egp] # row1 is eigenvector of eigenvalue1
        
        dataev.append(abs(evectors)**2*100.) 
        data.append(eigenvalues[egp])
        
    plt.gca().set_prop_cycle(None)
    ii = ax.plot(data)
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

# orbital_contrib(np.array(dataev))