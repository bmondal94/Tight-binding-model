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
structure = 'GaAs_test'
e_s_c = -5.505; e_s_a = -5.000; e_p_c = 3.818; e_p_a = 2.198
U_ss_c = -0.131; U_ss_a = 0.214; U_sx_c = -2.525/2; U_sx_a = 0.702/2
U_xx_c = -0.587; U_xx_a = 1.192; U_xx_c_pi = -2.046; U_xx_a_pi = -0.520
U_xy_c = .0; U_xy_a = .0
v_ss = -6.934
v_sc_p = 6.268; v_sa_p = 3.218
v_xx = 1.580; v_xy = 6.848

H = np.zeros((8,8),dtype = np.complex64)
diag = np.array([e_s_c,e_s_a,e_p_c,e_p_c,e_p_c,e_p_a,e_p_a,e_p_a],dtype=np.complex64) # Make the diagonal elements
np.fill_diagonal(H, diag)

n = 20
xticks = [0,n]
xtickslabel = [r"$\Gamma$",'X']

def definefigure():
    fig, ax = plt.subplots(num=1)
    print("*** Considering individual orbital energy and comparing with Vogl, here a==Ga, c==As.")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabel)
    ax.axhline(y=0,c='k',ls='--')
    ax.set_xlim(0,n)
    return fig,ax

fig,ax = definefigure()
ax.set_ylabel("E (eV)")
ax.set_title("Band structure (Chadi) for %s ([$V_{sa-pc},V_{sc-pa}$])" % structure)

splis = [0] #1*np.linspace(0,10,11)
ims = []
snn=1
for i in range(len(splis)):
    data = []
    dataev = []
    #v_sa_p = 3.5
    #v_sc_p = splis[i]
    for kx in np.linspace(0,1,n+1):
        g0 = cos(kx*pi/2.)
        g1 = sin(kx*pi/2.)*1j; g1s = -g1
        # Make the off-diagonal parts
        H[1,0] = H[0,1] = v_ss*g0
        H[2,1] = -v_sa_p*g1; H[1,2] = -v_sa_p*g1s
        H[5,0] = v_sc_p*g1s; H[0,5] = v_sc_p*g1
        H[5,2] = H[6,3] = H[7,4] = H[2,5] = H[3,6] = H[4,7] = v_xx*g0
        H[7,3] = H[6,4] = v_xy*g1s;   H[3,7] = H[4,6] = v_xy*g1
        if snn:
            cxi = cos(kx*pi); sxi = sin(kx*pi)
            diag2 = np.array([2*U_ss_c*cxi+U_ss_c, 2*U_ss_a*cxi+U_ss_a, \
                              2*U_xx_c*cxi+ U_xx_c_pi,\
                              U_xx_c+(U_xx_c+U_xx_c_pi)*cxi, U_xx_c+(U_xx_c+U_xx_c_pi)*cxi,\
                                  2*U_xx_a*cxi+ U_xx_a_pi,\
                              U_xx_a+(U_xx_a+U_xx_a_pi)*cxi, U_xx_a+(U_xx_a+U_xx_a_pi)*cxi])
                                 
            np.fill_diagonal(H, diag+diag2)
            
            H[2,0] = -2j*U_sx_c*sxi; H[0,2] = 2j*U_sx_c*sxi
            H[5,1] = -2j*U_sx_a*sxi; H[1,5] = 2j*U_sx_a*sxi
            
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
    
    
#%%
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