#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:54:51 2020

@author: bmondal
"""

import numpy as np      
from scipy import linalg as LA
from math import pi
import matplotlib.pyplot as plt
from matplotlib import animation

np.set_printoptions(precision=3,suppress=True)

#%% Function definitions
HH0 = np.zeros((4,4)); H_00 = np.zeros((2,2))
def HamilTonian(b,d,e,h,jj,ke0,ke1,ke2,ke3, cal_orb=False, cal_cohp=False, store_H=False):
    '''
    Input: b,d,e,h,jj,ke0,ke1,ke2,ke3, add_KE=True, cal_orb=False, cal_cohp=False, store_H=False
            b=v_ss*g0,; 
            d=v_sAs_pGa * g1
            e = -v_sGa_pAs*g1
            h = v_xx * g0
            jj = v_xy * g1
            kin=np.array([[s_As_KE],[s_Ga_KE],[p_As_KE],[p_Ga_KE]])
            ke = kin*kpoints*kpoints
            ke0, ke1, ke2, ke3 = s_As_KE,s_Ga_KE,p_As_KE,p_Ga_KE
            a=e_s_As; c=e_s_Ga
            f=e_p_As; k=e_p_Ga
            EFERMI = fermi energy
            
            add_KE == adding kinetic energy (default: False)
            cal_orb == calculate orbital contributions from eigenvectors (default: False)
            cal_COHP == calculate COHP (default: False)
            stor_H == store the hamiltonian matrix (default: False)
    
    Return: data, data2, H_t, COHP, dataev
            Band_energies from 4x4 matrix, Band_energies from 2x2 matrix,
            Total Hamiltonian matrix, COHP matrix, Orbital_contributions matrix           
    '''
    
    data = []; data2 = []; dataev=[]; H_t = {}; COHP={}; kpp=0
    for B, D, E, H_,J,KE1,KE2,KE3,KE4 in zip(b,d,e,h,jj,ke0,ke1,ke2,ke3):
        Bs, Es, Ds, H_s, Js = np.conjugate((B, E, D, H_, J))
        a_, c_, f_, k_ = a+KE1, c+KE2, f+KE3, k+KE4 
        #=============================================================================
        H11 = np.array([[a_, B],[Bs, c_]])
        H12 = np.array([[0, D],[Es, 0]])
        H21 = np.array([[0, E],[Ds, 0]])
        H22 = np.array([[f_, H_],[H_s, k_]])
        H_p1 = np.bmat([[H11,H12],[H21,H22]])
    
        if cal_orb:
            enarray, eigenvectors = LA.eig(H_p1)
        else:
            enarray =  LA.eigvals(H_p1)
        egp = np.argsort(enarray) # Sort index of the energies    

        #*********** Using eigenvalues ******************************************
        # Energy eigenvalues ==> band energies
        #print(kpp,enarray.imag)
        data.append(enarray[egp].real) #+KE)
    
        # enarray = LA.eigvals(H_p1)
        # data.append(np.sort(enarray.real))
        #=============================================================================
        
        # Double degenerate p-orbitals
        H11_p = np.array([[f_, H_],[H_s, k_]])
        H12_p = np.array([[0, J],[Js, 0]])
        H_p2 = np.bmat([[H11_p,H12_p],[H12_p,H11_p]])
        # Double degenerate p-orbitals, Alternative form different basis
        # H11_p = np.array([[f_, H_-J],[H_s-Js,k_]])
        # H22_p = np.array([[f_, H_+J],[H_s+Js,k_]])
        # H12_p = np.array([[0, J],[Js, 0]])
        # H_p2 = np.bmat([[H11_p,H12_p],[H_00,H22_p]])
        
        if cal_orb:
            enarray, eigenvectors = LA.eig(H_p2)
        else:
            enarray =  LA.eigvals(H_p2)
        
        enarray = LA.eigvals(H_p2) #+KE
        data2.append(np.sort(enarray.real)) #+KE)
        
        #*********** Eigenvectors ******************************************
        if cal_orb:
            evectors = eigenvectors.T[egp]  # row1 is eigenvector of eigenvalue1
            # Orbital contributions 
            dataev.append(abs(evectors)**2*100.)
        if cal_cohp:
            # COHP calculation
            for cc in range(len(evectors)):
                C = evectors[cc]
                COHP[(kpp,cc)] = C.conjugate().reshape((len(C),1)) * C * H_p1
        if store_H:
        # Save the final matrix
            H_tt = np.bmat([[H_p1,HH0],[HH0,H_p2]])
            H_t[(kpp,0)], H_t[(kpp,1)] = H_tt.real, H_tt.imag
    
        kpp += 1
        
    return data, data2, H_t, COHP, dataev

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

def onClick(event):
    global anim_running
    if anim_running:
        Ani[I].event_source.stop()
        anim_running = False
    else:
        Ani[I].event_source.start()
        anim_running = True

# %% Next try 
f=e_p_As = 4.537; v_xx =1.373
k=e_p_Ga = 7.8
a=e_s_As = -4.522;c=e_s_Ga = -0.094;
v_ss = -6.867
v_sAs_pGa = 4.322; v_sGa_pAs = 2.385
v_xy = 6.5
EFERMI = 4.0915
kin=np.array([[0.694],[.194],[0.694],[0.694]])
#kin = 0.694
#%%
structure = 'GaAs_vogl'
a=e_s_As = -10.3431; c=e_s_Ga = -0.09569; f=e_p_As = 1.0414; k=e_p_Ga = 2.16686
v_ss = -5.0513; v_xx =1.9546; v_xy = 6.534
v_sAs_pGa = 1.48; v_sGa_pAs =1.839 
EFERMI = 0
# Ga-s, As-s, Ga-p, As-p
#kin=np.array([[0.],[0.],[0.],[0.]])
kin=np.array([[2],[2],[2],[2]])
# %%
##################### Preparing K-point path ##################################
k1 = np.linspace(0,1,41)
k11 = pi*0.5*k1
g0 = np.cos(k11)
g1 = np.sin(k11) * 1j

#%%
###################### Prepare interaction parameters #########################
b = v_ss * g0; h = v_xx * g0; jj = v_xy * g1
d = v_sAs_pGa * g1; e = -v_sGa_pAs*g1
# KE
ke = kin*k1*k1
ke = kin*np.cos(k11)

# %%
################# Final plot with total matrix ################################
data, data2, _, _, _ = HamilTonian(b,d,e,h,jj,ke[0],ke[1],ke[2],ke[3])
## Rescaling with Fermi energy
data = np.array(data) - EFERMI
data2 = np.array(data2) - EFERMI
#%%
plt.figure(1)
plt.gca().set_prop_cycle(None)
plt.plot(k1,data,'k--')
plt.plot(k1,data2,'m-')
plt.xlabel('k-point')
plt.ylabel("Energy (eV)")

plt.xlim(0,1)
plt.xticks([0,1],["$\Gamma$",'X'])
plt.plot([],[],'m-',label='KE')
#plt.legend()
# %%
########################### Vary the s-p mixing parameters ####################
#v_sAs_pGa = 0; v_sGa_pAs = -2.5

fig,ax = plt.subplots()
ax.set_ylabel("Energy (eV)")
ax.set_xlabel('k-point')
ax.set_xlim(0,1)
ax.set_xticks([0,1])
ax.set_xticklabels(["$\Gamma$",'X'])
ims = []
lim = 10 #abs(v_ss)
VsAspGa = np.linspace(0,lim,51)
full_data = []

for var in VsAspGa:
    #h = var * g0
    e = -var*g1
    #d = var * g11; e = -var*g11   
    data, data2, H_t, _, _ = HamilTonian(b,d,e,h,jj,ke[0],ke[1],ke[2],ke[3])       
    full_data.append(data)
    plt.gca().set_prop_cycle(None)
    ii = ax.plot(k1,data) + ax.plot(k1,data2,color='gray')
    #p=ax.text(0.22, 0.95,"$V_{As_{p_i}-Ga_{p_j}}$ = %6.3f, $V_{As_{p_i}-Ga_{p_i}}$ = %6.3f"%(v_xy,var),transform=ax.transAxes)
    p=ax.text(0.22, 0.95,"$V_{As_s-Ga_p}$ = %6.3f, $V_{Ga_s-As_p}$ = %6.3f"%(v_sAs_pGa,var),transform=ax.transAxes)
    ims.append(ii+[p])
    
Ani = animation.ArtistAnimation(fig,ims,interval=500,repeat=True, blit=True,repeat_delay=1000)      
anim_running = True   
fig.canvas.mpl_connect('key_press_event', onClick)
fig.canvas.mpl_connect('button_press_event', onClick)
    
# %%=============================================================================
################### With back ground gray old paths ##############################
ims = []
fig,ax = plt.subplots()
ax.set_ylabel("Energy (eV)")
ax.set_xlabel('k-point')
ax.set_xlim(0,1)
for i in range(len(full_data)):
    j=0; ii=[]
    while j < i :
        ii += ax.plot(k1,full_data[j],c="lightgray") #+ ax.plot(k1,data2)
        j+=1
    plt.gca().set_prop_cycle(None)
    ii += ax.plot(k1,full_data[i])
    p=ax.text(0.22, 0.95,"$V_{As_s-Ga_p}$ = %6.3f, $V_{Ga_s-As_p}$ = %6.3f"%(VsAspGa[i],v_sGa_pAs),transform=ax.transAxes)
    ims.append(ii+[p])
    
Ani = animation.ArtistAnimation(fig,ims,interval=500,repeat=True, blit=True,repeat_delay=1000)       
anim_running = True   
fig.canvas.mpl_connect('key_press_event', onClick)
fig.canvas.mpl_connect('button_press_event', onClick)

#%% Roots of quadratic equation
LobsterFit = False
funcdata = {}
rootlim = np.linspace(-13, 5, num=200) # defines the band energy range (ultimate y-axis limit in bandstructure).
if LobsterFit:
    print("The variables are used from actual DFT band data fit from LobsterBand.py")
else:
    poptt = [a,c,f,k,v_xx,v_ss,v_sGa_pAs,v_sAs_pGa]

param = poptt
for I in range(len(k1)):
    funcdata[I] = f4by4((rootlim,k1[I]),*param)

fig,ax = plt.subplots()
ax.set_ylabel("$f$")
ax.set_xlabel('x')
ax.set_ylim(-200,200)
im = ax.plot(rootlim,funcdata[0])[0]
text=ax.text(0.8,0.95,"$k_x$ = %.2f"%k1[0],transform=ax.transAxes)
ax.axhline(y=0,color='k',ls='--')
ax.axvline(x=0,color='k',ls='--')

def updateData(I):
    im.set_data(rootlim,funcdata[I])
    text.set_text("$k_x$ = %.2f"%k1[I])
    return im, text

anim_running = True   
fig.canvas.mpl_connect('key_press_event', onClick)
fig.canvas.mpl_connect('button_press_event', onClick)
Ani = animation.FuncAnimation(fig,updateData,interval=500,frames=len(k1),repeat=True, blit=False,repeat_delay=1)

#%% Quartic equation solution
ck1 = np.cos(k11)**2
sk1 = np.sin(k11)**2
a,c,f,k,vxx,vss,vsgapas,vsaspga = poptt
vxxck1 = vxx*vxx*ck1
vssck1 = vss*vss*ck1
vsgapassk1 = vsgapas*vsgapas*sk1
vsaspgask1 = vsaspga*vsaspga*sk1

B = -(a+c+f+k)
C = (a*c+f*k)+(a+c)*(f+k)-(vxxck1+vssck1+vsgapassk1+vsaspgask1)
D = -(a*c*(f+k)+f*k*(a+c)) + (a+c)*vxxck1 + (f+k)*vssck1 + (a+k)*vsgapassk1 + (f+c)*vsaspgask1
E = a*c*f*k - a*c*vxxck1 - f*k*vssck1 - a*k*vsgapassk1 - f*c*vsaspgask1 + (vss*vxx*ck1 - vsgapas*vsaspga*sk1)**2

p = (8*C - 3*B*B)/8.
q = (B*B*B - 4*B*C + 8*D)/8.
Dt = 256*E*E*E - 192*B*D*E*E - 128*C*C*E*E + 144*C*D*D*E - 27*D*D*D*D +\
    144*B*B*C*E*E -6*B*B*D*D*E - 80*B*C*C*D*E + 18*B*C*D*D*D +16*C*C*C*C*E+\
        -4*C*C*C*D*D -27*B*B*B*B*E*E +18*B*B*B*C*D*E -4*(B*D)**3 -4*B*B*C*C*C*E +B*B*C*C*D*D
Dt_0 = C*C - 3*B*D + 12*E
Dt_1 = 2*C*C*C - 9*B*C*D + 27*B*B*E + 27*D*D - 72*C*E

if (Dt.any() < 0):
   print(r"$\Delta<0$ case is not allowed. This will give always 2 complex roots")
elif(Dt.all() >0):
    phi = np.arccos(0.5*Dt_1/(Dt_0**(1.5)))
    S =0.5*np.sqrt(-(2/3.)*p + (2/3.)*np.sqrt(Dt_0)*np.cos(phi/3.))
else:
    print(r"$\Delta=0$ multiple root")


root2 = -(B/4.) -S + 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
root1 = -(B/4.) -S - 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
root4 = -(B/4.) +S + 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
root3 = -(B/4.) +S - 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))

plt.figure(1)
plt.plot(k1,root1,label='root1')
plt.plot(k1,root2,label='root2')
plt.plot(k1,root3,label='root3')
plt.plot(k1,root4,label='root4')
plt.legend()

# %%
####################################################################
E_G_s_gap = 14.13
#alpha_s = 0.25*(e_s_As - e_s_Ga)**2
cg_s = 0.25* E_G_s_gap**2
beta_s = np.linspace(0,45,31)
alpha_s = - beta_s + cg_s

plt.figure()
plt.xlabel(r'-$V_{ss}$')
plt.ylabel(r'$\Delta E_s /2$')
plt.plot(np.sqrt(beta_s), np.sqrt(alpha_s), "*-")

#%% Plot the hamiltonian matrix
orbitals = ["$s_a$","$s_c$","$pz_a$","$pz_c$","$px_a$","$px_c$","$py_a$","$py_c$"]
t=list(H_t.values())
vminn, vmaxx = np.amin(t), np.amax(t)
NBASISFUNCTION = 8
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,6)) 
title_text = "Total Hamiltonian matrix\n Real part \t\t\t\t\t\t\t\t\t  Imaginary part ".expandtabs()
fig.suptitle(title_text)                              

for j in range(2):
    im = axs[j].matshow(H_t[(0,j)], vmin=vminn, vmax = vmaxx)
#fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.12, 0.01, 0.75])
ccbar=fig.colorbar(im, cax=cbar_ax)
ccbar.ax.tick_params(labelsize=10)
ticks = np.arange(NBASISFUNCTION)

def updateData(I):
    for j in range(2):
        axs[j].clear()
        axs[j].matshow(H_t[(I,j)], vmin=vminn, vmax = vmaxx) 
        axs[j].set_title("k-point {kp}".format(kp=I), y=-0.1)
        axs[j].set_xticks(ticks)
        axs[j].set_yticks(ticks)
        axs[j].set_xticklabels(orbitals,fontsize=10)
        axs[j].set_yticklabels(orbitals,fontsize=10)
        axs[j].vlines([1.5,3.5,5.5],[-0.5,-0.5,3.5],[3.5,7.5,7.5],colors='k')
        axs[j].hlines([1.5,3.5,5.5],[-0.5,-0.5,3.5],[3.5,7.5,7.5],colors='k')
        plt.setp(axs[j].get_xticklabels(), rotation=90, ha="center")
        # Loop over data dimensions and create text annotations.
        for k in ticks:
            for l in ticks:
                axs[j].text(l, k, "{:.2f}".format(H_t[(I,j)][k, l]),ha="center", va="center", color="w")

anim_running = True   
fig.canvas.mpl_connect('key_press_event', onClick)
fig.canvas.mpl_connect('button_press_event', onClick)
simulation = animation.FuncAnimation(fig, updateData, frames=40, interval=500, repeat=False, blit=False) #, repeat_delay=1000)

savefig = False
if savefig:
    print('*** Saving movie')
    movdirname = "/home/bmondal/MyFolder/LobsterAnalysis/Movies/GaAs"
    filname = movdirname+'/GaAs_TB_H-matrix.mp4'
    simulation.save(filname, fps=2, dpi=300)
