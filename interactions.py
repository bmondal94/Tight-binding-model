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
######################### Initial parameter set ###############################
# %% Vogl_original
structure = 'GaAs_vogl'
a=e_s_As = -8.3431; c=e_s_Ga = -2.6569; f=e_p_As = 1.0414; k=e_p_Ga = 3.6686
v_ss = -6.4513; v_xx =1.9546; v_xy = 5.0779
v_sAs_pGa = 4.48; v_sGa_pAs = 5.7839 

# %% Almost same as Cohen
a=e_s_As = -7.0431;v_ss = -6.8
c=e_s_Ga = -4.5; f=e_p_As = 0
v_sAs_pGa = 7.0; v_sGa_pAs = -3.7
k=e_p_Ga = 4.0686;v_xx =0
v_xy = 4.2

# %% Next try 
f=e_p_As = 4.537; v_xx =1.373
k=e_p_Ga = 7.8

a=e_s_As = -2.522;c=e_s_Ga = -0.094;
v_ss = -6.867

v_sAs_pGa = 4.322; v_sGa_pAs = 6.385

v_xy = 6.5

EFERMI = 4.0915
kin = 0.694
#%%
structure = 'GaAs_vogl'
a=e_s_As = -10.3431; c=e_s_Ga = -0.9569; f=e_p_As = 1.0414; k=e_p_Ga = 3.6686
v_ss = -5.0513; v_xx =1.9546; v_xy = 6.534
v_sAs_pGa = 2.48; v_sGa_pAs = 2.7839 
kin=0.66
# %%
##################### Preparing K-point path ##################################
k1 = np.linspace(0,1,41)
k11 = pi*0.5*k1
g0 = np.cos(k11)
g1 = np.sin(k11) * 1j


#%%
###################### Prepare interaction parameters #########################
# g00 = 1+(g0+g1); g11 = 1-(g0+g1)
# b = v_ss * g00; h = v_xx * g00; jj = v_xy * g11
# d = v_sAs_pGa * g11; e = -v_sGa_pAs*g11

#%%
###################### Prepare interaction parameters #########################
b = v_ss * g0; h = v_xx * g0; jj = v_xy * g1
d = v_sAs_pGa * g1; e = -v_sGa_pAs*g1
# KE
ke = kin*k11*k11
# %%
########### Plot the magnitude of different interaction parameters ############
InteParam = [abs(b),abs(h),abs(jj),abs(d),abs(e)] 
InteParam = [b.real,h.real,jj.real,d.real,e.real] 
InteParam = [b.imag,h.imag,jj.imag,d.imag,e.imag] 
ll = ['$V_{ss}$','$V_{xx}$','$V_{xy}$','$V_{As_s-Ga_p}$','$V_{Ga_p-As_s}$']
plt.figure()
plt.title("Interaction parameters")
plt.xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
plt.ylabel("$V_{ij}$ (eV)")
for II in range(len(InteParam)):
    plt.plot(k1,InteParam[II], label=ll[II])
plt.legend()
plt.show()


# %%
##################### When no s-p mixing, only s-s and p-p mixing #############
E_G_s_gap = 2*np.sqrt(((a-c)*0.5)**2 + abs(b)**2)
E_s_G2 = (a+c)*0.5 + E_G_s_gap *0.5
E_s_G1 = (a+c)*0.5 - E_G_s_gap *0.5
E_G_p_gap = 2*np.sqrt(((f-k)*0.5)**2 + abs(h)**2)
E_p_G2 = (f+k)*0.5 + E_G_p_gap *0.5
E_p_G1 = (f+k)*0.5 - E_G_p_gap *0.5

plt.figure(5)
plt.plot(k1,E_s_G1,"--", label = "$E_{As_s-Ga_s}(\Gamma)_1$")
plt.plot(k1,E_p_G1,"--", label = "$E_{As_p-Ga_p}(\Gamma)_1$")
plt.plot(k1,E_s_G2,"--", label = "$E_{Ga_s-As_s}(\Gamma)_2$")
plt.plot(k1,E_p_G2,"--", label = "$E_{As_p-Ga_p}(\Gamma)_2$")
plt.xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
plt.ylabel("E (eV)")
plt.title("Pure s-s and p-p mixing (no s-p mixing)")
plt.legend()


# %%
################## No s-s and p-p mixing, pure s-p mixing #####################
E_x_sp_gap = 2*np.sqrt(((a-k)*0.5)**2 + (abs(d))**2)
E_sp_x2 = (a+k)*0.5 + E_x_sp_gap *0.5
E_sp_x1 = (a+k)*0.5 - E_x_sp_gap *0.5
E_x_sp_gap2 = 2*np.sqrt(((c-f)*0.5)**2 + (abs(e))**2)
E_sp_x4 = (f+c)*0.5 + E_x_sp_gap2 *0.5
E_sp_x3 = (f+c)*0.5 - E_x_sp_gap2 *0.5

plt.figure(7)
plt.gca().set_prop_cycle(None)
plt.plot(k1,E_sp_x1,".-", label = "$E_{As_s-Ga_p}(X)_1$")
plt.plot(k1,E_sp_x3,".-", label = "$E_{As_p-Ga_s}(X)_1$")
plt.plot(k1,E_sp_x2,".-", label = "$E_{Ga_p-As_s}(X)_2$")
plt.plot(k1,E_sp_x4,".-", label = "$E_{Ga_s-As_p}(X)_2$")
plt.xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
plt.ylabel("E (eV)")
plt.title("Pure s-p mixing (no s-s or p-p mixing)")
plt.legend()


# %%
################# Final plot with total matrix ################################
data = []; data2 = []; dataev=[]; H_t = {}; COHP={}; kpp=0
HH0 = np.zeros((4,4)); H_00 = np.zeros((2,2))

for B, D, E, H_,J,KE in zip(b,d,e,h,jj,ke):
    Bs, Es, Ds, H_s, Js = np.conjugate((B, E, D, H_, J))
    a_, c_, f_, k_ = a+KE-0.1, c+KE-0.1, f+KE, k+KE 
    #a_, c_, f_, k_ = a, c, f, k
    #=============================================================================
    H11 = np.array([[a_, B],[Bs, c_]])
    H12 = np.array([[0, D],[Es, 0]])
    H21 = np.array([[0, E],[Ds, 0]])
    H22 = np.array([[f_, H_],[H_s, k_]])
    H_p1 = np.bmat([[H11,H12],[H21,H22]])

    enarray, eigenvectors = LA.eig(H_p1)
    egp = np.argsort(enarray) # Sort index of the energies

    #*********** Using eigenvectors ******************************************
    evectors = eigenvectors.T[egp]  # row1 is eigenvector of eigenvalue1
    # COHP calculation
    for cc in range(len(evectors)):
        C = evectors[cc]
        COHP[(kpp,cc)] = C.conjugate().reshape((len(C),1)) * C * H_p1
    
    # Orbital contributions 
    dataev.append(abs(evectors)**2*100.)
    
    
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
    
    enarray = LA.eigvals(H_p2) #+KE
    data2.append(np.sort(enarray.real))
    
    # Save the final matrix
    H_tt = np.bmat([[H_p1,HH0],[HH0,H_p2]])
    H_t[(kpp,0)], H_t[(kpp,1)] = H_tt.real, H_tt.imag

    kpp += 1
    
## Rescaling with Fermi energy
#data = np.array(data) - EFERMI
#data2 = np.array(data2) - EFERMI
#%%
plt.figure(1)
plt.gca().set_prop_cycle(None)
plt.plot(k1,data)
plt.plot(k1,data2)
plt.xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
plt.ylabel("E (eV)")


# %% 
################ Plot only E(s)_gap ###########################################
plt.figure(77)
plt.title("$E_{s}^{gap}$")
plt.plot(k1,E_G_s_gap,label = "$V_{ss}=%6.3f$"%v_ss)
plt.xlabel("k-points")
plt.ylabel("E(eV)")
plt.legend()

# %%
########################### Vary the s-p mixing parameters ####################
#v_sAs_pGa = 0; v_sGa_pAs = -2.5

fig,ax = plt.subplots()
ax.set_ylabel("Energy (eV)")
ax.set_xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
ims = []
lim = 10 #abs(v_ss)
VsAspGa = np.linspace(0,lim,51)
full_data = []

for var in VsAspGa:
    #h = var * g0
    e = -var*g1
    #d = var * g11; e = -var*g11
    data = []; data2 = []
    for B, D, E, H_,J,KE in zip(b,d,e,h,jj,ke):
        Bs, Es, Ds, H_s, Js = np.conjugate((B, E, D, H_, J))
        #print(B,D,E,H_)
#=============================================================================
        H11 = np.array([[a, B],[Bs, c]])
        H12 = np.array([[0, D],[Es, 0]])
        H21 = np.array([[0, E],[Ds, 0]])
        H22 = np.array([[f, H_],[H_s,k]])
        H_p1 = np.bmat([[H11,H12],[H21,H22]])
        
        enarray = LA.eigvals(H_p1)+KE
        data.append(np.sort(enarray.real))
        
#=============================================================================
        
        # Double degenerate p-orbitals
        H11_p = np.array([[f, H_],[H_s,k]])
        H12_p = np.array([[0, J],[Js, 0]])
        H_p2 = np.bmat([[H11_p,H12_p],[H12_p,H11_p]])
        
        enarray = LA.eigvals(H_p2)+KE
        data2.append(np.sort(enarray.real))
        
    full_data.append(data)
    plt.gca().set_prop_cycle(None)
    ii = ax.plot(k1,data) #+ ax.plot(k1,data2)
    #p=ax.text(0.22, 0.95,"$V_{As_{p_i}-Ga_{p_j}}$ = %6.3f, $V_{As_{p_i}-Ga_{p_i}}$ = %6.3f"%(v_xy,var),transform=ax.transAxes)
    p=ax.text(0.22, 0.95,"$V_{As_s-Ga_p}$ = %6.3f, $V_{Ga_s-As_p}$ = %6.3f"%(v_sAs_pGa,var),transform=ax.transAxes)
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
    
# %%=============================================================================
################### With back ground gray old paths ##############################
ims = []
fig,ax = plt.subplots()
ax.set_ylabel("Energy (eV)")
ax.set_xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
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

def onClick(event):
    global anim_running
    if anim_running:
        Ani.event_source.stop()
        anim_running = False
    else:
        Ani.event_source.start()
        anim_running = True
        
anim_running = True   
fig.canvas.mpl_connect('key_press_event', onClick)
fig.canvas.mpl_connect('button_press_event', onClick)

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

def onClick(event):
    global anim_running
    if anim_running:
        simulation.event_source.stop()
        anim_running = False
    else:
        simulation.event_source.start()
        anim_running = True
        
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
