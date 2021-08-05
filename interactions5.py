#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:23:05 2021

@author: bmondal
"""
import sys
import numpy as np      
from scipy import linalg as LA
from math import pi
import matplotlib.pyplot as plt
from matplotlib import animation

np.set_printoptions(precision=3,linewidth=100,suppress=True)
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.getH(), rtol=rtol, atol=atol)
######################### Initial parameter set ###############################
# %% 
structure = 'GaAs_vogl'
a=e_s_As = -8.3431; c=e_s_Ga = -2.6569; f=e_p_As = 2.414; k=e_p_Ga = 2.1
v_ss = -6.4513; v_xx =6.9546; v_xy = 1.
v_sAs_pGa = 1.48; v_sGa_pAs = 1.7839 
#%% Vogl_original
# structure = 'GaAs_vogl'
# a=e_s_As = -1.3431; c=e_s_Ga = 0.6569; f=e_p_As = 0.0414; k=e_p_Ga = 0.06686
# v_ss = -0.4513; v_xx =0.9546; v_xy = 1.3
# v_sAs_pGa = 0.95; v_sGa_pAs = 0.82

# %%
##################### Preparing K-point path ##################################
k1 = np.linspace(0,1,41)
k11 = pi*0.5*k1
g0 = np.cos(k11)
g1 = np.sin(k11) * 1j


#%%
###################### Prepare interaction parameters #########################
# g5 = (g1-g0)
# g6 = g7 =  ((g0+g1)) #-np.ones(len(g0),dtype=float)
# g8 = (-g5)
#g0 = 1+g0+g1
g5 = g1
g6 = g7 =  ((g0+g1)) #-np.ones(len(g0),dtype=float)
g8 = g0

#%%
b = v_ss * g0; h = v_xx * (-g5); jj = v_xy * (-g8) 
l = v_sAs_pGa * g8 *0
n = v_sGa_pAs * g8*0
m = v_sAs_pGa * g6 ; p = v_sGa_pAs * g6; q = v_xy * (-g7)
m_ = v_sAs_pGa * g7 ; p_ = v_sGa_pAs * g7; q_ = v_xy * (-g6)

#%%
def HamiltonianEVAL_v4(a,k,f,c,b,l,m,n,p,h,jj,q,m_,p_,q_):
    data1 = []; data2 = []; H_t = {}
    for ind, (B, L, M, N, P, H_, J, Q, M_,P_,Q_) in enumerate(zip(b,l,m,n,p,h,jj,q,m_,p_,q_)):
        Bs, Ls, Ms, Ns, Ps, H_s, Js, Qs, M_s,P_s,Q_s = np.conjugate((B, L, M, N, P, H_, J, Q, M_,P_,Q_))
        H11 = np.matrix([[a,   B,  0,   L,  0,  2*M], 
                        [Bs,  c,  Ns,  0,  2*Ps, 0], 
                        [0,   N,  f,   H_, 0,  2*Q], 
                        [Ls,  0,  H_s, k,  2*Qs, 0],
                        [0,   P,  0,   Q,  f,        (H_+J)],
                        [Ms,  0,  Qs,  0,  (H_s+Js), k]])

        H22 = np.matrix([[f,       H_-J], 
                        [H_s-Js,  k]]) 
 
        if not (check_symmetric(H11)):
            pass
            #print("*** Non hermitian matrix found for matrix index %d"%ind)
            #print(H11)
            #sys.exit()
        
        H_t[(ind,0)], H_t[(ind,1)] = H11.real, H11.imag
        
        enarray = LA.eigvals(H11)
        enarray2 = LA.eigvals(H22)
        #print(enarray.imag)
        data1.append(np.sort(enarray.real))
        data2.append(np.sort(enarray2.real))
    return data1, data2, H_t

data1,data2,Haa = HamiltonianEVAL_v4(a,k,f,c,b,l,m,n,p,h,jj,q,m_,p_,q_)

#%% 8*8 full matrix format
b = v_ss * g0
l = v_sAs_pGa * g5
n = v_sGa_pAs * g5
m = v_sAs_pGa * g5; p = v_sGa_pAs * g5
m_ = v_sAs_pGa * g5; p_ = v_sGa_pAs * g5
h1 = v_xx * (-g8); h2 = v_xx * (-g6); h3 = v_xx * (-g7)
q1 = v_xy * (-g8); q2 = v_xy * (-g8); q3 = v_xy * (-g6); q4 = v_xy * (-g7)
j1 = v_xy * (-g6); j2 = v_xy * (-g7)

def HamiltonianEVAL_v6(a,k,f,c,b,l,m,n,p,m_,p_,h1, h2, h3, q1, q2, q3, q4, j1, j2):
    data1 = [];H_t = {}
    for ind, (B, L, M, N, P, M_,P_, H1, H2, H3, Q1, Q2, Q3, Q4, J1, J2) in enumerate(zip(b,l,m,n,p,m_,p_,h1, h2, h3, q1, q2, q3, q4, j1, j2)):
        H1s, H2s, H3s, Q1s, Q2s, Q3s, Q4s, J1s, J2s = np.conjugate((H1, H2, H3, Q1, Q2, Q3, Q4, J1, J2))
        Bs, Ls, Ms, Ns, Ps, M_s,P_s= np.conjugate((B, L, M, N, P, M_,P_))
        H = np.matrix([[a,  B,  0,  0,  0,   L,  M,  M_],
                       [Bs, c,  Ns, Ps, P_s, 0,  0,  0  ],
                       [0,  N,  f,  0,  0,   H1, Q1, Q2 ],
                       [0,  P,  0,  f,  0,   Q3, H2, J1 ],
                       [0,  P_, 0,  0,  f,   Q4, J2, H3 ],
                       [Ls, 0, H1s, Q1s, Q2s,k,  0,  0  ],
                       [Ms, 0, Q3s, H2s, J1s,0,  k,  0  ],
                       [M_s,0, Q4s, J2s, H3s,0,  0,  k  ]])

        if not (check_symmetric(H)):
            pass
            #print("*** Non hermitian matrix found for matrix index %d"%ind)
            #print(H11)
            #sys.exit()
        
        H_t[(ind,0)], H_t[(ind,1)] = H.real, H.imag
        
        enarray = LA.eigvals(H)
        print(enarray.imag)
        data1.append(np.sort(enarray.real))

    return data1, H_t

data1,Haa = HamiltonianEVAL_v6(a,k,f,c,b,l,m,n,p,m_,p_,h1, h2, h3, q1, q2, q3, q4, j1, j2)

#%%
def HamiltonianEVAL_v3(a,k,f,c,b,l,m,n,p,h,jj,q,m_,p_,q_):
    data = []; H_t = {}
    for ind, (B, L, M, N, P, H_, J, Q, M_,P_,Q_) in enumerate(zip(b,l,m,n,p,h,jj,q,m_,p_,q_)):
        Bs, Ls, Ms, Ns, Ps, H_s, Js, Qs, M_s,P_s,Q_s = np.conjugate((B, L, M, N, P, H_, J, Q, M_,P_,Q_))
        H11 = np.array([[a,   B,  0,   L], 
                        [Bs,  c,  Ns,  0], 
                        [0,   N,  f,   H_], 
                        [Ls,  0,  H_s, k]])
        H12 = np.array([[0,   M,  0,   M_], 
                        [Ps,  0,  P_s,  0], 
                        [0,   Q,  0,   Q_], 
                        [Qs,  0,  Q_s,  0]])
        H21 = np.array([[0,   P,  0,   Q], 
                        [Ms,  0,  Qs,  0], 
                        [0,   P_,  0,   Q_], 
                        [M_s,  0,  Q_s,  0]])
        H22 = np.array([[f,   H_, 0,   J], 
                        [H_s, k,  Js,  0], 
                        [0,   J,  f,   H_], 
                        [Js,  0,  H_s, k]])
        #print(H22)
        H = np.bmat([[H11,H12],[H21,H22]])
        
        if not (check_symmetric(H)):
            print("*** Non hermitian matrix found for matrix index %d"%ind)
            print(H)
            sys.exit()
        
        H_t[(ind,0)], H_t[(ind,1)] = H.real, H.imag
        
        enarray = LA.eigvals(H)
        #print(enarray.imag)
        data.append(np.sort(enarray.real))
    return data, H_t

data,Ha = HamiltonianEVAL_v3(a,k,f,c,b,l,m,n,p,h,jj,q,m_,p_,q_)
#%%
plt.figure()
plt.gca().set_prop_cycle(None)
linestyles = ['-', '--', '-.', ':',(0, (5, 10)),(0, (3, 10, 1, 10)),(0, (3, 1, 1, 1)),(0, (3, 10, 1, 10, 1, 10))]
y = np.array(data)
for lnd, linestyle in enumerate(linestyles[:len(y[0])]):
    plt.plot(k1, y[:,lnd],linestyle=linestyle)
plt.xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
plt.ylabel("E (eV)")


#%%
varied_list = ['$V_{As_s-Ga_s}$', '$V_{Ga_s-As_p}$', '$V_{As_s-Ga_p}$']
varied = varied_list[1]
Mydata = {}
vari = np.linspace(0, 10, 51)
for ind, var in enumerate(vari):
    #a = -var
    #k = k-0.1
    #b = var * g0
    #h = var * (-g5)
    #q = var * (-g7); jj = var * (-g8); q_ = var * (-g6)
    n = var * (g8); p = var * (g6);  p_ = var * (g7)
    #l = (var) * (-g8); m = (var) * (-g6); m_ = (var) * (-g7)
    Mydata[ind],_,Haa = HamiltonianEVAL_v4(a,k,f,c,b,l,m,n,p,h,jj,q,m_,p_,q_)
    #Mydata[ind],Ham = HamiltonianEVAL_v3(a,k,f,c,b,l,m,n,p,h,jj,q,m_,p_,q_)

#%%
fig,ax = plt.subplots()
ax.set_ylabel("Energy (eV)")
ax.set_xlabel(r'$k_x$-points ($\Gamma \rightarrow X$)')
#ax.set_ylim(-9,+7)
im = ax.plot(k1,Mydata[0])
text=ax.text(0.7,0.95,"%s = %6.3f"%(varied,vari[0]),transform=ax.transAxes)
#text =ax.text(0.8,0.9,"$V_{As_s-Ga_p}$ = %6.3f, $V_{Ga_s-As_p}$ = %6.3f"%(v_sAs_pGa,v_sGa_pAs),transform=ax.transAxes)

def updateData(I):
    for i in range(4):
        im[i].set_data(k1,np.array(Mydata[I])[:,i])
    text.set_text("%s = %6.3f"%(varied,vari[I]))
    return im, text

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
Ani = animation.FuncAnimation(fig,updateData,interval=200,frames=50,repeat=True, blit=False,repeat_delay=1) 

#%%
f=e_p_As = -3.5; v_sGa_pAs =  -2. ; k=e_p_Ga = 3.; v_xx =0.5
h = v_xx * g5
p = v_sGa_pAs * g6
al = f-p-p.conjugate()
be = (h-p)*(h.conjugate()-p.conjugate())
fact = np.sqrt((al-k)**2 * 0.25 + be)
ave = (al+k)/2
l1 = ave + fact
l2 = ave - fact

plt.figure()
plt.plot(ave.real)
plt.plot(l1.real)
plt.plot(l2.real)
