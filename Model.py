# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:45:55 2017
Translated from Matlab
Original author: David Parra
@autminutesor: alejandro
"""
## @namespace Model
# Created Wed Apr  5 11:45:55 2017
#Original author
# David Parra
# translated from Matlab by:
# Alejandro Pena-Bello
# alejandro.penabello@unige.ch
# Script that simulates the output of a PV panel of 235 W.
# The script has been tested in Linux and Windows
# INPUTS (df,phi,res,beta,gamma)
# ------
# df: DataFrame; includes Temperature and GHI
# phi: float; latitude where the panel will be installed
# res: float; temporal resolution
# beta: float; inclination
# gamma: float; azimuth
# OUTPUTS
# ------
# W_PVpanelS: numpy array; PV panel power output
# TODO
# ----
# It needs to be cleaner
# Requirements
# ------------
# Pandas, numpy, itertools,sys,glob,multiprocessing, time

import pandas as pd
import sympy as sp
import numpy as np
import scipy.io
from scipy import optimize
import warnings
import math
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
def Ftiltedradiation(G,beta,n,gamma,minutes,phi):

    #Function which obtains the tilted radiation as a function of:
     #   Horizontal Radiation G
     #   Inclination of the surface (PV panel).
     #   n, number of the day in the year.

     #   Of course, this function depends also on the latitude and azimuth
     #   angle, but I will not consider them as inputs for this functions.
     #   phi is the latitude, gamma is the surface azimuth angle.
    #   I tranform the angles in radianes
    phi=phi*(np.pi/180)
    gamma=gamma*(np.pi/180)

    #I have decided to work with vectors
    Gd=np.zeros(minutes)
    t=np.zeros(minutes)
    G0nh=np.zeros(minutes)
    Kt=np.zeros(minutes)

     #   I define the diffuse reflectance of the surroundings
    ro_g=0.3

     # I obtain the declination
    delta=23.45*np.sin(2*np.pi*(284+n+1)/365)
    delta=delta*(np.pi/180)

     # I obtain the extrarrestrial radiation
     # Solar constant
    Gsc=1353
    G0n=Gsc*(1+0.033*np.cos(2*np.pi*(n+1)/365))

     #I obtain the real solar time with one minute frequency.
     #   minute
    res=int((24*60)/minutes)
    for i in range(minutes):
        j=i+1
        t[i]=(j-1)*1/60*res


    w=15*(12-t)*np.pi/180
     #   I obtain the extrarrestrial radiation on a horizontal plane
    G0nh=G0n*(np.sin(delta)*np.sin(phi)+np.cos(delta)*np.cos(phi)*np.cos(w))

    #OJO esto no esta dentro de ningun for
    for i in range(minutes):
        if G0nh[i]<0:
           G0nh[i]=0


     #   I obtain both the sunrise and sunset solar time for the horizontal
     #   surface
    wsh=np.arccos(-np.tan(phi)*np.tan(delta))



    for i in range(minutes):
         #   I obtain the clearness index
        if abs(w[i])<wsh:
            Kt[i]=G[i]/G0nh[i]
        else:
            Kt[i]=0

        if Kt[i]>0.9:
            Kt[i]=0
         #I obtain the beam and diffuse components of hourly radiation (The Orgill
         # and Hollands correlation)
        if Kt[i]<0.35:
            Gd[i]=G[i]*(1-0.249*Kt[i])
        elif 0.35<=Kt[i] and Kt[i]<0.75:
            Gd[i]=G[i]*(1.557-1.84*Kt[i])
        else:
            Gd[i]=0.177*G[i]


    Gb=np.transpose(G)-Gd


     # I obtain the geometric factor Rb: ratio of beam radiation on the tilted
     # surface so hat on a horizontal surface at any time.
    Num=np.sin(delta)*np.sin(phi)*np.cos(beta)-np.sin(delta)*np.cos(phi)*np.sin(beta)*np.cos(gamma)+np.cos(delta)*np.cos(phi)*np.cos(beta)*np.cos(w)+np.cos(delta)*np.sin(phi)*np.sin(beta)*np.cos(gamma)*np.cos(w)+np.cos(delta)*np.sin(beta)*np.sin(gamma)*np.sin(w)

    Den=np.sin(delta)*np.sin(phi)+np.cos(delta)*np.cos(phi)*np.cos(w)
    ############
    Rb=Num/Den

    for i in range(minutes):
        if Rb[i]>5:
            Rb[i]=5  #I adjust it
        if Rb[i]<0:
            Rb[i]=0


    G_t=(Gb)*Rb+Gd*((1+np.cos(beta))/2)+np.transpose(G)*ro_g*((1-np.cos(beta))/2)

    return np.transpose(G_t);

def FPVpanel(T,G_t,minutes):
     #Function which obtains the performance of PV panel.
     #   The main inputs are:
     #   Tilted solar radiation.
     #   Outdoor temperature

     #   The PV parameters such open-circuit voltage are not as inputs.  But the
     #   must be upgrated if a new model of PV panels is used.

     #   Also the PV script to obtain both series and Parallel resistance must
     #   be run firstly in order to calculate them.

     # Datos de partida

    Iscn=5.83
    Vocn=51.2
     # Imp=5.45
     # Vmp=42.3
     # Pmax=230

    Gn=1000
    Tn=298.15

    Ki=0.00175
    Kv=-0.128

     # Eg=1.602176462*10^-19
    a=1.3
    q=1.60217646E-19
    k=1.3806503E-23

    Ns_cell=72

     #  I have already obtained the values of the resistances at nominal
         #  conditions
    Rs=0.36
    Rp=625.6969
     #   -----------------------------------------------------------------------



     #   I obtain the solar cell temperature using the NOCT
    Tsc=np.zeros(minutes)
    NOTC=44


     #   I initilize new vectors
    Vmpp=np.zeros(minutes)
    Impp=np.zeros(minutes)
    Pmax=np.zeros(minutes)
    V_PV=np.zeros(minutes)
    W_PV=np.zeros(minutes)
    I_PV=np.zeros(minutes)
    M=100
    I=np.zeros(M)
     #     V=zeros(1,M)

     #     P=zeros(1,M)
     #     I0=zeros(1,M)
     #   I obtain the solar cell temperature using the NOCT
    Tsc=T+(NOTC-20)*G_t/800

    for i in range(minutes):
          #   I obtain the solar cell temperature using the NOCT
     #          Tsc(i,1)=T(i,1)+(NOTC-20)*G_t(i,1)/800

        Taux=Tsc[i]
        Gaux=G_t[i]
    #   I think that according to this model only the temperature affects
     #   the Voc
        VocT=Vocn+Kv*(Taux-Tn)
        V=np.linspace(0,VocT,M)
     #   Obtengo la corriente de saturacion
        Vt=Ns_cell*k*Taux/q
        I0=(Iscn+Ki*(Taux-Tn))/(np.exp((Vocn+Kv*(Taux-Tn))/(a*Vt))-1)
     #   Obtengo el resto de parametros
     #   Obtengo la corriente debido a la radiacion
        Ipvn=((Rp+Rs)/Rp)*Iscn
        Ipv=(Ipvn+Ki*(Taux-Tn))*Gaux/Gn
     # Resuelvo la ecuacion principal para parios valores de V
        aux=Iscn
        for n in range(M):

            Vaux=V[n]
            ##OJO ACA

            #y=x-Ipv+I0[0]*(sp.exp((Vaux+Rs*x)/(Vt*a))-1)+(Vaux+Rs*x)/Rp
            f = lambda x : x-Ipv+I0*(np.exp((Vaux+Rs*x)/(Vt*a))-1)+(Vaux+Rs*x)/Rp

            Iaux=scipy.optimize.fsolve(f,aux)
            #print(Iaux)
            if Iaux<0:
                Iaux=0

            I[n]=Iaux
            aux=Iaux


           #   I develop the MPP Tracking system.

        P=V*I

        Pmaxaux = max(P)
#        if Pmaxaux>235:
#            Pmaxaux=235
        index=np.argmax(P)
        Pmax[i]=Pmaxaux
        Vmpp[i]=V[index]
        Impp[i]=I[index]
     # Isim=Impp(i,1)
     # Vsim=Vmpp(i,1)
        V_PV[i]=Vmpp[i]
        W_PV[i]=Vmpp[i]*Impp[i]
        I_PV[i]=Impp[i]


    return W_PV,V_PV,I_PV;

########################################MAIN###################################
def inputs(df,phi,res,beta,gamma):
    beta=(beta/180)*np.pi
    N_array=11
    N_housePV=1
    Np=11
    Ns=1
    if df.Temperature.mean()<60:
        df.Temperature=df.Temperature+273.15
    angles=11

    G_tS=[]
    W_PVpanelS=[]
    #W_PVpanel_gamma=np.zeros([year_minutes,11]) # 11 posiciones de gamma from -50 to +50
    #G_tS_gamma=np.zeros([year_minutes,angles])
    #Aux=np.array([ -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50 ])
    #gamma=Aux[5]
    #for gamma in Aux:

    days=int(df.shape[0]*res/(24*60))
    for n in range(days):

        G=np.array(df.GHI[df.index.dayofyear==n+1].values)
        T=np.array(df.Temperature[df.index.dayofyear==n+1].values)

        #   Tilted radiation
        #   I obtain tminutese best inclination of tminutese panel
        G_t=Ftiltedradiation(G,beta,n,gamma,len(G),phi)
     #CHECKED UNTIL HERE-------------------------------------------------------------------------

    #       G_t=G
        #--------------------------------------------------------------------------
        #   PV Model.    #
        [W_PV,V_PV,I_PV] = FPVpanel(T,G_t,len(G))
        #--------------------------------------------------------------------------
        #   Array model
    #         [ V_array,I_array,W_array ] = FPVarray( V_PV,I_PV,Ns,Np )
    #         #--------------------------------------------------------------------------
    #
    #         #   Inverter model for domestic appliances.
    #
    #        [ W_invD,eta_invD ] = FinverterD( W_array )
        G_t=G_t.reshape(len(G))
        G_tS=np.append(G_tS,G_t)
        W_PVpanelS=np.append(W_PVpanelS,W_PV)
#        for i in range(minutes):
#            # Tilted radiatonsum()
#            G_tS[(n)*minutes+i]=G_t[i]
             #   PV energy
            #W_PVpanelS[(n-1)*minutes+i]=W_PV[i]
        #print(n)
    return W_PVpanelS
