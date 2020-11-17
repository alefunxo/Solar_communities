# -*- coding: utf-8 -*-
## @namespace Psycho_paper_community
# Created on Wed Feb 28 09:47:22 2018
# Author
# Alejandro Pena-Bello
# alejandro.penabello@unige.ch
# Script developed for the project developed together with the Consumer Decision and Sustainable Behavior Lab to include the user preferences in the charging and discharging of the battery. We use a deterministic approach and include probabilities to discharge the battery to the grid in the frame of a community with P2P energy trading.
# The script has been tested in Linux and Windows
# INPUTS
# ------
# Inputs are automatically saved in the 'Output' file
# OUTPUTS
# ------
# Outputs are automatically saved in the 'Output' file
# TODO
# ----
# User Interface
# Requirements
# ------------
# Pandas, numpy, itertools,sys,glob,multiprocessing, timeimport pandas as pd
# <img src='C:\Users\alejandro\Documents\GitHub\Psycho\Psycho_flowchart.jpg'>
# DC-coupled PV-Battery system with integrated inverter used in this study.
# <img src='C:\Users\alejandro\Documents\GitHub\Psycho\DC-coupled system (2).jpg'>

import numpy as np
import matplotlib.pyplot as plt
import paper_classes_2 as pc
import pandas as pd
import time

def find_interval_PQ(x, partition):
    '''
    Description
    -----------
    find_interval at which x belongs inside partition. Returns the index i.

    Parameters
    ------
    x: float; numerical value
    partition: array; sequence of numerical values
    Returns
    ------
    i: index; index for which applies
    partition[i] < x < partition[i+1], if such an index exists.
    -1 otherwise
    TODO
    ------
    '''
    
    for i in range(0, len(partition)):
        #print(partition)
        if x<partition[1]:
            return 1
        elif x < partition[i]:
            return i-1
        
    return -1
def find_interval(x, partition):
    '''
    Description
    -----------
    find_interval at which x belongs inside partition. Returns the index i.

    Parameters
    ------
    x: float; numerical value
    partition: array; sequence of numerical values
    Returns
    ------
    i: index; index for which applies
    partition[i] < x < partition[i+1], if such an index exists.
    -1 otherwise
    TODO
    ------
    '''

    for i in range(0, len(partition)):
        if x < partition[i]:
            return i-1
    return -1

def sell_prob(SOC,price,Capacity,case,path):
    '''
    Description
    -----------
    This function returns the selling probability based on the SOC, price, capacity and case. As well as the results of the survey which are stored as .txt
    Parameters
    ------
    SOC: float; Battery state of charge at time t
    price: float; Electricity price at time t
    Capacity: float; Battery capacity
    case: string; either PV or batt from where the user is willing to sale
    path: string; path where the .txt are found
    ------
    out: int; Either 1 or 0 if sell or not
    TODO
    ------
    '''
    #define the matrix of probs
    if case=='PV':
        prob_mat=pd.read_table(path+'Input/P_selling_by_price_and_autarky2.txt',sep='\t',index_col=[0])
        #prob_mat=pd.read_table(path+'Input/surplus_cluster1_p_selling.txt',sep='\t',index_col=[0])
        #prob_mat=pd.read_table(path+'Input/surplus_cluster2_p_selling.txt',sep='\t',index_col=[0])
    else:#when selling from battery
        prob_mat=pd.read_table(path+'Input/P_selling_by_price_and_autarky_2.txt',sep='\t',index_col=[0])
        #prob_mat=pd.read_table(path+'Input/nosurplus_cluster1_p_selling.txt',sep='\t',index_col=[0])
        #prob_mat=pd.read_table(path+'Input/nosurplus_cluster2_p_selling.txt',sep='\t',index_col=[0])
    if SOC==0:
        out=0
    else:
        I=np.array([0,0.05,0.2,0.35,0.5,0.65,0.8,0.95,1.0001])#define the interval of SOC
        ind=find_interval(SOC,I*Capacity)
        if ind==-1:
            prob=prob_mat.loc[1,str(price)]
        else:
            prob=prob_mat.loc[I[ind],str(price)]
        out=np.random.choice(2,1,p=[1-prob,prob]) 
    return out

def PV_SC(df,Batt,Conv_eff,Inv_eff,endo):
    '''
    Description
    -----------
    This function calculates the energy flows of each house WHITHOUT consumer behaviour.

    Parameters
    ------
    df: DataFrame;
    Batt: Battery class; Battery characteristics
    Conv_eff: float; Converter efficiency
    Inv_eff: float; Inverter efficiency
    endo: int; Defines the last element of the index we want to simulate

    ------
    df: DataFrame; with energy flows for each house
    TODO
    ------

    '''
    
    for i in range(len(df.index[:endo])):
        if df.gen[i]<df.demand[i]/Conv_eff/Inv_eff:#No surplus including losses
            if i==0:#CI
                df.grid_load[i]=df.demand[i]-df.gen[i]*Conv_eff*Inv_eff
            else:
                if df.SOC[i-1]>Batt.SOC_min:#Discharge battery to the load
                    df.E_dis[i]=min(df.SOC[i-1]-Batt.SOC_min,
                                        (df.demand[i]-df.gen[i]*Conv_eff*Inv_eff)/Inv_eff)
                    df.Batt_load[i]=df.E_dis[i]*Inv_eff#after inverter
                    df.SOC[i]=df.SOC[i-1]-df.E_dis[i]
                    df.Batt_losses[i]=df.E_dis[i]*(1-Inv_eff)#+inherent to the batt (eff)
                    df.PV_load[i]=df.gen[i]*Conv_eff*Inv_eff#after inverter
                    df.PV_losses[i]=df.gen[i]*(1-Conv_eff*Inv_eff)
                    df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]

                else:#Use energy from the grid
                    df.E_dis[i]=0
                    df.Batt_load[i]=df.E_dis[i]*Inv_eff
                    df.grid_load[i]=df.demand[i]-df.gen[i]*Conv_eff*Inv_eff#-df.Batt_load[i], is zero
                    df.PV_load[i]=df.gen[i]*Conv_eff*Inv_eff
                    df.SOC[i]=df.SOC[i-1]
                    df.PV_losses[i]=df.gen[i]*(1-Conv_eff*Inv_eff)
        else:#Surplus
            #include here the probability function
            if i==0:#CI
                df.grid_load[i]=df.demand[i]
            else:
                df.PV_load[i]=df.demand[i]#First, cover local demand (after inverter)
                if df.SOC[i-1]<Batt.SOC_max:#then if battery not full, charge
                    df.PV_batt[i]=min((df.gen[i]*Conv_eff-df.demand[i]/Inv_eff),
                                      (Batt.SOC_max-df.SOC[i-1])*Conv_eff)
                    df.SOC[i]=df.SOC[i-1]+df.PV_batt[i]*Batt.Efficiency
                    df.Batt_losses[i]=df.PV_batt[i]*(1-Batt.Efficiency)

                    df.PV_grid[i]=(df.gen[i]*Conv_eff-df.PV_batt[i])*Inv_eff-df.PV_load[i]
                    df.grid_load[i]=0
                else:#otherwise, export to the grid
                    df.PV_batt[i]=0
                    df.PV_grid[i]=df.gen[i]*Conv_eff*Inv_eff-df.PV_load[i]
                    df.SOC[i]=df.SOC[i-1]
                    df.grid_load[i]=0
                df.PV_losses[i]=df.gen[i]-df.PV_grid[i]-df.PV_batt[i]-df.PV_load[i]
    return df

# If the client agreees (Probability function) we discharge 1 kWh of the battery

def PV_SC_probs(df,system_param,flag_community):#,discharge_time):
    '''
    Description
    -----------
    This function calculates the energy flows of each house including the consumer behaviour (decision on whether the consumer wants to sell electricity from PV instead of charging its battery when surplus
    and whether the consumer wants to sell electricity form its battery in the morning when there is no surplus, after cover their own demand). The decisions are taken each hour and are valid for 1 kWh of selling.

    Parameters
    ------
    df: DataFrame;
    Batt: Battery class; Battery characteristics
    Conv_eff: float; Converter efficiency
    Inv_eff: float; Inverter efficiency
    endo: int; Defines the last element of the index we want to simulate
    path:string; defines the path
    flag_community:df; indicates when there is community surplus
    discharge_time:string;['surplus','surplus_morning','surplus_evening','morning','morning_evening','evening','surplus_morning_evening']
    ------
    df: DataFrame; with energy flows for each house
    TODO
    ------

    '''
    Batt=system_param['Batt']
    Conv_eff=system_param['Conv_eff']
    Inv_eff=system_param['Inv_eff']
    endo=system_param['endo']
    path=system_param['path']
    kWh_dis=system_param['kWh_dis']
    probs_applied=system_param['probs_applied']# 1 for surplus only, 2 for morning+surplus 3 for morning+surplus+evening 4 for evening and surplus
    #The choice from the user should be done only if PV_GEN<LOAD
    for i in range(len(df.index[:endo])):
        if df.gen[i]<df.demand[i]/Conv_eff/Inv_eff:#No surplus including losses
            if i==0:#CI
                df.grid_load[i]=df.demand[i]-df.gen[i]*Conv_eff*Inv_eff
            else:
                if df.SOC[i-1]>Batt.SOC_min:#Discharge battery to the load
                    aux=min(df.SOC[i-1]-Batt.SOC_min,
                                        (df.demand[i]-df.gen[i]*Conv_eff*Inv_eff)/Inv_eff)
                    df.PV_grid[i]=0
                    df.PV_batt[i]=0#No surplus
                    df.PV_load[i]=df.gen[i]*Conv_eff*Inv_eff#after inverter
                    df.PV_losses[i]=df.gen[i]*(1-Conv_eff*Inv_eff)
                    if df.SOC[i-1]-aux>Batt.SOC_min:
                        #print(df.index.hour[i])
                        if probs_applied==2:
                            if (sell_prob(df.SOC[i-1]-aux,df.prices[i],Batt.Capacity,'batt',path)&(df.index.hour[i]<12)):
                                #Sell from the battery to the community
                                #print('######UUUUU#####')
                                df.flag[i]=1#Sell from battery
                                df.Batt_load[i]=aux*Inv_eff#after inverter
                                df.E_dis[i]=min(kWh_dis,df.SOC[i-1]-Batt.SOC_min-aux)#the client agree to sell 1Kwh from batt
                                df.flag[i]=1
                                df.Batt_grid[i]=(df.E_dis[i])*Inv_eff
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]-aux
                                df.Batt_losses[i]=(df.E_dis[i]+aux)*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                                df.E_dis[i]=df.E_dis[i]+aux
                            else:
                                df.E_dis[i]=aux
                                df.Batt_grid[i]=0
                                df.Batt_load[i]=df.E_dis[i]*Inv_eff#after inverter
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]
                                df.Batt_losses[i]=df.E_dis[i]*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                        elif probs_applied==3:
                            if (sell_prob(df.SOC[i-1]-aux,df.prices[i],Batt.Capacity,'batt',path)):
                                #Sell from the battery to the community
                                #print('######UUUUU#####')
                                df.flag[i]=1#Sell from battery
                                df.Batt_load[i]=aux*Inv_eff#after inverter
                                df.E_dis[i]=min(kWh_dis,df.SOC[i-1]-Batt.SOC_min-aux)#the client agree to sell 1Kwh from batt
                                df.flag[i]=1
                                df.Batt_grid[i]=(df.E_dis[i])*Inv_eff
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]-aux
                                df.Batt_losses[i]=(df.E_dis[i]+aux)*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                                df.E_dis[i]=df.E_dis[i]+aux
                            else:
                                df.E_dis[i]=aux
                                df.Batt_grid[i]=0
                                df.Batt_load[i]=df.E_dis[i]*Inv_eff#after inverter
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]
                                df.Batt_losses[i]=df.E_dis[i]*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                        elif probs_applied==4:
                            if (sell_prob(df.SOC[i-1]-aux,df.prices[i],Batt.Capacity,'batt',path)&(df.index.hour[i]>12)):
                                #Sell from the battery to the community
                                #print('######UUUUU#####')
                                df.flag[i]=1#Sell from battery
                                df.Batt_load[i]=aux*Inv_eff#after inverter
                                df.E_dis[i]=min(kWh_dis,df.SOC[i-1]-Batt.SOC_min-aux)#the client agree to sell 1Kwh from batt
                                df.flag[i]=1
                                df.Batt_grid[i]=(df.E_dis[i])*Inv_eff
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]-aux
                                df.Batt_losses[i]=(df.E_dis[i]+aux)*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                                df.E_dis[i]=df.E_dis[i]+aux
                            else:
                                df.E_dis[i]=aux
                                df.Batt_grid[i]=0
                                df.Batt_load[i]=df.E_dis[i]*Inv_eff#after inverter
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]
                                df.Batt_losses[i]=df.E_dis[i]*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                        elif probs_applied==1:
                                df.E_dis[i]=aux
                                df.Batt_grid[i]=0
                                df.Batt_load[i]=df.E_dis[i]*Inv_eff#after inverter
                                df.SOC[i]=df.SOC[i-1]-df.E_dis[i]
                                df.Batt_losses[i]=df.E_dis[i]*(1-Inv_eff)#+inherent to the batt (eff)
                                df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]
                    else:
                        df.E_dis[i]=aux
                        df.Batt_grid[i]=0
                        df.Batt_load[i]=df.E_dis[i]*Inv_eff#after inverter
                        df.SOC[i]=df.SOC[i-1]-df.E_dis[i]
                        df.Batt_losses[i]=df.E_dis[i]*(1-Inv_eff)#+inherent to the batt (eff)
                        df.grid_load[i]=df.demand[i]-df.PV_load[i]-df.Batt_load[i]

                else:#Use energy from the grid
                    df.E_dis[i]=0
                    df.Batt_load[i]=df.E_dis[i]*Inv_eff
                    df.grid_load[i]=df.demand[i]-df.gen[i]*Conv_eff*Inv_eff#-df.Batt_load[i], is zero
                    df.PV_load[i]=df.gen[i]*Conv_eff*Inv_eff
                    df.SOC[i]=df.SOC[i-1]
                    df.PV_losses[i]=df.gen[i]*(1-Conv_eff*Inv_eff)
        else:#Surplus

            if i==0:#CI
                df.grid_load[i]=df.demand[i]-df.gen[i]*Conv_eff*Inv_eff
                #include here the probability function
                #if sell_prob(df.SOC[i-1],df_prices[i]):
                    #df.Batt_load[i]=min(1,df.SOC[i-1]-Batt.SOC_min)
                #else:
                    #df.Batt_load[i]=0
            else:

                df.PV_load[i]=df.demand[i]#First, cover local demand (after inverter)
                #Here it depends on the question. Is the question do you prefer to sell 1kWh rather than store it in this hour?
                #Or is it do you prefer to sell rather than store in this hour?
                #here we take the first approach.

                if sell_prob(df.SOC[i-1],df.prices[i],                Batt.Capacity,'PV',path)&(flag_community[i]==False):#No surplus for the whole community
                    df.flag[i]=2
                    if df.SOC[i-1]<Batt.SOC_max-0.0001:#then if battery not full, charge
                        if (df.gen[i]*Conv_eff-df.demand[i]/Inv_eff)>kWh_dis:
                            df.flag[i]=3#charge
                            df.PV_batt[i]=min((df.gen[i]*Conv_eff-df.demand[i]/Inv_eff)-1,
                                              (Batt.SOC_max-df.SOC[i-1])*Conv_eff)
                            df.SOC[i]=df.SOC[i-1]+df.PV_batt[i]*Batt.Efficiency
                            df.Batt_losses[i]=df.PV_batt[i]*(1-Batt.Efficiency)
                            df.PV_grid[i]=(df.gen[i]*Conv_eff-df.PV_batt[i])*Inv_eff-df.PV_load[i]
                            df.grid_load[i]=0
                        else:
                            df.flag[i]=4#instead of charging you sell
                            df.PV_batt[i]=0
                            df.SOC[i]=df.SOC[i-1]+df.PV_batt[i]*Batt.Efficiency
                            df.Batt_losses[i]=df.PV_batt[i]*(1-Batt.Efficiency)
                            df.PV_grid[i]=(df.gen[i]*Conv_eff-df.PV_batt[i])*Inv_eff-df.PV_load[i]
                            df.grid_load[i]=0
                        df.PV_losses[i]=df.gen[i]-df.PV_grid[i]-df.PV_batt[i]-df.PV_load[i]
                        #df.PV_losses[i]=(df.PV_grid[i]+df.PV_load[i])*(1-Conv_eff*Inv_eff)+df.PV_batt[i]*(1-Conv_eff)

                    else:#otherwise, export to the grid
                        df.PV_batt[i]=0
                        df.PV_grid[i]=df.gen[i]*Conv_eff*Inv_eff-df.PV_load[i]
                        df.SOC[i]=df.SOC[i-1]
                        df.grid_load[i]=0
                        df.PV_losses[i]=df.gen[i]*(1-Conv_eff*Inv_eff)

                else:
                    df.Batt_load[i]=0
                    df.E_dis[i]=0
                    if df.SOC[i-1]<Batt.SOC_max:#then if battery not full, charge
                        df.PV_batt[i]=min((df.gen[i]*Conv_eff-df.demand[i]/Inv_eff),
                                          (Batt.SOC_max-df.SOC[i-1])*Conv_eff)
                        df.SOC[i]=df.SOC[i-1]+df.PV_batt[i]*Batt.Efficiency
                        df.Batt_losses[i]=df.PV_batt[i]*(1-Batt.Efficiency)
                        df.PV_grid[i]=(df.gen[i]*Conv_eff-df.PV_batt[i])*Inv_eff-df.PV_load[i]
                        df.grid_load[i]=0
                    else:#otherwise, export to the grid
                        df.PV_batt[i]=0
                        df.PV_grid[i]=df.gen[i]*Conv_eff*Inv_eff-df.PV_load[i]
                        df.SOC[i]=df.SOC[i-1]
                        df.grid_load[i]=0
                    df.PV_losses[i]=df.gen[i]-df.PV_grid[i]-df.PV_batt[i]-df.PV_load[i]
    return df

# Let's first create the community, it consists of 74 households, 37 of which have PV (on which the prices are based and can be found in Price_setup.ipynb) and 18 (floor of 25%) will have a battery. The first step is to match PV with demand, then among those houses choose which ones will have a battery and then for the latter run the simulation. For the whole community calculate different indicators in the last step.

#def Price_definition_PQ_curves(prices, PV_penetration,Batt_penetration,reso,path,day_sel):
    #inputs={'df_demand':df_demand,'df_generation':df_sel,'df_prices':Prices,'selection_PV':selection_PV,'selection_PV_Batt':selection_PV_Batt,'flag':flag_surplus_community}
    #return inputs
def Price_definition(prices, PV_penetration,Batt_penetration,reso,path,day_sel):
    '''
    Description
    -----------
    This function takes the prices and PV_penetration as inputs, reads the df of generation in Munich and the PV_size_distribution, and RANDOMLY (unless we are in mode test or looking at a particular day of summer or winter) chooses the sizes for the X houses (RANDOMLY selected unless in test mode) and produces an output at 15 min resolution that can be resampled @ 1h ir resample==True

    Parameters
    ------
    prices:array; array of the different prices we want to use (steps)
    PV_penetration:float; PV penetration
    Batt_penetration: float; Battery penetration
    reso: string; 1h or 15m resolution
    path:string; path where the .txt are found
    day_sel:string; selects mode, winter, summer, test or other

    Return
    ------
    inputs: dict; Dictionary that includes the demand, pv generation, prices, flag of surplus in the community and the houses that include PV and PV+Batt

    TODO
    ------

    '''
    print('################################################')
    print('Getting prices')
    df_gen_comb=pd.read_csv(path+'Input/DE_gen_15_min_Energy.csv', encoding='utf8', sep=',',engine='python',parse_dates=[0],                   infer_datetime_format=True,index_col=0)
    df_gen_comb.index = df_gen_comb.index.tz_localize('UTC').tz_convert('CET')
    if reso=='1h':
        df_demand=pd.read_csv(path+'Input/DE_load_15_min_Power.csv', encoding='utf8', sep=',',
                              engine='python',index_col=0, parse_dates=[0],infer_datetime_format=True )/4
        df_demand=df_demand.resample('1H').sum()
        df_demand.index = df_demand.index.tz_localize('UTC').tz_convert('CET')
        df_gen_comb=df_gen_comb.resample('1H').sum()
        df_demand.index=df_gen_comb.index
    else:
        df_demand=pd.read_csv(path+'Input/DE_load_15_min_Power.csv', encoding='utf8', sep=',',
                              engine='python',index_col=0, parse_dates=[0],infer_datetime_format=True )/4
        df_demand.index = df_demand.index.tz_localize('UTC').tz_convert('CET')
        df_demand.index=df_gen_comb.index

    #We decide among the whole community which houses will have PV and PV and Batt

    selection_PV=np.random.choice(df_demand.columns,int(np.floor(df_demand.shape[1]*PV_penetration)), replace=False)
    selection_PV_Batt=np.random.choice(selection_PV,int(np.floor(df_demand.shape[1]*PV_penetration*Batt_penetration)), replace=False)
    if day_sel in ['winter','summer','test']:
        selection_PV=['41','24','65','0','69','8','49','63','2','1','14','42','67','10','6','9','38','50']
        selection_PV_Batt=['69','41','10','63','50','9','0','49','38','67','1','2','42','24','8','65','6','14']
    # We have to define the prices for the community based on PV penetration. We can include some variation in azimuth and angle but in general, I guess, only three or four should be ok. The df loaded has 66 combinations.

    # As an example I will take the 66 PV profiles for different azimuths and inclinations, but this input data must be modified once we define the final community

    # The PV size for every house must be defined. If the household have a Battery the PV size comes from another distribution

    PV_sizes=pd.read_csv(path+'Input/PV_size_distribution.csv',index_col=0,header=None)
    PV_sizes_batt=pd.read_csv(path+'Input/PV_size_distribution.csv',index_col=0,header=None)
    #The output of this function: should be a combined df with
    #df_prices,df_generation,df_demand
    count_batt, division_batt = np.histogram(PV_sizes_batt)
    prob_size_batt=count_batt/PV_sizes_batt.size
    sizes_PV_batt=np.random.choice(np.arange(1, 11),int(np.floor(PV_penetration*Batt_penetration*74)), p=prob_size_batt)
    #Delete after tests
    if day_sel in ['winter','summer','test']:
        sizes_PV_batt=[7,7,6,4,10,9,5,6,5,8,6,7,5,10,9,3,5,7]

    # Select the PV profiles randomly and include the size in the name of the columns
    newcols_batt=[]
    #We select from all the combinations of azimuth and inclination randomly
    selection_batt=np.random.choice(df_gen_comb.columns,int(np.floor(PV_penetration*Batt_penetration*74)))
    count, division = np.histogram(PV_sizes)
    prob_size=count/PV_sizes.size
    sizes=np.random.choice(np.arange(1, 11),int(np.floor(PV_penetration*74)), p=prob_size)
    if day_sel in ['winter','summer','test']:
        sizes=[3,3,9,10,9,6,4,10,3,7,5,10,6,4,10,10,5,9]
        selection_batt=['PV_-50_35','PV_-10_20','PV_30_40','PV_50_35','PV_40_40','PV_-40_45','PV_10_45','PV_-50_30','PV_-30_35','PV_10_20','PV_20_30','PV_-50_45','PV_30_25','PV_10_40','PV_-50_30','PV_-50_25','PV_50_35','PV_10_25']
  # Select the PV profiles randomly and include the size in the name of the columns
    newcols=[]
    #We select from all the combinations of azimuth and inclination randomly

    selection=np.random.choice(df_gen_comb.columns,len(selection_PV)-len(selection_batt))

    j=0
    k=0
    m=0
    df_sel=pd.DataFrame()
    df_sel_batt=pd.DataFrame()
    df_residual_load=pd.DataFrame()
    for i in df_demand.columns:
        if i in selection_PV_Batt:
            if j==0:
                df_sel_batt=df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]
            else:
                df_sel_batt=pd.concat([df_sel_batt, df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]], axis=1, join='inner')
            if m==0:
                df_residual_load=df_demand.loc[:,i]-df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]
            else:
                df_residual_load=pd.concat([df_residual_load, df_demand.loc[:,i]-df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]], axis=1, join='inner')
            newcols_batt.append('hh_'+str(i)+'_'+selection_batt[j]+'_size_'+str(sizes_PV_batt[j]))

            j+=1
        elif i in selection_PV:

            if k==0:
                df_sel=df_gen_comb[selection[k]]*sizes[k]
            else:
                df_sel=pd.concat([df_sel, df_gen_comb[selection[k]]*sizes[k]], axis=1, join='inner')
            if m==0:
                df_residual_load=df_demand.loc[:,i]-df_gen_comb[selection[k]]*sizes[k]
            else:
                df_residual_load=pd.concat([df_residual_load, df_demand.loc[:,i]-df_gen_comb[selection[k]]*sizes[k]], axis=1, join='inner')
            newcols.append('hh_'+str(i)+'_'+selection[k]+'_size_'+str(sizes[k]))

            k+=1
        else:
            if m==0:
                df_residual_load=df_demand.loc[:,i]
            else:
                df_residual_load=pd.concat([df_residual_load, df_demand.loc[:,i]], axis=1, join='inner')
        m+=1
    df_sel.columns=newcols
    df_sel_batt.columns=newcols_batt
    if Batt_penetration==1:
        df_sel=df_sel_batt
    else:
        df_sel=pd.concat([df_sel, df_sel_batt], axis=1, join='inner')
    df_residual_load.columns=df_demand.columns
    #The matching from PV-demand is done first.
    flag_surplus_community=df_residual_load.sum(axis=1)
    flag_surplus_community[flag_surplus_community>0]=0
    flag_surplus_community[flag_surplus_community<0]=1
    aux=df_residual_load.sum(axis=1)

    aux_max_day=aux.groupby(aux.index.dayofyear).max()

    step_daily=aux_max_day/(prices.size+1)
    aux[aux<0]=0
    Prices=pd.DataFrame(index=df_sel.index)
    Prices['step']=0
    Prices.loc[Prices.index.hour==0,'step']=step_daily.values
    Prices.step=Prices.step.replace(to_replace=0, method='ffill')
    for i in range(len(prices)):
        Prices.loc[aux>=Prices.step*i,'prices']=prices[i]
        Prices.loc[aux>=(Prices.step*(prices.size+1)*(3**i-3))/(3**(i)),'prices']=prices[i]
    print(Prices)
    for i in range(len(prices)):
        Prices.loc[aux>=Prices.step*i,'prices']=prices[i]
    #print(Prices)
    Prices=Prices.drop(columns=['step'])

    print(argsdsaf)
    if reso=='1h':
        df_sel.to_csv(path+'Input/DE_gen_1_h_Energy_sizes_{}.csv'.format(PV_penetration))
        Prices.to_csv(path+'Input/DE_price_1_h_{}.csv'.format(PV_penetration))
    else:
        df_sel.to_csv(path+'Input/DE_gen_15_min_Energy_sizes_{}.csv'.format(PV_penetration))
        Prices.to_csv(path+'Input/DE_price_15_min_{}.csv'.format(PV_penetration))
    print('Prices ok')
    print('################################################')
    if day_sel=='winter':
        #Prices[Prices.index.hour<23]=.1
        #Prices[Prices.index.hour<7]=.20
        #Prices[Prices.index.hour>17]=.28
        df_demand=df_demand[(df_demand.index.month==1)&(df_demand.index.day==1)]
        df_sel=df_sel[(df_sel.index.month==1)&(df_sel.index.day==1)]
        Prices=Prices[(Prices.index.month==1)&(Prices.index.day==1)]
        print(Prices)
    elif day_sel=='summer':
        #Prices[Prices.index.hour<23]=.16
        #Prices[Prices.index.hour<10]=.22
        #Prices[Prices.index.hour>17]=.28
        df_demand=df_demand[(df_demand.index.month==6)&(df_demand.index.day==1)]
        df_sel=df_sel[(df_sel.index.month==6)&(df_sel.index.day==1)]
        Prices=Prices[(Prices.index.month==6)&(Prices.index.day==1)]
        print(Prices)
    inputs={'df_demand':df_demand,'df_generation':df_sel,'df_prices':Prices,'selection_PV':selection_PV,'selection_PV_Batt':selection_PV_Batt,'flag':flag_surplus_community}
    return inputs
def create_community(inputs):
    
    '''
    Description
    -----------
    Define the houses with PV and Battery
    This function takes the PV and Batt penetration as inputs, reads the df of generation in Munich and the PV_size_distribution, 
    and RANDOMLY (unless we are in mode test or looking at a particular day of summer or winter) chooses the sizes 
    for the X houses (RANDOMLY selected unless in test mode) 

    Parameters
    ------
    inputs: dict; Dictionary that includes PV_penetration,Batt_penetration, resolution (reso),path,day_selection (day_sel)

    Return
    ------
    inputs: dict; Updated dictionary that includes the demand, pv generation, flag of surplus in the community and the houses that include PV and PV+Batt

    TODO
    ------

    '''
    print('################################################')
    print('Creating the community')
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Loading PV combinations')
    df_gen_comb=pd.read_csv(inputs['path']+'Input/DE_gen_15_min_Energy.csv', encoding='utf8', sep=',',engine='python',
                            date_parser=lambda col: pd.to_datetime(col, utc=True),infer_datetime_format=True,index_col=0)
    df_gen_comb.index = df_gen_comb.index.tz_convert('CET')
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Loading demand')
    if inputs['reso']=='1h':
        df_demand=pd.read_csv(inputs['path']+'Input/DE_load_15_min_Power.csv', encoding='utf8', sep=',',
                              engine='python',date_parser=lambda col: pd.to_datetime(col, utc=True),infer_datetime_format=True,index_col=0)/4
        df_demand.index=df_demand.index.tz_convert('CET')
        df_demand=df_demand.resample('1H').sum()
        #df_demand.index = df_demand.index.tz_localize('UTC').tz_convert('CET')
        df_gen_comb=df_gen_comb.resample('1H').sum()
        df_demand.index=df_gen_comb.index
    else:
        df_demand=pd.read_csv(inputs['path']+'Input/DE_load_15_min_Power.csv', encoding='utf8', sep=',',engine='python', 
                              date_parser=lambda col: pd.to_datetime(col, utc=True),infer_datetime_format=True,index_col=0)/4
        df_demand.index=df_demand.index.tz_convert('CET')
        #df_demand.index = df_demand.index.tz_localize('UTC').tz_convert('CET')
        df_demand.index=df_gen_comb.index
    selection_PV=np.random.choice(df_demand.columns,int(np.floor(df_demand.shape[1]*inputs['PV_penetration'])), replace=False)
    selection_PV_Batt=np.random.choice(selection_PV,int(np.floor(df_demand.shape[1]*inputs['PV_penetration']*inputs['Batt_penetration'])), replace=False)

    #We decide among the whole community which houses will have PV and PV and Batt

    if inputs['day_sel'] in ['winter','summer','test']:
        selection_PV=['41','24','65','0','69','8','49','63','2','1','14','42','67','10','6','9','38','50']
        selection_PV_Batt=['69','41','10','63','50','9','0','49','38','67','1','2','42','24','8','65','6','14']
    # We have to define the prices for the community based on PV penetration. We can include some variation in azimuth and angle but in general, I guess, only three or four should be ok. The df loaded has 66 combinations.

    # As an example I will take the 66 PV profiles for different azimuths and inclinations, but this input data must be modified once we define the final community

    # The PV size for every house must be defined. If the household have a Battery the PV size comes from another distribution
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Creating distributions')
    PV_sizes=pd.read_csv(inputs['path']+'Input/PV_size_distribution.csv',index_col=0,header=None)
    PV_sizes_batt=pd.read_csv(inputs['path']+'Input/PV_size_distribution.csv',index_col=0,header=None)#change this input for the real distribution of PV+battery
    #The output of this function: should be a combined df with
    #df_prices,df_generation,df_demand
    count_batt, division_batt = np.histogram(PV_sizes_batt)
    prob_size_batt=count_batt/PV_sizes_batt.size
    possible_sizes=np.arange(1,11)#for residential we consider from 1 to 10
    amount_PV_batt_systems=int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*74))
    sizes_PV_batt=np.random.choice(possible_sizes,amount_PV_batt_systems, p=prob_size_batt)
    #Delete after tests
    if inputs['day_sel'] in ['winter','summer','test']:
        sizes_PV_batt=[7,7,6,4,10,9,5,6,5,8,6,7,5,10,9,3,5,7]
        
    # Select the PV profiles randomly and include the size in the name of the columns
    newcols_batt=[]
    #We select from all the combinations of azimuth and inclination randomly
    selection_batt=np.random.choice(df_gen_comb.columns,int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*74)))
    count, division = np.histogram(PV_sizes)
    prob_size=count/PV_sizes.size
    amount_PV_systems=int(np.floor(inputs['PV_penetration']*74))
    sizes=np.random.choice(possible_sizes,amount_PV_systems, p=prob_size)
    if inputs['day_sel'] in ['winter','summer','test']:
        sizes=[3,3,9,10,9,6,4,10,3,7,5,10,6,4,10,10,5,9]
        selection_batt=['PV_-50_35','PV_-10_20','PV_30_40','PV_50_35','PV_40_40','PV_-40_45','PV_10_45','PV_-50_30','PV_-30_35','PV_10_20','PV_20_30','PV_-50_45','PV_30_25','PV_10_40','PV_-50_30','PV_-50_25','PV_50_35','PV_10_25']
  # Select the PV profiles randomly and include the size in the name of the columns
    newcols=[]
    #We select from all the combinations of azimuth and inclination randomly

    selection=np.random.choice(df_gen_comb.columns,len(selection_PV)-len(selection_batt))
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Assingning PV to households')
    j=0
    k=0
    m=0
    df_sel=pd.DataFrame()
    df_sel_batt=pd.DataFrame()
    df_residual_load=pd.DataFrame()
    for i in df_demand.columns:
        if i in selection_PV_Batt:
            if j==0:
                df_sel_batt=df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]
            else:
                df_sel_batt=pd.concat([df_sel_batt, df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]], axis=1, join='inner')
            if m==0:
                df_residual_load=df_demand.loc[:,i]-df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]
            else:
                df_residual_load=pd.concat([df_residual_load, df_demand.loc[:,i]-df_gen_comb[selection_batt[j]]*sizes_PV_batt[j]], axis=1, join='inner')
            newcols_batt.append('hh_'+str(i)+'_'+selection_batt[j]+'_size_'+str(sizes_PV_batt[j]))

            j+=1
        elif i in selection_PV:

            if k==0:
                df_sel=df_gen_comb[selection[k]]*sizes[k]
            else:
                df_sel=pd.concat([df_sel, df_gen_comb[selection[k]]*sizes[k]], axis=1, join='inner')
            if m==0:
                df_residual_load=df_demand.loc[:,i]-df_gen_comb[selection[k]]*sizes[k]
            else:
                df_residual_load=pd.concat([df_residual_load, df_demand.loc[:,i]-df_gen_comb[selection[k]]*sizes[k]], axis=1, join='inner')
            newcols.append('hh_'+str(i)+'_'+selection[k]+'_size_'+str(sizes[k]))

            k+=1
        else:
            if m==0:
                df_residual_load=df_demand.loc[:,i]
            else:
                df_residual_load=pd.concat([df_residual_load, df_demand.loc[:,i]], axis=1, join='inner')
        m+=1

    df_sel.columns=newcols
    df_sel_batt.columns=newcols_batt
    if inputs['Batt_penetration']==1:
        df_sel=df_sel_batt
    else:
        if df_sel.shape[0]==0:
            df_sel=df_sel_batt.copy()
        else:
            df_sel=pd.concat([df_sel, df_sel_batt], axis=1, join='inner')
    df_residual_load.columns=df_demand.columns
    #The matching from PV-demand is done first.
    flag_surplus_community=df_residual_load.sum(axis=1)
    flag_surplus_community[flag_surplus_community>0]=0
    flag_surplus_community[flag_surplus_community<0]=1
    inputs.update({'df_demand':df_demand,'df_generation':df_sel,'selection_PV':selection_PV,'selection_PV_Batt':selection_PV_Batt,'flag':flag_surplus_community})
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Finished')
    return inputs

def price_probability(inputs):
    '''
    Description
    -----------
    Define the price structure according to the inputs (how the community is created), the choice of the probability, 
    whether it is for the 50% or other probability according to the psychology data, the amount of kWh that will be 
    discharged every time a decision is made (1 or 2 kWh are the "normal" options) and the case, whether it is the
    price created from the probability of selling from PV or the battery in the morning or the evening.
    
    Parameters
    ------
    inputs:        dict; Dictionary that includes prob_choice (choice of the probability to create the prices 50% or other),
                    prices, kWh_dis (amount of energy to be discharged at every time a decision is made),  
                    case (included in ['PV', 'Batt'] the price may be created from the probability of selling from PV or the battery

    Return
    ------
    inputs: dict; Input dictionary updated with prices

    TODO
    ------

    '''
    net_demand=inputs['df_demand'].sum(axis=1)-inputs['df_generation'].sum(axis=1)
    prob_mat_PV=pd.read_csv(inputs['path']+'Input/P_selling_by_price_and_autarky2.txt',sep='\t',index_col=[0])
    prob_mat_Batt=pd.read_csv(inputs['path']+'Input/P_selling_by_price_and_autarky_2.txt',sep='\t',index_col=[0])
    if len(inputs['prob_choice'])==1:
        q_supply_midday=int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*inputs['df_demand'].shape[1]))*prob_mat_PV.loc[inputs['prob_choice'][0],:]*inputs['kWh_dis']
        q_supply_other=int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*inputs['df_demand'].shape[1]))*prob_mat_Batt.loc[inputs['prob_choice'][0],:]*inputs['kWh_dis']
        df_prices=pd.DataFrame(index=net_demand.index)
        df_prices.loc[:,'prices']=net_demand.apply(lambda x:inputs['prices'][find_interval_PQ(x,q_supply_midday)])
        df_prices.loc[(df_prices.index.hour<10)|(df_prices.index.hour>17),'prices']=net_demand[(net_demand.index.hour<10)|(net_demand.index.hour>17)].apply(lambda x:inputs['prices'][find_interval_PQ(x,q_supply_other)])
        inputs.update({'df_prices':df_prices})
        return inputs
    elif len(inputs['prob_choice'])==3:
        q_supply_morning=int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*inputs['df_demand'].shape[1]))*prob_mat_Batt.loc[inputs['prob_choice'][0],:]*inputs['kWh_dis']
        q_supply_midday=int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*inputs['df_demand'].shape[1]))*prob_mat_PV.loc[inputs['prob_choice'][1],:]*inputs['kWh_dis']
        q_supply_evening=int(np.floor(inputs['PV_penetration']*inputs['Batt_penetration']*inputs['df_demand'].shape[1]))*prob_mat_Batt.loc[inputs['prob_choice'][2],:]*inputs['kWh_dis']
        df_prices=pd.DataFrame(index=net_demand.index)
        df_prices.loc[:,'prices']=net_demand.apply(lambda x:inputs['prices'][find_interval_PQ(x,q_supply_midday)])
        df_prices.loc[df_prices.index.hour<10,'prices']=net_demand[net_demand.index.hour<10].apply(lambda x:inputs['prices'][find_interval_PQ(x,q_supply_morning)])
        df_prices.loc[df_prices.index.hour>17,'prices']=net_demand[net_demand.index.hour>17].apply(lambda x:inputs['prices'][find_interval_PQ(x,q_supply_evening)])

        inputs.update({'df_prices':df_prices})
        return inputs
    else:
        print('Warning: this function only takes either one choice of probability for the whole day or three separated for morning, midday and evening')
        return

    
def no_community_approach(inputs, system_param):
    '''
    Description
    -----------
    This function defines the dataframe selecting the appropriate house and its PV generation if it has then passes it to the PV_SC function (wihtout consumer behaviour). Finally, defines the different energy flows.

    Parameters
    ------

    inputs: dict; includes demand, generation, prices and the selected houses for PV and battery
    system_param: dict; includes the battery characteristics, the inverter and converter efficiencies, the path and the end of the index in case we do not want to simulate the whole year

    Return
    ------
    df_no_comm: DataFrame; includes the balance of the community with the number of the house in a single df
    TODO
    ------

    '''
    print('################################################')
    print('Simulating without community exchange')
    df_no_comm=pd.DataFrame()
    j=0
    for i in inputs['df_demand'].columns:
        #print(i)
        col=[]
        if i in inputs['selection_PV']:
            col=[s for s in list(inputs['df_generation']) if 'hh_'+str(i)+'_' in s]
            stack=np.hstack((np.array(np.zeros([len(inputs['df_demand'].index),12])),np.reshape(np.tile(i,inputs['df_demand'].shape[0]),(len(inputs['df_demand'].index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(inputs['df_demand'].index)
            df1=pd.concat([inputs['df_demand'].loc[:,i],inputs['df_generation'].loc[:,col],aux,inputs['df_prices'][:]],axis=1)
            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load','Batt_grid','grid_load','PV_losses','Batt_losses','flag','type','df','prices']
            if system_param['day_sel'] in ['winter','summer']:
                df1.SOC[0]=system_param['Batt'].Capacity*0.5
            if i in inputs['selection_PV_Batt']:# PV and Batt
                df_no_prob=PV_SC(df1,system_param['Batt'],system_param['Conv_eff'],system_param['Inv_eff'],system_param['endo'])
                df_no_prob.loc[:,'type']=2
                df_no_prob=df_no_prob.reset_index()
                if i=='0':
                    df_no_comm=df_no_prob.copy()

                else:
                    df_no_comm=df_no_comm.append(df_no_prob, ignore_index=True)
            else:#PV only
                df_no_prob=df1.copy()
                df_no_prob.loc[:,'type']=1#0 is no PV; 1 only PV and 2 PV+Batt

                df_no_prob.loc[:,'PV_load']=df1[['demand','gen']].min(axis=1)
                df_no_prob.loc[:,'PV_grid']=(df1.gen)*system_param['Inv_eff']-df1[['demand','gen']].min(axis=1)
                df_no_prob.loc[:,'PV_losses']=df1.gen*(1-system_param['Inv_eff'])
                df_no_prob.loc[:,'grid_load']=df_no_prob.loc[:,'demand']-df_no_prob.loc[:,'PV_load']
                df_no_prob=df_no_prob.reset_index()
                if i=='0':
                    df_no_comm=df_no_prob.copy()

                else:
                    df_no_comm=df_no_comm.append(df_no_prob, ignore_index=True)
            j+=1
        else:
            stack=np.hstack((np.array(np.zeros([len(inputs['df_demand'].index),13])),np.reshape(np.tile(i,inputs['df_demand'].shape[0]),(len(inputs['df_demand'].index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(inputs['df_demand'].index)
            df1=pd.concat([inputs['df_demand'].loc[:,i],aux,inputs['df_prices'][:]],axis=1)
            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load', 'Batt_grid','grid_load','PV_losses', 'Batt_losses','flag','type','df','prices']

            df_no_prob=df1.copy()
            df_no_prob.loc[:,'type']=0#0 is no PV; 1 only PV and 2 PV+Batt
            df_no_prob.loc[:,'grid_load']=df1.demand
            df_no_prob=df_no_prob.reset_index()
            if i=='0':
                df_no_comm=df_no_prob.copy()
            else:
                df_no_comm=df_no_comm.append(df_no_prob, ignore_index=True)

    # We have to change PV_load to include the losses when we use only PV.

    df_no_comm.PV_load.fillna(0)
    df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'PV_load']=df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'PV_load']+df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'PV_grid']
    df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'grid_load']=df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'grid_load']-df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'PV_grid']
    df_no_comm.loc[(df_no_comm.type==1)&(df_no_comm.PV_grid<0),'PV_grid']=0

    print(df_no_comm.sum().round(2))
    print('End of simulation without community exchange')
    print('###################################################')
    print('################################################')
    print('Simulation with community exchange')
    return(df_no_comm)

def community_approach(inputs, system_param):
    '''
    Description
    -----------
    This function defines the dataframe selecting the appropriate house and its PV generation if it has then passes it to the PV_SC_probs function to include the consumer behaviour. Finally, defines the different energy flows.

    Parameters
    ------

    inputs: dict; includes demand, generation, prices and the selected houses for PV and battery
    system_param: dict; includes the battery characteristics, the inverter and converter efficiencies, the path and the end of the index in case we do not want to simulate the whole year

    Return
    ------

    df_comm: DataFrame; includes the balance of the community with the number of the house in a single df
    TODO
    ------

    '''
    df_comm=pd.DataFrame()
    j=0
    for i in inputs['df_demand'].columns:
        #print(i)
        col=[]
        if i in inputs['selection_PV']:
            col=[s for s in list(inputs['df_generation']) if 'hh_'+str(i)+'_' in s]

            stack=np.hstack((np.array(np.zeros([len(inputs['df_demand'].index),12])),np.reshape(np.tile(i,inputs['df_demand'].shape[0]),(len(inputs['df_demand'].index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(inputs['df_demand'].index)
            df1=pd.concat([inputs['df_demand'].loc[:,i],inputs['df_generation'].loc[:,col],aux,inputs['df_prices'][:]],axis=1)

            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load',
                    'Batt_grid', 'grid_load','PV_losses','Batt_losses','flag','type','df','prices']
            if system_param['day_sel'] in ['winter','summer']:
                df1.SOC[0]=system_param['Batt'].Capacity*0.5
            if i in inputs['selection_PV_Batt']:# PV and Batt
                df_prob=PV_SC_probs(df1,system_param,inputs['flag'])
                df_prob.loc[:,'type']=2
                df_prob=df_prob.reset_index()
                if i=='0':
                    df_comm=df_prob.copy()

                else:
                    df_comm=df_comm.append(df_prob, ignore_index=True)
            else:#PV only
                df_prob=df1.copy()
                df_prob.loc[:,'type']=1#0 is no PV; 1 only PV and 2 PV+Batt

                df_prob.loc[:,'PV_load']=df1[['demand','gen']].min(axis=1)
                df_prob.loc[:,'PV_grid']=(df1.gen)*system_param['Inv_eff']-df1[['demand','gen']].min(axis=1)
                df_prob.loc[:,'PV_losses']=df1.gen*(1-system_param['Inv_eff'])
                df_prob.loc[:,'grid_load']=df_prob.loc[:,'demand']-df_prob.loc[:,'PV_load']
                df_prob=df_prob.reset_index()
                if i=='0':
                    df_comm=df_prob.copy()

                else:
                    df_comm=df_comm.append(df_prob, ignore_index=True)
            j+=1
        else:

            stack=np.hstack((np.array(np.zeros([len(inputs['df_demand'].index),13])),np.reshape(np.tile(i,inputs['df_demand'].shape[0]),(len(inputs['df_demand'].index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(inputs['df_demand'].index)
            df1=pd.concat([inputs['df_demand'].loc[:,i],aux,inputs['df_prices'][:]],axis=1)
            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load','Batt_grid','grid_load','PV_losses','Batt_losses','flag','type','df','prices']

            df_prob=df1.copy()
            df_prob.loc[:,'type']=0#0 is no PV; 1 only PV and 2 PV+Batt
            df_prob.loc[:,'grid_load']=df1.demand
            df_prob=df_prob.reset_index()
            if i=='0':
                df_comm=df_prob.copy()
            else:
                df_comm=df_comm.append(df_prob, ignore_index=True)

    df_comm.PV_load.fillna(0)
    df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'PV_load']=df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'PV_load']+df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'PV_grid']
    df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'grid_load']=df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'grid_load']-df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'PV_grid']
    df_comm.loc[(df_comm.type==1)&(df_comm.PV_grid<0),'PV_grid']=0

    df_comm.columns=['date','demand','gen','SOC','PV_batt', 'PV_load', 'PV_comm', 'E_dis','Batt_load',
                'Batt_comm', 'comm_load','PV_losses','Batt_losses','flag','type','df','prices']
    print(df_comm.sum().round(2))
    print('End of simulation with community exchange')
    print('###################################################')
    return(df_comm)

def community_psycho(Batt_penetration,PV_penetration,reso,path,day_sel,probs_applied):
    '''
    Description
    -----------
    This function calls the two functions no_community_approach and community_approach to calculate the balance of the houses when they are individually operated and when they are in a community in order to have a comparison.
    Three scenarios are analyzed.
        1. No community: when every house is connected directly to the grid and does not share within a community.
        2. Community No behaviour: when every house is in the community and there is only one physical connection to the grid. There is not consumer behaviour.
        3. Community behaviour: when every house is in the community and there is only one physical connection to the grid. Consumer behaviour is included.
    For scenarios 1 and 2 no_community_approach is used and then the data is processed differently (no consumer behaviour involved).
    For scenario 3 community_approach is used (includes consumer behaviour in the form of probability distributions)

    In a first instance the prices are calculated using Price_definition function
    'Output/community_{}_{}_{}.csv'.format(PV_penetration,Batt_penetration,probs_applied))
    'Output/no_community_{}_{}_{}.csv'.format(PV_penetration,Batt_penetration,probs_applied))
    Parameters
    ------
    Batt_penetration: float; Battery penetration in the community
    PV_penetration: float; PV penetration in the community
    reso: float; resolution either 1h or 15 minutes
    path:  string; path where the project is kept
    day_sel: string; either winter, summer or other to simulate one day on winter, summer or all the year (test)

    Return
    ------
    Gen_balance: float; Balance of generation must be zero
    Batt_balance float; Balance of the battery must be zero
    Demand_balance float; Balance of the demand must be zero
    TODO
    ------

    '''

    if (day_sel=='winter') or (day_sel=='summer'):
        endo=96
    else:
        endo=8760
    Conv_eff=0.98
    Inv_eff=0.95
    prices=np.array([0, 0.07,.10,.13,.16,.19,.22,.25,.28])
    prob_choice=[0.35,0.5,.95]
    kWh_dis=2
    case='PV'
    Batt=pc.Battery(10,'NMC')
    inputs={'Conv_eff':Conv_eff, 'Inv_eff':Inv_eff,'prices':prices,'Batt':Batt,'PV_penetration':PV_penetration,'probs_applied':probs_applied,
                   'Batt_penetration':Batt_penetration,'prob_choice':prob_choice,'kWh_dis':kWh_dis,'case':case,'reso':reso,'path':path,'day_sel':day_sel}
    inputs=create_community(inputs)
    inputs=price_probability(inputs)

    
    
    #inputs=Price_definition(prices,PV_penetration,Batt_penetration,reso,path,day_sel)

    system_param={'Batt':Batt,'Conv_eff':Conv_eff,'Inv_eff':Inv_eff,'endo':endo,'day_sel':day_sel,'probs_applied':probs_applied,'path':path,'kWh_dis':kWh_dis}

    print('################################################')
    print('Simulation begins')
    df_no_comm=no_community_approach(inputs,system_param)
    df_comm=community_approach(inputs,system_param)
    print('End of the simulation')
    print('################################################')

    df_no_comm.to_csv(path+'Output/no_community_{}_{}_{}.csv'.format(PV_penetration,Batt_penetration,probs_applied))
    df_comm.to_csv(path+'Output/community_{}_{}_{}.csv'.format(PV_penetration,Batt_penetration,probs_applied))

    Gen_balance=(df_comm.gen-(df_comm.PV_batt+df_comm.PV_load+df_comm.PV_comm+df_comm.PV_losses))[:endo].sum()

    Batt_balance=(df_comm.PV_batt-(df_comm.Batt_load+df_comm.Batt_comm+df_comm.Batt_losses))[:endo].sum()

    Demand_balance=(df_comm.demand-(df_comm.PV_load+df_comm.Batt_load+df_comm.comm_load))[:endo].sum()
    return[Gen_balance,Batt_balance,Demand_balance]
