import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import paper_classes_2 as pc
import Post_processing as pp
import itertools
from functools import wraps
import time
from termcolor import colored
import csv
import os
import datetime as dt
import warnings
import pickle
warnings.filterwarnings('error')

def save_obj(obj, name ):
    with open('../Output/'+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../Output/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class bcolors:
    '''Print with colors'''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def dict_to_csv(my_dict,filename):
    '''saves the dictionary to a csv using as input the dictionary and the name without the format (.csv)'''
    flag=False
    if os.path.exists(filename+'.csv'):
        flag=True
    with open(filename+'.csv', 'a+') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f)
        if flag==False:
            w.writerow(my_dict.keys())
        w.writerow(my_dict.values())
        
def fn_timer(function):
    '''Measures the time a function takes to run'''
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.__name__, str(t1-t0))
               )
        return result
    return function_timer



def plot_dispatch(pv, demand, E, week=30):
    """ Visualize dispatch algorithm for a specific week for a single household
    Parameters:
        demand (pd.Series): demand production
        pv (pd.Series): pv production
        E (dict):  Energy flows. Dictionary of pd.Series: res_pv, grid2load, store2inv, LevelOfCharge
    """

    sliced_index = (pv.index.week==week)
    pv_sliced = pv[sliced_index]
    demand_sliced = demand[sliced_index]
    self_consumption = E['inv2load'][sliced_index]
    
    direct_self_consumption = np.minimum(pv_sliced,demand_sliced)# E['inv2load'][sliced_index]
    indirect_self_consumption = self_consumption-direct_self_consumption
    res_pv_sliced = E['res_pv'][sliced_index]
    grid2load_sliced = E['grid2load'][sliced_index]
    store2inv_sliced = E['store2inv'][sliced_index]
    LevelOfCharge = E['LevelOfCharge'][sliced_index]
    inv2grid = E['inv2grid'][sliced_index]
    grid2load = E['grid2load'][sliced_index]
    aux=np.maximum(0,self_consumption)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(17, 4*3), frameon=False,
                             gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.04})

    #fig, ax = plt.subplots(figsize=(17, 4))
    axes[0].plot(demand_sliced.index, demand_sliced, color='black', lw=2,label='demand')
    axes[0].plot(pv_sliced.index, pv_sliced, color='black',ls='--', lw=2,label='PV')
    axes[0].fill_between(direct_self_consumption.index, 0, direct_self_consumption, color='orange', alpha=.8, label='DSC')
    axes[0].fill_between(pv_sliced.index, self_consumption, pv_sliced , where=pv_sliced<demand_sliced,color='blue', hatch='//',
                         alpha=.3,label='ISC')
    axes[0].fill_between(pv_sliced.index, direct_self_consumption, pv_sliced ,where=pv_sliced>demand_sliced, color='gold', alpha=.3,label='Excess PV')

    axes[0].fill_between(grid2load_sliced.index,self_consumption,demand_sliced,color='red',alpha=.2, label='grid2load')
        

    #axes[0].plot(grid2load_sliced.index, grid2load_sliced, color='red', ls=":", lw=1)
    axes[0].set_ylim([0, axes[0].get_ylim()[1] ])
    axes[0].set_ylabel('Power (kW)')

    axes[1].fill_between(LevelOfCharge.index, 0, LevelOfCharge, color='grey', alpha=.2, label='SOC')
    axes[1].set_ylabel('State of Charge (kWh)')

    axes[2].fill_between(inv2grid.index, 0, inv2grid, color='green', alpha=.2,label='injected2grid')
    axes[2].fill_between(inv2grid.index, 0, -grid2load, color='red', alpha=.2,label='grid drawn')
    axes[2].set_ylabel('In/out from grid (kW)')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    return

def plot_dispatch_comm(pv, demand, E, week=30,flag=False):
    """ Visualize dispatch algorithm for a specific week for the whole community
    Parameters:
        demand (pd.Series): demand production
        pv (pd.Series): pv production
        E (dict):  Energy flows. Dictionary of pd.Series: res_pv, grid2load, store2inv, LevelOfCharge
        flag (bool): fill_between
    """

    sliced_index = (pv.index.week==week)
    pv_sliced = pv[sliced_index]
    demand_sliced = demand[sliced_index]
    self_consumption = E['inv2load'][sliced_index]
    
    direct_self_consumption = np.minimum(pv_sliced,demand_sliced)# E['inv2load'][sliced_index]
    indirect_self_consumption = self_consumption-direct_self_consumption
    res_pv_sliced = E['res_pv'][sliced_index]
    grid2load_sliced = E['grid2load'][sliced_index]
    store2inv_sliced = E['store2inv'][sliced_index]
    LevelOfCharge = E['LevelOfCharge'][sliced_index]
    inv2grid = E['inv2grid'][sliced_index]
    grid2load = E['grid2load'][sliced_index]
    aux=np.maximum(0,self_consumption)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(17, 4*3), frameon=False,
                             gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.04})

    #fig, ax = plt.subplots(figsize=(17, 4))
    axes[0].plot(demand_sliced.index, demand_sliced, color='black', lw=2,label='demand')
        
    if flag:
        axes[0].plot(pv_sliced.index, pv_sliced, color='green', lw=2,label='pv')
        axes[0].plot(direct_self_consumption.index, direct_self_consumption, color='yellow', lw=2,label='DSC')
        axes[0].plot(indirect_self_consumption.index, indirect_self_consumption, color='orange', lw=2,label='ISC')
        axes[0].plot(grid2load_sliced.index, grid2load_sliced, color='red', lw=2,label='grid')

    else:
        axes[0].fill_between(direct_self_consumption.index, 0, direct_self_consumption, color='orange', alpha=.8, label='DSC')
        axes[0].fill_between(pv_sliced.index, self_consumption, pv_sliced ,where=pv_sliced<demand_sliced, color='blue', hatch='//',
                             alpha=.3,label='ISC')
        axes[0].fill_between(pv_sliced.index, direct_self_consumption, pv_sliced , color='gold', alpha=.3,label='Excess PV')

        axes[0].fill_between(grid2load_sliced.index,self_consumption,demand_sliced,color='red',alpha=.2, label='grid2load')
    axes[0].set_ylim([0, axes[0].get_ylim()[1] ])
    axes[0].set_ylabel('Power (kW)')

    axes[1].fill_between(LevelOfCharge.index, 0, LevelOfCharge, color='grey', alpha=.2, label='SOC')
    axes[1].set_ylabel('State of Charge (kWh)')

    axes[2].fill_between(inv2grid.index, 0, inv2grid, color='green', alpha=.2,label='injected2grid')
    axes[2].fill_between(inv2grid.index, 0, -grid2load, color='red', alpha=.2,label='grid drawn')
    axes[2].set_ylabel('In/out from grid (kW)')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    return

def print_analysis_prices(pv, demand,retail,export, param, E,isCommunity=False,hh=None):
    """ Print statistics and information of the dispatched solution
    Arguments
        pv (pd.Series): PV timeseries
        demand (pd.Series): demand timeseries
        param (dict): dictionary of technical parameters
        E (dict): dictionary of energy flows as estimated by the algorithm
    Returns
        none
    'store2inv': store2inv, # DC
        'inv2curt':inv2curt # DC
        'pv2store': pv2store, # DC
        'inv2load': inv2load, # AC
        'grid2load': grid2load, # AC
        'pv2inv': pv2inv, # AC
        'LevelOfCharge': LevelOfCharge,
        'inv2grid': inv2grid, #AC
    """
    RemainingSOC=E['LevelOfCharge'][-1]
    timestep = param['timestep']
    SelfConsumption = np.sum(E['inv2load']) * timestep # AC
    TotalFromGrid = np.sum(E['grid2load']) * timestep # AC
    TotalToGrid = np.sum(E['inv2grid']) * timestep # AC
    TotalLoad = demand.sum() * timestep # AC
    #TotalBattToLoad = np.sum(E['store2load']) * timestep # AC
    TotalBattToGrid = np.sum(E['store2grid']) * timestep # AC
    TotalPV = pv.sum() * timestep # DC
    TotalBatteryGeneration = np.sum(E['store2inv']) * timestep # DC
    TotalBatteryConsumption = np.sum(E['pv2store']) * timestep # DC
    if 'inv_losses' in E.keys():
        BatteryLosses=E['batt_losses'].sum()*timestep
        InverterLosses=E['inv_losses'].sum()*timestep
    else:
        BatteryLosses = TotalBatteryConsumption * (1 - param['BatteryEfficiency'])
        InverterLosses = (TotalPV - BatteryLosses-RemainingSOC) * (1 - param['InverterEfficiency'])
    SelfConsumptionRate = SelfConsumption / TotalPV * 100             # in %
    SelfSufficiencyRate = SelfConsumption / TotalLoad * 100
    Bill=((E['grid2load'] * timestep) * retail - (E['inv2grid'] * timestep ) * export).sum()
    Batt_revenue=((E['store2load']*param['InverterEfficiency']*timestep*retail-
                   E['pv2store']*param['InverterEfficiency']*timestep*export)).sum()
    
    print ('Total yearly consumption: {:1g} kWh'.format(TotalLoad))
    print ('Total PV production: {:1g} kWh'.format(TotalPV))
    print ('Self Consumption: {:1g} kWh'.format(SelfConsumption))
    print ('Total fed to the grid: {:1g} kWh'.format(TotalToGrid))
    print ('Total bought from the grid: {:1g} kWh'.format(TotalFromGrid))
    print ('Self consumption rate (SCR): {:.3g}%'.format(SelfConsumptionRate))
    print ('Self sufficiency rate (SSR): {:.3g}%'.format(SelfSufficiencyRate))
    print ('Amount of energy provided by the battery: {:1g} kWh'.format(TotalBatteryGeneration))
    print ('Total battery losses: {:1g} kWh, i.e., {:1g}% of the total PV'.format(BatteryLosses,BatteryLosses/TotalPV*100))
    #print('Total energy from battery to the load {:1g} kWh'.format(TotalBattToLoad))
    print('Total energy from battery to the grid {:1g} kWh'.format(TotalBattToGrid))
    #print ('Total inverter losses: {:1g} kWh'.format(InverterLosses))
    #print ('Total inverter losses: {:1g} kWh'.format(InverterLosses))
    print ('Total inverter losses: {:1g} kWh, i.e., {:1g}% of the total PV'.format(InverterLosses,InverterLosses/TotalPV*100))
    
    
    TotalCurtailment=np.sum(E['inv2curt'])*timestep # DC
    print ('Total curtailment : {:1g} kWh'.format(TotalCurtailment))    
    residue = TotalPV + TotalFromGrid - TotalToGrid - BatteryLosses - InverterLosses - TotalLoad - TotalCurtailment - RemainingSOC
    print ('Residue (check): {:1g} kWh'.format(residue))
    PV_check = TotalPV - SelfConsumption - TotalToGrid - BatteryLosses - InverterLosses - TotalCurtailment - RemainingSOC
    print ('PV Residue (check): {:1g} kWh'.format(PV_check))
    
    print(bcolors.WARNING + 'Maximum power injected into the grid is {:1g} kW'.format(E['inv2grid'].max())+bcolors.ENDC)
    print(bcolors.WARNING + 'Maximum power drained from the grid is {:1g} kW'.format(E['grid2load'].max())+bcolors.ENDC)
    print (bcolors.WARNING + 'Total bill: {:1g}\n\n'.format(Bill)+bcolors.ENDC)
    print (bcolors.WARNING + 'Total Batt_revenue: {:1g}\n\n'.format(Batt_revenue)+bcolors.ENDC)
    
    if isCommunity==False:
        AverageDepth = TotalBatteryGeneration / (365 * param['BatteryCapacity'])
        Nfullcycles = 365 * AverageDepth        
        print ('Number of equivalent full cycles per year: {:1g} '.format(Nfullcycles))
        print ('Average Charging/Discharging depth: {:1g}\n\n'.format(AverageDepth))
        
        out = { 'SCR': SelfConsumptionRate, # 
                'SSR':SelfSufficiencyRate, # 
                'EFC': Nfullcycles, # 
                'Demand_peak': E['grid2load'].max(), # 
                'Inj_peak': E['inv2grid'].max(), #
                'avg_dod': AverageDepth, #
                'bill': Bill,
                'Batt_revenue':Batt_revenue,
                'Batt_penetration':param['batt_penetration'],
                'PV_penetration':param['pv_penetration'],
                'seed':param['seed'],
                'hh':hh
                }
    else:
        out = { 'SCR': SelfConsumptionRate, # 
                'SSR':SelfSufficiencyRate, # 
                'EFC': None, # 
                'Demand_peak': E['grid2load'].max(), # 
                'Inj_peak': E['inv2grid'].max(), #
                'avg_dod': None, #
                'bill': Bill,
                'Batt_revenue':Batt_revenue,
                'Batt_penetration':param['batt_penetration'],
                'PV_penetration':param['pv_penetration'],
                'seed':param['seed'],
                'hh':hh
                }
    return out

def dispatch_max_sc(pv, demand, inv_size,param, return_series=False):
    """ Self consumption maximization pv + battery dispatch algorithm.
    The dispatch of the storage capacity is performed in such a way to maximize self-consumption:
    the battery is charged when the PV power is higher than the load and as long as it is not fully charged.
    It is discharged as soon as the PV power is lower than the load and as long as it is not fully discharged.
    Arguments:
        pv (pd.Series): Vector of PV generation, in kW DC (i.e. before the inverter)
        demand (pd.Series): Vector of household consumption, kW
        param (dict): Dictionary with the simulation parameters:
                timestep (float): Simulation time step (in hours)
                BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                BatteryEfficiency: Battery round-trip efficiency, -
                InverterEfficiency: Inverter efficiency, -
                MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
        return_series(bool): if True then the return will be a dictionary of series. Otherwise it will be a dictionary of ndarrays.
                        It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
    Returns:
        dict: Dictionary of Time series
    """
    bat_size_e_adj = param['BatteryCapacity']
    bat_size_p_adj = param['MaxPower']
    n_bat = param['BatteryEfficiency']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    # We work with np.ndarrays as they are much faster than pd.Series
    Nsteps = len(pv)
    LevelOfCharge = np.zeros(Nsteps)
    #inv2grid = np.zeros(Nsteps)
    inv_array=np.tile(inv_size/n_inv,len(pv))
    pv2store = np.zeros(Nsteps)
    store2inv = np.zeros(Nsteps)
    inv2curt = np.zeros(Nsteps)
    flagsell = np.zeros(Nsteps)
    flag_12h = np.zeros(Nsteps)
    store2grid = np.zeros(Nsteps)
    store2load = np.zeros(Nsteps)
    #grid2store = np.zeros(Nsteps) # TODO Always zero for now.

    #Load served by PV
    pv2load_dc = np.array([pv, demand / n_inv,inv_array]).min(axis=0)  # DC direct self-consumption, with inverter limitation

    #Residual load
    res_load = demand - (pv2load_dc * n_inv)  # AC
    inv2load = pv2load_dc * n_inv  # AC

    #Excess PV
    res_pv = pv-pv2load_dc # DC

    #PV to storage after eff losses
    pv2inv = pv2load_dc*n_inv # AC

    #first timestep = 0
    LevelOfCharge[0] = 0  # bat_size_e_adj / 2  # DC


    for i in range(1,Nsteps):
        #PV to storage
        if LevelOfCharge[i-1] >= bat_size_e_adj:  # if battery is full
                pv2store[i] = 0
        else: #if battery is not full

            if LevelOfCharge[i-1] + res_pv[i] * n_bat * timestep > bat_size_e_adj:  # if battery will be full after putting excess
                pv2store[i] = min((bat_size_e_adj - LevelOfCharge[i-1]) / timestep, bat_size_p_adj)
            else:
    #                pv2store[i] = min(res_pv[i], bat_size_p_adj)
                pv2store[i] = min(res_pv[i] * n_bat, bat_size_p_adj)
        #Storage to load
        if pv2store[i]==0:# modification to original algorithm (The battery cannot charge and discharge at the same time)
            store2inv[i] = min(bat_size_p_adj,(inv_size/n_inv-pv2load_dc[i]),  # DC
                       res_load[i] / n_inv,
                       LevelOfCharge[i-1] / timestep) #modif to original, store2inv=pv2store*n_bat

        #SOC
        LevelOfCharge[i] = min(LevelOfCharge[i-1] - (store2inv[i] - pv2store[i]*n_bat ) * timestep,  # DC
                               bat_size_e_adj)#modif to original, store2inv=pv2store*n_bat

    pv2grid_dc=np.array([pv-pv2store,inv_array]).min(axis=0)-pv2load_dc # DC
    pv2inv= (pv2grid_dc+pv2load_dc)*n_inv # AC
    inv2curt=pv-pv2grid_dc-pv2load_dc-pv2store # DC

    inv2load = (pv2load_dc + store2inv) * n_inv  # AC
    inv2grid = pv2grid_dc * n_inv  # AC
    grid2load = demand - inv2load  # AC
    #MaxDischarge = np.minimum(LevelOfCharge[i-1]*BatteryEfficiency/timestep,MaxPower)
    batt_losses=pv2store*(1-n_bat)
    inv_losses=(pv2grid_dc+pv2load_dc+store2inv)*(1-n_inv)
    #Potential Grid to storage  # TODO: not an option for now in this strategy
    # GridPurchase = False
    
    out = { 'pv2inv': pv2inv, # AC
            'res_pv':res_pv, # DC
            'pv2store': pv2store, # DC
            'inv2load': inv2load, # AC
            'grid2load': grid2load, # AC
            'store2inv': store2inv, # DC
            'inv2curt':inv2curt, # DC
            'LevelOfCharge': LevelOfCharge, # kWh
            'inv2grid': inv2grid, #AC            
            'inv_losses':inv_losses,
            'batt_losses':batt_losses,
            'flag_sell':flagsell,
            'flag_12h':flag_12h,             
            'store2grid':store2grid,
            'store2load':store2inv*n_inv
            }
    if not return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
    return out
def dispatch_max_sc_bhv(pv, demand, bhv,prices_binned,inv_size,param , return_series=False, bins_soc=[0,33,66,100],bins_price=[0.04,0.12,0.20,0.28],kW_dis=4):
    """ Self consumption maximization pv + battery dispatch algorithm.
    The dispatch of the storage capacity is performed in such a way to maximize self-consumption:
    the battery is charged when the PV power is higher than the load and as long as it is not fully charged.
    It is discharged as soon as the PV power is lower than the load and as long as it is not fully discharged.
    Arguments:
        pv (pd.Series): Vector of PV generation, in kW DC (i.e. before the inverter)
        demand (pd.Series): Vector of household consumption, kW
        param (dict): Dictionary with the simulation parameters:
                timestep (float): Simulation time step (in hours)
                BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                BatteryEfficiency: Battery round-trip efficiency, -
                InverterEfficiency: Inverter efficiency, -
                MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
                prices: Community prices for sell and buy
        bhv (pd.DataFrame): pd.DataFrame with price, SOC, surplus time for sell and willingness to participate in the community
        return_series(bool): if True then the return will be a dictionary of series. Otherwise it will be a dictionary of ndarrays.
                        It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
    Returns:
        dict: Dictionary of Time series
    """
    
    bat_size_e_adj = param['BatteryCapacity']
    bat_size_p_adj = param['MaxPower']
    n_bat = param['BatteryEfficiency']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    # We work with np.ndarrays as they are much faster than pd.Series
    Nsteps = len(pv)
    LevelOfCharge = np.zeros(Nsteps)
    inv2grid = np.zeros(Nsteps)
    inv_array=np.tile(inv_size/n_inv,len(pv))
    pv2store = np.zeros(Nsteps)
    store2inv = np.zeros(Nsteps)
    inv2curt = np.zeros(Nsteps)
    
    flagsell = np.zeros(Nsteps)
    store2load = np.zeros(Nsteps)
    store2grid = np.zeros(Nsteps)
    flag_12h=np.full((Nsteps), False)
    #grid2store = np.zeros(Nsteps) # TODO Always zero for now.
    
    #Load served by PV
    pv2load_dc = np.array([pv, demand / n_inv,inv_array]).min(axis=0)  # DC direct self-consumption, with inverter limitation

    #Residual load
    res_load = demand - (pv2load_dc * n_inv)  # AC
    inv2load = pv2load_dc * n_inv  # AC

    #Excess PV
    res_pv = pv-pv2load_dc # DC

    #PV to storage after eff losses
    pv2inv = pv2load_dc*n_inv # AC

    #first timestep = 0
    LevelOfCharge[0] = 0  # bat_size_e_adj / 2  # DC

    #get the sp hours
    delta=dt.timedelta(hours=12)
    sp_hour=res_pv.ne(0).groupby([res_pv.index.month,res_pv.index.day]).idxmax().droplevel(1).reset_index(drop=True)
    twelve_bf=sp_hour-delta

    for i in range(1,Nsteps):    
        #PV to storage
        if LevelOfCharge[i-1] >= bat_size_e_adj:  # if battery is full
                pv2store[i] = 0
        else: #if battery is not full
            if LevelOfCharge[i-1] + res_pv[i] * n_bat * timestep > bat_size_e_adj:  # if battery will be full after putting excess
                pv2store[i] = min((bat_size_e_adj - LevelOfCharge[i-1]) / timestep, bat_size_p_adj)
            else:
#                pv2store[i] = min(res_pv[i], bat_size_p_adj)
                pv2store[i] = min(res_pv[i] * n_bat, bat_size_p_adj)
        #Storage to load
        if (pv2store[i]==0)&(res_pv[i]==0): 
            #According to prosumer behaviour the battery will inject into the community or not only if there is not PV suplus
            store2load[i] = min(bat_size_p_adj,(inv_size/n_inv-pv2load_dc[i]), #Power
                           res_load[i] / n_inv, #Power
                           LevelOfCharge[i-1] / timestep)  # Power; all in DC
            if (pv.index[i]<=sp_hour[int(np.floor((i)/96))]) & (pv.index[i]>twelve_bf[int(np.floor((i)/96))]):
                #if timestep is between twelve hours before and the surplus hour
                try:
                    flag_12h[i]=True
                    if np.array(bhv.loc[(bhv.Price_binned==prices_binned[i])&
                                        (bhv.SOC_binned==np.digitize(LevelOfCharge[i-1],bins=bins_soc))&
                                        (bhv.surplus_time=='12h'),'sell'].values):#if sell
                        #surplus_time==12h
                        # sell 1 kWh of energy to the community and cover residual load if excess
                        #print('12hsell')
                        flagsell[i]=1
                        store2grid[i] = min(bat_size_p_adj-store2load[i],(inv_size/n_inv-pv2load_dc[i]-store2load[i]),  #Power
                                               kW_dis / n_inv, #Power
                                               (LevelOfCharge[i-1]) / timestep-store2load[i])# Power; all in DC
                        store2inv[i] = (store2load[i]+store2grid[i]) # DC

                    # here should take into account whether the prosumer desires to sell or not according to his preferences (SOC,price)
                    else: # no sell then cover only res_load
                        #print('out12h')
                        store2inv[i] = (store2load[i]+store2grid[i]) #DC
                except Exception as e:
                    # Ugly but provisional SOC bin from 0 to 1
                    if np.array(bhv.loc[(bhv.Price_binned==prices_binned[i])&
                                        (bhv.SOC_binned==1)&
                                        (bhv.surplus_time=='12h'),'sell'].values):#if sell
                        #surplus_time==12h
                        # sell 1 kWh of energy to the community and cover residual load if excess
                        #print('12hsell')
                        flagsell[i]=1
                        store2grid[i] = min(bat_size_p_adj-store2load[i],(inv_size/n_inv-pv2load_dc[i]-store2load[i]),  #Power
                                               kW_dis / n_inv, #Power
                                               (LevelOfCharge[i-1]) / timestep-store2load[i])# Power; all in DC
                        store2inv[i] = (store2load[i]+store2grid[i]) # DC
                    else: # no sell then cover only res_load
                        #print('out12h')
                        store2inv[i] = (store2load[i]+store2grid[i]) #DC
                    
            else: 
                try:
                
                    if np.array(bhv.loc[(bhv.Price_binned==prices_binned[i])&
                                        (bhv.SOC_binned==np.digitize(LevelOfCharge[i-1],bins=bins_soc))&
                                        (bhv.surplus_time=='12h+'),'sell'].values):#if sell
                        #surplus_time==12h+
                        # sell 1 kWh of energy to the community and cover residual load if excess                    
                        flagsell[i]=2
                        store2grid[i] = min(bat_size_p_adj-store2load[i],(inv_size/n_inv-pv2load_dc[i]-store2load[i]),  #Power
                                               kW_dis / n_inv, #Power
                                               (LevelOfCharge[i-1]) / timestep-store2load[i])# Power; all in DC
                        store2inv[i] = (store2load[i]+store2grid[i]) # DC

                    # here should take into account whether the prosumer desires to sell or not according to his preferences (SOC,price)
                    else: # no sell then cover only res_load
                        #print('out12h+')
                        store2inv[i] = (store2load[i]+store2grid[i]) #DC
                except Exception as e:
                    # Ugly but provisional SOC bin from 0 to 1
                    if np.array(bhv.loc[(bhv.Price_binned==prices_binned[i])&
                                        (bhv.SOC_binned==1)&
                                        (bhv.surplus_time=='12h+'),'sell'].values):#if sell
                        #surplus_time==12h
                        # sell 1 kWh of energy to the community and cover residual load if excess
                        #print('12hsell')
                        flagsell[i]=1
                        store2grid[i] = min(bat_size_p_adj-store2load[i],(inv_size/n_inv-pv2load_dc[i]-store2load[i]),  #Power
                                               kW_dis / n_inv, #Power
                                               (LevelOfCharge[i-1]) / timestep-store2load[i])# Power; all in DC
                        store2inv[i] = (store2load[i]+store2grid[i]) # DC
                    else: # no sell then cover only res_load
                        #print('out12h')
                        store2inv[i] = (store2load[i]+store2grid[i]) #DC
        #SOC
        LevelOfCharge[i] = min(LevelOfCharge[i-1] - (store2inv[i] - pv2store[i]*n_bat ) * timestep,  # DC
                               bat_size_e_adj)#modif to original, store2inv=pv2store*n_bat

    pv2grid_dc=np.array([pv-pv2store,inv_array]).min(axis=0)-pv2load_dc # DC
    pv2inv= (pv2grid_dc+pv2load_dc)*n_inv # AC
    inv2curt=pv-pv2grid_dc-pv2load_dc-pv2store # DC

    inv2load = (pv2load_dc + store2load) * n_inv  # AC
    inv2grid = (pv2grid_dc+store2grid) * n_inv  # AC
    grid2load = demand - inv2load  # AC
    #MaxDischarge = np.minimum(LevelOfCharge[i-1]*BatteryEfficiency/timestep,MaxPower)
    batt_losses=pv2store*(1-n_bat)
    inv_losses=(pv2grid_dc+pv2load_dc+store2inv)*(1-n_inv)
 
    out = {'pv2inv': pv2inv, # AC
            'res_pv':res_pv, # DC
            'pv2store': pv2store, # DC
            'inv2load': inv2load, # AC
            'grid2load': grid2load, # AC
            'store2inv': store2inv, # DC
            'inv2curt':inv2curt, # DC
            'LevelOfCharge': LevelOfCharge, # kWh
            'inv2grid': inv2grid, # AC
            'inv_losses':inv_losses,
            'batt_losses':batt_losses,
            'flag_sell':flagsell,
            'flag_12h':flag_12h,             
            'store2grid':store2grid*n_inv, # AC
            'store2load':store2load*n_inv # AC
            
            }
    if not return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
        
    return out
@fn_timer
def flag_selection(df,list_product,community_size,seed):
    '''
    Function that randomly selects the households that will have PV only and those with PV and battery
    ''' 
    np.random.seed(seed)
    list_names=['sub_'+str(i[0])+'_'+str(i[1]) for i in list_product]
    for i in list_names:
        df[i]=False
    dict_pv_df={}
    dict_batt_df={}
    j=0
    l=0
    for i in list_pv_penetration: 
        #print('-----')
        dict_pv_df[j]=np.random.choice(df.index,int(len(df)*(i/100)), replace=False)
        for k in list_batt_penetration:
            #print(int(len(dict_pv_df[j])*(k/100)))
            dict_batt_df[l]=np.random.choice(dict_pv_df[j],int(len(dict_pv_df[j])*(k/100)), replace=False)
            df.iloc[dict_batt_df[l],l+2]=True
            l+=1
        j+=1
    return df    
    
@fn_timer
def main():
    first_time=True
    community_size=50
    timestep=0.25 # in hours 0.25 is 15 minutes
    seed=1
    pv_penetration=50
    batt_penetration=50
    #ILR=1.2
    
    retail_price_sc=np.ones(pv2inv.shape[0])*0.29
    export_price_sc=np.ones(pv2inv.shape[0])*0.04
    
    param_tech = {'BatteryCapacity': 10,
              'BatteryEfficiency': .91,
              'InverterEfficiency': .94,
              'timestep': timestep,
              'MaxPower': 4,
              'pv_penetration':pv_penetration,
              'batt_penetration':batt_penetration,
              'seed':seed
             }
    param_tech_no_batt = {'BatteryCapacity': 0,
                  'BatteryEfficiency': .91,
                  'InverterEfficiency': .94,
                  'timestep': timestep,
                  'MaxPower': 0,
                  'pv_penetration':pv_penetration,
                  'batt_penetration':batt_penetration,
                  'seed':seed
                 }
    param_tech_comm = {'BatteryCapacity': 0,
                  'BatteryEfficiency': 1,
                  'InverterEfficiency': 1,
                  'timestep': timestep,
                  'MaxPower': 0,
                  'pv_penetration':pv_penetration,
                  'batt_penetration':batt_penetration,
                  'seed':seed
                 }
    
    
    df_CH=pd.read_csv('../Input/df_CH_Marzia3.csv', encoding='utf8', sep=',',usecols=[*range(0, 669)],
                              engine='python',date_parser=lambda col: pd.to_datetime(col, utc=True),infer_datetime_format=True,index_col=0)
    df_CH.index=df_CH.index.tz_convert('Europe/Brussels')
    if timestep==1:
        df_CH=df_CH.resample('1H').sum()
    else:
        df_CH_kW=df_CH/timestep #data must be in kW
    max_PV=10
    if first_time:   
        path='../Input/'
        df_pv_ch=pd.read_excel(path+'PV_beneficiaires_Swiss.xlsx')
        df_nat=df_pv_ch[(df_pv_ch['Anlage_Projekt-Bezeichnung']=='natürliche Person')]
        df_nat_pv=df_nat[(df_nat['Anlage_Energieträger']=='Photovoltaik')]
        df_nat_pv_15=df_nat_pv[df_nat_pv['Leistung [kW]']<max_PV]['Leistung [kW]'].reset_index(drop=True)
    if not first_time:
        selection=pd.read_csv('../Input/selection.csv')
    nested_out={}
    j=0
    k=0
    PV_size_comm=0
    for i in selection.index:
        print(i, end='')
        if selection.loc[i,'sub_'+str(pv_penetration)+'_100']:#all with PV
            inv_size=max(param_tech['MaxPower'],selection.PV_size[i])#selection.PV_size[i]/ILR
            PV_size_comm+=selection.PV_size[i]
            if selection.loc[i,'sub_'+str(pv_penetration)+'_'+str(batt_penetration)]: #if battery
                nested_out[i]=dispatch_max_sc(df_CH_kW.E_PV*selection.PV_size[i],df_CH_kW.loc[:,str(selection.name[i])],inv_size,param_tech)
                j+=1
            else: #if only PV battery=0 kWh
                nested_out[i]=dispatch_max_sc(df_CH_kW.E_PV*selection.PV_size[i],df_CH_kW.loc[:,str(selection.name[i])],inv_size,param_tech_no_batt)
                k+=1
        else: #No PV
            nested_out[i]=dispatch_max_sc(df_CH_kW.E_PV*0,df_CH_kW.loc[:,str(selection.name[i])],0,param_tech_no_batt)
    
    #get all the data in a single dict
    pv2inv=pd.DataFrame()
    res_pv=pd.DataFrame()
    pv2store=pd.DataFrame()
    inv2load=pd.DataFrame()
    grid2load=pd.DataFrame()
    store2inv=pd.DataFrame()
    inv2curt=pd.DataFrame()
    LevelOfCharge=pd.DataFrame()
    inv2grid=pd.DataFrame()
    batt_losses=pd.DataFrame()
    inv_losses=pd.DataFrame()
    store2load=pd.DataFrame()
    store2grid=pd.DataFrame()
    for i in nested_out.keys():
        pv2inv['pv2inv'+str(i)]=nested_out[i]['pv2inv']
        inv2curt['inv2curt'+str(i)]=nested_out[i]['inv2curt']
        pv2store['pv2store'+str(i)]=nested_out[i]['pv2store']
        inv2load['inv2load'+str(i)]=nested_out[i]['inv2load']
        batt_losses['batt_losses'+str(i)]=nested_out[i]['batt_losses']
        inv_losses['inv_losses'+str(i)]=nested_out[i]['inv_losses']
        store2inv['store2inv'+str(i)]=nested_out[i]['store2inv']
        LevelOfCharge['LevelOfCharge'+str(i)]=nested_out[i]['LevelOfCharge']
        res_pv['res_pv'+str(i)]=nested_out[i]['res_pv']#not important, recalculated later
        inv2grid['inv2grid'+str(i)]=nested_out[i]['inv2grid']#not important, recalculated later
        grid2load['grid2load'+str(i)]=nested_out[i]['grid2load']#not important, recalculated later
        store2load['store2load'+str(i)]=nested_out[i]['store2load']
        store2grid['store2grid'+str(i)]=nested_out[i]['store2grid']
    nested_dict={'pv2inv':pv2inv.sum(axis=1),'pv2store':pv2store.sum(axis=1),'inv2load':inv2load.sum(axis=1),
                 'grid2load':grid2load.sum(axis=1),'store2inv':store2inv.sum(axis=1),'LevelOfCharge':LevelOfCharge.sum(axis=1),
                 'inv2grid':inv2grid.sum(axis=1),'res_pv':res_pv.sum(axis=1),'store2grid':store2grid.sum(axis=1),
                 'store2load':store2load.sum(axis=1),'inv_losses':inv_losses.sum(axis=1),
                 'batt_losses':batt_losses.sum(axis=1),'inv2curt':inv2curt.sum(axis=1)}

    out_comm_res=dispatch_max_sc(nested_dict['inv2grid'],nested_dict['grid2load'],1e10,param_tech_comm)
    out_comm_final={}
    out_comm_final['pv2inv']=nested_dict['pv2inv'] # DC from hh with PV
    out_comm_final['pv2store']=nested_dict['pv2store'] # DC PV 2 batt from hh with PV
    out_comm_final['inv2load']=(nested_dict['inv2load']+out_comm_res['inv2load']) # AC from all hh
    out_comm_final['store2inv']=nested_dict['store2inv'] # DC  from hh with PV and batt
    out_comm_final['LevelOfCharge']=nested_dict['LevelOfCharge'] # kWh  from hh with PV and batt
    out_comm_final['res_pv']=out_comm_res['res_pv'] # DC  from hh with PV
    out_comm_final['inv2grid']=out_comm_res['inv2grid'] # AC from hh with PV
    out_comm_final['grid2load']=out_comm_res['grid2load'] # AC  from all hh
    out_comm_final['inv2curt']=nested_dict['inv2curt'] # DC  from hh with PV
    out_comm_final['batt_losses']=nested_dict['batt_losses']
    out_comm_final['inv_losses']=nested_dict['inv_losses']
    out_comm_final['store2load']=nested_dict['store2load']
    out_comm_final['store2grid']=nested_dict['store2grid']
    out_comm_final['LevelOfCharge']=nested_dict['LevelOfCharge']
    out_comm_final['PV_size_comm']=PV_size_comm
    out_comm_final['df']=df_CH
    out_comm_final['param_tech']=param_tech
    out_comm_final['nested_dict']=nested_dict
    out_comm_final['selection']=selection
    out=print_analysis_prices(df_CH_kW.E_PV*PV_size_comm, df_CH_kW.loc[:,df.iloc[:,0]].sum(axis=1),retail_price_sc,export_price_sc,
                      param_tech_comm,out_comm_final,isCommunity=True)

    dict_to_csv(out,'../Output/test1')
    







if __name__== '__main__':
    main()
