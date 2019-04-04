# -*- coding: utf-8 -*-
## @namespace Post_processing
# Created on Wed Feb 28 09:47:22 2018
# Author
# Alejandro Pena-Bello
# alejandro.penabello@unige.ch
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
# Pandas, numpy, itertools,sys,glob,multiprocessing, time
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import paper_classes_2 as pc
import time
import os


# PV distribution script
#
# 1. I have to correct the timestamp of PV which is in UTC while all the other dates are in local time. Double check the source.
#
# 2. After loading the GHI and Temp data we look for missing data and fill those in.
#
# 3. We model PV generation for different inclinations and orientations of the PV array (azimuth from -50 to +50 and inclinations from 20 to 45º, for a total of 66 combinations). The PV generation is normalized for a 1 kW array.
#
# 4. We resample the demand data from 1 minute to 15 minutes resolution.
#
#

# Price setup script
#
# 1. We define the PV penetration rate (50%). We define the prices to be used [0.07,.10,.13,.16,.19,.22,.25,.28].
#
# 2. We choose PV profiles and sizes randomly (taking into account the German PV size distribution).
#
# 3. We assign the prices according to the PV profile (low prices when high PV generation and high prices in the other case).
#
#

# Community script
#
# 1. We select battery penetration in the community.
# 2. We resample to 1h resolution.
# 3. We define the battery to be a 10 kWh
# 4. Create the community, it consists of 74 households, 37 of which have PV (on which the prices are based and can be found in Price_setup.ipynb) and 18 (floor of 25%) will have a battery. Match PV with demand, then among those houses choose which ones will have a battery and then for the latter run the simulation. This is done **randomly**.
# 5. Once the community is ready we enter in a for and calculate for every household the different energy flows. This include the consumer choices when surplus and without surplus (we have to update the numbers only).

# Post_processing script
#
#
# 1. We load the community results with consumer behaviour as well as the results when there is no community at all (i.e. individual households). The results when consumer behaviour is not included are the same as without community but we assume that they are now in a pool.
#
# 2. We calculate the local PV self-consumption including direct and indirect (through the battery).
#
# 3. We calculate Autarky (the proportion of electricity a property consumes, that can be produced independent of the grid).

# # ACHTUNG!
#
# This is in energy terms we cannot give the power terms since we are working on 1h resolution! (P=E/t)
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)
def graphs_gral(aux_comm,pdf):
    plt.figure(figsize=(10,10))
    aux_comm.gen.plot(label='Community generation')
    aux_comm.demand.plot(label='Community demand')
    plt.xlabel('Time',size=18)
    plt.ylabel('kWh',size=18)
    plt.legend()
    pdf.savefig()

    return

def community_ind(aux_no_comm,aux_comm):

    Direct_in_house_comm=(aux_comm.PV_load.sum())/aux_comm.gen.sum()#PVSC direct
    Indirect_in_house_comm=(aux_comm.PV_batt.sum())/aux_comm.gen.sum()#combined Batt_load+Batt_comm
    Total_out_house_comm=((aux_comm.comm_demand-aux_comm.Batt_comm).sum())/aux_comm.gen.sum()
    # comm_demand=aux_no_comm[['grid_load','PV_grid']].min(axis=1) is the smallest between grid and PV export
    Total_out_grid_comm=(aux_comm.comm_grid.sum())/aux_comm.gen.sum()
    Total_losses_comm=(aux_comm.PV_losses.sum())/aux_comm.gen.sum()
    (aux_no_comm.PV_load.sum())/aux_no_comm.gen.sum()
    (aux_no_comm.PV_batt.sum())/aux_no_comm.gen.sum()
    Direct_in_house_comm+Indirect_in_house_comm+Total_out_house_comm+Total_out_grid_comm+Total_losses_comm


    Direct_in_house=(aux_no_comm.PV_load.sum())/aux_no_comm.gen.sum()#DSC
    Indirect_in_house=(aux_no_comm.PV_batt.sum())/aux_no_comm.gen.sum()#PV_batt=Batt_load+Batt_losses
    Total_out_house=(aux_no_comm.comm_demand.sum())/aux_no_comm.gen.sum()
    Total_out_grid=(aux_no_comm.comm_grid.sum())/aux_no_comm.gen.sum()
    Total_losses=(aux_no_comm.PV_losses.sum())/aux_no_comm.gen.sum()

    Direct_in_house_comm=(aux_comm.PV_load.sum())/aux_comm.gen.sum()
    Indirect_in_house_comm=(aux_comm.PV_batt.sum())/aux_comm.gen.sum()#combined Batt_load+Batt_comm
    Total_out_house_comm=((aux_comm.comm_demand-aux_comm.Batt_comm).sum())/aux_comm.gen.sum()
    Total_out_grid_comm=(aux_comm.comm_grid.sum())/aux_comm.gen.sum()
    Total_losses_comm=(aux_comm.PV_losses.sum())/aux_comm.gen.sum()
    data_no_comm = [Direct_in_house,Indirect_in_house,Total_out_house,Total_out_grid,Total_losses]
    data_comm = [Direct_in_house_comm,Indirect_in_house_comm,Total_out_house_comm,Total_out_grid_comm,Total_losses_comm]

    return[data_no_comm,data_comm]
def graph_prices(df,pdf):
    plt.subplots(1,1,figsize=(10, 10))#, subplot_kw=dict(aspect="equal"))
    axes=df.groupby(df.loc[:,'date'].dt.hour).prices.mean().plot()
    axes.set_xlabel('Hour of the day')
    plt.suptitle('Mean prices throughout the year')
    #df[df.df==0].prices.plot()
    #plt.tight_layout()
    pdf.savefig()
    return


def graph_autarky(aux_no_comm,aux_comm,pdf):
    grid=(aux_no_comm.grid_load.sum())/aux_no_comm.demand.sum()
    self_PV=(aux_no_comm.PV_load.sum()+aux_no_comm.Batt_load.sum())/aux_no_comm.demand.sum()
    grid_comm=(aux_comm.grid_demand.sum())/aux_comm.demand.sum()
    self_PV_comm=(aux_comm.PV_load.sum()+aux_comm.Batt_load.sum()+aux_comm.comm_demand.sum())/aux_comm.demand.sum()
    grid_nb=(aux_no_comm.grid_demand.sum())/aux_no_comm.demand.sum()
    self_PV_nb=(aux_no_comm.PV_load.sum()+aux_no_comm.Batt_load.sum()+aux_no_comm.comm_demand.sum())/aux_no_comm.demand.sum()

    fig, axs = plt.subplots(1,3,figsize=(10, 10), subplot_kw=dict(aspect="equal"))
    data = [grid,self_PV]
    data2 = [grid_comm,self_PV_comm]
    data3 = [grid_nb,self_PV_nb]
    labels = ['Electricity imported from the grid','Electricity from the community']

    wedges, texts, autotexts = axs[0].pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))
    axs[2].legend(wedges, labels,
              title="Autarky",
              loc="center left",
              bbox_to_anchor=(.5, -.7, 0.5, 1))
    plt.setp(autotexts, size=14, weight="bold")
    wedges, texts, autotexts = axs[1].pie(data2, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))
    plt.setp(autotexts, size=14, weight="bold")
    wedges, texts, autotexts = axs[2].pie(data3, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    axs[0].set_title("No community",size=18)

    axs[1].set_title("Community \nwith user behaviour",size=18)
    axs[2].set_title("Community \nwithout user behaviour",size=18)
    plt.setp(autotexts, size=14, weight="bold")
    plt.tight_layout()
    pdf.savefig()
    return
    # ## If we do the same calculations taking a community approach but when the households cannot sell from the battery or choose to sell instead of charge their battery:

def graph_sc(data,data2,pdf):

    fig, axs = plt.subplots(1,2,figsize=(10, 10), subplot_kw=dict(aspect="equal"))

    labels = ['Direct SC','Battery discharge','Consumed inside the community','Exported to grid','Total_losses']


    wedges, texts, autotexts = axs[1].pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))
    axs[0].legend(wedges, labels,
              title="Self-Consumption",
              loc="center right",
              bbox_to_anchor=(.5, -.7, 0.5, 1))
    plt.setp(autotexts, size=14, weight="bold")
    wedges, texts, autotexts = axs[0].pie(data2, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))
    axs[1].set_title("Self-consumption community\n no user behaviour",size=18)

    axs[0].set_title("Self-consumption community\nwith user behaviour",size=18)
    plt.setp(autotexts, size=14, weight="bold")
    plt.tight_layout()
    pdf.savefig()
    # # Attention!
    # through battery includes the amount of electricity used from PV to charge the battery and later can be used to go to the community or to the house, moreover, it includes the losses inherent to the battery.
    # Comm_load includes only PV not battery to community
    return
    # ## Here the important messages are:
    # ### 1. Reduction of duck-head
    # ### 2. Reduction of injection to the grid (3.6%)
    # ### 3. Slight increase of losses (reduction of PV but increase due to battery. In total, increase)

def graph_average(aux_no_comm,aux_comm,pdf):
    # # Now in average for the whole year, we have:
    aux_comm_mean=aux_comm.groupby(aux_comm.index.hour).mean()
    aux_no_comm_mean=aux_no_comm.groupby(aux_no_comm.index.hour).mean()
    plt.figure(figsize=(10,10))
    (aux_comm_mean.gen).plot(label='Community generation')
    (aux_comm_mean.demand).plot(label='Original community demand')
    
    (aux_comm_mean.comm_demand).plot(label='Exported from the community to \n cover community demand')
    (aux_comm_mean.comm_grid).plot(label='Exported from the community \n to the grid')
    (aux_comm_mean.grid_demand).plot(label='Imported from the grid \n to the community (new demand)')
    #(aux_no_comm_mean.grid_demand).plot(label='grid to community no user ')
    #plt.ylim(-1,100)
    plt.title('',size=18)
    plt.xlabel('Time',size=18)
    plt.ylabel('kWh',size=18)
    plt.legend(loc=2)
    plt.title('Average values per year')
    plt.tight_layout()
    pdf.savefig()
    plt.figure(figsize=(10,10))
    (aux_comm_mean.demand).plot(label='Original community demand')
    (aux_comm_mean.demand-aux_comm_mean.grid_demand).plot(label='Original community residual demand')
    (aux_comm_mean.grid_demand).plot(label='Imported from the grid \n to the community (new demand)')
    (aux_no_comm_mean.grid_demand).plot(label='Imported from the grid \n to the community (new demand) \n without user behaviour')
    (aux_comm_mean.grid_demand*0).plot(label='zero')
    #plt.title('Comparison SOC between community with and w/o user behaviour.',size=18)
    #plt.ylim(-1,100)
    plt.xlabel('Time',size=18)
    plt.ylabel('kWh',size=18)
    plt.tight_layout()
    plt.legend(loc=2)
    pdf.savefig()
    plt.figure(figsize=(10,10))
    (aux_comm_mean.gen).plot(label='Community generation')
    (aux_comm_mean.comm_grid).plot(label='Exported from the community to the grid')
    (aux_no_comm_mean.comm_grid).plot(label='Exported from \n the community to the grid \n without user behaviour')
    (aux_comm_mean.grid_demand*0).plot(label='zero')
    plt.title('Comparison SOC between community with and w/o user behaviour.',size=18)
    plt.xlabel('Time',size=18)
    plt.ylabel('kWh',size=18)
    plt.legend(loc=2)
    plt.figure(figsize=(10,10))

    (aux_comm_mean.SOC).plot(label='SOC with behaviour')
    (aux_no_comm_mean.SOC).plot(label='SOC without behaviour')

    plt.title('Comparison SOC between community with and w/o user behaviour.',size=18)
    #(aux_comm_mean.grid_demand*0).plot(label='zero')
    plt.xlabel('Time',size=18)
    plt.ylabel('SOC',size=18)
    plt.legend(loc=2)
    pdf.savefig()
    return


def Post_processing(Batt_penetration,PV_penetration,print_,path,probs_applied):

    df_no_comm=pd.read_csv(path+'Output/no_community_{}_{}_{}.csv'.format(PV_penetration,Batt_penetration,probs_applied),encoding='utf8', sep=',',header=0,
        engine='python',date_parser=lambda col: pd.to_datetime(col, utc=True), parse_dates=[1],infer_datetime_format=True,index_col=0,
    names=['index','date','demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load','Batt_grid','grid_load','PV_losses','Batt_losses','flag','type','df','prices'])
    df_no_comm.loc[:,'date'] =pd.DatetimeIndex(df_no_comm.loc[:,'date']).tz_convert('CET')
    #df_no_comm.loc[:,'date'].dt.hour

    df_comm=pd.read_csv(path+'Output/community_{}_{}_{}.csv'.format(PV_penetration,Batt_penetration,probs_applied),encoding='utf8', sep=',',header=0,
        engine='python',date_parser=lambda col: pd.to_datetime(col, utc=True), parse_dates=[1],infer_datetime_format=True,index_col=0,
                        names=['index','date','demand','gen','SOC','PV_batt', 'PV_load', 'PV_comm', 'E_dis','Batt_load','Batt_comm','comm_load','PV_losses','Batt_losses','flag','type','df','prices'])

    df_comm.loc[:,'date'] =pd.DatetimeIndex(df_comm.loc[:,'date']).tz_convert('CET')

    df_no_comm['bill']=df_no_comm.prices*df_no_comm.grid_load-df_no_comm.PV_grid*df_no_comm.prices
    df_comm['bill']=df_comm.prices*df_comm.comm_load-(df_comm.PV_comm+df_comm.Batt_comm)*df_comm.prices
    # We can calculate the import and export to the grid. First we sum every profile in the "microgrid" to get the totals.

    aux_comm=df_comm.groupby(df_comm.date).sum()
    aux_no_comm=df_no_comm.groupby(df_no_comm.date).sum()


    # ### Local PV includes instantaneous consumption (from PV to load) and deferred consumption
    # ### (PV to Batt to load).

    aux_comm['export_comm']=aux_comm['PV_comm']+aux_comm['Batt_comm']
    aux_comm['comm_demand']=aux_comm[['comm_load','export_comm']].min(axis=1)
    aux_comm['comm_grid']=aux_comm['export_comm']-aux_comm['comm_demand']
    aux_comm['grid_demand']=aux_comm['comm_load']-aux_comm['comm_demand']

    # # Energy Balance
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Energy Balance:')
    print('Demand Balance:')
    print((aux_comm.demand-aux_comm.Batt_load-aux_comm.PV_load-aux_comm.comm_demand-aux_comm.grid_demand).sum())
    print('Generation Balance:')
    print((aux_comm.gen-aux_comm.PV_batt-aux_comm.PV_load-aux_comm.PV_comm-aux_comm.PV_losses).sum())
    print('Battery Balance:')
    print((aux_comm.PV_batt-aux_comm.Batt_load-aux_comm.Batt_comm-aux_comm.Batt_losses).sum())
    print(aux_comm.tail())
    print('Community Balance:')
    print(((aux_comm.Batt_comm+aux_comm.PV_comm)-aux_comm.comm_demand-aux_comm.comm_grid).sum())

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    aux_no_comm['comm_demand']=aux_no_comm[['grid_load','PV_grid']].min(axis=1)
    aux_no_comm['comm_grid']=aux_no_comm['PV_grid']-aux_no_comm['comm_demand']
    aux_no_comm['grid_demand']=aux_no_comm['grid_load']-aux_no_comm['comm_demand']
    print(aux_comm.tail())
    aux_comm_bill=df_comm.groupby(df_comm.df).sum()
    aux_no_comm_bill=df_no_comm.groupby(df_no_comm.df).sum()
    #print(aux_comm_bill)
    #print(aux_no_comm_bill)
    aux_no_comm_bill.to_csv(path+'Input/bill.csv')
    aux_comm_bill.to_csv(path+'Input/bill_comm.csv')

    df=pd.DataFrame(np.array([aux_comm_bill.type,aux_comm_bill.bill,aux_no_comm_bill.bill]).T)


    print('amount of times energy when surplus was exchanged was: {}'.format(df_comm[(df_comm.flag==4)].flag.sum()/4))
    print('amount of times energy when no surplus was exchanged was: {}'.format(df_comm[(df_comm.flag==1)].flag.sum()))

    [data_no_comm,data_comm]=community_ind(aux_no_comm,aux_comm)
    if print_:
        file=path+'Output/Output_{}_{}_{}_train.pdf'.format(PV_penetration*100,Batt_penetration*100,probs_applied)
        print(file)
        with  PdfPages(file) as pdf:
            
            graphs_gral(aux_comm,pdf)
            graph_average(aux_no_comm,aux_comm,pdf)
            graph_autarky(aux_no_comm,aux_comm,pdf)
            graph_sc(data_no_comm,data_comm,pdf)
            graph_prices(df_comm,pdf)
            df.columns=['types','bill_comm','bill']

            plt.figure(figsize=(10,10))
            labels = ['No PV No Battery','PV Only','PV and Battery']
            axes=df.reset_index().boxplot(column='bill_comm', by='types')
            axes.set_xticklabels(labels)
            
            plt.title('With community')
            pdf.savefig()
            
            plt.figure(figsize=(10,10))
            labels = ['No PV No Battery','PV Only','PV and Battery']
            axes=df.reset_index().boxplot(column='bill', by='types')
            axes.set_xticklabels(labels)
            plt.suptitle('Without community')
            
            pdf.savefig()
            d = pdf.infodict()
            plt.close('all')
