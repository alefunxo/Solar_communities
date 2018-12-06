# -*- coding: utf-8 -*-
## @namespace Germany_PVdistribution
# Created on Wed Feb 28 09:47:22 2018
# Author
# Alejandro Pena-Bello
# alejandro.penabello@unige.ch
# Script developed for the project developed together with the Consumer Decision and Sustainable Behavior Lab to include the user preferences in the charging and discharging of the battery.
# The script has been tested in Linux and Windows
# This script includes five functions
# TODO
# ----
# User Interface
# Requirements
# ------------
# Pandas, numpy, itertools,sys,glob,multiprocessing, time
import pandas as pd
import matplotlib.pyplot as plt
import Model as M
import numpy as np
import os
import sys

# ## Munich GHI and Temperature 2015
# source: soda-pro.com

def PV_generation(path):
    '''
    Description
    -----------
    This function get input data from Input_data_PV3.csv and fill missing data if needed and filter for 2015.

    Parameters
    ----------
    path : string ; Is the path where all the input data will be stored.

    Returns
    ------
    df: DataFrame; Dataframe includes temperature and GHI with timestamp

    TODO
    ------
    Do it more general
    '''
    print('##############################')
    print('PV_Gen')
    df=pd.read_csv(path+'Input/Input_data_PV3.csv',
                 encoding='utf8', sep=';',engine='python',index_col=12,parse_dates=[12],infer_datetime_format=True )

    df.index = df.index.tz_localize('UTC').tz_convert('CET')

    df=df[df.index.year==2015]
    df.GHI=df.GHI/.25

    # ## Filling missing data
    if df.Temperature.isnull().sum():
        df['Temperature']=df['Temperature']-273.15#To °C
        df['Date']=df.index.date
        df['TimeOnly']=df.index.time
        test_pivot=df.pivot_table(values='Temperature', columns='TimeOnly', index='Date')
        test_filled=test_pivot.fillna(method='ffill')
        test_long = test_filled.stack()
        test_long.name='aux'
        test_long = test_long.reset_index()
        test_long['Time'] = test_long.apply(lambda r : pd.datetime.combine(r['Date'], r['TimeOnly']), axis='columns')
        test_long = test_long[['Time', 'aux']]
        test_long=test_long.sort_values(['Time'])
        #test_long=test_long.set_index('Time')
        test_long.index=df.index
        df['Temperature']=test_long['aux']

    if df.GHI.isnull().sum():

        df['Date']=df.index.date
        df['TimeOnly']=df.index.time
        test_pivot=df.pivot_table(values='GHI', columns='TimeOnly', index='Date')
        test_filled=test_pivot.fillna(method='ffill')
        test_long = test_filled.stack()
        test_long.name='aux'
        test_long = test_long.reset_index()
        test_long['Time'] = test_long.apply(lambda r : pd.datetime.combine(r['Date'], r['TimeOnly']), axis='columns')
        test_long = test_long[['Time', 'aux']]
        test_long=test_long.sort_values(['Time'])
        test_long.index=df.index
        df['GHI']=test_long['aux']
    return (df)

def PV_output_inclinations(azimuths,inclinations,df,res,phi):
    '''
    Description
    -----------
    This function will generate PV outputs for different azimuths and inclinations taking into account the inputs. df must contain a column called GHI and one Temperature it will put all the outputs in a csv file located in a folder called PV_Gen. The name follows this nomenclature: PV_Generation_Gamma_Beta.csv where beta is inclination and gamma is azimuth

    Parameters
    ----------
    df: DataFrame; includes Temperature and GHI
    phi: float; latitude where the panel will be installed
    res: float; temporal resolution
    inclinations: numpy array; inclination
    azimuths: numpy array; azimuth

    Returns
    ------

    TODO
    ------
    '''
    print('##############################')
    print('PV_output_inclinations')
    i=0
    for gamma in azimuths:
        for beta in inclinations:
            print(i)
            out=M.inputs(beta=beta,gamma=gamma,df=df,phi=phi,res=15)
            print(out)
            df_out=pd.DataFrame(out)
            df_out=df_out.set_index(df.index)
            name_file=path+'PV_Gen/PV_Generation_'+str(gamma)+'_'+str(beta)+'.csv'
            df_out.to_csv(name_file)
            i+=1
    return
def Distribution(path):
    '''
    Description
    -----------
    This function gets the distribution of PV size from Germany for sizes smaller than 10kW. The PV size distribution is saved in the Input folder under the name PV_size_distribution.csv


    Parameters
    ----------
    path : string ; Is the path where all the input data is stored.

    Returns
    ------

    TODO
    ------
    Do it more general and for other countries

    '''
    print('##############################')
    print('Distribution')
    df=pd.read_csv(path+'Input/105_devices_utf8.csv', encoding='utf8', sep=';',engine='python',header=3)

    df_sol=df[df.Anlagentyp=='Solarstrom']
    cap_sol=df_sol['Nennleistung(kWp_el)']

    cap_sol=cap_sol.apply(lambda x: x.replace('.',''))
    cap_sol=cap_sol.apply(lambda x: x.replace(',','.'))
    cap_sol=cap_sol.astype(float)
    res_sol=cap_sol[cap_sol<10]
    res_sol=res_sol.reset_index(drop=True)
    res_sol.to_csv(path+'Input/PV_size_distribution.csv')
    return()

def German_load(path):

    '''
    Description
    -----------
    This function gets the yearly and daily average of the german load from DE_load_15_min_Power and put it in a folder called Input as csv. For this it reads a the file from DE_load_15_min_Power.csv which comes from the paper Representative electrical load profiles of residential buildings in Germany with an original temporal resolution of one second Tjaden et al. reshaped to 15 minutes resolution. German demand curves (2010)

    Parameters
    ----------
    path : string ; Is the path where all the input data is stored.

    Returns
    ------

    TODO
    ------
    Do it more general and for other countries
    '''
    print('##############################')
    print('German_load')

    df_15power=pd.read_csv('C:/Users/alejandro/Dropbox/0. PhD/Python/Paper_psycho/Input/DE_load_15_min_Power.csv',
                       index_col=[0],parse_dates=[0],infer_datetime_format=True )
    df_15power.index=df_15power.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    a=(df_15power.mean(axis=1)/4)
    a.to_csv(path+'Input/German_yearly_average_load_curve_kWh.csv')
    b=(a.groupby([a.index.hour,a.index.minute]).mean())
    b.to_csv(path+'Input/German_daily_average_load_curve_kWh.csv')
    return()

def PV_gen_munich(path):
    '''
    Description
    -----------
    This function reads the outputs in PV Gen and put them together in a df (normalized @ 1kW) delivered in a csv in the Input folder, called DE_gen_15_min_Energy.csv

    Parameters
    ----------
    path : string ; Is the path where all the input data is stored.

    Returns
    ------

    TODO
    ------
    Do it more general and for other countries
    '''
    print('##############################')
    print('PV_gen_munich')
    path2=path+'PV_gen/'
    mat=np.array(['Azimuth','Inclination','PV_output','Capacity_factor'])

    for file in os.listdir(path2):

        #We want to have the PV_output, Capacity_factor, Inclination and Azimuth in a table (PV_munich)
        df=pd.read_csv(path2+file,
                     encoding='utf8', sep=',',engine='python',parse_dates=[0],infer_datetime_format=True,index_col=0)

        aux=file.split('_')
        arr=np.array([aux[2], aux[3].split('.')[0], (df.sum()/4/230).values[0],(df.sum()/4/230/(365*24)).values[0]])
        mat=np.vstack((mat,arr))

    PV_munich=pd.DataFrame(mat[1:].astype(float).round(2),columns=mat[0])
    PV_munich.sort_values('PV_output',ascending=False)

    result=pd.read_csv(path2+os.listdir(path2)[0], encoding='utf8', sep=',',
                       engine='python',parse_dates=[0],infer_datetime_format=True,index_col=0)
    result.columns=['PV_'+os.listdir(path2)[0].split('_')[2]+'_'+os.listdir(path2)[0].split('_')[3].split('.')[0]]
    result.index = result.index.tz_localize('UTC').tz_convert('CET')
    for file in os.listdir(path2)[1:]:
        df=pd.read_csv(path2+file, encoding='utf8', sep=',',engine='python',parse_dates=[0],infer_datetime_format=True,index_col=0)
        df.columns=['PV_'+file.split('_')[2]+'_'+file.split('_')[3].split('.')[0]]
        df.index = df.index.tz_localize('UTC').tz_convert('CET')
        result = pd.concat([result, df], axis=1, join='inner')

    # ### Normalize to 1 kW array
    result=(result/4/230)
    #result=result.drop('Index')
    #result.index=pd.to_datetime(result.index)
    result.to_csv(path+'Input/DE_gen_15_min_Energy.csv')
    return()
