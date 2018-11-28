
# coding: utf-8

# # Psycho paper

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt

import Model as M
import numpy as np
import os

# ## Skip up to German demand curves (GHI, T, PV Gen and load need to be run only once)

# # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ## Munich GHI and Temperature 2015
# source: soda-pro.com
# ACHTUNG! GHI is in Wh/m2 we need it in W/m2

# In[3]:
def PV_generation():
    print('##############################')
    print('PV_Gen')
    df=pd.read_csv('C:/Users/alejandro/Documents/GitHub/Psycho//Input/Input_data_PV3.csv',
                 encoding='utf8', sep=';',engine='python',index_col=12,parse_dates=[12],infer_datetime_format=True )

    df.index = df.index.tz_localize('UTC').tz_convert('CET')

    #print(df.keys())
    #df=pd.read_csv('Input/Input_data_PV2.csv',
    #                 encoding='utf8', sep=',',engine='python',index_col=12,
    #                parse_dates=[12],infer_datetime_format=True )
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
#azimuths=[-50,-40,40,50]
#inclinations=[20,25,30,35,40,45]
#phi=48.1351 #Munich
def PV_output_inclinations(azimuths,inclinations,df,res,phi):
    '''Will generate PV outputs for different azimuths and inclinations taking into account the inputs
    df must contain a column called GHI and one Temperature
    it will put all the outputs in a csv file located in a folder called PV_Gen'''
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
            name_file='C:/Users/alejandro/Documents/GitHub/Psycho//PV_Gen/PV_Generation_'+str(gamma)+'_'+str(beta)+'.csv'
            df_out.to_csv(name_file)
            i+=1
    return
def Distribution():
    '''Get the distribution of PV size from Germany for sizes smaller than 10kW'''
    print('##############################')
    print('Distribution')
    df=pd.read_csv('C:/Users/alejandro/Documents/GitHub/Psycho/Input/105_devices_utf8.csv', encoding='utf8', sep=';',engine='python',header=3)

    df_sol=df[df.Anlagentyp=='Solarstrom']
    cap_sol=df_sol['Nennleistung(kWp_el)']

    cap_sol=cap_sol.apply(lambda x: x.replace('.',''))
    cap_sol=cap_sol.apply(lambda x: x.replace(',','.'))
    cap_sol=cap_sol.astype(float)
    res_sol=cap_sol[cap_sol<10]
    res_sol=res_sol.reset_index(drop=True)
    res_sol.to_csv('C:/Users/alejandro/Documents/GitHub/Psycho/Input/PV_size_distribution.csv')
    return()

# # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # From here can be run again

# ## German demand curves (2010)
# see Representative electrical load profiles of residential buildings in Germany with an original temporal resolution of one second Tjaden et al. reshaped to 15 minutes resolution

# In[2]:
def German_load():

    '''Get the yearly and daily average of the german load from DE_load_15_min_Power and put it in a folder called Input as csv'''
    print('##############################')
    print('German_load')

    df_15power=pd.read_csv('C:/Users/alejandro/Dropbox/0. PhD/Python/Paper_psycho/Input/DE_load_15_min_Power.csv',
                       index_col=[0],parse_dates=[0],infer_datetime_format=True )
    df_15power.index=df_15power.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    a=(df_15power.mean(axis=1)/4)
    a.to_csv('C:/Users/alejandro/Documents/GitHub/Psycho/Input/German_yearly_average_load_curve_kWh.csv')
    b=(a.groupby([a.index.hour,a.index.minute]).mean())
    b.to_csv('C:/Users/alejandro/Documents/GitHub/Psycho/Input/German_daily_average_load_curve_kWh.csv')
    return()

    # ## PV generation Munich
def PV_gen_munich():
    '''read the outputs in PV Gen and put them together in a df (normalized @ 1kW) delivered in a csv in the Input folder, called DE_gen_15_min_Energy.csv'''
    print('##############################')
    print('PV_gen_munich')
    path='C:/Users/alejandro/Documents/GitHub/Psycho/PV_gen'
    mat=np.array(['Azimuth','Inclination','PV_output','Capacity_factor'])

    for file in os.listdir(path):
        #We want to have the PV_output, Capacity_factor, Inclination and Azimuth in a table (PV_munich)
        df=pd.read_csv(path+file,
                     encoding='utf8', sep=',',engine='python',parse_dates=[0],infer_datetime_format=True,index_col=0)

        aux=file.split('_')
        arr=np.array([aux[2], aux[3].split('.')[0], (df.sum()/4/230).values[0],(df.sum()/4/230/(365*24)).values[0]])
        mat=np.vstack((mat,arr))

    PV_munich=pd.DataFrame(mat[1:].astype(float).round(2),columns=mat[0])
    PV_munich.sort_values('PV_output',ascending=False)
    result=pd.read_csv(path+os.listdir(path)[0], encoding='utf8', sep=',',
                       engine='python',parse_dates=[0],infer_datetime_format=True,index_col=0)
    result.columns=['PV_'+os.listdir(path)[0].split('_')[2]+'_'+os.listdir(path)[0].split('_')[3].split('.')[0]]
    result.index = result.index.tz_localize('UTC').tz_convert('CET')
    for file in os.listdir(path)[1:]:
        df=pd.read_csv(path+file, encoding='utf8', sep=',',engine='python',parse_dates=[0],infer_datetime_format=True,index_col=0)
        df.columns=['PV_'+file.split('_')[2]+'_'+file.split('_')[3].split('.')[0]]
        df.index = df.index.tz_localize('UTC').tz_convert('CET')
        result = pd.concat([result, df], axis=1, join='inner')

    # ### Normalize to 1 kW array
    result=(result/4/230)
    #result=result.drop('Index')
    #result.index=pd.to_datetime(result.index)
    result.to_csv('C:/Users/alejandro/Documents/GitHub/Psycho/Input/DE_gen_15_min_Energy.csv')
    return()
