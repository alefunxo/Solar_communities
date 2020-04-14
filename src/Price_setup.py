
# coding: utf-8

# # Community prices definition

# ## Get the PV profiles of the community

# The generation profiles are named using PV_Gamma_Inclination syntax


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Price_definition(prices, PV_penetration,reso):
    '''Takes the prices and PV_penetration as inputs, reads the df of generation in Munich and the PV_size_distribution, and RANDOMLY chooses the sizes for the X houses (RANDOMLY selected) and produces an output at 15 min resolution that can be resampled @ 1h ir resample==True'''
    print('################################################')
    print('Getting prices')
    df=pd.read_csv('Input/DE_gen_15_min_Energy.csv', encoding='utf8', sep=',',engine='python',parse_dates=[0],
                   infer_datetime_format=True,index_col=0)

    df.index = df.index.tz_localize('UTC').tz_convert('CET')
    print(df.head())
    # We have to define the prices for the community based on PV penetration. We can include some variation in azimuth and angle but in general, I guess, only three or four should be ok. The df loaded has 66 combinations.

    # As an example I will take the 66 PV profiles for different azimuths and inclinations, but this input data must be modified once we define the final community

    # The PV size for every house must be defined

    PV_sizes=pd.read_csv('Input/PV_size_distribution.csv',index_col=0,header=None)

    count, division = np.histogram(PV_sizes)
    prob_size=count/PV_sizes.size
    sizes=np.random.choice(np.arange(1, 11),int(np.floor(PV_penetration*74)), p=prob_size)
    # Select the PV profiles randomly and include the size in the name of the columns
    newcols=[]
    selection=np.random.choice(df.columns,int(np.floor(PV_penetration*74)))
    j=0
    df_sel=df.copy()
    for i in selection:
        df_sel.loc[:,i]=df[i]*sizes[j]
        newcols.append(i+'_size_'+str(sizes[j]))
        j+=1

    df_sel=df_sel[selection]
    df_sel.columns=newcols

    aux=df_sel.sum(axis=1)

    step=aux.max()/(prices.size+1)

    Prices=pd.DataFrame(index=df_sel.index)

    for i in range(len(prices)):
        Prices.loc[aux>=step*i,'prices']=prices[i]

    df_sel.to_csv('Input/DE_gen_15_min_Energy_sizes_{}.csv'.format(PV_penetration))
    Prices.to_csv('Input/DE_price_15_min_{}.csv'.format(PV_penetration))

    if reso=='1h':
        df_sel=df_sel.resample('1H').sum()

        Prices=pd.DataFrame(index=df_sel.index)

        for i in range(len(prices)):
            Prices.loc[aux>=step*i,'prices']=prices[i]

        df_sel.to_csv('Input/DE_gen_1h_Energy_sizes_{}.csv'.format(PV_penetration))
        Prices.to_csv('Input/DE_price_1h_{}.csv'.format(PV_penetration))
    print('Prices ok')
    print('################################################')

    return
