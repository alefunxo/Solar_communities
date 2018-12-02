import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import paper_classes_2 as pc
import time

# # Choose Battery penetration in the community

# <img src='Psycho_flowchart.jpg'>

# # DC-coupled PV-Battery system with integrated inverter used in this study.

# <img src='DC-coupled system (2).jpg'>

# Be careful! Prices depend on PV penetration if PV penetration is adjusted, prices must be adjusted as well.


# create the df test. We need demand, generation, prices, SOC, PV_batt, PV_load, PV_grid, Batt_load, Batt_grid, grid_load.

def find_interval(x, partition):
    """ find_interval -> i
        partition is a sequence of numerical values
        x is a numerical value
        The return value "i" will be the index for which applies
        partition[i] < x < partition[i+1], if such an index exists.
        -1 otherwise
    """

    for i in range(0, len(partition)):
        if x < partition[i]:
            return i-1
    return -1

def sell_prob(SOC,price,Capacity,case,path):
    #define the matrix of probs
    if case=='PV':
        prob_mat=pd.read_table(path+'Input/P_selling_by_price_and_autarky2.txt',sep='\t',index_col=[0])
    else:#when selling from battery
        prob_mat=pd.read_table(path+'Input/P_selling_by_price_and_autarky_2.txt',sep='\t',index_col=[0])
    if SOC==0:
        out=0
    else:

        I=np.array([0,0.05,0.2,0.35,0.5,0.65,0.8,0.95,1.0001])#define the interval of SOC
        ind=find_interval(SOC,I*Capacity)#find the interval of SOC
        if ind==-1:
            prob=prob_mat.loc[1,str(price)]
        else:
            prob=prob_mat.loc[I[ind],str(price)]
        out=np.random.choice(2,1,p=[1-prob,prob])
    return out #either 1 or 0 if sell or not

def PV_SC(df,Batt,Conv_eff,Inv_eff,endo):
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

# # If the client agreees (Probability function) we discharge 1 kWh of the battery

def PV_SC_probs2(df,Batt,Conv_eff,Inv_eff,endo,path):
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
                        if (sell_prob(df.SOC[i-1]-aux,df.prices[i],Batt.Capacity,'batt',path)&(df.index.hour[i]<12)):
                            #Sell from the battery to the community
                            #Is this Working???
                            #print('######UUUUU#####')
                            df.flag[i]=1#Sell from battery
                            df.Batt_load[i]=aux*Inv_eff#after inverter
                            df.E_dis[i]=min(1,df.SOC[i-1]-Batt.SOC_min-aux)#the client agree to sell 1Kwh from batt
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
                if sell_prob(df.SOC[i-1],df.prices[i],Batt.Capacity,'PV',path):
                    df.flag[i]=2
                    if df.SOC[i-1]<Batt.SOC_max-0.0001:#then if battery not full, charge
                        if (df.gen[i]*Conv_eff-df.demand[i]/Inv_eff)>1:
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

def Price_definition(prices, PV_penetration,Batt_penetration,reso,path,day_sel):
    '''Takes the prices and PV_penetration as inputs, reads the df of generation in Munich and the PV_size_distribution, and RANDOMLY chooses the sizes for the X houses (RANDOMLY selected) and produces an output at 15 min resolution that can be resampled @ 1h ir resample==True'''

    print('################################################')
    print('Getting prices')
    df_gen_comb=pd.read_csv(path+'Input/DE_gen_15_min_Energy.csv', encoding='utf8', sep=',',engine='python',parse_dates=[0],
                   infer_datetime_format=True,index_col=0)

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

    # We have to decide if the choice is made randomly (it will change every time we rerun the script) or not.

    #We decide among the whole community which houses will have PV and PV and Batt

    selection_PV=np.random.choice(df_demand.columns,int(np.floor(df_demand.shape[1]*PV_penetration)), replace=False)
    selection_PV_Batt=np.random.choice(selection_PV,int(np.floor(df_demand.shape[1]*PV_penetration*Batt_penetration)), replace=False)

    # We have to define the prices for the community based on PV penetration. We can include some variation in azimuth and angle but in general, I guess, only three or four should be ok. The df loaded has 66 combinations.

    # As an example I will take the 66 PV profiles for different azimuths and inclinations, but this input data must be modified once we define the final community

    # The PV size for every house must be defined. If the household have a Battery the PV size comes from another distribution

    PV_sizes=pd.read_csv(path+'Input/PV_size_distribution.csv',index_col=0,header=None)
    PV_sizes_batt=pd.read_csv(path+'Input/PV_size_distribution.csv',index_col=0,header=None)
    #The output of this function: should be a combined df with
    #df_prices,df_generation,df_demand
    count_batt, division_batt = np.histogram(PV_sizes_batt)
    prob_size_batt=count_batt/PV_sizes_batt.size
    sizes_batt=np.random.choice(np.arange(1, 11),int(np.floor(PV_penetration*Batt_penetration*74)), p=prob_size_batt)
    # Select the PV profiles randomly and include the size in the name of the columns
    newcols_batt=[]
    #We select from all the combinations of azimuth and inclination randomly
    selection_batt=np.random.choice(df_gen_comb.columns,int(np.floor(PV_penetration*Batt_penetration*74)))
    count, division = np.histogram(PV_sizes)
    prob_size=count/PV_sizes.size
    sizes=np.random.choice(np.arange(1, 11),int(np.floor(PV_penetration*74)), p=prob_size)
    # Select the PV profiles randomly and include the size in the name of the columns
    newcols=[]
    #We select from all the combinations of azimuth and inclination randomly
    selection=np.random.choice(df_gen_comb.columns,len(selection_PV)-len(selection_batt))
    j=0
    k=0
    m=0
    df_sel=pd.DataFrame()
    df_sel_batt=pd.DataFrame()
    df_com=pd.DataFrame()
    for i in df_demand.columns:
        if i in selection_PV_Batt:
            if j==0:
                df_sel_batt=df_gen_comb[selection_batt[j]]*sizes_batt[j]
            else:
                df_sel_batt=pd.concat([df_sel_batt, df_gen_comb[selection_batt[j]]*sizes_batt[j]], axis=1, join='inner')
            if m==0:
                df_com=df_demand.loc[:,i]-df_gen_comb[selection_batt[j]]*sizes_batt[j]
            else:
                df_com=pd.concat([df_com, df_demand.loc[:,i]-df_gen_comb[selection_batt[j]]*sizes_batt[j]], axis=1, join='inner')
            newcols_batt.append('hh_'+str(i)+'_'+selection_batt[j]+'_size_'+str(sizes_batt[j]))

            j+=1
        elif i in selection_PV:

            if k==0:
                df_sel=df_gen_comb[selection[k]]*sizes[k]
            else:
                df_sel=pd.concat([df_sel, df_gen_comb[selection[k]]*sizes[k]], axis=1, join='inner')
            if m==0:
                df_com=df_demand.loc[:,i]-df_gen_comb[selection[k]]*sizes[k]
            else:
                df_com=pd.concat([df_com, df_demand.loc[:,i]-df_gen_comb[selection[k]]*sizes[k]], axis=1, join='inner')
            newcols.append('hh_'+str(i)+'_'+selection[k]+'_size_'+str(sizes[k]))

            k+=1
        else:
            if m==0:
                df_com=df_demand.loc[:,i]
            else:
                df_com=pd.concat([df_com, df_demand.loc[:,i]], axis=1, join='inner')
        m+=1

    df_sel.columns=newcols
    df_sel_batt.columns=newcols_batt
    if Batt_penetration==1:
        df_sel=df_sel_batt
    else:
        df_sel=pd.concat([df_sel, df_sel_batt], axis=1, join='inner')


    df_com.columns=df_demand.columns
    #The matching from PV-demand is done first.
    aux=df_com.sum(axis=1)

    step=aux.max()/(prices.size+1)


    aux[aux<0]=0


    Prices=pd.DataFrame(index=df_sel.index)

    for i in range(len(prices)):
        Prices.loc[aux>=step*i,'prices']=prices[i]
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
        Prices[Prices.index.hour<23]=.16
        Prices[Prices.index.hour<10]=.22

        Prices[Prices.index.hour>17]=.28

        df_demand=df_demand[(df_demand.index.month==6)&(df_demand.index.day==1)]
        df_sel=df_sel[(df_sel.index.month==6)&(df_sel.index.day==1)]
        Prices=Prices[(Prices.index.month==6)&(Prices.index.day==1)]
        print(Prices)
    return [df_demand,df_sel,Prices,selection_PV,selection_PV_Batt]
def community_psycho(Batt_penetration,PV_penetration,reso,path,day_sel):
    '''

        'Output/community_{}_{}.csv'.format(PV_penetration,Batt_penetration))
        'Output/no_community_{}_{}.csv'.format(PV_penetration,Batt_penetration))
    '''
    if (day_sel=='winter') or (day_sel=='summer'):
        endo=96
    else:
        endo=8760
    Conv_eff=0.98
    Inv_eff=0.95
    prices=np.array([0.07,.10,.13,.16,.19,.22,.25,.28])
    #prices=np.flip(prices)
    Batt=pc.Battery(10,'test')
    [df_demand,df_generation,df_prices,selection_PV,selection_PV_Batt]=Price_definition(prices, PV_penetration,Batt_penetration,reso,path,day_sel)

    print('################################################')
    print('Simulation begins')
    # In order to have a comparison point we pool the houses and operating individually the battery (not taking into account the community approach)
    print('################################################')
    print('Simulating without community exchange')
    df_no_comm=pd.DataFrame()
    j=0
    for i in df_demand.columns:
        #print(i)
        col=[]
        if i in selection_PV:
            col=[s for s in list(df_generation) if 'hh_'+str(i)+'_' in s]

            stack=np.hstack((np.array(np.zeros([len(df_demand.index),12])),np.reshape(np.tile(i,df_demand.shape[0]),(len(df_demand.index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(df_demand.index)
            df1=pd.concat([df_demand.loc[:,i],df_generation.loc[:,col]
                          ,aux,df_prices[:]],axis=1)

            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load',
                    'Batt_grid', 'grid_load','PV_losses','Batt_losses','flag','type','df','prices']
            df1.SOC[0]=5
            if i in selection_PV_Batt:# PV and Batt
                df_no_prob=PV_SC(df1,Batt,Conv_eff,Inv_eff,endo)
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
                df_no_prob.loc[:,'PV_grid']=(df1.gen)*Inv_eff-df1[['demand','gen']].min(axis=1)
                df_no_prob.loc[:,'PV_losses']=df1.gen*(1-Inv_eff)
                df_no_prob.loc[:,'grid_load']=df_no_prob.loc[:,'demand']-df_no_prob.loc[:,'PV_load']
                df_no_prob=df_no_prob.reset_index()
                if i=='0':
                    df_no_comm=df_no_prob.copy()

                else:
                    df_no_comm=df_no_comm.append(df_no_prob, ignore_index=True)
            j+=1
        else:
            stack=np.hstack((np.array(np.zeros([len(df_demand.index),13])),np.reshape(np.tile(i,df_demand.shape[0]),(len(df_demand.index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(df_demand.index)
            df1=pd.concat([df_demand.loc[:,i],aux,df_prices[:]],axis=1)
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
    df_comm=pd.DataFrame()
    j=0
    for i in df_demand.columns:
        #print(i)
        col=[]
        if i in selection_PV:
            col=[s for s in list(df_generation) if 'hh_'+str(i)+'_' in s]

            stack=np.hstack((np.array(np.zeros([len(df_demand.index),12])),np.reshape(np.tile(i,df_demand.shape[0]),(len(df_demand.index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(df_demand.index)
            df1=pd.concat([df_demand.loc[:,i],df_generation.loc[:,col]
                          ,aux,df_prices[:]],axis=1)

            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load',
                    'Batt_grid', 'grid_load','PV_losses','Batt_losses','flag','type','df','prices']
            df1.SOC[0]=5
            if i in selection_PV_Batt:# PV and Batt
                df_prob=PV_SC_probs2(df1,Batt,Conv_eff,Inv_eff,endo,path)
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
                df_prob.loc[:,'PV_grid']=(df1.gen)*Inv_eff-df1[['demand','gen']].min(axis=1)
                df_prob.loc[:,'PV_losses']=df1.gen*(1-Inv_eff)
                df_prob.loc[:,'grid_load']=df_prob.loc[:,'demand']-df_prob.loc[:,'PV_load']
                df_prob=df_prob.reset_index()
                if i=='0':
                    df_comm=df_prob.copy()

                else:
                    df_comm=df_comm.append(df_prob, ignore_index=True)
            j+=1
        else:

            stack=np.hstack((np.array(np.zeros([len(df_demand.index),13])),np.reshape(np.tile(i,df_demand.shape[0]),(len(df_demand.index),1)))).astype(float)
            aux=pd.DataFrame(stack).set_index(df_demand.index)
            df1=pd.concat([df_demand.loc[:,i],aux,df_prices[:]],axis=1)
            df1.columns=['demand','gen','SOC','PV_batt', 'PV_load', 'PV_grid', 'E_dis','Batt_load',
                    'Batt_grid', 'grid_load','PV_losses','Batt_losses','flag','type','df','prices']

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
    df_comm.to_csv(path+'Output/community_{}_{}.csv'.format(PV_penetration,Batt_penetration))
    df_no_comm.to_csv(path+'Output/no_community_{}_{}.csv'.format(PV_penetration,Batt_penetration))

    Gen_balance=(df_comm.gen-(df_comm.PV_batt+df_comm.PV_load+df_comm.PV_comm+df_comm.PV_losses))[:endo].sum()

    Batt_balance=(df_comm.PV_batt-(df_comm.Batt_load+df_comm.Batt_comm+df_comm.Batt_losses))[:endo].sum()

    Demand_balance=(df_comm.demand-(df_comm.PV_load+df_comm.Batt_load+df_comm.comm_load))[:endo].sum()
    return[Gen_balance,Batt_balance,Demand_balance]
