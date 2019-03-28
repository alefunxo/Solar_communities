# -*- coding: utf-8 -*-
## @namespace Main
# Created on Wed Feb 28 09:47:22 2018
# Author
# Alejandro Pena-Bello
# alejandro.penabello@unige.ch
#Main script developed for the project developed together with the Consumer Decision and Sustainable Behavior Lab to include the user preferences in the charging and discharging of the battery. We use a deterministic approach and include probabilities to discharge the battery to the grid in the frame of a community with P2P energy trading.
# The script has been tested in Linux and Windows
# INPUTS
# ------
# Inputs are automatically saved in the 'Input' file
# OUTPUTS
# ------
# Outputs are automatically saved in the 'Output' file
# TODO
# ----
# User Interface
# Requirements
# ------------
# Pandas, numpy, itertools,sys,glob,multiprocessing, time

import pandas as pd
import matplotlib.pyplot as plt
import Germany_PVdistribution as D_PV
import Price_setup as ps
import Post_processing as pp
from functools import wraps
import Psycho_paper_community as psy_com
import time
import os
import argparse
import numpy as np
import itertools
import sys
import glob
def fn_timer(function):
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
@fn_timer
def set_distributions(path):
    '''
    Description
    -----------
    Calls functions from the Germany_PVdistribution module
    PV_generation, defines the azimuths, inclinations and latitude to be simulated using PV_output_inclinations. Calls the following functions: Distribution, German_load and PV_gen_munich. Only the last one is necessary in order to simulate the correct amount of PV penetration. The other functions must be called only the first time the script is used in order to create the input data (PV for different inclinations, the PV distribution and the load).

    Returns
    ------

    TODO
    ------
    Do it interactive with the user in order to avoid manual changes when some functions are not needed
    '''
    #df=D_PV.PV_generation(path)
    #azimuths=[-30,-20,-10,0,10,20,30]
    azimuths=[-50,-40,-30,-20,-10,0,10,20,30,40,50]
    inclinations=[20,25,30,35,40,45]
    phi=48.1351 #Munich
    #D_PV.PV_output_inclinations(azimuths,inclinations,df,15,phi)
    #D_PV.Distribution(path)
    #D_PV.German_load(path)
    D_PV.PV_gen_munich(path)#Once all run, re-run this one.
    return
@fn_timer
def main():
    '''
    Description
    -----------
    This function handle the deterministic model. It can be use in three modes, choice=True will allow to choice the PV and Battery penetration of the community. If choice=False the script will be run for different PV penetrations and Battery penetrations (predefined, PV_array and Batt_array). pp_only=True will only use predefined outputs (included already in the Output folder) and plot the results in a pdfself. If print_=False the script will not save the outputs in a pdf.

    Parameters
    ----------
    choice: Boolean ; Allows to run for different PV penetrations and Battery penetration or for a predefined set.
    pp_only: Boolean ; Use predefined outputs (included already in the Output folder) and plot the results in a pdfself
    print_: Boolean ; If true the script will save the outputs in a pdf.

    Returns
    ------
    Nothing. The results are stored in the folder 'Outputs'

    TODO
    ------
    Do it interactive with the user in order to avoid manual changes when some functions are not needed
    '''
    for i in range(4):
        i=i+1
        if sys.platform=='win32':
            path='C:/Users/alejandro/Documents/GitHub/Psycho/'
        else:
            path='./'
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('Welcome')
        choice=True
        print_=True
        pp_only=False
        probs_applied=i# 1 for surplus only, 2 for morning+surplus 3 for morning+surplus+evening 4 for evening and surplus
        if pp_only==False:
            #set_distributions(path)
            if choice:
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    'PV_penetration', choices=range(0,101),type=int,
                    help='Choice PV penetration between 0 and 100%')
                parser.add_argument(
                    'Batt_penetration', choices=range(0,101),type=int,
                    help='Choice Battery penetration (among houses with PV) between 0 and 100%')
                args=parser.parse_args()
                day_sel='other'
                print(args)
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                print('PV and battery penetration selected')
                print('PV penetration: {}%'.format(args.PV_penetration))
                print('Battery penetration: {}%'.format(args.Batt_penetration))
                reso='1h'
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                [Gen_balance,Batt_balance,Demand_balance]=psy_com.community_psycho(args.Batt_penetration/100,args.PV_penetration/100,reso,path,day_sel,probs_applied)
                print('Generation Balance:')
                print(Gen_balance)
                print('Battery Balance:')
                print(Batt_balance)
                print('Demand Balance:')
                print(Demand_balance)
                pp.Post_processing(args.Batt_penetration/100,args.PV_penetration/100,print_,path,probs_applied)
            else:
                PV_array=[20,25,30,35,40,45,50]
                Batt_array=[25,50,75,100]
                for PV in PV_array:
                    for Batt in Batt_array:
                        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                        print('PV and battery penetration selected')
                        print('PV penetration: {}%'.format(PV))
                        print('Battery penetration: {}%'.format(Batt))
                        reso='1h'
                        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                        [Gen_balance,Batt_balance,Demand_balance]=psy_com.community_psycho(Batt/100,PV/100,reso,path,day_sel,probs_applied)
                        print('Generation Balance:')
                        print(Gen_balance)
                        print('Battery Balance:')
                        print(Batt_balance)
                        print('Demand Balance:')
                        print(Demand_balance)
                        pp.Post_processing(Batt/100,PV/100,print_,path)
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        else:
            "Only printing results"
            parser = argparse.ArgumentParser()
            parser.add_argument(
                'PV_penetration', choices=range(0,101),type=int,
                help='Choice PV penetration between 0 and 100%')
            parser.add_argument(
                'Batt_penetration', choices=range(0,101),type=int,
                help='Choice Battery penetration (among houses with PV) between 0 and 100%')
            args=parser.parse_args()
            print(args)
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print('PV and battery penetration selected')
            print('PV penetration: {}%'.format(args.PV_penetration))
            print('Battery penetration: {}%'.format(args.Batt_penetration))
            reso='1h'
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            pp.Post_processing(args.Batt_penetration/100,args.PV_penetration/100,print_,path,probs_applied)
if __name__== '__main__':
    main()
