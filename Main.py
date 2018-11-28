
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
def set_distributions():
    #df=D_PV.PV_generation()
    #azimuths=[-30,-20,-10,0,10,20,30]
    azimuths=[-50,-40,-30,-20,-10,0,10,20,30,40,50]
    inclinations=[20,25,30,35,40,45]
    phi=48.1351 #Munich
    #D_PV.PV_output_inclinations(azimuths,inclinations,df,15,phi)
    #D_PV.Distribution()
    #D_PV.German_load()
    D_PV.PV_gen_munich()#Once all run, re-run this one.
@fn_timer
def main():

    choice=False
    print_=True
    pp_only=False
    if pp_only==False:
        set_distributions()
        if choice:
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
            #ps.Price_definition(prices, args.PV_penetration/100,reso)
            [Gen_balance,Batt_balance,Demand_balance]=psy_com.community_psycho(args.Batt_penetration/100,args.PV_penetration/100,reso)
            print('Generation Balance:')
            print(Gen_balance)
            print('Battery Balance:')
            print(Batt_balance)
            print('Demand Balance:')
            print(Demand_balance)
            pp.Post_processing(args.Batt_penetration/100,args.PV_penetration/100,print_)
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
                    #ps.Price_definition(prices, PV/100,reso)
                    [Gen_balance,Batt_balance,Demand_balance]=psy_com.community_psycho(Batt/100,PV/100,reso)
                    print('Generation Balance:')
                    print(Gen_balance)
                    print('Battery Balance:')
                    print(Batt_balance)
                    print('Demand Balance:')
                    print(Demand_balance)
                    pp.Post_processing(Batt/100,PV/100,print_)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')




    else:
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
        pp.Post_processing(args.Batt_penetration/100,args.PV_penetration/100,print_)
if __name__== '__main__':
    main()
