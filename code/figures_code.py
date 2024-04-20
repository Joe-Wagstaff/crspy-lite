# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:53:17 2024

@author: joedw
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configparser import RawConfigParser
import calendar

current_directory = os.getcwd() 
wd = current_directory
nld = RawConfigParser()
nld.read("config_files\config.ini")

### TOGGLE CORRECTION FUNCTIONS ###

def mod_output_toggle(f_p, f_h, f_i, f_v, df):

    """This function outputs the corrected nuetron count
    toggle the boolean statements to select which (if any) correction factors to apply.
    The function can run for column names formatted with names from both CRSPY_light and CRSPY.

    Parameters
    ----------
    f_p : Boolean
        whether to apply presure correction factor         
    f_h : Boolean
        wether to apply humidity corection factor  
    f_i : Boolean
        wether to apply cosmic ray intensity corection factor 
    f_v : Boolean
        wether to apply above ground biomass corection factor
    df : dataframe
        dataframe of level1 CRNS data

    """
    if f_p==True:
        try: 
            f_p_c = df['f_pressure']
        except:
            pass
        
        try:
            f_p_c = df['fbar']
        except:
            pass
    else:
        f_p_c=1
    
    if f_h==True:
        try:
            f_h_c = df['f_humidity']
        except:
            pass
        try:
            f_h_c = df['fawv']
        except:
            pass
        
    else:
        f_h_c=1
        
    if f_i==True:
        try:
            f_i_c = df['f_intensity']
        except:
            pass
        
        try:
            f_i_c = df['finten']
        except:
            pass
        
    else:
        f_i_c=1
       
    if f_v==True:
        try:
            f_v_c = df['f_vegetation']
        except:
            pass
        
        try:
            f_v_c = df['fagb']
        except:
            pass
    else:
        f_v_c=1
        
    mod = df['MOD']
    
    
    return mod * f_p_c * f_h_c * f_i_c * f_v_c

def mod_output(f_p, f_h, f_i, f_v, mod):
    
##########################
### PLOTTING FUNCTIONS ###

    """mod_corr gives the corrected neutron count 

    Parameters
    ----------
    f_p : float
        corection factor for presure       
    f_h : float
        corection factor for humidity
    f_i : float
        corection factor for cosmic ray intensity
    f_v : float
        corection factor for above ground biomass
    mod : int
        neutron count

    """
    return mod * f_p * f_h * f_i * f_v
##########################

def site_data_plots(df, site_features, monthly_means_plt, typical_year, daily_sm_eff_depth_plt, qa_plots):
    
    """report_plots will output some figures for my RP3 report
        Toggle booleans when the function is called to select what plots are outputed.

    Parameters
    ----------
    df : DataFrame 
        DataFrame of final CRNS data.
    
    site_features : Boolean
        Wether to output plot of site features. This includes subplots for daily temperature,
        daily precipitation and daily SM (with the caluclated error)
    
    monthly_means_plt: Boolean
        whether to output plot of a typical year of SM for the site. This includes an estimated 
        error as well
     
    daily_sm_eff_depth_plt: Boolean
        whether to plot a subplot of daily raw SM (without error) and a subplot of estimated 
        sensor measurement depth.
        
    """

    if site_features==True:
    
        print("Outputting site_features plot...")

        df['DT'] = pd.to_datetime(df['DT'])
        df['YEAR'] = df['DT'].dt.year
        years = df['YEAR'].unique()
        df['MONTH'] = df['DT'].dt.month
        months = df['MONTH'].unique()
        df['DAY'] = df['DT'].dt.day
        days = df['DAY'].unique()
    
        df_by_year_month_day = df.groupby([df['DT'].dt.year, df['DT'].dt.month, df['DT'].dt.day])
        year_month_day_series = df['DT'].dt.to_period('D')
        year_month_day_series = (year_month_day_series.drop_duplicates()).reset_index()
        year_month_day_series = year_month_day_series.astype(str)
        year_month_day_col = year_month_day_series['DT']
        
        #precipitation
        precip = df_by_year_month_day['RAIN'].mean()
        
        #max precipitation for site from metadata. Default is no limit. 
        max_precip_val = nld['metadata']['max_precipitation']
        if max_precip_val.isnumeric() or (max_precip_val.count('.') == 1): 
            precip = precip.clip(upper=float(max_precip_val))
        
        else:
            precip=precip
        
        #temperature
        e_temp = df_by_year_month_day['E_TEM'].mean()
        
        # we want sm +/- the sm errors
        df['SM_PLUS_ERR'] = df['SM_RAW'] + abs(df['SM_PLUS_ERR']) 
        df['SM_MINUS_ERR'] = df['SM_RAW'] - abs(df['SM_MINUS_ERR'])
        
        sm_raw = df_by_year_month_day['SM_RAW'].mean()
        sm_plus_err = df_by_year_month_day['SM_PLUS_ERR'].mean()
        sm_minus_err = df_by_year_month_day['SM_MINUS_ERR'].mean()

        #figure assembly 
        plt.rcParams['font.size'] = 16
    
        fig, axs1 = plt.subplots(3, sharex=True, figsize=(15, 9))
        
        #axs1[0].set_title("Site info - " + nld.get('metadata','site_name') + ", " + nld.get('metadata','country') + "_" + nld.get('metadata','sitenum'))
        axs1[0].plot(year_month_day_col, e_temp, lw=0.8, color='black', label='Temperature')
        axs1[0].set_ylabel("Daily Temp ($^o$C)")
        #axs1[0].set_title("Site features - "+nld['metadata']['site_name']+", "+nld['metadata']['country'])
        
        axs1[1].plot(year_month_day_col, precip, lw=0.8, color='black', label='Precipitation')
        axs1[1].set_ylabel("Daily Rainfall (mm)")
        
        axs1[2].plot(year_month_day_col, sm_raw, lw=0.8, color='black', label='SM')
        axs1[2].fill_between(year_month_day_col, sm_minus_err, sm_plus_err, alpha=0.5, edgecolor='#CC4F1B', facecolor='firebrick', label='Error')
        axs1[2].set_ylabel('Daily SM (cm$^3$ / cm$^3$)')
        axs1[2].set_xlabel("Time (years)")
        axs1[2].legend()
        
        plt.xticks(np.arange(0, len(year_month_day_col), 365))
        plt.xticks(fontsize=14)
        
        #labelling subplots a, b, c, etc...
        for i, ax in enumerate(axs1.flat):
            label = chr(ord('a') + i)  
            ax.text(1.02, 0.16, f'({label})', transform=ax.transAxes, fontsize=16, va='top', weight='bold')
        
        plt.tight_layout()
        fig.savefig(wd+"/outputs/figures/site_features.png", dpi=350)
        plt.show()   
        
        #convert +/- error columns back.
        df['SM_PLUS_ERR'] = abs(df['SM_PLUS_ERR'] - df['SM_RAW'])
        df['SM_MINUS_ERR'] = abs(df['SM_MINUS_ERR'] - df['SM_RAW'])
        
    else:
        print("Skipping site_features plot...")
    
    if monthly_means_plt==True:
        print("Outputting montly_means plt...")
        
        #want the exact errors for error bar plot. This should already be the case.
        #df['SM_PLUS_ERR'] = abs(df['SM_PLUS_ERR'] - df['SM'])
        #df['SM_MINUS_ERR'] = abs(df['SM_MINUS_ERR'] - df['SM'])
        
        #sm month plot (monthly)
        df['DT'] = pd.to_datetime(df['DT'])
        df['YEAR'] = df['DT'].dt.year
        years = df['YEAR'].unique()
        df['MONTH'] = df['DT'].dt.month
        months = df['MONTH'].unique()
        month_names = [calendar.month_name[month_num] for month_num in months]
      
        df_by_month = df.groupby([df['DT'].dt.month])
        month_series = df['DT'].dt.to_period('M')
        month_series = (month_series.drop_duplicates()).reset_index()
        month_series = month_series.astype(str)
        month_col = month_series['DT']
        
        sm_raw_avg_month = df_by_month['SM_RAW'].mean()
        sm_plus_err_avg_month = (df_by_month['SM_PLUS_ERR'].mean()).to_numpy()
        sm_minus_err_avg_month = (df_by_month['SM_MINUS_ERR'].mean()).to_numpy()
        
        m = sm_raw_avg_month.mean()  #mean
        var = ((sm_raw_avg_month - m) **2) / int(len(sm_raw_avg_month)) #variance
        sd = var.apply(np.sqrt) #standard deviation is equivalent to the error
        
        #changing arrays to series reset indexes to avoid reindexing errors
        month_names = pd.Series(month_names)
        month_names = month_names.reset_index(drop=True)
        
        sm_raw_avg_month = pd.Series(sm_raw_avg_month)
        sm_raw_avg_month = sm_raw_avg_month.reset_index(drop=True)
        
        sm_plus_err_avg_month = pd.Series(sm_plus_err_avg_month)
        sm_plus_err_avg_month = sm_plus_err_avg_month.reset_index(drop=True)
        
        sm_minus_err_avg_month = pd.Series(sm_minus_err_avg_month)
        sm_minus_err_avg_month = sm_minus_err_avg_month.reset_index(drop=True)
        
        sd = pd.Series(sd)
        sd = sd.reset_index(drop=True)
    
        #reindexing data to start from january rather than april
        df_fig = pd.DataFrame({})
        df_fig['MONTH'] = month_names
        df_fig['SM_RAW'] = sm_raw_avg_month
        df_fig['SM_PLUS_ERR'] = sm_plus_err_avg_month
        df_fig['SM_MINUS_ERR'] = sm_minus_err_avg_month
        df_fig['STANDARD_DEVIATION'] = sd
    
        df_fig.set_index('MONTH', inplace=True)

        #locating jan 01 (where the typical year should start from)
        january_index = df_fig.index.to_list().index(('January'))

        # Rearrange the DataFrame once to locate all values below jan 01 and twice to locate all values above jan 01
        rearranged_df = df_fig.iloc[january_index:]
        rearranged_df1 = df_fig.iloc[:january_index]
        
        #then merge the two rearranged df to create a df of the complete year from 01 jan to 31 dec
        merged_df = pd.concat([rearranged_df, rearranged_df1])
        
        #redefining variables that have been reindexed
        month_names = merged_df.index
        sm_raw_avg_month = merged_df['SM_RAW']
        sm_plus_err_avg_month = merged_df['SM_PLUS_ERR']
        sm_minus_err_avg_month = merged_df['SM_MINUS_ERR']
        sd = merged_df['STANDARD_DEVIATION']
    
        #creating figure
        plt.rcParams['font.size'] = 16
    
        fig, ax = plt.subplots(figsize=(15,4.5))
        
        plt.errorbar(month_names, sm_raw_avg_month, yerr=sd, xerr=None, label='SM with error',
                     fmt='-', marker='.',color='blue', ecolor='red', lw=1.5, elinewidth=1.0,
                     capsize=3.0)
        
        plt.xticks(rotation=45, ha="right", fontsize=12)
        ax.set_title("Monthly Means - "+nld['metadata']['site_name']+", "+nld['metadata']['country'])
        ax.set_ylabel("Monthly SM (cm$^3$ / cm$^3$)")
        ax.set_xlabel('Time (Years)')
        plt.legend()
        plt.tight_layout()
        fig.savefig(wd+"/outputs/figures/monthly_means.png", dpi=350)
        plt.show()
        
    else:
        print("Skipping monthly_means_plt...")
        pass
    
    if typical_year==True:
        
        print("Outputting typical year plot...")

        df['DT'] = pd.to_datetime(df['DT'])
        df['YEAR'] = df['DT'].dt.year
        years = df['YEAR'].unique()
        df['MONTH'] = df['DT'].dt.month
        months = df['MONTH'].unique()
        
        #grouping df by month and day so that the values each year can be averaged 
        df_by_day = df.groupby([df['DT'].dt.month, df['DT'].dt.day])
        dates = df['DT']
        startdate = dates[0].date()
        dt_index = pd.date_range(start=startdate, periods=366, freq='D')
        
        #precipitation
        precip = df_by_day['RAIN'].mean()
        #max precipitation for site from metadata. Default is no limit. 
        max_precip_val = str(3.0)#nld['metadata']['max_precipitation']
        if max_precip_val.isnumeric() or (max_precip_val.count('.') == 1): 
            precip = precip.clip(upper=float(max_precip_val))
        
        else:
            precip=precip
    
        #temperature
        e_temp = df_by_day['E_TEM'].mean()
        e_temp = e_temp.reset_index(drop=True)
       
        # we want sm +/- the sm errors
        df['SM_PLUS_ERR'] = df['SM_RAW'] + abs(df['SM_PLUS_ERR']) 
        df['SM_MINUS_ERR'] = df['SM_RAW'] - abs(df['SM_MINUS_ERR'])
        
        sm_raw = df_by_day['SM_RAW'].mean()
        sm_plus_err = df_by_day['SM_PLUS_ERR'].mean()
        sm_minus_err = df_by_day['SM_MINUS_ERR'].mean()
    
        #reset indexes
        e_temp = e_temp.reset_index(drop=True)
        precip = precip.reset_index(drop=True)
        sm_raw = sm_raw.reset_index(drop=True)
        sm_plus_err = sm_plus_err.reset_index(drop=True)
        sm_minus_err = sm_minus_err.reset_index(drop=True)
    
        #defining a new df for the figure with the values for a typical year
        df_fig = pd.DataFrame({})
        df_fig['DATE'] = dt_index
        df_fig['E_TEM'] = e_temp
        df_fig['RAIN'] = precip
        df_fig['SM_RAW'] = sm_raw
        df_fig['SM_PLUS_ERR'] = sm_plus_err
        df_fig['SM_MINUS_ERR'] = sm_minus_err
        
        #creating a new index of the month-days
        df_fig['DATE'] = df_fig['DATE'].dt.strftime('%m-%d')
        df_fig.set_index('DATE', inplace=True)
        
        #locating jan 01 (where the typical year should start from)
        january_1_index = df_fig.index.to_list().index(('01-01'))

        # Rearrange the DataFrame once to locate all values below jan 01 and twice to locate all values above jan 01
        rearranged_df = df_fig.iloc[january_1_index:]
        rearranged_df1 = df_fig.iloc[:january_1_index]
        
        #then merge the two rearranged df to create a df of the complete year from 01 jan to 31 dec
        merged_df = pd.concat([rearranged_df, rearranged_df1])
        
        #redefining variables
        dt_index = merged_df.index
        e_temp = merged_df['E_TEM']
        precip = merged_df['RAIN']
        sm_raw = merged_df['SM_RAW']
        sm_plus_err = merged_df['SM_PLUS_ERR']
        sm_minus_err = merged_df['SM_MINUS_ERR']
        
        #figure assembly and plotting
        plt.rcParams['font.size'] = 16
        
        fig, axs1 = plt.subplots(3, sharex=True, figsize=(15, 9))
        
        #axs1[0].set_title("Site info - " + nld.get('metadata','site_name') + ", " + nld.get('metadata','country') + "_" + nld.get('metadata','sitenum'))
        axs1[0].plot(dt_index, e_temp, lw=1.5, color='black', label='Temperature')
        axs1[0].set_ylabel("Daily Temp ($^o$C)")
        #axs1[0].set_title("Site features - "+nld['metadata']['site_name']+", "+nld['metadata']['country'])
        
        axs1[1].plot(dt_index, precip, lw=1.5, color='black', label='Precipitation')
        axs1[1].set_ylabel("Daily Rainfall (mm)")
        
        axs1[2].plot(dt_index, sm_raw, lw=1.5, color='black', label='SM')
        axs1[2].fill_between(dt_index, sm_minus_err, sm_plus_err, alpha=0.5, edgecolor='#CC4F1B', facecolor='firebrick', label='Error')
        axs1[2].set_ylabel('Daily SM (cm$^3$ / cm$^3$)')
        axs1[2].set_xlabel("Time (months)")
        axs1[2].legend()
       
        #setting x-labels
        plt.xticks(np.arange(0, len(dt_index), 31)) #defining the no. ticks (12) 
        dt_index = '2016' + '-' + dt_index.astype(str) #required to convert to dt
        dt_index = pd.to_datetime(dt_index, format='%Y-%m-%d')  #converting to dt
        axs1[2].set_xticklabels(dt_index.strftime('%b').unique()) #setting xlabels to be month names (%b).
        plt.xticks(fontsize=12, rotation=25) #changing font and orientation.
        
        #labelling subplots a, b, c, etc...
        for i, ax in enumerate(axs1.flat):
            label = chr(ord('a') + i)  
            ax.text(1.02, 0.16, f'({label})', transform=ax.transAxes, fontsize=16, va='top', weight='bold')

        plt.tight_layout()
        fig.savefig(wd+"/outputs/figures/typical_year.png", dpi=350)
        plt.show()   
        
        #convert +/- error columns back.
        df['SM_PLUS_ERR'] = abs(df['SM_PLUS_ERR'] - df['SM_RAW'])
        df['SM_MINUS_ERR'] = abs(df['SM_MINUS_ERR'] - df['SM_RAW'])
        
    if daily_sm_eff_depth_plt==True:
       
        print("Outputting daily_sm_eff_depth_plt plot...")

        ##sm comparison
        df['DT'] = pd.to_datetime(df['DT'])
        df['YEAR'] = df['DT'].dt.year
        years = df['YEAR'].unique()
        df['MONTH'] = df['DT'].dt.month
        months = df['MONTH'].unique()
        df['DAY'] = df['DT'].dt.day
        days = df['DAY'].unique()
    
        df_by_year_month_day = df.groupby([df['DT'].dt.year, df['DT'].dt.month, df['DT'].dt.day])
        year_month_day_series = df['DT'].dt.to_period('D')
        year_month_day_series = (year_month_day_series.drop_duplicates()).reset_index()
        year_month_day_series = year_month_day_series.astype(str)
        year_month_day_col = year_month_day_series['DT']
    
        #daily SM
        sm_raw_by_day = df_by_year_month_day['SM_RAW'].mean()
    
        #daily effective depth
        eff_z_by_day = df_by_year_month_day['D86avg'].mean()
        
        #figure assembly 
        
        plt.rcParams['font.size'] = 16
    
        fig, axs = plt.subplots(2, sharex=True, figsize=(15, 3))
        
        axs[0].set_title("Daily SM and effective depth - "+nld['metadata']['site_name']+", "+nld['metadata']['country']+"_"+nld['metadata']['sitenum'])
        axs[0].plot(year_month_day_col, sm_raw_by_day, lw=1.3, color='blue')
        axs[0].set_ylabel("Daily SM (cm$^3$ / cm$^3$)")

        axs[1].plot(year_month_day_col, eff_z_by_day, lw=1.3, color='red')
        axs[1].set_ylabel("Effective Depth (cm)")
        axs[1].set_xlabel("Time (years)")
     
        plt.xticks(np.arange(0, len(year_month_day_col), 365))
        plt.xticks(fontsize=14)
        plt.tight_layout()
        fig.savefig(wd+"/outputs/figures/daily_sm_eff_depth_plt.png", dpi=350)
        plt.show()     
        
    else:
        print("Skipping daily_sm_eff_depth_plt plot...")
        
    if qa_plots == True:
        
        print("Outputting Quality Assessment plots...")
        
        dtime = pd.to_datetime(df['DT'], format= "%Y-%m-%d %H:%M:%S")
        dfdt1 = df.set_index(dtime)
        
        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(15, 3))
        
        axs[0].plot(dfdt1['I_RH'], lw=0.5, color='red', label ='Internal relative humidity')
        axs[0].set_ylabel("Internal Relative\nhumidity (%)")
        axs[0].set_xlabel("Time (years)")
        
        axs[1].plot(dfdt1['BATT'], lw=0.5, color='red', label ='Battery level')
        axs[1].set_ylabel("Battery Level (Volts)")
        axs[1].set_xlabel("Time (years)")
        
        plt.tight_layout()
        fig.savefig(wd+"/outputs/figures/qa_plots.png", dpi=350)
        plt.show()    
    
    else:
        print("Skipping Quality Assessment plots...")
    
    return df

###  IMPORTING DATA AND CALLING PLOTTING FUNCTIONS ###

#importing final data
final_data_path = wd +"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['sitenum']+"_final.txt"
final_data = pd.read_csv(final_data_path, sep="\t")
df = pd.DataFrame(final_data)
df.replace(int(nld['general']['noval']), np.nan, inplace=True)

#importing crspy data 
try:
    level_1_crspy_path = "crspy_files_from_dan\AUS_SITE_018\AUS_SITE_018_final.txt"  #change this to be hard coded.
    df2= pd.read_csv(level_1_crspy_path, sep="\t")
    df2.replace(int(nld['general']['noval']), np.nan, inplace=True)
except:
    pass

#Calling plotting functions

# *** toggle booleans (set to either 'True' or 'False') to output different plots. *** #

site_data_plots(df=df,
            site_features=True, 
            monthly_means_plt=False,
            typical_year=True,
            daily_sm_eff_depth_plt=False,
            qa_plots=False
            )

#~~~~~~~~ OPTIONAL CRSPY COMPARISON ~~~~~~~~~#

def crspy_comparison_plt(df, crspy_data_filepath, comparison_plt, mean_bias):
    """crpsy_comparison_plt outputs CRSPY and CRSPY_lite data at several key processing stages. 
    These include subplots of daily corrected neutron counts, estimated daily raw SM and 
    daily effective depth.
    
    Parameters
    ----------
    
    df : DataFrame 
        DataFrame of final CRNS data.
        
    crspy_data_filepath : str
        string of filepath for final data outputed by crspy
    
    comparison_plt : Boolean
        Wether to output comparison plot.
        
    mean_bias : Boolean
        wether to calculate the mean bias between the CRSPY and CRSPY lite outputs.Pleas note that
        this can't be run without running 'comparison plts'.
    
    """
    crspy_data_filepath = crspy_data_filepath
    if os.path.isfile(crspy_data_filepath) and comparison_plt==True:
        
        print("Outputting outputs_comparison plot...")
        
        df2= pd.read_csv(crspy_data_filepath, sep="\t")
        df2.replace(int(nld['general']['noval']), np.nan, inplace=True)

        df = df[df.index <= 46535]
        ##mod corr comparison
        df['DT'] = pd.to_datetime(df['DT'])
        df['YEAR'] = df['DT'].dt.year
        years = df['YEAR'].unique()
        df['MONTH'] = df['DT'].dt.month
        months = df['MONTH'].unique()
        df['DAY'] = df['DT'].dt.day
        days = df['DAY'].unique()
    
        df_by_year_month_day = df.groupby([df['DT'].dt.year, df['DT'].dt.month, df['DT'].dt.day])
        year_month_day_series = df['DT'].dt.to_period('D')
        year_month_day_series = (year_month_day_series.drop_duplicates()).reset_index()
        year_month_day_series = year_month_day_series.astype(str)
        year_month_day_col = year_month_day_series['DT']
        
        #assigning mod output.
        mod_output = mod_output_toggle(f_p=True, f_h=True, f_i=True, f_v=True, df=df)
        mod_output = mod_output.round(0)
        df['MOD_OUTPUT'] = mod_output
        mod_corr_by_day = df_by_year_month_day['MOD_OUTPUT'].mean()
        
        #crspy data
        df2['DT'] = pd.to_datetime(df2['DT'])
        df2['YEAR'] = df2['DT'].dt.year
        years = df2['YEAR'].unique()
        df2['MONTH'] = df2['DT'].dt.month
        months = df2['MONTH'].unique()
        df2['DAY'] = df2['DT'].dt.day
        days = df2['DAY'].unique()
    
        df2_by_year_month_day = df2.groupby([df2['DT'].dt.year, df2['DT'].dt.month, df2['DT'].dt.day])
        c_year_month_day_series = df2['DT'].dt.to_period('D')
        c_year_month_day_series = (c_year_month_day_series.drop_duplicates()).reset_index()
        c_year_month_day_series = c_year_month_day_series.astype(str)
        c_year_month_day_col = c_year_month_day_series['DT']
        
        mod_output_crspy = mod_output_toggle(f_p=True, f_h=True, f_i=True, f_v=True, df=df2) 
        mod_output_crspy=mod_output_crspy.round(0)
        df2['MOD_OUTPUT'] = mod_output_crspy
        c_mod_corr_by_day = df2_by_year_month_day['MOD_OUTPUT'].mean()
        
        ##sm comparison
        df['DT'] = pd.to_datetime(df['DT'])
        df['YEAR'] = df['DT'].dt.year
        years = df['YEAR'].unique()
        df['MONTH'] = df['DT'].dt.month
        months = df['MONTH'].unique()
        df['DAY'] = df['DT'].dt.day
        days = df['DAY'].unique()
    
        df_by_year_month_day = df.groupby([df['DT'].dt.year, df['DT'].dt.month, df['DT'].dt.day])
        year_month_day_series = df['DT'].dt.to_period('D')
        year_month_day_series = (year_month_day_series.drop_duplicates()).reset_index()
        year_month_day_series = year_month_day_series.astype(str)
        year_month_day_col = year_month_day_series['DT']
    
        sm_raw_by_day = df_by_year_month_day['SM_RAW'].mean()
        
        #crspy data
        df2['DT'] = pd.to_datetime(df2['DT'])
        df2['YEAR'] = df2['DT'].dt.year
        years = df2['YEAR'].unique()
        df2['MONTH'] = df2['DT'].dt.month
        months = df2['MONTH'].unique()
        df2['DAY'] = df2['DT'].dt.day
        days = df2['DAY'].unique()
    
        df2_by_year_month_day = df2.groupby([df2['DT'].dt.year, df2['DT'].dt.month, df2['DT'].dt.day])
        c_year_month_day_series = df2['DT'].dt.to_period('D')
        c_year_month_day_series = (c_year_month_day_series.drop_duplicates()).reset_index()
        c_year_month_day_series = c_year_month_day_series.astype(str)
        c_year_month_day_col = c_year_month_day_series['DT']
        
        c_sm_raw_by_day = df_by_year_month_day['SM_RAW'].mean()
      
        #effective depth
        eff_z_by_day = df_by_year_month_day['D86avg'].mean()
        
        c_eff_z_by_day = df2_by_year_month_day['D86avg'].mean()
        
        #fist year of df
        df = df[df.index <= 46535] #balancing with crspy df
        
        one_year_month_day_col = year_month_day_col[0:364]
        one_c_year_month_day_col = c_year_month_day_col[0:364]
        
        one_mod_corr_by_day = mod_corr_by_day[0:364]
        one_c_mod_corr_by_day =c_mod_corr_by_day[0:364]
        
        one_sm_raw_by_day = sm_raw_by_day[0:364]
        one_c_sm_raw_by_day = c_sm_raw_by_day[0:364]
        
        one_eff_z_by_day = eff_z_by_day[0:364]
        one_c_eff_z_by_day = c_eff_z_by_day[0:364]
        
        #figure assembly 
        
        plt.rcParams['font.size'] = 16
    
        fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(16, 9))
    
        #whole length of df
        axs[0,0].plot(year_month_day_col, mod_corr_by_day, lw=1.5, linestyle='--', color='crimson', label='CRSPY_lite data')  
        axs[0,0].plot(c_year_month_day_col, c_mod_corr_by_day, lw=0.5, color='black', label='CRSPY data') 
        axs[0,0].set_ylabel("Daily Corrected\nNeutron Counts")
        axs[0,0].legend()
 
        axs[1,0].plot(year_month_day_col, sm_raw_by_day, lw=1.5, linestyle='--', color='crimson')
        axs[1,0].plot(year_month_day_col, c_sm_raw_by_day, lw=0.5, color='black')
        axs[1,0].set_ylabel("Daily SM (cm$^3$ / cm$^3$)")
        axs[1,0].legend()
        
        axs[2,0].plot(year_month_day_col, eff_z_by_day, lw=1.5, color='crimson', linestyle='--')
        axs[2,0].plot(c_year_month_day_col, c_eff_z_by_day, lw=0.5, color='black')
        axs[2,0].set_ylabel("Effective Depth (cm)")
        #axs[2,0].set_xlabel("Time (Year-Month-Day)")
        axs[2,0].legend()
        
        #first year of df
        axs[0,1].plot(one_year_month_day_col, one_mod_corr_by_day, lw=1.5, linestyle='--', color='crimson', )  
        axs[0,1].plot(one_c_year_month_day_col, one_c_mod_corr_by_day, lw=0.5, color='black') 
        axs[0,1].set_ylabel("Daily Corrected\nNeutron Counts")
        axs[0,1].legend()
        
        axs[1,1].plot(one_year_month_day_col, one_sm_raw_by_day, lw=1.5, linestyle='--', color='crimson')
        axs[1,1].plot(one_c_year_month_day_col, one_c_sm_raw_by_day, lw=0.5, color='black')
        axs[1,1].set_ylabel("Daily SM (cm$^3$ / cm$^3$)")
        axs[1,1].legend()
       
        axs[2,1].plot(one_year_month_day_col, one_eff_z_by_day, lw=1.5, color='crimson', linestyle='--')
        axs[2,1].plot(one_c_year_month_day_col, one_c_eff_z_by_day, lw=0.5, color='black')
        axs[2,1].set_ylabel("Effective Depth (cm)")
        axs[2,1].legend()
        #axs[2,1].set_xlabel("Time (months)")
     
        #plt.suptitle("CRSPY vs CRSPY_lite comparison - " + nld.get('metadata','site_name') + ", " + nld.get('metadata','country') + "_" + nld.get('metadata','sitenum'))
        fig.text(0.5, 0.0035, "Time (Year-Month)", ha='center')
        
        #xticklabels
        #col 1
        dt_index = pd.to_datetime(year_month_day_col)
        dt_df = pd.DataFrame({})
        dt_df['DT'] = dt_index
        dt_df.set_index('DT', inplace=True)
        dt_index = dt_df.index
        start_month = str(dt_index[0].strftime('%m'))
        dt_index = dt_index.strftime('%Y').unique()
        dt_index = dt_index.astype(str) + '-' + start_month
    
        #col 2
        dt_index2 = pd.to_datetime(year_month_day_col)
        dt_df2 = pd.DataFrame({})
        dt_df2['DT'] = dt_index2
        dt_df2.set_index('DT', inplace=True)
        dt_index2 = dt_df2.index
        dt_index2 = dt_index2.strftime('%Y-%m').unique()
        dt_index2 = dt_index2[:12]
         
        #xticks
        col_one_ticks = np.arange(0, len(year_month_day_col), 365)
        col_two_ticks = np.arange(0, len(one_year_month_day_col), 31)
      
        empty_ticks = []
        
        #setting xticks and xticklabels
        axs[0,0].set_xticks(empty_ticks)
        axs[1,0].set_xticks(empty_ticks)
        axs[2,0].set_xticks(col_one_ticks)
        axs[2,0].set_xticklabels(dt_index)
        axs[0,1].set_xticks(empty_ticks)
        axs[1,1].set_xticks(empty_ticks)
        axs[2,1].set_xticks(col_two_ticks)
        axs[2,1].set_xticklabels(dt_index2)

        #formatting ticks
        axs[2,0].tick_params(axis='x', rotation=25, labelsize=11)
        axs[2,1].tick_params(axis='x', rotation=25, labelsize=11)
        
        #labelling subplots a, b, c, etc...
        for i, ax in enumerate(axs.flat):
            label = chr(ord('a') + i)  
            ax.text(1.02, 0.16, f'({label})', transform=ax.transAxes, fontsize=16, va='top', weight='bold')
    
        plt.tight_layout()
    
        fig.savefig(wd+"/outputs/figures/comparison_report.png", dpi=350)
        plt.show()     
    
        if mean_bias == True:
            print("Calculating mean bias")
        
            #mean bias of corrected counts
           
            N_corr_crpsy_lite = df['MOD_CORR']
            N_corr_crspy = df2['MOD_CORR']
            
            SM_corr_crpsy_lite = df['SM_RAW']
            SM_corr_crspy = df2['SM_RAW']
            
            ED_corr_crpsy_lite = df['D86avg']
            ED_corr_crspy = df2['D86avg']
            
            '''
            length_crspy_lite = len(N_corr_crpsy_lite)
            length_crspy = len(N_corr_crspy)
            
            if length_crspy_lite > length_crspy:
                N_corr_crpsy_lite = N_corr_crpsy_lite.head(length_crspy)
            
            elif length_crspy_lite < length_crspy:
                N_corr_crspy = N_corr_crspy.head(length_crspy_lite)

            else:
                pass
            '''
            N_m_bias = (N_corr_crpsy_lite - N_corr_crspy) / len(N_corr_crpsy_lite)
            N_m_bias  = (round(N_m_bias.sum(), 4))
        
            print("The mean bias for neutron counts = " + str(N_m_bias))
            
            SM_m_bias = (SM_corr_crpsy_lite - SM_corr_crspy) / len(SM_corr_crpsy_lite)
            SM_m_bias  = (round(SM_m_bias.sum(), 4))
        
            print("The mean bias for SM = " + str(SM_m_bias))
            
            ED_m_bias = (ED_corr_crpsy_lite - ED_corr_crspy) / len(ED_corr_crpsy_lite)
            ED_m_bias  = (round(ED_m_bias.sum(), 4))
        
            print("The mean bias for effective depth = " + str(ED_m_bias))
           
        else:
            print("Skipping mean_bias ...")
            
    else:
        print("Skipping crspy_comparison_plt. If 'comparison_plt' is set to 'True', please check filepath is valid")
        
    return df

#path of the final data outputted by CRSPY for a comparison between the two packages.
crspy_data_filepath = "C://RP3_2//CRNS_code_project//crspy_files_from_dan//AUS_SITE_018//AUS_SITE_018_final.txt"

df = crspy_comparison_plt(df=df,
                     crspy_data_filepath=crspy_data_filepath,
                     comparison_plt=False,
                     mean_bias=False
                     )




