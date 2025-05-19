
"""
@author: Joe Wagstaff
@institution: University of Bristol

"""

#general imports
import sys
import os
import json
import pandas as pd
import argparse

#crspy-lite processing functions
from crspy_lite_code import processing_functions as pf

#crspy-lite visualisation functions
from crspy_lite_code import visualisation_functions as vf


# Check user Python version
python_version = (3, 7)  # tuple of (major, minor) version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = "Module only runs on python version >= %s" % python_version_str
    raise Exception(msg)


def full_process_wrapper(default_dir):

    '''Full process wrapper of the data processing for crspy-lite. 
        The wrapper is divided into 7 sections which correspond to the same 7 processing sections in "processing_functions.py" 
        This wrapper is recommended for more experienced users. Functions can be activated/deactivated depending on user needs.
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (1) SETTING UP WORKING DIRECTORY AND CONFIG FILE ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #check default directory exists
    if os.path.exists(default_dir):
       print("Default directory detected")

    else:
       print("Taking the default directory as the working directory") 
       wd = os.getcwd()
       default_dir = wd

    #create folder structure from the default_dir
    pf.initial(default_dir)

    #config file creation
    config_file = os.path.join(default_dir, "inputs/config.json")
    
    #check to see if the file already exists
    if not os.path.exists(config_file): 
       print("Creating config file....")
       pf.create_config_file(default_dir)
       print("Done")

    else:
        print("config.json already exists. Skipping config file creation.")

    #open config file
    try:
        with open(config_file, "r") as file:
            config_data = json.load(file)

    except FileNotFoundError:
        print(f"Error: The file '{config_file}' was not found.")

    except json.JSONDecodeError:
        print(f"Error: The file '{config_file}' is not a valid JSON file.")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (2) RAW DATA TIDYING ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #read in input data (known as "raw" data). The default is a filepath however crspy-lite can import data from a url as well.
    raw_data_filepath = config_data['filepaths']['raw_data_filepath']
    raw_data_url = config_data['filepaths']['raw_data_url']

    if raw_data_filepath == None:
        try:
            raw_data = pf.retrieve_data_url(url=raw_data_url)
        except:
            print("url data retrieval unsuccessful")

    else:
        try:
            os.path.exists(raw_data_filepath) #is the path valid? 
            raw_data = pf.read_file(file_path=raw_data_filepath)   #read in raw data with in built function.
        except FileNotFoundError:
            print("Input data file not found. Please check the path is inputted correctly into the config.json file.")

    raw_data.to_csv(default_dir+"/inputs/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_input.txt", 
            header=True, index=False, sep="\t") #save raw data to inputs folder

    #tidy input data
    pf.tidy_headers(df=raw_data) #remove whitespace present in cols
    pf.flip_df(df=raw_data) #check if the dates are the write way around
    raw_data = pf.resample_to_hourly(df=raw_data) #resamples the input data to hourly intevals - only works if the df is redefined (hence "raw_data = pf" ...)
    pf.prepare_data(df=raw_data, config_data=config_data) #fixes input data to counts on the hour.
    
    tidy_data = raw_data #assign tidied df a new name: 'tidy_data'

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (3) DATA IMPORTS & CALCULATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~~~~~ NMDB data import ~~~~~~~#
    #creating copy of tidy data df that can be broken up to use for various parts of the neutron monitoring data process.
    nmdb_df_times = pd.DataFrame({})
    tidy_data['DT'] = tidy_data['DT'].astype(str)
    nmdb_df_times['DATE'] = tidy_data['DT'].str.split().str[0]
    nmdb_df_times['TIME'] = tidy_data['DT'].str.split().str[1]
    nmdb_df_times['DT'] = nmdb_df_times['DATE'] + ' ' + nmdb_df_times['TIME']

    #Assigning start & end dates.
    startdate = nmdb_df_times['DATE'].iloc[1]
    enddate = nmdb_df_times['DATE'].iloc[-1]

    #import nmdb data (currently set for Junfraijoch station).
    pf.nmdb_get(startdate, enddate, station="JUNG", default_dir=default_dir)

    #reading nmdb data.
    nmdb_data_path = default_dir + "/outputs/data/nmdb_station_counts.txt"
    nmdb_data = pd.read_csv(nmdb_data_path, header=None, delimiter = ';')
    nmdb_data.columns = ['DT', 'N_COUNT'] #adding column headings

    #adding nmdb counts to tidied df.
    tidy_data = pd.merge(nmdb_data, tidy_data, on='DT', how='inner')

    #~~~~~~~ CUT-OFF RIGIDITY CALC~~~~~~~#
    #check to see if the user inputted a value in the config file. Otherwise calculate using rc_retrieval function.
    if config_data['metadata']['rc'] is None:

        Rc = pf.rc_retrieval(latitude=config_data['metadata']['latitude'], longitude=config_data['metadata']['longitude'])
        #adding rc to metadata.
        config_data['metadata']['rc'] = str(Rc)
        with open(config_file, "w") as file:
            json.dump(config_data, file, indent=4)

    else:
        Rc = float(config_data['metadata']['rc'])

    #~~~~~~~ BETA COEFFICIENT CALC ~~~~~~~#
    #check to see if the user inputted a value in the config file. Otherwise calculate using betacoeff function.
    if config_data['metadata']['beta_coeff'] is None:

        beta_coeff, x0 = pf.betacoeff(latitude=float(config_data['metadata']['latitude']),elevation=float(config_data['metadata']['elev']),Rc=Rc)
        #adding beta_coeff to metadata.
        config_data['metadata']['beta_coeff'] = beta_coeff
        with open(config_file, "w") as file:
            json.dump(config_data, file, indent=4)

    else:
        beta_coeff = config_data['metadata']['beta_coeff']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (4) CORRECTION FACTOR FUNCTIONS ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~~~~~ PRESSURE CORRECTION ~~~~~~~#
    f_p = pf.pressure_correction(press = tidy_data['PRESS'] , beta_coeff = float(beta_coeff), 
                  p0 = float(config_data['metadata']['reference_press']))
    
    #~~~~~~~ HUMIDITY CORRECTION ~~~~~~~#
    saturation_vapour_press = pf.es(temp=tidy_data['TEMP']) #in Pa
    actual_vapour_press = pf.ea(saturation_vapour_press, RH = tidy_data['E_RH'])  #in Pa 
    absolute_humidity = pf.pv(actual_vapour_press, temp=tidy_data['TEMP'])  #in g/m3
    f_h = pf.humidity_correction(pv=absolute_humidity, pv0=float(config_data['general']['pv0']))

    #~~~~~~~ INCOMING NEUTRON INTENSITY CORRECTION ~~~~~~~#
    intensity_correction_method = config_data['general']['intensity_correction_method']

    if intensity_correction_method == 'mcjannet':
        ##McJannet correction method (reccommended)
        #calculate variables
        site_atmospheric_depth = pf.atmospheric_depth(elevation=float(config_data['metadata']['elev']), latitude=float(config_data['metadata']['latitude']))
        ref_atmospheric_depth = pf.atmospheric_depth(elevation=3570, latitude=46.55)    #elev & latitude currently set for Jungfraujoch NMDB station
        reference_Rc = 4.49  #currently set for Jungfraujoch NMDB station
        tau, K = pf.location_factor(site_atmospheric_depth, Rc, ref_atmospheric_depth, reference_Rc)
        #calculate correction
        f_i = pf.intensity_correction_McJannet2023(station_ref = float(config_data['general']['jung_ref']),station_count = tidy_data['N_COUNT'],tau=tau)
    
    elif intensity_correction_method == 'hawdon': 
        ##Hawdon correction method    
        RcCorr_result = pf.RcCorr(Rc, Rc_ref=4.49) #Rc_ref set for Jungfraujoch station
        f_i = pf.intensity_correction_Hawdon2014(station_ref=float(config_data['general']['jung_ref']), station_count=tidy_data['N_COUNT'],RcCorr=RcCorr_result)
    
    else:
        print("Invalid input. Available methods are 'mcjannet' [recommended] and 'hawdon'.") 
    
    #~~~~~~~ BIOMASS CORRECTION ~~~~~~~#
    f_v = 1 #currently no biomass correction available in crspy-lite

    #~~~~~~~ APPLY CORRECTIONS ~~~~~~~~#
    mod_corr_result = pf.mod_corr(f_p, f_h, f_i, f_v, mod=tidy_data['MOD'])

    #Define new df and add new cols to it
    corrected_data = tidy_data 
    corrected_data['MOD_CORR'] = mod_corr_result.round(0)  #nuetron counts must be integers
    corrected_data['F_PRESSURE'] = round(f_p, (4))  #4dp is sufficiently accurate for the corrections
    corrected_data['F_HUMIDITY'] = round(f_h, (4))
    corrected_data['F_INTENSITY'] = round(f_i, (4))
    corrected_data['F_VEGETATION'] = round(f_v, (4))

    #~~~~~~~~~~~~~~~~~~~~~#
    #~~ (5) CALIBRATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~#

    #check to see if the user inputted an N0 number in the config file. Otherwise calculate using n0_calibration function (note that this requires calibration data).
    if config_data['metadata']['n0'] is None:
        N0 = pf.n0_calibration(corrected_data=corrected_data, country=config_data['metadata']['country'], 
                               sitenum=config_data['metadata']['sitenum'], 
                               defineaccuracy=0.01, 
                               calib_start_time="09:00:00",
                               calib_end_time="17:00:00",
                               config_data=config_data,
                               calib_data_filepath=config_data['filepaths']['calib_data_filepath'], 
                               default_dir=config_data['filepaths']['default_dir'],
                               sm_calc_method=config_data['general']['sm_calc_method'])
    else:
        N0 = config_data['metadata']['n0']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (6) QUALITY ANALYSIS ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #Apply quality control checks to corrected df. See 'flag_and_remove' function for more detail on what is defined as eroneous data
    qa_data = pf.flag_and_remove(df=corrected_data, N0=int(N0), config_data=config_data)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (7) SOIL MOISTURE ESTIMATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    processed_sm_data = pf.thetaprocess(df=qa_data, N0=N0, sm_calc_method=config_data['general']['sm_calc_method'], 
                config_data=config_data, default_dir=config_data['filepaths']['default_dir'])
   
    processed_sm_data.to_csv(default_dir+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_processed.txt", 
            header=True, index=False, sep="\t")
   
    #~~~ END OF DATA PROCESSING! ~~~#
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ POST-PROCESSING: DATA VISUALISATION ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~ STANDARD PLOTS ~~~#
    vf.standard_plots(df=processed_sm_data, config_data=config_data)
    
    #~~~ TYPICAL YEAR OF SM PLOT ~~~#
    vf.typical_year(df=processed_sm_data, config_data=config_data)

    #~~~ BOKEH HTML INTERACTIVE PLOT PAGE ~~~#
    vf.plot_page(df=processed_sm_data)

    #~~~ END OF DATA VISUALISATION! ~~~#


    return


def main():
    parser = argparse.ArgumentParser(description="Run CRSPY Lite processing.")
    parser.add_argument("default_dir", help="File directory you wish to run the crspy_lite from")
    args = parser.parse_args()

    full_process_wrapper(args.default_dir)

if __name__ == "__main__":
    main()



