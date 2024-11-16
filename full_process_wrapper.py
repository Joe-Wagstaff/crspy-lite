

"""
@author: Joe Wagstaff
@institution: University of Bristol

"""

#general imports
import sys
import os
import json
import pandas as pd
import warnings

#import crspy-lite functions
from processing_functions import initial, create_config_file  # PROCESSING STEP (1)
from processing_functions import resample_to_hourly, tidy_headers, dropemptycols, prepare_data # (2)
from processing_functions import nmdb_get, rc_retrieval, betacoeff     # (3)
from processing_functions import pressfact_B, es, rh, ea, dew2vap, pv, humfact, finten, RcCorr, agb, mod_corr # (4)
from processing_functions import n0_calibration #(5)
from processing_functions import flag_and_remove # (6)
from processing_functions import thetaprocess # (7)

# Check user Python version
python_version = (3, 7)  # tuple of (major, minor) version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = "Module only runs on python version >= %s" % python_version_str
    raise Exception(msg)


def process_data():

    '''Full process wrapper of the data processing for crspy-lite. 
        The wrapper is divided into 7 sections which correspond to the same 7 processing sections in "processing_functions.py" 
    
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (1) SETTING UP WORKING DIRECTORY AND CONFIG FILE ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #file setup
    current_directory = os.getcwd()
    wd = current_directory
    initial(wd)

    #config file creation
    config_file = "config_files/config.json"
    
    if not os.path.exists(config_file):
       print("Creating config file....")
       create_config_file(wd)
       print("Done")

    else:
        print("config.json already exists. Skipping config file creation.")

    print("Variables can be edited by opening the config.json manually by opening the config.json file and saving your changes.")
    
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

    #read in input data (know as "raw" data).
    raw_data_file_path = config_data['filepaths']['raw_data_filepath'].strip('"')   #brought in to avoid reading filepath bug (when a filepath is copied into the config file it adds extra '') 
    
    try:
        os.path.exists(raw_data_file_path)    
        raw_data = pd.read_csv(raw_data_file_path, header=0, delimiter='\t')

    except FileNotFoundError:
        print("Input data file not found. Please check the path is inputted correctly into the config.json file.")

    #optional resampling function 
    while True:
        response = input("Does the input data need to be resampled to hourly time intevals? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Resampling dataframe...")
            resampled_df = resample_to_hourly(df=raw_data)
            resampled_df.reset_index(drop=True, inplace=True) #reseting index
            raw_data = resampled_df
            print("Done.")
            break
        
        elif response == 'n':
            print("Skipping resampling function.")
            break
        
        else:
            print("Invalid input. Please enter 'y' or 'n'. ")

    #mandatory data tidying functions
    tidy_headers(df=raw_data)
    prepare_data(df=raw_data, config_data=config_data)

    #output cleaned data file
    output_tidy_path = wd+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_tidy.txt"
    raw_data.to_csv(output_tidy_path, index=False, header=True, sep='\t')

    #assign df as tidy data for next processing steps
    tidy_data = raw_data


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (3) DATA IMPORTS & CALCULATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~~~~~ NMDB data import ~~~~~~~#
    #creating copy of tidy data df that can be broken up to use for various parts of the JUNG process.
    Jung_df_times = pd.DataFrame({})
    tidy_data['DT'] = tidy_data['DT'].astype(str)
    Jung_df_times['DATE'] = tidy_data['DT'].str.split().str[0]
    Jung_df_times['TIME'] = tidy_data['DT'].str.split().str[1]
    Jung_df_times['DT'] = Jung_df_times['DATE'] + ' ' + Jung_df_times['TIME']
    Jung_df_times.to_csv(wd+"/outputs/nmdb_outputs/Jung_df_times.txt", sep='\t', index=False)

    #Assigning start & end dates
    startdate = Jung_df_times['DATE'].iloc[1]
    enddate = Jung_df_times['DATE'].iloc[-1]

    #calling nmdb_get function
    nmdb_get(startdate, enddate, station="JUNG", wd=wd, config_data=config_data)

    #reading nmdb data
    nmdb_data_path = wd+"/outputs/nmdb_outputs/nmdb_tmp.txt"
    nmdb_data = pd.read_csv(nmdb_data_path, header=None, delimiter = ';')
    nmdb_data.columns = ['DT', 'N_COUNT'] #adding column headings

    #adding nmdb counts to tidied df
    tidy_data = pd.merge(nmdb_data, tidy_data, on='DT', how='inner')

    #moving nmdb counts column to the end of the df
    column_to_move = 'N_COUNT'
    moved_column = tidy_data.pop(column_to_move)
    tidy_data[column_to_move] = moved_column

    #~~~~~~~ CUT-OFF RIGIDITY CALC~~~~~~~#

    Rc = rc_retrieval(wd, config_data)

    
    #~~~~~~~ BETA COEFFICIENT CALC ~~~~~~~#
    
    #check to see if the user inputted a value. Otherwise calculate using betacoeff function.
    if config_data['metadata']['betacoeff'] is None:
        
        beta_coeff, x0 = betacoeff(lat=float(config_data['metadata']['latitude']),
                            elev=float(config_data['metadata']['elev']),
                            r_c= Rc,
                            wd=wd,
                            config_data=config_data
        )

    else:
        beta_coeff = config_data['metadata']['betacoeff']
        print("Beta coefficient aquired from metadata.")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (4) CORRECTION FACTOR FUNCTIONS ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    print("Correcting neutron counts...")

    #some definitions.
    temp = tidy_data['TEMP']  #taking external temp column (Celsius)
    RH = tidy_data['E_RH']  #taking humidity column (%)

    #pressure correction.
    f_p = pressfact_B(press = tidy_data['PRESS'] , B = float(config_data['metadata']['beta_coeff']), 
                  p0 = float(config_data['metadata']['reference_press']))
    
    #humidity
    es_series = 100 * es(temp) #converting es from hPa to Pa
    ea_series = ea(es_series, RH)  #in Pa 
    pv_series = 1000 * pv(ea_series, temp)  #converting from kg/m3 to g/m3
    f_h = humfact(pv_series, pv0 = float(config_data['general']['pv0']))

    #cosmic ray intensity
    finten_result = finten(jung_ref = float(config_data['general']['jung_ref']), jung_count = tidy_data['N_COUNT']) 
    RcCorr_result = RcCorr(Rc)
    f_i = ((finten_result - 1) * RcCorr_result) + 1

    #biomass
    ''' agbweight = str(config_data['metadata']['agbweight'])
    if agbweight.isnumeric() or (agbweight.count('.') == 1):
        agbval = float(agbweight)
        f_v = agb(agbval)
        print(f_v)
    else:
        f_v = 1
        print("No agbval from metadata. Treating f_v as equal to " + str(f_v))'''

    f_v = 1

    #applying corrections
    mod_corr_result = mod_corr(f_p, f_h, f_i, f_v, mod = tidy_data['MOD'])
    print("Done")

    #outputing corrected (level1) data
    level_1_data = tidy_data 
    mod_corr_result = mod_corr_result.round(0)  #rounding corrected nuetron counts column to 0dp
    #adding corrected nuetron counts column and correction factors to df
    level_1_data['MOD_CORR'] = round(mod_corr_result)   
    level_1_data['f_pressure'] = round(f_p, (4))
    level_1_data['f_humidity'] = round(f_h, (4))
    level_1_data['f_intensity'] = round(f_i, (4))
    level_1_data['f_vegetation'] = round(f_v, (4))
    output_level_1_path = wd+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_level1.txt"
    level_1_data.to_csv(output_level_1_path, index=False, header=True, sep='\t')
    print("Done")

    #~~~~~~~~~~~~~~~~~~~~~#
    #~~ (5) CALIBRATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~#

    # Brought in to stop warning around missing data
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Sensor calibration (N0)...")

    #while and if statements to prompt whether the user (a) has an n0 number and if not (b) has calibration data.
    while True:
        response = input("Do you have an N0 number? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Taking N0 from metadata...")
            #check if n0 exists in metadata
            try: 
                if config_data['metadata']['n0'] is None:
                    N0 = input(f"{"Please input N0 number"}: ").strip()
                    config_data['metadata']['n0'] = N0
                    #save n0 input to config file
                    with open(config_file, "w") as file:
                        json.dump(config_data, file, indent=4)
                    print("Successfully aquired. N0 = "+str(N0))
        
                else:
                    N0 = config_data['metadata']['n0']
                    print("Successfully aquired. N0 = "+str(N0))
            except:
                    N0 = input(f"{"Please input N0 number"}: ").strip()
                    config_data['metadata']['n0'] = N0
                    #save n0 input to config file
                    with open(config_file, "w") as file:
                        json.dump(config_data, file, indent=4)
                    print("Successfully aquired. N0 = "+str(N0))
            
            break

        elif response == 'n':
            print("Calibrating site using Shcron et al. (2017) method...")
            calib_data_filepath = "/calibration_data"
            if os.path.exists(calib_data_filepath):
                print("Calibration data detected. Calibrating site...")
                n0_calibration(country=config_data['metadata']['country'], 
                               sitenum=config_data['metadata']['sitenum'], 
                               defineaccuracy=0.01, 
                               calib_start_time="09:00:00",
                               calib_end_time="17:00:00",
                               config_data=config_data)
                N0 = config_data['metadata']['n0']
        
            else:
                print("Calibration data NOT detected.")
                calib_data_filepath = input(f"{"Please input calibration data filepath"}: ").strip('"') 
                try:
                    os.path.exists(calib_data_filepath) #does the users filepath exist
                    print("Calibration data detected. Calibrating site... ")
                    n0_calibration(country=config_data['metadata']['country'], 
                               sitenum=config_data['metadata']['sitenum'], 
                               defineaccuracy=0.01, 
                               calib_start_time="09:00:00",
                               calib_end_time="17:00:00",
                               config_data=config_data)
                    N0 = config_data['metadata']['n0']
                    #save calibration data filepath to config file
                    config_data['filepaths']['calib_data_filepath'] = calib_data_filepath
                    with open(config_file, "w") as file:
                        json.dump(config_data, file, indent=4)

                except FileNotFoundError:
                    print("Calibration filepath invalid. Please check the input is correct.")

            break
        
        else:
            print("Invalid input. Please enter 'y' or 'n'. ")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (6) QUALITY ANALYSIS ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #quality control checks applied to level 1 df. See 'flag_and_remove' function for more detail on what is defined as eroneous data
    print("Performing quality control checks on corrected neutron counts...")
    qa_data = flag_and_remove(df=level_1_data, N0=N0, 
               country=str(config_data['metadata']['country']), 
               sitenum=str(config_data['metadata']['sitenum']), config_data=config_data)

    #outputting quality controlled df
    qa_data.to_csv(wd+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_qa.txt",   
           header=True, index=False, sep="\t", mode='w')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (7) SOIL MOISTURE ESTIMATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    print("Ready to calculate soil moisture")
    
    while True:
        response = input(f"{"Do you want to estimate soil moisture using the 'desilets' [standard] method or the 'kohli' method? (d/k): "}").strip().lower()
        
        if response == 'd':
            print("Estimating soil moisture using the desilets method...")
            final_sm_data = thetaprocess(df=qa_data, country=config_data['metadata']['country'],
                     sitenum=config_data['metadata']['sitenum'], N0=N0, theta_method="desilets", config_data=config_data)
            final_sm_data.to_csv(wd+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_final.txt", 
                    header=True, index=False, sep="\t")
            print("Done.")
            break
        
        elif response == 'k':
            print("Estimating soil moisture using the kohli method...")
            final_sm_data = thetaprocess(df=qa_data, country=config_data['metadata']['country'],
                     sitenum=config_data['metadata']['sitenum'], N0=N0, theta_method="kohli", config_data=config_data)
            final_sm_data.to_csv(wd+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_final.txt", 
                    header=True, index=False, sep="\t")
            print("Done.")
            break
        
        else:
            print("Invalid input. Please enter 'd' for desilets method or 'k' for kohli. If unsure the desilets method is recommended.")

    ### END OF DATA PROCESSING ###
    print("Data processing complete! Go to 'outputs' folder to view processed data.")



    
#
process_data()

