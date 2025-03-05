
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

#crspy-lite processing functions
import processing_functions as pf

#crspy-lite visualisation functions
import visualisation_functions as vf


# Check user Python version
python_version = (3, 7)  # tuple of (major, minor) version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = "Module only runs on python version >= %s" % python_version_str
    raise Exception(msg)


def wrapper_beginner():

    '''Full process wrapper of the data processing for crspy-lite. 
        The wrapper is divided into 7 sections which correspond to the same 7 processing sections in "processing_functions.py" 
        This beginner version contains prompts in the terminal for users to follow along at each stage of the data processing.
    
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (1) SETTING UP WORKING DIRECTORY AND CONFIG FILE ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #file setup
    print("File setup...")
    while True:
        response = input(f"Would you like the default directory (for folder creation and outputs) to be the working directory (y/n)? :")

        if response == 'y':
            wd = os.getcwd()
            default_dir = wd
            pf.initial(default_dir)
            print("Folders created.")
            break

        elif response == 'n':
            default_dir = input(f"Please input the filepath of an alternative directory: ").strip('"')
            pf.initial(default_dir)
            print("Folders created.")
            break

        else: 
            print("Invalid input. Please enter 'y' or 'n'. ")
   
    #config file creation
    config_file = os.path.join(default_dir, "inputs/config.json")
    
    if not os.path.exists(config_file):
       print("Creating config file....")
       pf.create_config_file(default_dir)
       print("Done")

    else:
        print("config.json already exists. Skipping config file creation.")

    print("Variables can be edited manually by opening the config.json file and saving your changes.")
    
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

    #read in input data (known as "raw" data).
    raw_data_file_path = config_data['filepaths']['raw_data_filepath']

    try:
        os.path.exists(raw_data_file_path)    
        
    except FileNotFoundError:
        print("Input data file not found. Please check the path is inputted correctly into the config.json file.")

    raw_data = pf.read_file(file_path=raw_data_file_path) #function that reads in the input data, and automatically adjusts for different delimiters.

    #tidy columns headers function
    pf.tidy_headers(df=raw_data) #remove whitespace present in cols

    #optional resampling function 
    while True:
        response = input("Does the input data need to be resampled to hourly time intevals? (y/n) [y is default]: ").strip().lower()
        
        if response == 'y':
            print("Resampling dataframe...")
            raw_data = pf.resample_to_hourly(df=raw_data, config_data=config_data, default_dir=default_dir)  #rewrite the raw_data as the new resampled df
            print("Done.")
            break
        
        elif response == 'n':
            print("Skipping resampling function.")
            break
        
        else:
            print("Invalid input. Please enter 'y' or 'n'. ")

    #data tidying function
    pf.prepare_data(df=raw_data, config_data=config_data, default_dir=config_data['filepaths']['default_dir'])
    tidy_data = raw_data #assign manipulated df a new name: 'tidy_data'

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

    #Assigning start & end dates.
    startdate = Jung_df_times['DATE'].iloc[1]
    enddate = Jung_df_times['DATE'].iloc[-1]

    #calling nmdb_get function to get nmdb data.
    pf.nmdb_get(startdate, enddate, station="JUNG", default_dir=config_data['filepaths']['default_dir'])

    #reading nmdb data.
    nmdb_data_path = default_dir + "/outputs/data/nmdb_station_counts.txt"
    nmdb_data = pd.read_csv(nmdb_data_path, header=None, delimiter = ';')
    nmdb_data.columns = ['DT', 'N_COUNT'] #adding column headings

    #adding nmdb counts to tidied df.
    tidy_data = pd.merge(nmdb_data, tidy_data, on='DT', how='inner')

    #moving nmdb counts column to the end of the df.
    column_to_move = 'N_COUNT'
    moved_column = tidy_data.pop(column_to_move)
    tidy_data[column_to_move] = moved_column

    #~~~~~~~ CUT-OFF RIGIDITY CALC~~~~~~~#

    #check to see if the user inputted a value. Otherwise calculate using betacoeff function.
    if config_data['metadata']['rc'] is None:
        
        print("Calculating cutoff rigidity for site...")

        Rc = pf.rc_retrieval(latitude=config_data['metadata']['latitude'], longitude=config_data['metadata']['longitude'])
        
        #adding rc to metadata.
        config_data['metadata']['rc'] = str(Rc)
        with open(config_file, "w") as file:
            json.dump(config_data, file, indent=4)

        print("Done.")

    else:
        Rc = float(config_data['metadata']['rc'])
        print("Cut off rigidity aquired from metadata.")


    #~~~~~~~ BETA COEFFICIENT CALC ~~~~~~~#
    
    #check to see if the user inputted a value. Otherwise calculate using betacoeff function.
    if config_data['metadata']['beta_coeff'] is None:
        
        print("Calculating beta coefficient...")

        beta_coeff, x0 = pf.betacoeff(lat=float(config_data['metadata']['latitude']),
                            elev=float(config_data['metadata']['elev']),
                            Rc= Rc,
                            
        )

        #adding beta_coeff to metadata.
        config_data['metadata']['beta_coeff'] = beta_coeff
        with open(config_file, "w") as file:
            json.dump(config_data, file, indent=4)

        print("Done")

    else:
        beta_coeff = config_data['metadata']['beta_coeff']
        print("Beta coefficient aquired from metadata.")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (4) CORRECTION FACTOR FUNCTIONS ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    print("Correcting neutron counts...")

    #~~~~~~~ PRESSURE CORRECTION ~~~~~~~#

    f_p = pf.pressure_correction(press = tidy_data['PRESS'] , beta_coeff = float(beta_coeff), 
                  p0 = float(config_data['metadata']['reference_press']))
    
    #~~~~~~~ HUMIDITY CORRECTION ~~~~~~~#

    saturation_vapour_press = pf.es(temp=tidy_data['TEMP']) #in Pa
    actual_vapour_press = pf.ea(saturation_vapour_press, RH = tidy_data['E_RH'])  #in Pa 
    absolute_humidity = pf.pv(actual_vapour_press, temp=tidy_data['TEMP'])  #in g/m3
    
    f_h = pf.humidity_correction(pv=absolute_humidity, pv0=float(config_data['general']['pv0']))

   #~~~~~~~ INCOMING NEUTRON INTENSITY CORRECTION ~~~~~~~#

    while True:
        response = input("Would you like to correct for incoming neutron intensity using the 'McJannet2023' [default] method or the'Hawdon2014' method? (m/h) [m is default]: ").strip().lower()

        if response == 'm':

            print("Correcting for neutron intensity using 'McJannet' method...")

            #First need to find some variables for McJannet correction
            site_atmospheric_depth = pf.atmospheric_depth(elev=float(config_data['metadata']['elev']), latitude=float(config_data['metadata']['latitude']))
            ref_atmospheric_depth = pf.atmospheric_depth(elev=3570, latitude=46.55)    #currently set for Jungfraujoch NMDB station
            reference_Rc = 4.49  #currently set for Jungfraujoch NMDB station
            tau, K = pf.location_factor(site_atmospheric_depth, Rc, ref_atmospheric_depth, reference_Rc)

            #calculate correction
            f_i = pf.intensity_correction_McJannet2023(station_ref = float(config_data['general']['jung_ref']),
                                                            station_count = tidy_data['N_COUNT'],
                                                            tau=tau)
        
            print("Done")
            break
       
        elif response == 'h':

            print("Correcting for neutron intensity using 'Hawdon2024' method...")
           
            RcCorr_result = pf.RcCorr(Rc, Rc_ref=4.49) #Rc_ref set for Jungfraujoch station
            f_i = pf.intensity_correction_Hawdon2014(station_ref=float(config_data['general']['jung_ref']), 
                                                    station_count=tidy_data['N_COUNT'],
                                                    RcCorr=RcCorr_result) 

            print("Done")
            break

        else:
            print("Invalid input. Please enter 'm' for McJannet2023 method or 'h' for Hawdon2014. If unsure the McJannet method is recommended.") 

    #~~~~~~~ BIOMASS CORRECTION ~~~~~~~#
    f_v = 1 #currently no biomass correction available in crspy-lite

    #applying corrections
    mod_corr_result = pf.mod_corr(f_p, f_h, f_i, f_v, mod=tidy_data['MOD'])
    print("Done")

    #outputing corrected data
    corrected_data = tidy_data #take tidy dataframe and add new columns to it.

    #adding corrected nuetron counts column and correction factors to df
    corrected_data['MOD_CORR'] = mod_corr_result.round(0)  #nuetron counts must be integers
    corrected_data['F_PRESSURE'] = round(f_p, (4))
    corrected_data['F_HUMIDITY'] = round(f_h, (4))
    corrected_data['F_INTENSITY'] = round(f_i, (4))
    corrected_data['F_VEGETATION'] = round(f_v, (4))

    #~~~~~~~~~~~~~~~~~~~~~#
    #~~ (5) CALIBRATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~#

    while True:
        response = input(f"Do you want to estimate soil moisture using the 'desilets' [default] method or the 'kohli' method? (d/k): ").strip().lower()

        if response == 'd':
            sm_calc_method = 'desilets'
            break

        elif response == 'k':
            sm_calc_method = 'kohli'
            break

        else:
            KeyError("Invalid input. Please enter 'd' for desilets method or 'k' for kohli. If unsure the desilets method is recommended.")

    #adding sm_calc_method to metadata.
    config_data['general']['sm_calc_method'] = str(sm_calc_method)
    with open(config_file, "w") as file:
        json.dump(config_data, file, indent=4)


    print("Sensor calibration (N0)...")
    
    while True:
        response = input("Do you have an N0 number? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Taking N0 from metadata...")
            #check if n0 exists in metadata
            try: 
                if config_data['metadata']['n0'] is None:
                    N0 = input(f"Please input N0 number: ").strip()
                    config_data['metadata']['n0'] = N0
                    #save n0 input to config file
                    with open(config_file, "w") as file:
                        json.dump(config_data, file, indent=4)
                    print("Successfully aquired. N0 = "+str(N0))
        
                else:
                    N0 = config_data['metadata']['n0']
                    print("Successfully aquired. N0 = "+str(N0))
            except:
                    N0 = input(f"Please input N0 number: ").strip()
                    config_data['metadata']['n0'] = N0
                    #save n0 input to config file
                    with open(config_file, "w") as file:
                        json.dump(config_data, file, indent=4)
                    print("Successfully aquired. N0 = "+str(N0))
            
            break

        elif response == 'n':
            print("Calibrating site using Shcron et al. (2017) method...")
            calib_data_filepath = "/inputs/calibration_data.csv"    #default filepath for calibration data
            if os.path.exists(calib_data_filepath):
                print("Calibration data detected. Calibrating site...")
                N0 = pf.n0_calibration(corrected_data=corrected_data, country=config_data['metadata']['country'], 
                               sitenum=config_data['metadata']['sitenum'], 
                               defineaccuracy=0.01, 
                               calib_start_time="09:00:00",
                               calib_end_time="17:00:00",
                               config_data=config_data,
                               calib_data_filepath=calib_data_filepath,
                               default_dir=config_data['filepaths']['default_dir'],
                               sm_calc_method=config_data['general']['sm_calc_method'])
        
                #save calibration data filepath to config file
                config_data['filepaths']['calib_data_filepath'] = calib_data_filepath
                with open(config_file, "w") as file:
                    json.dump(config_data, file, indent=4)
        
            else:
                print("Calibration data NOT detected.")
                calib_data_filepath = input(f"Please input calibration data filepath: ").strip('"') 
                try:
                    os.path.exists(calib_data_filepath) #does the users filepath exist
                    print("Calibration data detected. Calibrating site... ")
                    N0 = pf.n0_calibration(corrected_data=corrected_data, country=config_data['metadata']['country'], 
                               sitenum=config_data['metadata']['sitenum'], 
                               defineaccuracy=0.01, 
                               calib_start_time="09:00:00",
                               calib_end_time="17:00:00",
                               config_data=config_data,
                               calib_data_filepath=calib_data_filepath, 
                               default_dir=config_data['filepaths']['default_dir'],
                               sm_calc_method=config_data['general']['sm_calc_method'])

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
    qa_data = pf.flag_and_remove(df=corrected_data, N0=int(N0), 
               country=str(config_data['metadata']['country']), 
               sitenum=str(config_data['metadata']['sitenum']), config_data=config_data)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (7) SOIL MOISTURE ESTIMATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    print("Ready to calculate soil moisture.")
    print("Estimating soil moisture with the "+str(sm_calc_method)+" method...")

    final_sm_data = pf.thetaprocess(df=qa_data, country=config_data['metadata']['country'],
                sitenum=config_data['metadata']['sitenum'], N0=N0, sm_calc_method=config_data['general']['sm_calc_method'], 
                config_data=config_data, default_dir=config_data['filepaths']['default_dir'])
   
    final_sm_data.to_csv(default_dir+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_processed.txt", 
            header=True, index=False, sep="\t")
    print("Done.")
         

    ### END OF DATA PROCESSING ###
    print("Data processing complete! Go to 'outputs' folder to view processed data.")


def wrapper_general(intensity_correction_method, sm_calc_method):

    '''Full process wrapper of the data processing for crspy-lite. 
        The wrapper is divided into 7 sections which correspond to the same 7 processing sections in "processing_functions.py" 
        This wrapper is recommended for more experienced users. Functions can be activated/deactivated depending on user needs.
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (1) SETTING UP WORKING DIRECTORY AND CONFIG FILE ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #establish default directory
    wd = os.getcwd()
    default_dir = r"C:\Users\joedw\OneDrive - University of Bristol\Documents\Civil Engineering\Y4\RP4\data\IAEA\Austria\Petzenkirchen\test1" #wd #default_dir defaulted to the working directory. Edit here for a different directory.

    #create folder structure from the default_dir
    pf.initial(default_dir)

    #config file creation
    config_file = os.path.join(default_dir, "inputs/config.json")
    
    if not os.path.exists(config_file): #check to see if the file already exists
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

    ## User must edit this before proceeding - but how??

    #save user preferences to config file
    config_data['general']['intensity_correction_method'] = intensity_correction_method
    config_data['general']['sm_calc_method'] = sm_calc_method
    with open(config_file, "w") as file:
            json.dump(config_data, file, indent=4)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (2) RAW DATA TIDYING ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #read in input data (known as "raw" data).
    raw_data_file_path = config_data['filepaths']['raw_data_filepath']
    raw_data_url = config_data['filepaths']['raw_data_url']

    if raw_data_file_path == None:
        
        try:
            raw_data = pf.retrieve_data_url(url=raw_data_url)
        except:
            print("url data retrieval unsuccessful")
        try:
            raw_data = pf.IAEA_data_reformat(df=raw_data)
        except:
            print("url reformatting data unsuccessful")

    else:
        try:
            os.path.exists(raw_data_file_path) #is the path valid? 
            raw_data = pf.read_file(file_path=raw_data_file_path)   #read in raw data with in built function.
        except FileNotFoundError:
            print("Input data file not found. Please check the path is inputted correctly into the config.json file.")

    #tidy input data
    pf.tidy_headers(df=raw_data) #remove whitespace present in cols
    
    raw_data = pf.resample_to_hourly(df=raw_data, config_data=config_data, default_dir=default_dir) #resamples the input data to hourly intevals, #rewrites the raw_data as the new resampled df

    pf.prepare_data(df=raw_data, config_data=config_data, default_dir=config_data['filepaths']['default_dir']) #fixes input data to counts on the hour.
    tidy_data = raw_data #assign manipulated df a new name: 'tidy_data'

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
    #check to see if the user inputted a value in the config file. Otherwise calculate using betacoeff function.
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

        beta_coeff, x0 = pf.betacoeff(lat=float(config_data['metadata']['latitude']),elev=float(config_data['metadata']['elev']),Rc=Rc)
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
    if intensity_correction_method == 'mcjannet':
        ##McJannet correction method (reccommended)
        #calculate variables
        site_atmospheric_depth = pf.atmospheric_depth(elev=float(config_data['metadata']['elev']), latitude=float(config_data['metadata']['latitude']))
        ref_atmospheric_depth = pf.atmospheric_depth(elev=3570, latitude=46.55)    #elev & latitude currently set for Jungfraujoch NMDB station
        reference_Rc = 4.49  #currently set for Jungfraujoch NMDB station
        tau, K = pf.location_factor(site_atmospheric_depth, Rc, ref_atmospheric_depth, reference_Rc)
        #calculate correction
        f_i = pf.intensity_correction_McJannet2023(station_ref = float(config_data['general']['jung_ref']),station_count = tidy_data['N_COUNT'],tau=tau)
    
    elif intensity_correction_method == 'hawdon': 
        ##Hawdon correction method    
        RcCorr_result = pf.RcCorr(Rc, Rc_ref=4.49) #Rc_ref set for Jungfraujoch station
        f_i = pf.intensity_correction_Hawdon2014(station_ref=float(config_data['general']['jung_ref']), station_count=tidy_data['N_COUNT'],RcCorr=RcCorr_result)
    
    else:
        print("Invalid input. Available methods are 'mcjannet' [reccommended] and 'hawdon'.") 
    
    #~~~~~~~ BIOMASS CORRECTION ~~~~~~~#
    f_v = 1 #currently no biomass correction available in crspy-lite

    #~~~~~~~ APPLY CORRECTIONS ~~~~~~~~#
    #f_p = f_p.fillna(1)  #change back!!!

    mod_corr_result = pf.mod_corr(f_p, f_h, f_i, f_v, mod=tidy_data['MOD'])

    corrected_data = tidy_data #define new df, called corrected_data
    #adding new cols to df
    corrected_data['MOD_CORR'] = mod_corr_result.round(0)  #nuetron counts must be integers
    corrected_data['F_PRESSURE'] = round(f_p, (4))
    corrected_data['F_HUMIDITY'] = round(f_h, (4))
    corrected_data['F_INTENSITY'] = round(f_i, (4))
    corrected_data['F_VEGETATION'] = round(f_v, (4))

    #~~~~~~~~~~~~~~~~~~~~~#
    #~~ (5) CALIBRATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~#
    #check to see if the user inputted a value in the config file. Otherwise calculate using n0_calibration function.
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
    #quality control checks applied to corrected df. See 'flag_and_remove' function for more detail on what is defined as eroneous data
    print("Performing quality control checks on corrected neutron counts...")
    qa_data = pf.flag_and_remove(df=corrected_data, N0=int(N0), country=str(config_data['metadata']['country']), 
                                 sitenum=str(config_data['metadata']['sitenum']), config_data=config_data)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ (7) SOIL MOISTURE ESTIMATION ~~# 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    processed_sm_data = pf.thetaprocess(df=qa_data, country=config_data['metadata']['country'],
                sitenum=config_data['metadata']['sitenum'], N0=N0, sm_calc_method=config_data['general']['sm_calc_method'], 
                config_data=config_data, default_dir=config_data['filepaths']['default_dir'])
   
    processed_sm_data.to_csv(default_dir+"/outputs/data/"+config_data['metadata']['country']+"_SITE_"+config_data['metadata']['sitenum']+"_processed.txt", 
            header=True, index=False, sep="\t")
   
    #~~~ END OF DATA PROCESSING! ~~~#
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ POST-PROCESSING: DATA VISUALISATION ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~ STANDARD PLOTS ~~~#
    #vf.standard_plots(df=processed_sm_data, config_data=config_data)



    return


#~~~~~ CALLING WRAPPER ~~~~~~#
print("Welcome to crspy-lite!")
print("crspy-lite is a python tool for single site CRNS soil moisture estimation and site visualisation.")

#wrapper_beginner()  #defaulted to the beginner wrapper. 

wrapper_general(intensity_correction_method='hawdon', sm_calc_method='desilets') #uncomment to use the general wrapper. 

'''
processed_data_path = r"C:\Users\joedw\OneDrive - University of Bristol\Documents\Civil Engineering\Y4\RP4\data\IAEA\Austria\Petzenkirchen\test1\outputs\data\AUSTRIA_SITE_1_processed.txt"
processed_data = pf.read_file(file_path=processed_data_path)
config_file = r"C:\Users\joedw\OneDrive - University of Bristol\Documents\Civil Engineering\Y4\RP4\data\IAEA\Austria\Petzenkirchen\test1\inputs\config.json"
with open(config_file, "r") as file:
        config_data = json.load(file)

vf.standard_plots(df=processed_data, config_data=config_data)
'''


'''while True:
        
    beginner = input(f"Would you like to run the beginner version of crspy-lite (this includes user prompts and guides you through each processing step? (y/n):")

    if beginner == 'y':
        print("Follow the steps along in the terminal. If you are unsure about what to do select the [default] optional or go to ... to learn more.")
        print("Beginning data processing...")
        wrapper_beginner()
        break

    elif beginner == 'n':
        print("Beginning data processing...")
        wrapper_advanced()
        break

    else: print("Invalid input. Please enter 'y' or 'n'. ")'''




