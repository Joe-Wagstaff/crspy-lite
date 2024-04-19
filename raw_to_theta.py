# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:16:00 2024

@author: joedw
"""

#import modules
import crspy 
import pandas as pd
from configparser import RawConfigParser
import os
import numpy as np
import datetime
import urllib
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import re
import math

#####################################
### RAW DATA TIDYING AND CLEANING ###
#####################################

#import raw data and read config files
current_directory = os.getcwd() 
wd = current_directory
nld = RawConfigParser()
nld.read(wd + '\config_files\config.ini')
raw_data_file_path = nld['filepaths']['raw_data_filepath']
raw_data = pd.read_csv(raw_data_file_path, header=0, delimiter='\t')

###########################
### TIDY DATA FUNCTIONS ###
###########################

def tidy_headers(df):

    """tidy_headers removes any spaces in the headers for the columns in the raw data file. 
    
    Parameters
    ----------
    df : dataframe  
        dataframe of raw data
    
    """
    df.columns = df.columns.str.strip()
    
    return df

def press_column(df):
    
    """press_column selects what pressure data the rest of the data uses. 
    Most CRNS sites should have a PRESS1 and a PRESS2 columns. 
    
    Parameters
    ----------
    df : dataframe  
        dataframe of raw data
    
    """

    if 'PRESS2' in df:
        
        try:
            df.drop(columns=['PRESS'], inplace=True)
        except:
            pass

        try:
            df.rename(columns={'PRESS2': 'PRESS'}, inplace=True)
            print("Using PRESS2 for pressure data")
        
        except:
            print("No PRESS2 column in df detected")
            pass
        
        try:
            df.drop(columns=['PRESS1'], inplace=True)
        except:
            pass
        
    elif 'PRESS1' in df:
        
        try:
            df.drop(columns=['PRESS'], inplace=True)
        except:
            pass
        
        try:
            df.rename(columns={'PRESS1': 'PRESS'}, inplace=True)
            print("Using PRESS1 for pressure data")
        
        except:
            print("No PRESS1 column in df detected")
            pass
    
    else:
        try:
            df['PRESS'] = df['PRESS']
            print("Using PRESS for pressure data")
        
        except:
            print("No PRESS column in df detected. Please check inputed raw data")
            pass
    
    return df
    
def dropemptycols(colstocheck, df, nld=nld):
    """dropemptycols drop any columns that are empty (i.e. all -999)

    Parameters
    ----------
    colstocheck : str
        string of column title to check
    df : dataframe  
        dataframe to check

    """
    nld=nld['general']
    for i in range(len(colstocheck)):
        col = colstocheck[i]
        if col in df:
            try:
                if df[col].mean() == int(nld['noval']):
                    df = df.drop([col], axis=1)
                else:
                    pass
            except:
                pass
        else:
            pass
    return df

def prepare_data(df, intentype=None, nld=nld):
    """prepare_data provided with the location of the raw data it will prepare the data.

    Steps include: 
        - Find the country and sitenum from text file title
        - Fix time to be on the hour rather than variable
        - Final tidy of columns

    Parameters
    ----------
    
    intentype : str, optional
        can be set to nearestGV if using the alternative method, by default None
    nld : dictionary
        nld should be defined in the main script (from name_list import nld), this will be the name_list.py dictionary. 
        This will store variables such as the wd and other global vars

    """

    # Remove leading white space - present in some SD card data
    df['TIME'] = df['TIME'].str.lstrip()
    # Ensure using dashes as Excel tends to convert to /s
    if '/' in df['TIME'][0]:
        df['TIME'] = df['TIME'].str.replace('/', '-')
    else:
        pass
    tmp = df['TIME'].str.split(" ", n=1, expand=True)
    df['DATE'] = tmp[0]
    df['TIME'] = tmp[1]
    new = df["TIME"].str.split(":", n=2, expand=True)
    df['DT'] = pd.to_datetime(df['DATE'])
    my_time = df['DT']
    time = pd.DataFrame()

    tseries = []
    for i in range(len(new)):  # Loop through with loc to append the hours onto a DateTime object
        # time = the datetime plus the hours and mins in time delta
        time = my_time[i] + datetime.timedelta(
            hours=int(new.loc[i, 0]), minutes=int(new.loc[i, 1]))
        tseries.append(time)  # Append onto tseries

    df['DT'] = tseries      # replace DT with tseries which now has time as well

    # Collect dates here to prevent issues with nan values after mastertime - for use in Jungrafujoch process
    startdate = str(df['DATE'].iloc[0])
    enddate = str(df['DATE'].iloc[-1])

    ### The Master Time ###
    """
    Master Time creates a time series from first data point to last data point with
    every hour created. This is to remedy the gaps in the data.
    
    DateTime is standardised to be on the hour (using floor). This can create issues
    with "duplicated" data points, usually when errors in logging have created data
    every half hour instead of every hour, for example. 
    
    The best way to address this currently is to retain the first instance of the 
    duplicate and discard the second.
    """
    print("Master Time process...")
    df['DT'] = df.DT.dt.floor(freq='H')
    dtcol = df['DT']
    df.drop(labels=['DT'], axis=1, inplace=True)  # Move DT to first col
    df.insert(0, 'DT', dtcol)

    df['dupes'] = df.duplicated(subset="DT")
    # Add a save for dupes here - need to test a selection of sites to see
    # whether dupes are the same.
    df.to_csv("outputs/data/" + nld['metadata']['country']+"_SITE_" + nld['metadata']['sitenum'] + "_DUPES.txt",
              header=True, index=False, sep="\t",  mode='w')
    df = df.drop(df[df.dupes == True].index)
    df = df.set_index(df.DT)
    if df.DATE.iloc[0] > df.DATE.iloc[-1]:
        raise Exception(
            "The dates are the wrong way around, see crspy.flipall() to fix it")

    idx = pd.date_range(
        df.DATE.iloc[0], df.DATE.iloc[-1], freq='1H', closed='left')
    df = df.reindex(idx, fill_value=np.nan)
    df = df.replace(np.nan, int(nld['general']['noval'])) # add replace to make checks on whole cols later

    df['DT'] = df.index
    print("Done")
    
    ### The Final Table ###
   
    print("Writing out table...")
    
    def movecol(col, location):
        """
        Move columns to a specific position.
        """
        tmp = df[col]
        df.drop(labels=[col], axis=1, inplace=True)  # Move DT to first col
        df.insert(location, col, tmp)

    # Move below columns as like them at the front
    movecol("MOD", 1)
    try:
        movecol("UNMOD", 2)
    except:
        df.insert(2, "UNMOD", np.nan)  # filler for if UNMOD is unavailble
    movecol("PRESS", 3)
    #movecol("TEMP", 4)
    
    df.drop(labels=['dupes'],
            axis=1, inplace=True)  # not required after here
    try:
        df.drop(labels=['fsol'], axis=1, inplace=True)
    except:
        pass
    # Add list of columns that some sites wont have data on - removes them if empty
    df = dropemptycols(df.columns.tolist(), df)
    df = df.round(3)
    df = df.replace(np.nan, int(nld['general']['noval']))
    # SD card data had some 0 values - should be nan
    df['MOD'] = df['MOD'].replace(0, int(nld['general']['noval']))
    # Change Order

    # Save Tidy data
    #df.to_csv(nld.get('config','default_dir') + "/data/crns_data/tidy/"+ nld.get('config','country') +"_SITE_" + nld.get('config','sitenum') +"_TIDY.txt",
       #       header=True, index=False, sep="\t",  mode='w')
    print("Done")
    
### CALLING TIDY DATA FUNCTIONS  ###

tidy_headers(df=raw_data)
press_column(df=raw_data)
dropemptycols(colstocheck='UNMOD', df=raw_data)
prepare_data(df=raw_data)

### OUTPUT TIDY DATA FILE  ###

output_tidy_path = wd+"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['sitenum']+"_tidy.txt"
raw_data.to_csv(output_tidy_path, index=False, header=True, sep='\t')

######################
### IMPORTING DATA ###
######################

#~~~~~~~~~ NMDB DATA (JUNG STATION) ~~~~~~~~~~#

def nmdb_get(startdate, enddate, station="JUNG", nld=nld):
    """nmdb_get will collect data for Junfraujoch station that is required to calculate f_intensity.
    Returns a dictionary that can be used to fill in values to the main dataframe
    of each site.

    Parameters
    ----------
    startdate : datetime
        start date of desire data in format YYYY-mm-dd
            e.g 2015-10-01
    enddate : datetime
        end date of desired data in format YYY-mm-dd
    station : str, optional
        if using different station provide the value here (NMDB.eu shows alternatives), by default "JUNG"
    nld : dictionary
        nld should be defined in the main script (from name_list import nld), this will be the name_list.py dictionary. 
        This will store variables such as the wd and other global vars

    Returns
    -------
    dict
        dictionary of neutron count data from NMDB.eu
    """
    # split for use in url
    sy, sm, sd = str(startdate).split("-")
    ey, em, ed = str(enddate).split("-")

    # Collect html from request and extract table and write to dict
    url = "http://nest.nmdb.eu/draw_graph.php?formchk=1&stations[]={station}&tabchoice=1h&dtype=corr_for_efficiency&tresolution=60&force=1&yunits=0&date_choice=bydate&start_day={sd}&start_month={sm}&start_year={sy}&start_hour=0&start_min=0&end_day={ed}&end_month={em}&end_year={ey}&end_hour=23&end_min=59&output=ascii"
    url = url.format(station=station, sd=sd, sm=sm, sy=sy, ed=ed, em=em, ey=ey)
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, features="html.parser")
    pre = soup.find_all('pre')
    pre = pre[0].text
    pre = pre[pre.find('start_date_time'):]
    pre = pre.replace("start_date_time   1HCOR_E", "")
    f = open("outputs/data/nmdb_tmp.txt", "w")
    f.write(pre)
    f.close()
    df = open("outputs/data/nmdb_tmp.txt", "r")
    lines = df.readlines()
    df.close()
    lines = lines[1:]
    dfneut = pd.DataFrame(lines)
    dfneut = dfneut[0].str.split(";", n=2, expand=True)
    cols = ['DATE', 'COUNT']
    dfneut.columns = cols
    dates = pd.to_datetime(dfneut['DATE'])
    count = dfneut['COUNT']

    dfdict = dict(zip(dates, count))
    
tidy_data_path = wd+"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['sitenum']+"_tidy.txt"
tidy_data = pd.read_csv(tidy_data_path, header=0, delimiter='\t')

#creating copy of tidy data df that can be broken up to use for various parts of the JUNG process.
Jung_df_times = pd.DataFrame({})
Jung_df_times['DATE'] = tidy_data['DT'].str.split().str[0]
Jung_df_times['TIME'] = tidy_data['DT'].str.split().str[1]
Jung_df_times['DT'] = Jung_df_times['DATE'] + ' ' + Jung_df_times['TIME']
Jung_df_times.to_csv(wd+"/outputs/data/Jung_df_times.txt", sep='\t', index=False)

startdate = Jung_df_times['DATE'].iloc[1]
enddate = Jung_df_times['DATE'].iloc[-1]

#calling nmdb_get function - imports data from nmdb database for start and end date of tidy_data df.
nmdb_get(startdate, enddate)

#reading nmdb data
nmdb_data_path = wd+"/outputs/data/nmdb_tmp.txt"
nmdb_data = pd.read_csv(nmdb_data_path, header=None, delimiter = ';')
nmdb_data.columns = ['DT', 'N_COUNT'] #adding column headings

#import tidy_data file and add nmdb counts to df
tidy_data_path = wd+"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['sitenum']+"_tidy.txt"
tidy_data = pd.read_csv (tidy_data_path, header=0, delimiter='\t')
tidy_data = pd.merge(nmdb_data, tidy_data, on='DT', how='inner')
#moving nmdb counts column to the end of the df
column_to_move = 'N_COUNT'
moved_column = tidy_data.pop(column_to_move)
tidy_data[column_to_move] = moved_column

#~~~~~~~~~ CUT-OFF RIGIDITY ~~~~~~~~~~#

def rc_retrieval(nld=nld):
    
    ''' Cut off rigidity (Rc) taken from metadata (if present) or otherswise webscraped from crnslab.org. 
        This is done using selenium which requires a web driver to interface with the chosen browser.
        Here it's coded for google chrome so ensure the relevent chromedriver (for your edition of chrome) is installed and saved in the same file as this python code.
        Otherswise please visit crnslab.org and retrieve an Rc value manually, then edit the namelist for it to be applied in the code. 
        Ensure units for Rc are GV.
       
        Parameters
        ----------
        
        nld : dictionary
            nld should be defined in the main script (from name_list import nld), this will be the name_list.py dictionary. 
            This will store variables such as the wd and other global vars
       
        Returns
        -------
       
         Rc: float
             cut off rigidity (GV)
       
        '''
    def Rc_webscrape(nld=nld):
        
        ''' Webscrapes cut off rigidity from crnslab.org. Read description of rc_retrieval fnc for more info.
        
        Parameters
        ----------
        
        nld : dictionary
            nld should be defined in the main script (from name_list import nld), this will be the name_list.py dictionary. 
            This will store variables such as the wd and other global vars
            
        Returns
        -------
       
         Rc: float
             cut off rigidity (GV)
             
        '''
        
        latitude = float(nld['metadata']['latitude'])
        if not -90 <= latitude <= 90:
            raise ValueError('Latitude value from metadata invalid to calculate Rc from crnslab.org. Must be between -90 and 90')
        
        longitude = float(nld['metadata']['longitude'])
        if not 0 <= longitude <= 360:
            raise ValueError('Longitude value from metadata invalid to calculate Rc from crnslab.org. Must be between 0 and 360')
      
        install_date = str(nld['metadata']['install_date'])
        day_month_year = install_date.split('/')
        year = int(day_month_year[2])
        if not 1900 <= year <= 2025:
            raise ValueError('Year of installation from metadata invalid to calculate Rc from crnslab.org. Must be between 1900 and 2025')
        
        url = "https://crnslab.org/util/rigidity.php"
        driver = webdriver.Chrome()
        driver.get(url)
        
        # Finding the input elements.
        latitude_input = driver.find_element(By.XPATH,'//*[@id="body"]/table/tbody/tr[1]/td[2]/div/input')  
        longitude_input = driver.find_element(By.XPATH,'//*[@id="body"]/table/tbody/tr[2]/td[2]/div/input')
        year_input = driver.find_element(By.XPATH,'//*[@id="body"]/table/tbody/tr[3]/td[2]/div/input')
       
        # Clearing any keys present in elements.
        latitude_input.clear()
        longitude_input.clear()
        year_input.clear()
        
        # Setting values.
        latitude_input.send_keys(latitude)
        longitude_input.send_keys(longitude)
        year_input.send_keys(year)
    
        # Finding calculate button element and perform a click.
        calculate_button = driver.find_element(By.XPATH,'//*[@id="body"]/div[2]/input')  
        calculate_button.click()
        
        # Waiting for the calculation to be completed (you may need to adjust the waiting time).
        driver.implicitly_wait(10)

        print('Retrieving Rc value...')
    
        # Copying all text in the results window.
        driver.switch_to.window(driver.window_handles[-1])
        output_text = driver.find_element(By.XPATH, '//*[@id="header"]').text
    
        # Closing the browser`
        driver.quit()
        
        # Writing the results into a texfile.
        f1 = open("outputs/data/rc_calc.txt", "w")
        f1.write(output_text)
        f1.close()
       
        #Locate Rc output in textfile and import into script.
        def find_rc_value(filename):
            
            ''' Locates the value of Rc in the textfile defined previously as the ouptut calcs from crnslab.org.'''
            
            try:
                # Open the text file in read mode with a different encoding
                with open(filename, 'r', encoding='latin-1') as file:
                    # Iterate through each line in the file
                    for line_number, line in enumerate(file, start=1):
                        # Search for the pattern 'Rc = number'
                        match = re.search(r'Rc = (\S+)', line)
                        if match:
                            # Extract the numeric value after 'Rc = '
                               rc_value = match.group(1)
                               return f"{rc_value}"
    
                # If 'Rc =' is not found
                return "'Rc =' not found in the file."
    
            except FileNotFoundError:
                return f"File not found: {filename}" 
        Rc = float(find_rc_value(filename = "outputs/data/rc_calc.txt"))
        
        return Rc
   
    print("Cut off rigidity...")
    
    try:
        Rc = str(nld['metadata']['rc'])
    
        if Rc.isnumeric() or str(nld['metadata']['rc']).count('.') == 1:
            print("Cut off rigidity taken from metadata")
            return float(Rc)
        
        else:
            Rc = Rc_webscrape()
            print("Cut off rigidity calculated from cnslab.org")
            return Rc
            
    except:
        Rc = Rc_webscrape()
        print("Cut off rigidity calculated from cnslab.org")
        return Rc        
    
    print("Done")

#call rc retrieval function
Rc = rc_retrieval()



############################################
### DEFINING CORRECTION FACTOR FUNCTIONS ###
############################################

#pressure
def pressfact_B(press, B, p0):
    """pressfact_B corrects neutrons for pressure changes

    Parameters
    ----------
    press : float
        pressure (mb)
    B : float
        beta coefficient e.g. 0.007
    p0 : int
        reference pressure (mb)

    Returns
    -------
    float
        number to multiply neutron counts by to correct
    """
    return np.exp(B*(press-p0))

#humidity
def es(temp):
    """es Calculate saturation vapour pressure (hPA) using average temperature
    Can be used to calculate actual vapour pressure (hPA) if using dew point temperature    

    Parameters
    ----------
    temp : float  
        temperature (C)

    Returns
    -------
    float 
        saturation vapour pressure (hPA) 
    """
    return 6.112*np.exp((17.67*temp)/(243.5+temp))

def rh(t, td): #not using this fnc atm as processing data from sites with relative humidity sensors. #RH = rh(t = temp, td = tidy_data['E_RH'])
    """rh Use temperature (C) and dewpoint temperature (C) to calculate relative humidity (%)

    Parameters
    ----------
    t : float   
        temperature (C)
    td : float
        dewpoint temperature (C)

    Returns
    -------
    float
        relative humidity (%)
    """
    return 100*np.exp((17.625*243.04*(td-t))/((243.04+t)*(243.04+td)))

def ea(es, RH):
    """ea Uses saturation vapour pressure (es - converted to Pascals) with relative humidity (RH) to 
    produce actual vapour pressure (Pascals)

    Parameters
    ----------
    es : float
        saturation vapour pressure (Pascal)
    RH : float
        relative humidity (%)

    Returns
    -------
    float
        actual vapour pressure (Pascals)
    """
    return es * (RH/100)

def dew2vap(dt):    #not using this fnc atm
    """dew2vap  gives vapour pressure (kPA). Taken from Shuttleworth (2012) Eq 2.21 rearranged

    Parameters
    ----------
    dt : float 
        dewpoint temperature (C)

    Returns
    -------
    float
        vapour pressure (kPA)
    """
    return np.exp((0.0707*dt-0.49299)/(1+0.00421*dt))

def pv(ea, temp):
    """pv works out absolute humidity using temperature (C) and vapour pressure unit (Pascals)

    Parameters
    ----------
    ea : float  
        Vapour Presure (Pascals)
    temp : float
        temperature (C)

    Returns
    -------
    float
        absolute humidity (ouput as kg/m^3)
    """
    return ea/(461.5*(temp+273.15))

def humfact(pv, pv0):
    """humfact gives the factorial to multiply Neutron counts by

    Parameters 
    ----------
    pv : float 
        absolute humidity
    pv0 : float
        reference absolute humidity

    Returns
    -------
    float
        factor to multiply neutrons by
    """
    return (1+0.0054*(pv-pv0))


#cosmic ray intensity

def finten(jung_ref, jung_count):
    """finten correction for incoming neutron intensity

    Parameters
    ----------
    jung_ref : float
        reference neutron count at JUNG (usually 01/05/2011)
    jung_count : float
        count at time[0]

    """
    return jung_ref / jung_count

def RcCorr(Rc):
    """RcCorr takes cutoff ridgity of site (Rc) and gives the RcCorr value required to 
    account for the difference to Jungfraujoch.

    Parameters
    ----------
    Rc : float
        cutoff ridgidity of site

    """
    return -0.075*(Rc-4.49)+1


#biomass
def agb(agbval):
    """agb Works out the effect of above ground biomass on neutron counts.
    agbval units are kg per m^2 of above ground biomass. Taken from Baatz et al
    (2015)

    Parameters
    ----------
    agbval : float
        above ground biomass value (kg/m2)

    """
    
    return 1/(1-(0.009*agbval))
    
### CALLING CORRECTION FACTOR FUNCTIONS ###

#Defining temperature and relative humidity variables.
temp = tidy_data['E_TEM']  #taking external temp column (Celsius)
RH = tidy_data['E_RH']  #taking humidity column (%)

#pressure correction
f_p = pressfact_B(press = tidy_data['PRESS'] , B = float(nld['metadata']['beta_coeff']), 
                  p0 = float(nld['metadata']['reference_press']))

#humidity (not using dew to vap fnc) 
es = 100 * es(temp) #converting es from hPa to Pa
ea = ea(es, RH)  #in Pa 
pv = 1000 * pv(ea, temp)  #converting from kg/m3 to g/m3
f_h = humfact(pv, pv0 = float(nld['general']['pv0']))

#cosmic ray intensity
finten = finten(jung_ref = float(nld['general']['jung_ref']), jung_count = tidy_data['N_COUNT']) 
RcCorr = RcCorr(Rc)
f_i = ((finten - 1) * RcCorr) + 1

#biomass
agbweight = str(nld['metadata']['agbweight'])
if agbweight.isnumeric() or (agbweight.count('.') == 1):
    agbval = float(agbweight)
    f_v = agb(agbval)
    print(f_v)
else:
    f_v = 1
    print("No agbval from metadata. Treating f_v as equal to " + str(f_v))

### APPLYING CORRECTION FACTORS ###

def mod_corr(f_p, f_h, f_i, f_v, mod):
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
    print("Correcting nuetron counts...")
    return mod * f_p * f_h * f_i * f_v

mod_corr = mod_corr(f_p, f_h, f_i, f_v, mod = tidy_data['MOD'])
print("Done")

### OUTPUT CORRECTED/LEVEL 1 DATA  ###

print("Outputing level 1 data...")
level_1_data = tidy_data 
mod_corr = mod_corr.round(0)  #rounding corrected nuetron counts column to 0dp
#adding corrected nuetron counts column and correction factors to df
level_1_data['MOD_CORR'] = round(mod_corr)   
level_1_data['f_pressure'] = round(f_p, (4))
level_1_data['f_humidity'] = round(f_h, (4))
level_1_data['f_intensity'] = round(f_i, (4))
level_1_data['f_vegetation'] = round(f_v, (4))
output_level_1_path = wd+"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['sitenum']+"_level1.txt"
level_1_data.to_csv(output_level_1_path, index=False, header=True, sep='\t')
print("Done")

#########################
### QUALITY ANALYSIS  ###
#########################

def flag_and_remove(df, N0, country, sitenum, nld=nld):
    """flag_and_remove identifies data that should be flagged based on the following criteria and removes it:
    Flags:
        1 = fast neutron counts more than 20% difference to previous count
        2 = fast neutron counts less than the minimum count rate (default == 30%, can be set in namelist)
        3 = fast neutron counts more than n0
        4 = battery below 10v


    Parameters
    ----------
    df : dataframe
        dataframe of the CRNS data
    N0 : int
        N0 number
    country : str
        string of country e.g. "USA"
    sitenum : str
        string o sitenum e.g. "011"
    nld : dictionary
        nld should be defined in the main script (from name_list import nld), this will be the name_list.py dictionary. 
        This will store variables such as the wd and other global vars

    """
    print("~~~~~~~~~~~~~ Flagging and Removing ~~~~~~~~~~~~~")
    print("Identifying erroneous data...")
    idx = df['DT']
    idx = pd.to_datetime(idx)
    
    #df = df.drop_duplicates(subset='DT', keep='first')
    
    # Need to save external variables as when deleting and reindexing they are lost
    # tmpitemp = df['I_TEM']
    #tmpetemp = df['E_TEM']
    #tmprain = df['RAIN']
   
    #external_vars = ['I_TEM', 'E_TEM', 'RAIN']
    #df = df.drop(columns=external_vars)
   
    # Reset index incase df is from another process and is DT 
    df = df.reset_index(drop=True)
    df2 = df.copy()
    df2['FLAG'] = 0  # initialise FLAG to 0
    
    # Flag to remove above N0*1.075
    df2.loc[df2.MOD_CORR > (N0 * 1.075), "FLAG"] = 3
    df = df.drop(df[df.MOD_CORR > (N0 * 1.075)].index)   # drop above N0

    df2.loc[(df2.MOD_CORR < (N0*(int(nld['general']['belown0'])/100))) &
            (df2.MOD_CORR != int(nld['general']['noval'])), "FLAG"] = 2
    # drop below 0.3 N0
    df = df.drop(df[df.MOD_CORR < (N0*(int(nld['general']['belown0'])/100))].index)
    df2.loc[(df2.BATT < 10) & (df2.BATT != int(nld['general']['noval'])), "FLAG"] = 4
    df = df.drop(df[df.BATT < 10].index)

    # Drop >20% diff in timestep
    
    moddiff = [0]
    tmpindex = list(df.index)
    #print(tmpindex)
    for i in range(len(tmpindex)-1):
        lateridx = tmpindex[i+1]
        earlieridx = tmpindex[i]
        later = df['MOD'][lateridx]
        earlier = df['MOD'][earlieridx]
        currentdiff = (later - earlier)
        moddiff.append(currentdiff)
    df['DIFF'] = moddiff

    prcntdiff = [0]
    for i in range(len(tmpindex)-1):
        lateridx = tmpindex[i+1]
        earlieridx = tmpindex[i]
        later = df['DIFF'][lateridx]
        earlier = df['MOD'][earlieridx]
        indvdiff = (later / earlier)*100
        prcntdiff.append(indvdiff)
    df['PRCNTDIFF'] = prcntdiff

    df['INDEX_TMP'] = df.index
    
    diff1 = df.loc[(df['PRCNTDIFF'] > int(nld['general']['timestepdiff'])), "INDEX_TMP"]
    #diff1 = diff1[0]
    diff2 = df.loc[(df['PRCNTDIFF'] < (-int(nld['general']['timestepdiff']))), "INDEX_TMP"]
    #diff2 = diff2[0]

    #df2 = df2.reset_index(drop=True)
    df2.loc[diff1, "FLAG"] = 1
    df2.loc[diff2, "FLAG"] = 1

    df = df.drop(df[df.PRCNTDIFF > int(nld['general']['timestepdiff'])].index)
    df = df.drop(df[df.PRCNTDIFF < (-int(nld['general']['timestepdiff']))].index)
   # df = df.reset_index(drop=True)

    # Fill in master time again after removing
    # Need this to handle below code
    
    df.replace(int(nld['general']['noval']), np.nan, inplace=True)
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")

    df = df.set_index(df.DT)
    
    idx2 = pd.date_range(
        df.DT.iloc[0], df
        .DT.iloc[-1], freq='1H', closed='left')

    dtcol = df['DT']
    df = df.rename(columns={'DT': 'DT2'})
    df = df.drop_duplicates(subset='DT2')
    df.drop(labels=['DT2'], axis=1, inplace=True)
    df = df.reindex(index=idx2, fill_value=np.nan)

    df2 = df2.rename(columns={'DT': 'DT2'})
    df2 = df2.drop_duplicates(subset='DT2')
    df2.drop(labels=['DT2'], axis=1, inplace=True)
    df2 = df2.reindex(index=idx2, fill_value=np.nan)

 
    # df['DT'] = pd.to_datetime(df['DT'])
    flagseries = df2['FLAG']
    df['FLAG'] = flagseries.values  # Place flag vals back into df
    df['DT'] = df.index
    
    df = df.drop(["DIFF", "PRCNTDIFF", "INDEX_TMP"], axis=1)

    insertat = len(df.columns)-3
    dtcol = df['FLAG']
    df.drop(labels=['FLAG'], axis=1, inplace=True)  # Move FLAG to first col
    df.insert(insertat, 'FLAG', dtcol)
    
    # Add the external variables back in
    #df['I_TEM'] = tmpitemp
    #df['E_TEM'] = tmpetemp
    #df['RAIN'] = tmprain
    #print(df['I_TEM'])
    #print(tmpitemp)
    
    df.replace({nld['general']['noval']: np.nan})
   # df = df.replace(np.nan, int(nld.get('general','noval')))
    print("Done")
    
    return df
    
### CALLING QA FUNCTION ###

qa_data = flag_and_remove(level_1_data, N0=int(nld['metadata']['n0']), 
               country=str(nld['metadata']['country']), 
               sitenum=str(nld['metadata']['sitenum']), nld=nld)

qa_data.to_csv(wd+"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['sitenum']+"_qa.txt",
           header=True, index=False, sep="\t", mode='w')

##################
### THETA CALC ###
##################

def theta_calc(a0, a1, a2, bd, N, N0, lw, wsom):
    """theta_calc standard theta calculation

    Parameters
    ----------
    a0 : float
        constant
    a1 : float
        constant
    a2 : float
        constant
    bd : float
        bulk density e.g. 1.4 g/cm3
    N : int
        Neutron count (corrected)
    N0 : int
        N0 number
    lw : float
        lattice water - decimal percent e.g. 0.002
    wsom : float
        soil organic carbon - decimal percent e.g, 0.02


    """
    return (((a0)/((N/N0)-a1))-(a2)-lw-wsom)*bd

def rscaled(r, p, Hveg, y):
    """rscaled rescales the radius based on below parameters.
       Used in theaprocess to find the effective depth of the sensor.
       
    Parameters
    ----------
    r : float
        radius from sensor (m)
    p : float
        pressure at site (mb)
    Hveg : float
        height of vegetation during calibration period (m)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """
    Fp = 0.4922/(0.86-np.exp(-p/1013.25))
    Fveg = 1-0.17*(1-np.exp(-0.41*Hveg))*(1+np.exp(-9.25*y))
    return(r / Fp / Fveg)

def D86(r, bd, y):
    """D86 Calculates the depth of sensor measurement (taken as the depth from which
    86% of neutrons originate)

    Parameters
    ----------
    r : float, int
        radial distance from sensor (m)
    bd : float
        bulk density (g/cm^3)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """

    return(1/bd*(8.321+0.14249*(0.96655+np.exp(-0.01*r))*(20+y)/(0.0429+y)))

def thetaprocess(df, country, sitenum, nld=nld):
    """thetaprocess takes the dataframe provided by previous steps and uses the theta calculations
    to give an estimate of soil moisture. 

    Constants are taken from meta data which is identified using the "country" and "sitenum"
    inputs. 

    Realistic values are provided by constraining soil moisture estimates to physically possible values.
        i.e. Nothing below 0 and max values based on porosity at site.

    Provides an estimated depth of measurement using the D86 function from Shcron et al., (2017)

    Gives running averages of measurements using a 12 hour window. To handle missing 
    data a minimum of 6 hours of data per 12 hour window is required, otherwise one
    missing hour could lead to large gaps in 12 hour means. 
    
    This function utilizes the desilets equation to calculate soil moisture.

    Parameters
    ----------
    df : dataframe 
        dataframe of CRNS data
    meta : dataframe
        dataframe of metadata
    country : str
        country e.g. "USA"
    sitenum : str  
        sitenum e.g. "011"

    nld : dictionary
        nld should be defined in the main script (from name_list import nld), this will be the name_list.py dictionary. 
        This will store variables such as the wd and other global vars
    """

    print("~~~~~~~~~~~~~ Estimate Soil Moisture ~~~~~~~~~~~~~")
    

    df.replace({nld.get('general','noval'): np.nan})
    
    #################
    ### CONSTANTS ###
    #################
    
    #reading in constants
    lw = float(nld['metadata']['lw'])
    soc = float(nld['metadata']['soc'])
    bd = float(nld['metadata']['bd'])
    N0 = int(nld['metadata']['n0'])
    
    #finding maximum soil moisture through either metadata or using bulk density, bd.
    def sm_max_calc(bd, density):
       
        """sm_max calculation using bulk density

        Parameters
        ----------
        bd : float
            bulk density
            
        density : float
            density

        """
        return (1-(bd/density))               

    try:
        sm_max = float(nld['metadata']['sm_max'])
        if sm_max.isnumeric() or (sm_max.count('.') == 1):
            print("SM_max in metadata. Using this value")
            
        else:
            print("No numerical SM_max value found from metadata. Calculating SM_max from bulk density data.")
            sm_max = sm_max_calc(bd=bd,density=float(nld['general']['density'])) 
    except:
        print("No SM_max value found from metadata. Calculating SM_max from bulk density data.")
        sm_max = sm_max_calc(bd=bd,density=float(nld['general']['density']))
        
    # convert SOC to water equivelant (see Hawdon et al., 2014)
    soc = soc * 0.556
    sm_min = 0  # Cannot have less than zero
    
    ###################
    ### IMPORT DATA ###
    ###################
    
    #defining MOD_ERR#
    def mod_error(mod):
       
        """sm_max calculation using bulk density

        Parameters
        ----------
        mod : int
            nuetron count (uncorrected)
            
        """
        m = mod.mean()  #mean
        sd = (abs(mod)).apply(np.sqrt)  #standard deviation
        cv = sd / m  #coefficient of variation
        mod_err = mod * cv
        return mod_err
  
    df['MOD_ERR'] = round(mod_error(mod = df['MOD']))
    
    print("Calculating soil moisture with estimated error...")
    
    # Create MOD count to min and max of error
    df['MOD_CORR_PLUS'] = df['MOD_CORR'] + df['MOD_ERR']
    df['MOD_CORR_MINUS'] = df['MOD_CORR'] - df['MOD_ERR']
    
    # Calculate soil moisture for +MOD_ERR and -MOD_ERR
    df['SM'] = theta_calc(a0 = float(nld['general']['a0']), a1 = float(nld['general']['a1']), 
                  a2 = float(nld['general']['a2']), bd = float(nld['metadata']['bd']),
                  N = df['MOD_CORR'], N0 = float(nld['metadata']['n0']), 
                  lw = float(nld['metadata']['lw']), wsom = float(nld['metadata']['soc']))
    
    df['SM_RAW'] = df['SM']

    df['SM_PLUS_ERR'] = theta_calc(a0 = float(nld['general']['a0']), a1 = float(nld['general']['a1']), 
                  a2 = float(nld['general']['a2']), bd = float(nld['metadata']['bd']),
                  N = df['MOD_CORR_MINUS'], N0 = float(nld['metadata']['n0']), 
                  lw = float(nld['metadata']['lw']), wsom = float(nld['metadata']['soc']))  # Find error (inverse relationship so use MOD minus for soil moisture positive Error)
    df['SM_PLUS_ERR'] = abs(df['SM_PLUS_ERR'] - df['SM'])

    df['SM_MINUS_ERR'] = theta_calc(a0 = float(nld['general']['a0']), a1 = float(nld['general']['a1']), 
               a2 = float(nld['general']['a2']), bd = float(nld['metadata']['bd']),
               N = df['MOD_CORR_PLUS'], N0 = float(nld['metadata']['n0']), 
                  lw = float(nld['metadata']['lw']), wsom = float(nld['metadata']['soc']))   
    df['SM_MINUS_ERR'] = abs(df['SM_MINUS_ERR'] - df['SM'])
    
    # Remove values above or below max and min vols
    df.loc[df['SM'] < sm_min, 'SM'] = 0
    df.loc[df['SM'] > sm_max, 'SM'] = sm_max

    df.loc[df['SM_PLUS_ERR'] < sm_min, 'SM_PLUS_ERR'] = 0
    df.loc[df['SM_PLUS_ERR'] > sm_max, 'SM_PLUS_ERR'] = sm_max

    df.loc[df['SM_MINUS_ERR'] < sm_min, 'SM_MINUS_ERR'] = 0
    df.loc[df['SM_MINUS_ERR'] > sm_max, 'SM_MINUS_ERR'] = sm_max
    print("Done")
    
    # Take 12 hour average
    print("Averaging and writing table...")
    df['SM_12h'] = df['SM'].rolling(int(nld['general']['smwindow']), min_periods=6).mean()

    ### EFFECTIVE DEPTH ###

    # Depth calcs - use new Schron style. Depth is given considering radius and bd   
    
    hveg=0 #setting height of vegetation to 0 as we assume its effect is negligible on the sensor depth. 
    
    df['rs10m'] = df.apply(lambda row: rscaled(
        10, row['PRESS'], hveg, (row['SM'])), axis=1)
    df['rs75m'] = df.apply(lambda row: rscaled(
        75, row['PRESS'], hveg, (row['SM'])), axis=1)
    df['rs150m'] = df.apply(lambda row: rscaled(
        150, row['PRESS'], hveg, (row['SM'])), axis=1)
    
    df['D86_10m'] = df.apply(lambda row: D86(
        row['rs10m'], bd, (row['SM'])), axis=1)
    df['D86_75m'] = df.apply(lambda row: D86(
        row['rs75m'], bd, (row['SM'])), axis=1)
    df['D86_150m'] = df.apply(lambda row: D86(
        row['rs150m'], bd, (row['SM'])), axis=1)
    df['D86avg'] = (df['D86_10m'] + df['D86_75m'] + df['D86_150m']) / 3
    df['D86avg_12h'] = df['D86avg'].rolling(window=int(nld['general']['smwindow']), 
                                            min_periods=6).mean()  
    # Replace nans with -999
    df.fillna(int(nld['general']['noval']), inplace=True)  
    df = df.round(3)
    df = df.drop(['rs10m', 'rs75m', 'rs150m', 'D86_10m','D86_75m',
                  'D86_150m', 'SM', 'MOD_CORR_PLUS', 'MOD_CORR_MINUS'], axis=1)
    print("Done")

    return df

### CALLING THETAPROCESS FUNCTION ###

final_sm_data = thetaprocess(df=qa_data, country=nld['metadata']['country'],
                     sitenum=nld['metadata']['sitenum'],nld=nld)

final_sm_data.to_csv(wd+"/outputs/data/"+nld['metadata']['country']+"_SITE_"+nld['metadata']['country']+"_final.txt", 
                  header=True, index=False, sep="\t")



### END! ###









