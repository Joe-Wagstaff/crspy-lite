
"""
@author: Joe Wagstaff, Daniel Power
@institution: University of Bristol

"""

#general imports
import os
import json
import pandas as pd
import numpy as np
import csv
import datetime
import urllib.request
from bs4 import BeautifulSoup
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#crspy-lite imports
from crspy_lite_code import data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~ (1) SETTING UP WORKING DIRECTORY AND CONFIG FILE ~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def initial(default_dir):
    
    '''Creates a file structure for all outputs to be saved to. Creates all files relative to the default directory specified by the user.
    
    Parameters:
    -----------
    
    default_dir : str
        Default directory for folder creation and outputs.
    
    '''
    try:
        os.mkdir(default_dir+"/inputs/")
    except:
        print("Folder already exists, skipping.")
        pass
    
    try:
        os.mkdir(default_dir+"/outputs/")
    except:
        print("Folder already exists, skipping.")
        pass
    
    try:
        os.mkdir(default_dir+"/outputs/figures/")
    except:
        print("Folder already exists, skipping.")
        pass

    try:
        os.mkdir(default_dir+"/outputs/data/")
    except:
        print("Folder already exists, skipping.")
        pass
    


def create_config_file(default_dir):  
    
    ''' Creates config file which is a namelist of general and metadata (site) variables. 
        Prompts the user to input metadata variables in the terminal.
    
    Parameters:
    -----------
    
    default_dir : str
        Default directory for folder creation and outputs.
    
    '''

    config_file = os.path.join(default_dir, "inputs/config.json")

    #general processing variables.
    general_variables = {
        "noval" : "-999",
        "jung_ref":"159",
        "defaultbd":"1.43",
        "cdtformat":"%d/%m/%Y",
        ";accuracy is for n0 calibration":"",
        "accuracy":"0.01",
        ";QA values are percentages (e.g. below 30% N0)":"",
        "belowN0":"30",
        "timestepdiff":"20",
        ";density=density of quartz":"",
        "density":"2.65",
        "smwindow":"12",
        "pv0":"0",
        "a0":"0.0808",
        "a1":"0.372",
        "a2":"0.115",
        "sm_calc_method" : "desilets",
        "intensity_correction_method": "mcjannet",
    }
    
    #variables to be inputted by the user
    metadata_variables = {
    "country": {"description": "Code of country the sensor is in", "default": None, "required": True},
    "sitenum": {"description": "Site number", "default": None, "required": True},
    "sitename": {"description": "Name of site", "default": None, "required": True},
    "install_date": {"description": "Date of sensor installation dd/mm/yyyy", "default": None, "required": False},
    "latitude": {"description": "Latitude of site (in degrees). Values in range -90 to 90", "default": None, "required": True},
    "longitude": {"description": "Longitude of site (in degrees). Values in range from 0 to 360", "default": None, "required": True},
    "elev": {"description": "Elevation of site above sea level (m)", "default": None, "required": True},
    "timezone": {"description": "Timezone of site [optional]", "default": None, "required": False},
    "rc": {"description": "Cut of Rigidity (in gv) [optional - crspy_lite can calculate this]", "default": None, "required": False},
    "lw": {"description": "Percent lattice water (as a number between 0 and 1)", "default": None, "required": True},
    "soc": {"description": "Percent soil organic carbon (in g/cm^3)", "default": None, "required": True},
    "bd": {"description": "Bulk density (g/cm^3)", "default": None, "required": True},
    "beta_coeff": {"description": "Store of the calculated beta coefficient for each individual site [optional - crspy_lite can calculate this]", "default": None, "required": False},
    "reference_press": {"description": "Reference pressure (in mb)", "default": None, "required": True},
    "n0": {"description": "Theoetical neutron count for absolute dry soil conditions [optional - If no N0 number is given crspy_lite will require calibration data to calculate N0]", "default": None, "required": False},
    "precip_max": {"description": "Maximum precipitation (in mm). Values above this are removed. [optional]", "default": None, "required": False},
    "sm_max": {"description": "Maximum value of soil moisture (in cm^3/cm^3). Values above this are removed. [optional]", "default": None, "required": False},
    "field_capacity": {"description": "Field Capacity: The amount of soil moisture or water content held in the soil after excess water has drained away [optional]", "default": None, "required": False},
    "wilting_point": {"description": "Wilting Point: The amount of water that is held so tightly by the soil matrix that roots cannot absorb it and plants will wilt [optional]", "default": None, "required": False},
    "raw_data_filepath": {"description": "File path of Input data [optional - but must give a file path OR url (see next prompt) of the input data]", "default": None, "required": False},
    "raw_data_url": {"description": "URL of Input data [optional - but must give a file path OR url of the input data]", "default": None, "required": False},
    "calib_data_filepath": {"description": "File path of calibration data [optional - but needed for site calibration if no N0 number is given]", "default": None, "required": False},
    
    }

    metadata = {}
    for key, details in metadata_variables.items():
        if details["required"]:
            # Required input: loop until a valid value is provided.
            while True:
                value = input(f"{details['description']}: ").strip()
                if value:
                    metadata[key] = value
                    break
                print(f"{key} is required. Please provide a value.")

        else:
            # Optional input: use default if no input provided.
            value = input(f"{details['description']} (default: {details['default']}): ").strip()
            metadata[key] = value if value else details["default"]

    config_data = {
        "general": general_variables,
        "metadata": metadata
    }
    
    # Save the combined data to a JSON file.
    with open(config_file, "w") as file:
        json.dump(config_data, file, indent=4)

    #load config file to edit it .
    with open(config_file, "r") as file:
        config_data = json.load(file)

    #adjusting config file sections to include a filepaths section, move filepaths to this section & establish a default directory.
    config_data['filepaths'] = {}
    
    raw_data_filepath = config_data['metadata']['raw_data_filepath']
    del config_data['metadata']['raw_data_filepath'] #no longer want it in this section so delete
    if isinstance(raw_data_filepath, str):  #can't .strip() unless its a string
        config_data['filepaths']['raw_data_filepath'] = raw_data_filepath.strip('"') #.strip('"') brought in to avoid filepath error from copying filepaths using ctrl+shift+c.
    else:
        config_data['filepaths']['raw_data_filepath'] = raw_data_filepath

    raw_data_url = config_data['metadata']['raw_data_url']
    del config_data['metadata']['raw_data_url']
    config_data['filepaths']['raw_data_url'] = raw_data_url

    calib_data_filepath = config_data['metadata']['calib_data_filepath']
    del config_data['metadata']['calib_data_filepath']
    if isinstance(calib_data_filepath, str):
        config_data['filepaths']['calib_data_filepath'] = calib_data_filepath.strip('"')
    else:
        config_data['filepaths']['calib_data_filepath'] = calib_data_filepath

    config_data['filepaths']['default_dir'] = default_dir  

    # Save the new section back to the JSON file.
    with open(config_file, "w") as file:
        json.dump(config_data, file, indent=4)

    print(f"Configuration saved to {config_file}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~ (2) RAW DATA TIDYING ~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~#

def retrieve_data_url(url):

    """Function that will retrieve input CRNS data from a weblink

    Parameters
    ----------
    url : str
        Weblink of CRNS input data.
    
    Returns
    -------
    df : DataFrame
        Input data from weblink.

    """
    print("Retrieving CRNS data from url...")
    try:
        response = urllib.request.urlopen(url) #get the html response
        html = response.read()   #rewad response
        soup = BeautifulSoup(html, "lxml") #pass html

        body = soup.find("body") # Find the <body> of the HTML
        lines = [line.strip() for line in body.find_all(text=True) if line.strip()] # Extract all the content split by <br> tags
        data_rows = [line.split(", ") for line in lines]  # Adjust delimiter if necessary

        # Create a DataFrame, assuming the first line contains headers
        if data_rows:
        
            headers = data_rows[1]  # Assuming the second line contains headers
            df = pd.DataFrame(data_rows[2:], columns=headers)   # '2:' Assumes the second line contains headers
            print("Done")
            return df
    
        else:
            print("No data found.")
            return None

    except Exception as e:  #catches error that might occur
        print(f"An error occurred: {e}")
        return None
    

def read_file(file_path):

    """Function to check what delimiter the input file has and then read it correctly.
    
    Parameters
    ----------
    file_path : str
        File path of the input data.
        
    Returns
    -------
    df : DataFrame
        DataFrame of input (or "raw") data.
    
    """

    # Read the first few lines of the file
    with open(file_path, 'r') as file:
        sample = file.read(4096)  # Read the first 4KB of the file

    # Use csv.Sniffer to detect the delimiter
    sniffer = csv.Sniffer()
    delimiter = ','  # Default to comma
    if sniffer.has_header(sample):
        header = True
    else:
        header = False

    try:
        delimiter = sniffer.sniff(sample).delimiter
    
    except csv.Error:
        print("Could not detect delimiter. Defaulting to comma.")

    print(f"Detected delimiter of input data: '{delimiter}'")

    # Load the file into a DataFrame with the detected delimiter
    df = pd.read_csv(file_path, delimiter=delimiter, header=0 if header else None)

    return df


def flip_df(df):

    """Optional function to flip the DataFrame if the dates are the wrong way round (from most recent to least recent).
    
    Parameters
    ----------
    df : DataFrame  
        Input data to reformat.
 
    Returns
    -------
    df : DataFrame
        Flipped DataFrame.

    """
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')
    df.set_index(df['DATETIME'], inplace=True)

    if df['DATETIME'].iloc[0] > df['DATETIME'].iloc[-1]:    #is the first date greater than the last date?
        
        df = df.iloc[::-1]  #if so, this line reorders the rows

    else:
        pass

    return df


def resample_to_hourly(df):

    """
    Resamples the input data to hourly intervals.

    Parameters
    ----------
    df : DataFrame  
        Input data to reformat.
        
    Returns
    -------
    df_resampled : DataFrame
        Resampled data.
    
    """
    
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], dayfirst=True)
    df.set_index(df['DATETIME'], inplace=True)

    df = df.drop(columns=['DATETIME'])

    sum_cols = ['MOD', 'RAIN']
    mean_cols = ['PRESS','TEMP', 'E_RH', 'I_RH', 'BATT']

    df[sum_cols] = df[sum_cols].apply(pd.to_numeric, errors='coerce')
    df[mean_cols] = df[mean_cols].apply(pd.to_numeric, errors='coerce')
    
    df_sum = df[sum_cols].resample('h').sum()
    df_mean = df[mean_cols].resample('h').mean()

    df_resampled = pd.concat([df_sum, df_mean], axis=1) #doesn't work if you concat df, need to create a new dataframe
    df_resampled.reset_index(inplace=True)  #resets the index, moving 'TIME' back to the cols
    
    return df_resampled
        

def tidy_headers(df):

    """Removes any spaces in the headers of the columns in the input data file. 
    
    Parameters
    ----------
    df : DataFrame  
        Input data.
    
        
    Returns
    -------
    df : DataFrame  
        Manipulated DataFrame.
   
     """
    
    df.columns = df.columns.str.strip()
    
    return df

def dropemptycols(colstocheck, df, config_data):
    
    """Drops any columns that are empty (i.e. all -999)

    Parameters
    ----------
    colstocheck : str
        string of column title to check.
    
    df : DataFrame  
        DataFrame to check.
    
    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.


    Returns
    -------
    df : DataFrame  
        Manipulated DataFrame.

    """
    for i in range(len(colstocheck)):
        col = colstocheck[i]
        if col in df:
            try:
                if df[col].mean() == int(config_data['noval']):
                    df = df.drop([col], axis=1)
                else:
                    pass
            except:
                pass
        else:
            pass
    
    return df

def movecol(df, col, location):
        
        """
        Move columns to a specific position.
        """
        tmp = df[col]
        df.drop(labels=[col], axis=1, inplace=True)
        df.insert(location, col, tmp)

def prepare_data(df, config_data):
    
    """Cleans/tidies the input data.

    Steps include: 
        - Fix time to be on the hour rather than variable
        - Tidy of columns

    Parameters
    ----------
    df: DataFrame 
        Input data.

    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.
    
    Returns
    -------
    df : DataFrame  
        Reformatted DataFrame.

    """
    print("Preparing data...")

    # Remove leading white space - present in some SD card data
    df['DATETIME'] = df['DATETIME'].astype(str)
    df['DATETIME'] = df['DATETIME'].str.lstrip() 
    
    # Ensure using dashes as Excel tends to convert to /s
    if '/' in df['DATETIME'][0]:
        df['DATETIME'] = df['DATETIME'].str.replace('/', '-')
    else:
        pass
    
    #split 'DATETIME' col into a 'DATE' col and a 'TIME' col. The datetime col is then reassempled under the name 'DT' (shorter name so simpler to code)
    tmp = df['DATETIME'].str.split(" ", n=1, expand=True) 
    df['DATE'] = tmp[0]
    df['TIME'] = tmp[1]
    new = df["TIME"].str.split(":", n=2, expand=True)
    df['DT'] = pd.to_datetime(df['DATE'], format='mixed')
    my_time = df['DT']
    time = pd.DataFrame()
    
    df.reset_index(drop=True, inplace=True)
    
    tseries = []
    for i in range(len(new)):  # Loop through with loc to append the hours onto a DateTime object
        # time = the datetime plus the hours and mins in time delta
        time = my_time[i] + datetime.timedelta(
            hours=int(new.loc[i, 0]), minutes=int(new.loc[i, 1]))
        tseries.append(time)  # Append onto tseries

    df['DT'] = tseries      # replace DT with tseries which now has time as well

    """
    THE MASTER TIME

    Master Time creates a time series from first data point to last data point with
    every hour created. This is to remedy the gaps in the data.
    
    DateTime is standardised to be on the hour (using floor). This can create issues
    with "duplicated" data points, usually when errors in logging have created data
    every half hour instead of every hour, for example. 
    
    The best way to address this currently is to retain the first instance of the 
    duplicate and discard the second.
    """
    
    df['DT'] = df.DT.dt.floor(freq='h')
    dtcol = df['DT']
    df.drop(labels=['DT'], axis=1, inplace=True)  # Move DT to first col
    df.insert(0, 'DT', dtcol)

    df['dupes'] = df.duplicated(subset="DT")
    df = df.drop(df[df.dupes == True].index)
    df = df.set_index(df.DT)
    if df.DATE.iloc[0] > df.DATE.iloc[-1]:
        raise Exception("The dates are the wrong way around.")

    idx = pd.date_range(
        df.DATE.iloc[0], df.DATE.iloc[-1], freq='1H')
    df = df.reindex(idx, fill_value=np.nan)
    df = df.replace(np.nan, int(config_data['general']['noval']))

    df['DT'] = df.index

    ### The Final Table ###
    # Move below columns as like them at the front
    movecol(df, "MOD", 1)
    movecol(df, "PRESS", 3)
    
    df.drop(labels=['dupes'],
            axis=1, inplace=True)  # not required after here
    try:
        df.drop(labels=['fsol'], axis=1, inplace=True)
    except:
        pass
    # Add list of columns that some sites wont have data on - removes them if empty
    df = dropemptycols(df.columns.tolist(), df, config_data)
    df = df.round(3)
    df = df.replace(np.nan, int(config_data['general']['noval']))
    # SD card data had some 0 values - should be nan
    df['MOD'] = df['MOD'].replace(0, int(config_data['general']['noval']))
    
    print("Done")

    return df

#~~~~~~~~~~~~~~~~~~~~~~#
#~~ (3) DATA IMPORTS ~~# 
#~~~~~~~~~~~~~~~~~~~~~~#

#~~~ NMDB data import ~~~#

def nmdb_get(startdate, enddate, station, default_dir):
    
    """Function will collect data for Junfraujoch station that is required to calculate the incoming neutron intensity correction factor.
    Returns a dictionary that can be used to fill in values to the main dataframe of each site.

    Parameters
    ----------
    startdate : datetime
        start date of desire data in format YYYY-mm-dd (e.g 2015-10-01)
    
    enddate : datetime
        end date of desired data in format YYY-mm-dd
    
    station : str
        Neutron monitoring station (default "JUNG")
   
    default_dir : str
        Default directory for folder creation and outputs.
        

    Returns
    -------
    dfdict : dict
        dictionary of neutron count data from NMDB.eu
 
    """
    # split for use in url
    sy, sm, sd = str(startdate).split("-")
    ey, em, ed = str(enddate).split("-")

    # Collect html from request and extract table and write to dict
    print("Retrieving data from the NMDB...")
    url = "http://nest.nmdb.eu/draw_graph.php?formchk=1&stations[]={station}&tabchoice=1h&dtype=corr_for_efficiency&tresolution=60&force=1&yunits=0&date_choice=bydate&start_day={sd}&start_month={sm}&start_year={sy}&start_hour=0&start_min=0&end_day={ed}&end_month={em}&end_year={ey}&end_hour=23&end_min=59&output=ascii"
    url = url.format(station=station, sd=sd, sm=sm, sy=sy, ed=ed, em=em, ey=ey)
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, features="html.parser")
    pre = soup.find_all('pre')
    pre = pre[0].text
    pre = pre[pre.find('start_date_time'):]
    pre = pre.replace("start_date_time   1HCOR_E", "")
    
    # Split lines and create a DataFrame
    lines = pre.strip().split("\n")[1:]  # Skip the header
    dfneut = pd.DataFrame(lines)
    dfneut = dfneut[0].str.split(";", n=2, expand=True)
    dfneut.columns = ['DATE', 'COUNT']

    # Convert COUNT to numeric and replace 0 or 0.00 with NaN
    dfneut['COUNT'] = pd.to_numeric(dfneut['COUNT'], errors='coerce')
    dfneut['COUNT'] = dfneut['COUNT'].replace(0, np.nan)

    # Write the cleaned data to the file
    output_path = f"{default_dir}/outputs/data/nmdb_station_counts.txt"
    dfneut.to_csv(output_path, sep=";", index=False, header=False)

    # Convert dates to datetime and create the dictionary
    dates = pd.to_datetime(dfneut['DATE'])
    count = dfneut['COUNT']
    dfdict = dict(zip(dates, count))

    print("Done")
    
    return dfdict


#~~~~~~~ CUT-OFF RIGIDITY CALC~~~~~~~#

def rc_retrieval(latitude, longitude):
    
    """Function to estimate the approximate cutoff rigidity for any point on Earth according to the
    tabulated data of Smart and Shea, 2019. 

    Parameters
    ----------
    latitude: float
        Geographic latitude in decimal degrees. Values in range -90 to 90
    
    longitude: float
        Geographic longitude in decimal degrees. Values in range from 0 to 360.
        Typical negative longitudes in the west hemisphere will fall in the range 180 to 360.


    Returns
    -------
    zq : float
        Cutoff rigidity in GV. Error is about +/- 0.3 GV

    References
    ----------
    Hawdon et al. (2014) : https://doi.org/10.1002/2013WR015138
    
    Smart, D. & Shea, Matthew. (2001). Geomagnetic Cutoff Rigidity Computer Program:
    Theory, Software Description and Example. NASA STI/Recon Technical Report N.

    Shea, M. A., & Smart, D. F. (2019, July). Re-examination of the First Five Ground-Level Events.
    In International Cosmic Ray Conference (ICRC2019) (Vol. 36, p. 1149).
  
    """

    xq = float(longitude)
    yq = float(latitude)

    if xq < 0:
        xq = xq * -1 + 180
    Z = np.array(data.cutoff_rigidity)
    x = np.linspace(0, 360, Z.shape[1])
    y = np.linspace(90, -90, Z.shape[0])
    X, Y = np.meshgrid(x, y)
    points = np.array((X.flatten(), Y.flatten())).T
    values = Z.flatten()
    zq = griddata(points, values, (xq, yq))
    zq = np.round(zq, 2)

    return float(zq)

#~~~~~~~ BETA COEFFICIENT CALC ~~~~~~~#

def betacoeff(latitude, elevation, Rc):
    
    """Function will calculate the beta coefficient for a site.

    Parameters
    ----------
    latitude : float
        latitude of site (degrees)
   
    elevation : float 
        elevation of site (m)
    
    Rc : float
        cutoff ridgidity of site (in GV)

        
    Returns
    -------
    beta_coeff : float
        Beta coefficient of the site.

    x0 : float
        Mean pressure (in mb)
    
    """

    # --- elevation scaling ---

    # --- Gravity correction ---

    # constants
    rho_rck = 2670
    x0 = (101325*(1-2.25577*(10**-5)*elevation)**5.25588)/100  # output in mb

    # variables
    z = -0.00000448211*x0**3 + 0.0160234*x0**2 - 27.0977*x0+15666.1

    # latitude correction
    g_lat = 978032.7*(1 + 0.0053024*(np.sin(np.radians(latitude))
                                     ** 2)-0.0000058*(np.sin(np.radians(2*latitude)))**2)

    # free air correction
    del_free_air = -0.3087691*z

    # Bouguer correction
    del_boug = rho_rck*z*0.00004193

    g_corr = (g_lat + del_free_air + del_boug)/100000

    # final gravity and depth
    g = g_corr/10
    x = x0/g

    # --- elevation scaling ---

    # parameters
    n_1 = 0.01231386
    alpha_1 = 0.0554611
    k_1 = 0.6012159
    b0 = 4.74235E-06
    b1 = -9.66624E-07
    b2 = 1.42783E-09
    b3 = -3.70478E-09
    b4 = 1.27739E-09
    b5 = 3.58814E-11
    b6 = -3.146E-15
    b7 = -3.5528E-13
    b8 = -4.29191E-14

    # calculations
    term1 = n_1*(1+np.exp(-alpha_1*Rc**k_1))**-1*(x-x0)
    term2 = 0.5*(b0+b1*Rc+b2*Rc**2)*(x**2-x0**2)
    term3 = 0.3333*(b3+b4*Rc+b5*Rc**2)*(x**3-x0**3)
    term4 = 0.25*(b6+b7*Rc+b8*Rc**2)*(x**4-x0**4)

    beta_coeff = abs((term1 + term2 + term3 + term4)/(x0-x))
    beta_coeff = str(round(beta_coeff, 5))

    return beta_coeff, x0


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~ (4) CORRECTION FACTOR FUNCTIONS ~~# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~ PRESSURE CORRECTION ~~~~~~~#

def pressure_correction(press, beta_coeff, p0):
   
    """Function corrects neutrons for pressure changes

    Parameters
    ----------
    press : Series
        pressure of site (mb).
    
    beta_coeff : float
        Beta coefficient for site. [Can be calculated using the function "betacoeff()"]
    
    p0 : int
        reference pressure (mb)

        
    Returns
    -------
    : float
        Pressure correction factor
    
    """
    
    return np.exp(beta_coeff * (press-p0))

#~~~~~~~ HUMIDITY CORRECTION ~~~~~~~#

def es(temp):
   
    """Calculates saturation vapour pressure (in Pa) using average temperature   

    Parameters
    ----------
    temp : Series  
        temperature (C)

        
    Returns
    -------
    : Series 
        saturation vapour pressure (in Pa) 
    """

    return (6.112*np.exp((17.67*temp)/(243.5+temp)))*100


def ea(es, RH):
    
    """Caluculates actual vapour pressure (in Pa) using saturation vapour pressure and relative humidity (RH).

    Parameters
    ----------
    es : Series
        saturation vapour pressure (in Pa). [Can be found using the function "es()"]
    
    RH : Series
        External relative humidity (%) at the site.

        
    Returns
    -------
    : Series
        actual vapour pressure (Pascals)
    
    """
   
    return es * (RH/100)

def pv(ea, temp):
    
    """Finds absolute humidity (in g/m^3) using actual vapour pressure (in Pa) and temperature (C) 

    Parameters
    ----------
    ea : Series  
        Vapour Presure (Pascals) [Can be found using the function "ea()"]
   
    temp : float
        temperature (C)

        
    Returns
    -------
    : Series
        Absolute humidity (in g/m^3).
    
    """

    return (ea/(461.5*(temp+273.15))) * 1000


def humidity_correction(pv, pv0):
    
    """humfact gives the factorial to multiply Neutron counts by

    Parameters 
    ----------
    pv : Series 
        Absolute humidity. [Can be found using the function "pv()"]
    
    pv0 : float
        Reference absolute humidity.

    Returns
    -------
    : Series
        Humidity correction factor.
   
     """

    return (1+0.0054*(pv-pv0))


#~~~~~~~ INCOMING NEUTRON INTENSITY CORRECTION ~~~~~~~#

#Hawdon et al. (2014) method.
def RcCorr(Rc, Rc_ref):
    
    """RcCorr takes cutoff ridgity of site (Rc) and gives the RcCorr value required to 
    account for the difference to the reference monitoring station. 

    Required for the Hawdon et al (2014) intensity correction.

    Parameters
    ----------
    Rc : float
        Cutoff ridgidity of site. [Can be estimated using the function "rc_retrieval()"]

    Rc_ref : float
        Cutoff ridigity of the reference monitoring station

    Returns
    -------
    gamma : float
        Amplitude scaling factor used to adjust for geomagnetic effects between the site & station.

    References
    ----------
    Hawdon et al. (2014) :  https://doi.org/10.1002/2013WR015138    

    """
    
    gamma = -0.075 * (Rc - Rc_ref) + 1

    return gamma


def intensity_correction_Hawdon2014(station_ref, station_count, RcCorr):
  
    """Incoming neutron intensity correction for the Hawdon et al. (2014) method.
   
    Note the intensity corrections are labelled as finten_1, finten_2 & finten_3 in the code as per the 
        labelling in the McJannet & Desilets (2023) paper.
     
    Parameters
    ----------
    jung_ref : float
        Reference neutron count at JUNG (usually 01/05/2011).
    
    jung_count : float
        Count at time[0].
    
    RcCorr : float
        Amplitude scaling factor used to adjust for geomagnetic effects between the site & station. [Found using the function "RcCorr()"]

    Returns
    -------
    f_i : Series
        Incoming neutron intensity correction factor

    References
    ----------
    Hawdon et al. (2014) :  https://doi.org/10.1002/2013WR015138

    """
    f_i = station_count / station_ref
    f_i = 1 / (((f_i - 1) * RcCorr) + 1)

    return f_i

#McJannet & Desilets (2023) method.
def atmospheric_depth(elevation, latitude):
    
    """Estimates the atmospheric depth for any point on the Earth.
    This function is required to calculate the location factor for the McJannet and Desilets (2023) intensity correction.
    
    Parameters
    ----------
    elevation : float
        Elevation of the site above sea level (m).
   
    latitude : float
        Geographic latitude in decimal degrees (ranging from -90 to 90).
        
    Returns
    -------
    atmospheric_depth : float
        Atmospheric depth in g/cm2.

    References
    ----------
    Atmosphere, U. S. (1976). US standard atmosphere. National Oceanic and Atmospheric Administration.

    McJannet, D. L., & Desilets, D. (2023). Incoming Neutron Flux Corrections for Cosmic‐Ray Soil and Snow Sensors Using the 
        Global Neutron Monitor Network. Water Resources Research, 59(4), e2022WR033889.
    
    """
    density_of_rock = 2670  # Density of rock in kg/m3
    air_pressure_sea_level = 1013.25  # Air pressure at sea level in hPa
    air_molar_mass = 0.0289644  # Air molar mass in kg/mol
    universal_gas_constant = 8.31432  # Universal gas constant in J/(mol*K)
    reference_temperature = 288.15  # Reference temperature Kelvin
    temperature_lapse_rate = -0.0065  # Temperature lapse rate in K/m

    # Gravity at sea-level calculation
    gravity_sea_level = 9.780327 * (
            1 + 0.0053024 * np.sin(np.radians(latitude)) ** 2 - 0.0000058 * np.sin(2 * np.radians(latitude)) ** 2)
    # Free air correction
    free_air = -3.086 * 10 ** -6 * elevation
    # Bouguer correction
    bouguer_corr = 4.193 * 10 ** -10 * density_of_rock * elevation
    # Total gravity
    gravity = gravity_sea_level + free_air + bouguer_corr

    # Air pressure calculation
    reference_air_pressure = air_pressure_sea_level * (
            1 + temperature_lapse_rate / reference_temperature * elevation) ** ((-gravity * air_molar_mass) / (
            universal_gas_constant * temperature_lapse_rate))

    # Atmospheric depth calculation
    atmospheric_depth = (10 * reference_air_pressure) / gravity
    
    return atmospheric_depth


def location_factor(site_atmospheric_depth, Rc, ref_atmospheric_depth, reference_Rc):
    
    """location_factor estimates the location factor (tau) between two sites according to McJannet and Desilets (2023)
    Function is required for the McJannet and Desilets (2023) incoming neutron intensity correction.
    
    Parameters
    ----------
    site_atmospheric_depth : float 
        Atmospheric depth at the site in g/cm2. [Can be estimated using the function "atmospheric_depth()"]
    
    Rc : float
        Cutoff rigidity at the site in GV. [Can be estimated using the function "Rc_retrieval()"]
    
    reference_atmospheric_depth : float
        Atmospheric depth at the reference location in g/cm2.
   
    reference_Rc : float
        Cutoff rigidity at the reference location in GV.
        
    Returns
    -------
    tau : float
        Location factor.

    K : float
        Scaling factor between chosen neutron monitor & CLMX monitor.

    References
    ----------
    McJannet and Desilets (2023) : https://doi.org/10.1029/2022WR033889

    """
    #Fixed variables based on Table 1 in McJannet & Desilet (2023). Note some differences in sign convention to simplify the code.
    c0 = -0.0009 
    c1 = 1.7699 
    c2 = 0.0064 
    c3 = 1.8855  
    c4 = 0.000013 
    c5 = -1.2237 
    epsilon = 1 #Accounts for the differences between epithermal neutrons measured by a CRNS and high energy neutrons measured by the neutrons monitors. No correction for this at present.

    #first find tau_new to account for differences between new neutron monitoring station and CLMX station.
    tau_new = epsilon * (c0 * ref_atmospheric_depth + c1) * (
            1 - np.exp(-(c2 * ref_atmospheric_depth + c3) * reference_Rc ** (c4 * ref_atmospheric_depth + c5)))

    #Quanitfy this difference by calculating the scaling factor, K.
    K = 1 / tau_new    # K = tau_CLMX / tau_new , where tau_CLMX is equal to 1 >see McJannet & Desilet (2023) paper.

    #Now find location factor, tau for the CRNS site.
    tau = epsilon * (c0 * site_atmospheric_depth + c1) * (
            1 - np.exp(-(c2 * site_atmospheric_depth + c3) * Rc ** (c4 * site_atmospheric_depth + c5)))
    
    return tau, K


def intensity_correction_McJannet2023(station_ref, station_count, tau):
    
    """Incoming neutron intensity correction for the McJannet and Desilets (2023) method. The author recommends using this correction method.
    
    Parameters
    ----------
    station_ref : float
        reference neutron count at neutron monitoring station (usually 01/05/2011).
    
    station_count : float
        count at time[0].

    tau : float
        location factor. [Found using the function "location_factor()"]
    
    Returns
    -------
    f_i : Series
        Incoming neutron intensity correction factor 

    References
    ----------
    McJannet and Desilets (2023) : https://doi.org/10.1029/2022WR033889

    """
   
    f_i = station_count / station_ref 

    f_i = 1 / ((tau * f_i) + 1 - tau)
    
    return f_i

#~~~~~~~ BIOMASS CORRECTION ~~~~~~~#

#currently no biomass correction available in crspy-lite

#~~~~~~~ APPLY CORRECTION FACTORS ~~~~~~~#

def mod_corr(f_p, f_h, f_i, f_v, mod):
    
    """mod_corr gives the corrected neutron count 

    Parameters
    ----------
    f_p : Series
        Corection factor for presure. [Found using the function "pressure_correction()"]     
   
    f_h : Series
        Corection factor for humidity. [Found using the function "humidity_correction()"]  
    
    f_i : Series
        Corection factor for incoming neutron intensity. [Found using either function "intensity_correction_McJannet2023()" or "intensity_correction_Hawdon2014()"]  
    
    f_v : Series
        Corection factor for above ground biomass.
    
    mod : Series
        Raw (uncorrected) neutron count.

        
    Returns
    -------
    : Series
        Corrected neutron count

    """
    
    return mod * f_p * f_h * f_i * f_v


#~~~~~~~~~~~~~~~~~~~~~#
#~~ (5) CALIBRATION ~~# 
#~~~~~~~~~~~~~~~~~~~~~#

#soil moisture (theta) calculation. Needed for calibration process.
def desilets_sm_calc(a0, a1, a2, bd, N, N0, lw, wsom):
    
    """Soil moisture processing equation for the Desilets equation.
    This is currently the standard method for soil moisture estimation in CRNS.

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

    Returns
    -------
    sm : Series
        Volumetric soil moisture (in cm3/cm3)

        
    References
    ----------
    Desilets et al. (2010) : https://doi.org/10.1029/2009WR008726

    """

    sm = (((a0)/((N/N0)-a1))-(a2)-lw-wsom)*bd
    
    return sm


def kohli_sm_calc(a0, a1, a2, bd, N, N0, lw, wsom):
    
    """kohli et al. (2021) soil moisture processing equation. For more info see> 

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

    Returns
    -------
    sm : Series
        Volumetric soil moisture (in cm3/cm3)

    References
    ----------
    Kohli et al (2021) : https://doi.org/10.3389/frwa.2020.544847

    """
    Nmax = N0 * ( (a0 + (a1 * a2)) / (a2) )
    ah0 = -a2
    ah1 = (a1 * a2) / (a0 + (a1 * a2))
    sm = ((ah0 * ( (1-(N/Nmax)) / (ah1 - (N/Nmax)) )) - lw - wsom) * bd
   
    return sm

# ~~~~ FUNCTIONS FROM SCHRON ET AL 2017 USED TO CALIBRATE THE SENSOR ~~~~~#
    
# Horizontal
def WrX(r, x, y):
    
    """Radial Weighting function for point measurements taken within 5m of sensor.

    Parameters
    ----------
    r : float
        rescaled distance from sensor (see rscaled function below)
    
    x : float
        Air Humidity from 0.1 to 0.50 in g/m^3
    
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3

    """

    x00 = 3.7
    a00 = 8735
    a01 = 22.689
    a02 = 11720
    a03 = 0.00978
    a04 = 9306
    a05 = 0.003632
    a10 = 2.7925e-002
    a11 = 6.6577
    a12 = 0.028544
    a13 = 0.002455
    a14 = 6.851e-005
    a15 = 12.2755
    a20 = 247970
    a21 = 23.289
    a22 = 374655
    a23 = 0.00191
    a24 = 258552
    a30 = 5.4818e-002
    a31 = 21.032
    a32 = 0.6373
    a33 = 0.0791
    a34 = 5.425e-004

    x0 = x00
    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x

    return((A0*(np.exp(-A1*r))+A2*np.exp(-A3*r))*(1-np.exp(-x0*r)))


def WrA(r, x, y):
    
    """Radial Weighting function for point measurements taken within 50m of sensor

    Parameters
    ----------
    r : [type]
        [description]
    
    x : [type]
        [description]
    
    y : [type]
        [description]
    """

    a00 = 8735
    a01 = 22.689
    a02 = 11720
    a03 = 0.00978
    a04 = 9306
    a05 = 0.003632
    a10 = 2.7925e-002
    a11 = 6.6577
    a12 = 0.028544
    a13 = 0.002455
    a14 = 6.851e-005
    a15 = 12.2755
    a20 = 247970
    a21 = 23.289
    a22 = 374655
    a23 = 0.00191
    a24 = 258552
    a30 = 5.4818e-002
    a31 = 21.032
    a32 = 0.6373
    a33 = 0.0791
    a34 = 5.425e-004

    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x

    return(A0*(np.exp(-A1*r))+A2*np.exp(-A3*r))


def WrB(r, x, y):
    
    """Radial Weighting function for point measurements taken over 50m of sensor

    Parameters
    ----------
    r : float
        rescaled distance from sensor (see rscaled function below)
    
    x : float
        Air Humidity from 0.1 to 0.50 in g/m^3
    
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    
    """
    b00 = 39006
    b01 = 15002337
    b02 = 2009.24
    b03 = 0.01181
    b04 = 3.146
    b05 = 16.7417
    b06 = 3727
    b10 = 6.031e-005
    b11 = 98.5
    b12 = 0.0013826
    b20 = 11747
    b21 = 55.033
    b22 = 4521
    b23 = 0.01998
    b24 = 0.00604
    b25 = 3347.4
    b26 = 0.00475
    b30 = 1.543e-002
    b31 = 13.29
    b32 = 1.807e-002
    b33 = 0.0011
    b34 = 8.81e-005
    b35 = 0.0405
    b36 = 26.74

    B0 = (b00-b01/(b02*y+x-0.13))*(b03-y)*np.exp(-b04*y)-b05*x*y+b06
    B1 = b10*(x+b11)+b12*y
    B2 = (b20*(1-b26*x)*np.exp(-b21*y*(1-x*b24))+b22-b25*y)*(2+x*b23)
    B3 = ((-b30+b34*x)*np.exp(-b31*y/(1+b35*x+b36*y))+b32)*(2+x*b33)

    return(B0*(np.exp(-B1*r))+B2*np.exp(-B3*r))


# Vertical
def D86(r, bd, y):
    
    """Calculates the depth of sensor measurement (taken as the depth from which 86% of neutrons originate)

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


def Wd(d, r, bd, y):
    
    """Weighting function to be applied on samples to calculate weighted impact of soil samples based on depth.

    Parameters
    ----------
    d : float
        depth of sample (cm)
    
    r : float,int
        radial distance from sensor (m)
    
    bd : float
        bulk density (g/cm^3)
    
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    
    """
    return(np.exp(-2*d/D86(r, bd, y)))


# Rescaled distance
def rscaled(r, p, y, Hveg=0):
    
    """Rescales the radius based on below parameters.

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


#calibration function
def n0_calibration(corrected_data, country, sitenum, defineaccuracy, calib_start_time, calib_end_time, config_data, calib_data_filepath, default_dir, sm_calc_method):
    
    """n0_calib the full calibration process

    Parameters
    ----------
    corrected_data : DataFrame
        Corrected input data (data with corrected neutron counts)

    country : str
        Country of the site
    
    sitenum : str
        Site number.
    
    defineaccuracy : float
        Accuracy that is desired usually 0.01 (default)
    
    calib_start_time : str
        start time of the calibration period in UTC time e.g."16:00:00" 
    
    calib_end_time : str
        end time of the calibration period in UTC time e.g."23:00:00" 
    
    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.
        
    calib_data_filepath : str
        File path of the calibration data.
    
    default_dir : str
        Default directory for folder creation and outputs.

    sm_calc_method : str
        Must either be 'desilets' for desilets method or 'kohli' for kohli method.

    
    Returns
    -------
    N0: str
        Theoretical neutron count for dry soil conditions (the calibration value for the site)

    References
    ----------
    Schron et al (2017) : https://doi.org/10.5194/hess-21-5009-2017

    """

    #reading in constants
    bd = float(config_data['metadata']['bd'])
    lw = float(config_data['metadata']['lw'])
    soc = float(config_data['metadata']['soc'])
    Hveg = 0  # Hveg not used due to lack of reliable data and small impact.

    if math.isnan(bd):  #checking if bd value is a number.
        print("Bulk density (bd) is nan value. Please add a valid value to the config file.")

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~ CALIBRATION DATA READ AND TIDY ~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Read in the calibration data and do some tidying to it
    """
    print("Fetching calibration data...")
    # Read in Calibration data
    df = pd.read_csv(calib_data_filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])  # Dtype into DATE
    # Remove the hour part as interested in DATE here
    df['DATE'] = df['DATE'].dt.date

    unidate = df.DATE.unique()  # Make object of unique dates (calib dates)
    print("Unique calibration dates found: "+str(unidate))
    print("Done")

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~ FINDING RADIAL DISTANCES OF MEASUREMENTS ~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Using the LOC column which combines directional heading and radial
    distance from sensor. It splits this column to give raidal distance
    in metres.
    
    """
    
    df['DIST'] = df['DIST'].astype(float)  # dtype is float

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~ PROVIDE A SINGLE NUMBER FOR DEPTH ~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Calibration data has the depth given as a range in meters. For easier
    handling this is converted into an average depth of the range.
    
    """

    df['DEPTH_AVG'] = df['DEPTH_AVG'].astype(float)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~ SEPERATE TABLES FOR EACH CALIB DAY ~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    It will look at the number of unique days identified above. 
    It will then create seperate tables for each calibration day. 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Numdays is a number from 1 to n that defines how many calibration days there are.
    numdays = len(unidate)

    # Dict that contains a df for each unique calibration day.
    dfCalib = dict()
    for i in range(numdays):
        dfCalib[i] = df.loc[df['DATE'] == unidate[i]]

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~ AVERAGE PRESSURE FOR EACH CALIB DAY ~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    We need average pressure for the functions further down the script. This will 
    find the average pressure.
    """

    #call in corrected CRNS data
    corrected_data = corrected_data.replace(int(config_data['general']['noval']), np.nan)   #first must make sure -999s are converted to nan values.
    lvl1 = corrected_data #renamed the corrected CRNS data to lvl1 for short
    lvl1['DATE'] = pd.to_datetime(lvl1['DT'], yearfirst=True)  # Use correct formatting
    lvl1['DATE'] = lvl1['DATE'].dt.date     # Remove the time portion
    
    #is there external RH - crspy-lite requires this so could just remove all the input conditions
    if lvl1['E_RH'].mean() == int(config_data['general']['noval']):
        isrh = False                         #Check if external RH is available
        print("No external relative humidity detected. Please check input data.")    
    else:
        isrh = True
   
    lvl1[lvl1 == int(config_data['general']['noval'])] = np.nan

    # Creates dictionary of dfs for calibration days found
    dflvl1Days = dict()
    for i in range(numdays):
        dflvl1Days[i] = lvl1.loc[lvl1['DATE'] == unidate[i]]

    # Create a dict of avg pressure for each Calibday
    avgP = dict()
    for i in range(len(dflvl1Days)):
        tmp = pd.DataFrame.from_dict(dflvl1Days[i])
        tmp = tmp[(tmp['DT'] > str(unidate[i])+' '+str(calib_start_time)) & (tmp['DT']
                                                               <= str(unidate[i])+' '+str(calib_end_time))]  # COSMOS time of Calib
        check = float(np.nanmean(tmp['PRESS'], axis=0))

        if np.isnan(check):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            avgP[i] = float(np.nanmean(tmp['PRESS'], axis=0))
        else:
            avgP[i] = check
        # Very few sites had no data at time of COSMOS calib - if thats the case use day average

    avgT = dict()
    for i in range(len(dflvl1Days)):
        tmp = pd.DataFrame.from_dict(dflvl1Days[i])
        tmp = tmp[(tmp['DT'] > str(unidate[i])+' '+str(calib_start_time)) & (tmp['DT']
                                                               <= str(unidate[i])+' '+str(calib_end_time))]  # COSMOS time of Calib
        check = float(np.nanmean(tmp['TEMP'], axis=0))
        # Very few sites had no data at time of COSMOS calib - if thats the case use day average
        if np.isnan(check):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            avgT[i] = float(np.nanmean(tmp['TEMP'], axis=0))
        else:
            avgT[i] = check
    
    # Introduce flux AH possibility
    try:
        avgAH = dict()
        for i in range(len(dflvl1Days)):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            tmp = tmp[(tmp['DT'] > str(unidate[i])+' '+str(calib_start_time)) & (tmp['DT']
                                                               <= str(unidate[i])+' '+str(calib_end_time))]  # COSMOS time of Calib
            check = float(np.nanmean(tmp['E_AH_FLUX'], axis=0))
            # Very few sites had no data at time of COSMOS calib - if thats the case use day average
            if np.isnan(check):
                tmp = pd.DataFrame.from_dict(dflvl1Days[i])
                avgAH[i] = float(np.nanmean(tmp['E_AH_FLUX'], axis=0))
            else:
                avgAH[i] = check
    except:
        pass

    if isrh:
        avgRH = dict()
        for i in range(len(dflvl1Days)):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            tmp = tmp[(tmp['DT'] > str(unidate[i])+' '+str(calib_start_time)) & (tmp['DT']
                                                               <= str(unidate[i])+' '+str(calib_end_time))]  # COSMOS time of Calib
            check = float(np.nanmean(tmp['E_RH'], axis=0))
            # Very few sites had no data at time of COSMOS calib - if thats the case use day average
            if np.isnan(check):
                tmp = pd.DataFrame.from_dict(dflvl1Days[i])
                avgRH[i] = float(np.nanmean(tmp['E_RH'], axis=0))
            else:
                avgRH[i] = check
        
    if isrh == False:
        avgVP = dict()
        for i in range(len(dflvl1Days)):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            tmp = tmp[(tmp['DT'] > str(unidate[i])+' '+str(calib_start_time)) & (tmp['DT']
                                                               <= str(unidate[i])+' '+str(calib_end_time))]  # COSMOS time of Calib
            check = float(np.nanmean(tmp['VP'], axis=0))
            # Very few sites had no data at time of COSMOS calib - if thats the case use day average
            if np.isnan(check):
                tmp = pd.DataFrame.from_dict(dflvl1Days[i])
                avgVP[i] = float(np.nanmean(tmp['VP']))
            else:
                avgVP[i] = check
    print("Done")

    ##TODO: Introduce a check here to see if data is available


    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~ THE BIG ITERATIVE LOOP - CALCULATE WEIGHTED THETA ~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Below is the main loop that will iterate through the process oultined by 
    Schron et al., (2017). It iterates the weighting procedure until the defined
    accuracy is achieved. It will then repeat for as many calibration days there
    are.
    """

    AvgTheta = dict()  # Create a dictionary to place the AvgTheta into

    for i in range(len(dflvl1Days)):  # for i in number of calib days...

        print("Calibrating to day "+str(i+1)+"...")
        # Assign first calib day df to df1
        df1 = pd.DataFrame.from_dict(dfCalib[i])
        df1['SWV'] = pd.to_numeric(df1['SWV'], errors='coerce') #coerce non numeric values to nan values
        if df1['SWV'].mean() > 1:
            print("crspy-lite has detected that your volumetric soil water units (in the calibration data) are not in decimal format and so has divided them by 100. If this is incorrect please adjust the units and reprocess your data")
            # added to convert - handled with message plus calc
            df1['SWV'] = df1['SWV']/100

        CalibTheta = df1['SWV'].mean()           # Unweighted mean of SWV
        # Assign accuracy as 1 to be compared to in while loop below
        Accuracy = 1

        # COSMOS data needs some processing to get profiles - check if already available
        if "PROFILE" not in df1.columns:
            # Create a profile tag for each profile
            profiles = df1.LOC.unique()             # Find unique profiles
            # Defines the number of profiles
            numprof = len(profiles)

            # Following mini-loop will append an integer value for each profile
            pfnum = []
            for row in df1['LOC']:
                for j in range(numprof):
                    if row == profiles[j, ]:
                        pfnum.append(j+1)
            df1['PROFILE'] = pfnum

            # Now loop the iteration until the defined accuracy is achieved
        while Accuracy > defineaccuracy:
            # Initial Theta
            thetainitial = CalibTheta  # Save a copy for comparison
            """
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~~~ DEPTH WEIGHTING ~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            According to Schron et al., (2017) the first thing to do is find the 
            weighted average for each profile. This is done 
            """
            # Rescale the radius using rscaled function
            df1['rscale'] = df1.apply(lambda row: rscaled(
                row['DIST'], avgP[i], Hveg, thetainitial), axis=1)

            # Calculate the depth weighting for each layer
            df1['Wd'] = df1.apply(lambda row: Wd(
                row['DEPTH_AVG'], row['rscale'], bd, thetainitial), axis=1)
            df1['thetweight'] = df1['SWV'] * df1['Wd']

            # Create a table with the weighted average of each profile
            depthdf = df1.groupby('PROFILE', as_index=False)[
                'thetweight'].sum()
            temp = df1.groupby('PROFILE', as_index=False)['Wd'].sum()
            depthdf['Wd_tot'] = temp['Wd']
            depthdf['Profile_SWV_AVG'] = depthdf['thetweight'] / \
                depthdf['Wd_tot']
            dictprof = dict(zip(df1.PROFILE, df1.DIST))
            dictprof2 = dict(zip(df1.PROFILE, df1.rscale))

            depthdf['Radius'] = depthdf['PROFILE'].apply(lambda x: dictprof.get(x, None))
            depthdf['rscale'] = depthdf['PROFILE'].apply(lambda x: dictprof2.get(x, None))

           

            """
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~ ABSOLUTE HUMIDITY CALULCATION ~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            Need to provide absolute air humidity for each calibration day. This will be 
            taken from data (vapour pressure and temperature) for the USA sites. 
            Using the pv function from hum_funcs.py this can be calculated by providing 
            vapour pressure and temperature.
            """
            if isrh == True:
                day1temp = avgT[i]
                day1rh = avgRH[i]
                day1es = es(day1temp)
                day1ea = ea(day1es, day1rh)
                day1hum = pv(day1ea, day1temp)

            else:
                day1temp = avgT[i]
                day1vp = avgVP[i]
                
                # Calculate absolute humidity (output will be kg m-3).
                day1hum = pv(day1vp, day1temp)
                # Multiply by 1000 to convert to g m-3 which is used by functions
                day1hum = day1hum * 1000
            
            try:
                if np.isnan(day1hum):
                    day1hum = avgAH[i]
                    print("Using flux AH")
            except:
                pass

            # Need to add value to each row for .loc application
            depthdf['day1hum'] = day1hum
            depthdf['TAVG'] = day1temp
            depthdf['Wr'] = "NaN"  # set up column

            """
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~ RADIAL WEIGHTING ~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            Once each profile has a depth averaged value, and we have the absolute
            humidity, we can begin to find the final weighted value of theta. Three
            different functions are utilised below depending on the distance from
            sensor.
            """
            # Below three lines applies WrN function based on radius of the measurement
            depthdf.loc[depthdf['Radius'] > 50, 'Wr'] = WrB(
                depthdf.rscale, depthdf.day1hum, (depthdf.TAVG / 100))
            depthdf.loc[(depthdf['Radius'] > 5) & (depthdf['Radius'] <= 50), 'Wr'] = WrA(
                depthdf.rscale, depthdf.day1hum, (depthdf.TAVG / 100))
            depthdf.loc[depthdf['Radius'] <= 5, 'Wr'] = WrX(
                depthdf.rscale, depthdf.day1hum, (depthdf.TAVG / 100))

            depthdf['RadWeight'] = depthdf['Profile_SWV_AVG'] * depthdf['Wr']

            FinalTheta = depthdf.sum()

            try:
                CalibTheta = FinalTheta['RadWeight'] / FinalTheta['Wr']
            except ZeroDivisionError:
                print("A zero division was attempted. This usually indicates missing data around the calibration date.")
            AvgTheta[i] = CalibTheta


            """
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~~ WRITE TABLES ~~~~~~~~~~~~~~~~~
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Can output the radius weighting and depth weighting here by uncommenting the two lines below.
            """
            os.chdir(default_dir + "/outputs/data")
            #depthdf.to_csv(country + '_SITE_'+ sitenum + '_RadiusWeighting' + str(unidate[i]) + '.csv', header=True, index=False,  mode='w')  
            #df1.to_csv(country + '_SITE_'+ sitenum + '_DepthWeighting' + str(unidate[i]) + '.csv', header=True, index=False,  mode='w')
            os.chdir(default_dir)

            Accuracy = abs((CalibTheta - thetainitial) / thetainitial)  # Calculate new accuracy
       
        print("Done")

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~ OPTIMISED N0 CALCULATION ~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    We now check the optimal N0 value for this sensor. This is completed by iterating
    through calculations of soil moisture for each calibration date with the N0 value
    being replaced with values from 0 - 10,000. We can then see what the accuracy is
    for each possible N0 value. 
    
    To optimise across calibration days these accuracy values are then summed for each
    N0 value across the calibration days. The lowest error value is then taken as the 
    correct N0.
    
    """
    print("Finding Optimised N0......")

    tmp = corrected_data   #temporary df of the corrected data.
    tmp = tmp.replace(int(config_data['general']['noval']), np.nan) #make sure -999s are nan values.
    tmp['MOD_CORR'] = tmp['MOD_CORR'].replace([np.inf, -np.inf], np.nan)  #brought in to avoid infinite values bug
    tmp['MOD_CORR'] = pd.to_numeric(tmp['MOD_CORR'], errors='coerce')   #make sure missing data is 'coerced' to nan values 
    n_avg = (np.nanmean(tmp['MOD_CORR']))
  
    if np.isnan(n_avg) or np.isinf(n_avg):
        print("Error: MOD_CORR contains only NaN values or results in infinity.")
        n_avg = 0  # or any other default value you want to use
    else:
        n_avg = int(n_avg)
   
    if n_avg >= 4000:
        print("Avg N for the site is  "+str(n_avg)+" due to presumed errors in raw data. It has been capped at 3,000")
        n_avg = 3000
        """
        TODO need to reconsider this for broader issue. What value for n_avg when averages are way off.
        For now hard code 3000, but this will vary depending on sensor/location. Perhaps some kind of mode?
        """
    else:
        print("Avg N is for the site is  "+str(n_avg))
    tmp['DATE'] = pd.to_datetime(tmp['DT'], format="%Y-%m-%d %H:%M:%S")
    
    tmp['DATE'] = tmp['DATE'].dt.date  # Remove the time portion to match above

    NeutCount = dict()  # Create a dict for appending time series readings for each calib day
    for i in range(numdays):  # Create a df for each day with the CRNS readings
        # Create a dictionary df of neutron time series data for each calibration day
        tmpneut = tmp.loc[tmp['DATE'] == unidate[i]]

        NeutCount[i] = tmpneut

        #Optional output - uncomment below if you wish to see the result.
        os.chdir(default_dir + "/outputs/data")
        #tmpneut.to_csv(country + '_SITE_'+sitenum+'_MOD_AVG_TABLE_' + str(unidate[i]) + '.csv', header=True, index=False,  mode='w')
        os.chdir(default_dir)

    avgN = dict()
    for i in range(len(NeutCount)):
        # Find the daily mean neutron count for each calibration day
        tmp = pd.DataFrame.from_dict(NeutCount[i])
        tmp = tmp[(tmp['DT'] > str(unidate[i])+' '+str(calib_start_time)) & (tmp['DT']
                                                               <= str(unidate[i])+' '+str(calib_end_time))]  # COSMOS time of Calib
        check = float(np.nanmean(tmp['MOD_CORR']))
        

       # Need another catch to stop errors with missing data
        if np.isnan(check):
            # Find the daily mean neutron count for each calibration day
            tmp = pd.DataFrame.from_dict(NeutCount[i])
            avgN[i] = float(np.nanmean(tmp['MOD_CORR']))
        else:
            avgN[i] = check

    RelerrDict = dict()
    with np.errstate(divide='ignore'):  # prevent divide by 0 error message
        for i in range(numdays):
            # Create a series of N0's to test from 0 to 10000
            N0 = pd.Series(range(n_avg, int(n_avg*2.5)))
            # Avg theta divided by 100 to be given as decimal
            vwc = AvgTheta[i]
            Nave = avgN[i]  # Taken as average for calibration period
            sm = pd.DataFrame(columns=['sm'])
            reler = pd.DataFrame(columns=['RelErr'])

            if sm_calc_method == 'desilets':

                for j in range(len(N0)):

                    sm.loc[j] = desilets_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), float(config_data['general']['a2']), 
                                            bd, Nave, N0.loc[j], lw, soc)
                
                    tmp = sm.iat[j, 0]
                    reler.loc[j] = abs(sm.iat[j, 0] - vwc)

                
            elif sm_calc_method == 'kohli':
                
                for j in range(len(N0)):
                
                    sm.loc[j] = kohli_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), float(config_data['general']['a2']), 
                                            bd, Nave, N0.loc[j], lw, soc)
                
                    tmp = sm.iat[j, 0]
                    reler.loc[j] = abs(sm.iat[j, 0] - vwc)
            else:
                KeyError("No other method to calculate soil moisture at this time. Please use either 'desilets' or 'kohli' method.")
                

            RelerrDict[i] = reler['RelErr']

            # Optional output for the calibration error
            os.chdir(default_dir + "/outputs/data")
            reler['N0'] = range(n_avg, int(n_avg*2.5))  # Add N0 for csv write
            #reler.to_csv(country + '_SITE_'+sitenum+'_error_' + str(unidate[i]) + '.csv',header=True, index=False,  mode='w')
            os.chdir(default_dir)

    """
                        N0 Optimisation
    
    Need to optimise N0 based on all calibration days. This is done by taking a total
    mean from all sites. e.g. when N0 = 2000, sum mean for each calibration day at N0=2000.
    
    This is then used as our error calculation so we minimise for that to give us N0.
    """

    totalerror = RelerrDict[0]         # Create a total error series
    # Range is number of calibration days - 1 (as already assigned day 1)
    for i in range(len(unidate)-1):
        # Create a tmp series of the next calibration day
        tmp = RelerrDict[i+1]
        # Sum these together for total mean (will continue until all days are summed)
        totalerror = totalerror+tmp

    minimum_error = min(totalerror)  # Find the minimum error value and assign

    totalerror = totalerror.to_frame()
    totalerror['N0'] = range(n_avg, int(n_avg*2.5))
    
    # Create object that maintains the index value at min
    minindex = totalerror.loc[totalerror.RelErr == minimum_error]
    N0 = minindex['N0'].item()
    
    #plotting the relative error for each value of N0 - the lowest error should indicate the true N0 value for the calibration!
    plt.plot(totalerror['N0'], totalerror['RelErr'])    
    plt.yscale('log')
    plt.xlabel('N0')
    plt.ylabel('Sum Relative Error (log scale)')
    plt.title('Sum Relative Error plot on log scale across all calibration days')
    plt.legend()
    os.chdir(default_dir + "/outputs/figures")  # Change wd to folder
    plt.savefig(country+"_"+sitenum+"Relative_Error_Plot.png")
    os.chdir(default_dir)
    plt.close()

    """
                            User Report
                            
    This report is going to outline all the variables and calculations that are used
    in the automated process. It is essential for the user to read through this and 
    identify if there is anything that seems incorrect.
    
    Examples:   Too many/few calibration days
                TempAvg is unrealistic
                Calibration Dates are incorrect
                Any values that appear beyond reasonable
                
    This should work correctly however mistakes are possible if, for example, the 
    data structure is wrong. This could cause knock on effects.
    
    """

    N0R = "The optimised N0 is calculated as: \nN0   |  Total Relative Error  \n" + str(minindex)
    R1 = "The site calibrated was site number " + sitenum + " in  " + str(country) + " and the name is " + str(config_data['metadata']['sitename'])
    R2 = "The bulk density was " + str(bd)
    R3 = "The vegetation height was " +str(Hveg)+ " (m)"
    R4 = "The user defined accuracy was "+str(defineaccuracy)
    R5 = "The soil organic carbon was "+str(soc) + " g/m^3"
    R6 = "The lattice water content was " + str(lw)
    R7 = "Unique calibration dates where on: \n" + str(unidate)
    RAvg = "Average neutron counts for each calib day where " + str(avgN)
    Rtheta = "The weighted field scale average of theta (from soil samples) was "+str(
        AvgTheta)
    R8 = "Please see the additional tables which hold calculations for each calibration date"

    os.chdir(default_dir + "/outputs/data")
    # Write the user report to the file below
    f = open(country + "_"+sitenum+"_Report.txt", "w")
    f.write(N0R + '\n\n' + R1 + '\n' + R2 + '\n' + R3 + '\n' + R4 + '\n' + R5 + '\n' + R6 + '\n' +
            '\n \n' + R7 + '\n\n' + RAvg + '\n \n' + Rtheta + '\n \n' + R8)
    f.close()

    # Write total error table - optional output, uncomment below.
    totalerror = pd.DataFrame(totalerror)
    totalerror['N0'] = range(n_avg, int(n_avg*2.5))
    #totalerror.to_csv(country+'_SITE_'+sitenum +'totalerror.csv', header=True, index=False,  mode='w')
    os.chdir(default_dir)

    config_data['metadata']['n0'] = N0
    #save n0 input to config file
    config_file = "inputs/config.json"
    with open(config_file, "w") as file:
        json.dump(config_data, file, indent=4)
   
    print("Calibration successful. N0 = "+str(N0))

    print("N0 for " + str(unidate) + "data = " + str(N0))

    return N0


#~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~ (6) QUALITY ANALYSIS ~~# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~#

def flag_and_remove(df, N0, config_data):
    
    """This function identifies erroneous data that should be flagged and removed based upon the following criteria:
    
    Flags:
        1 = fast neutron counts more than 20% difference to previous count
        2 = fast neutron counts less than the minimum count rate (default == 30%, can be edited in the config.json file)
        3 = fast neutron counts more than n0
        4 = battery below 10v

    Parameters
    ----------
    df : dataframe
        dataframe of the CRNS data
    
    N0: int
        Theoretical neutron count for dry soil conditions (the calibration value for the site)
    
    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.
   
    Returns
    -------
    df : DataFrame
        Quality controlled DataFrame for the site.

    """
    print("Performing quality control checks on corrected neutron counts...")

    idx = df['DT']
    idx = pd.to_datetime(idx)
   
    # Reset index incase df is from another process and is DT 
    df = df.reset_index(drop=True)
    df2 = df.copy()
    df2['FLAG'] = 0  # initialise FLAG to 0
    
    # Flag to remove above N0*1.075
    df2.loc[df2.MOD_CORR > (N0 * 1.075), "FLAG"] = 3
    df = df.drop(df[df.MOD_CORR > (N0 * 1.075)].index)   # drop above N0

    df2.loc[(df2.MOD_CORR < (N0*(int(config_data['general']['belowN0'])/100))) &
            (df2.MOD_CORR != int(config_data['general']['noval'])), "FLAG"] = 2
    
    # drop below 0.3 N0
    df = df.drop(df[df.MOD_CORR < (N0*(int(config_data['general']['belowN0'])/100))].index)
    df2.loc[(df2.BATT < 10) & (df2.BATT != int(config_data['general']['noval'])), "FLAG"] = 4
    df = df.drop(df[df.BATT < 10].index)

    # Drop >20% diff in timestep
    moddiff = [0]
    tmpindex = list(df.index)

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
    
    diff1 = df.loc[(df['PRCNTDIFF'] > int(config_data['general']['timestepdiff'])), "INDEX_TMP"]
 
    diff2 = df.loc[(df['PRCNTDIFF'] < (-int(config_data['general']['timestepdiff']))), "INDEX_TMP"]

    df2.loc[diff1, "FLAG"] = 1
    df2.loc[diff2, "FLAG"] = 1

    df = df.drop(df[df.PRCNTDIFF > int(config_data['general']['timestepdiff'])].index)
    df = df.drop(df[df.PRCNTDIFF < (-int(config_data['general']['timestepdiff']))].index)


    # Fill in master time again after removing
    # Need this to handle below code
    df.replace(int(config_data['general']['noval']), np.nan, inplace=True)
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")

    df = df.set_index(df.DT)
    
    idx2 = pd.date_range(
        df.DT.iloc[0], df
        .DT.iloc[-1], freq='1h')

    dtcol = df['DT']
    df = df.rename(columns={'DT': 'DT2'})
    df = df.drop_duplicates(subset='DT2')
    df.drop(labels=['DT2'], axis=1, inplace=True)
    df = df.reindex(index=idx2, fill_value=np.nan)

    df2 = df2.rename(columns={'DT': 'DT2'})
    df2 = df2.drop_duplicates(subset='DT2')
    df2.drop(labels=['DT2'], axis=1, inplace=True)
    df2 = df2.reindex(index=idx2, fill_value=np.nan)

    flagseries = df2['FLAG']
    df['FLAG'] = flagseries.values  # Place flag vals back into df
    df['DT'] = df.index
    
    df = df.drop(["DIFF", "PRCNTDIFF", "INDEX_TMP"], axis=1)

    insertat = len(df.columns)-3
    dtcol = df['FLAG']
    df.drop(labels=['FLAG'], axis=1, inplace=True)  # Move FLAG to first col
    df.insert(insertat, 'FLAG', dtcol)
    
    df.replace({config_data['general']['noval']: np.nan})
    
    print("Done")

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~ (7) SOIL MOISTURE ESTIMATION ~~# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def mod_error(mod):
    
    """Function to calculate the error of the neutron counts (based upon the coefficient of variation). 
    This function is required to estimate the soil moisture error, calculated in the function "thetaprocess()".

    Parameters
    ----------
    mod : Series
        nuetron count (uncorrected)
        

    Returns
    -------
    mod_err : Series
        Estimated error.

    """

    m = mod.mean()  #mean
    sd = (abs(mod)).apply(np.sqrt)  #standard deviation
    cv = sd / m  #coefficient of variation
    mod_err = mod * cv

    return mod_err


def sm_max_calc(bd, density):
    
    """Calculates a theoretical maximum of soil moisture for a site. This is based upon the bulk density.

    Parameters
    ----------
    bd : float
        bulk density
        
    density : float
        density

    Returns
    -------
    sm_max : float
        Maximum value of soil moisture for the site.
    
    """

    sm_max = (1-(bd/density)) 

    return sm_max


def thetaprocess(df, N0, sm_calc_method, config_data, default_dir):
    
    """This function takes the dataframe provided by the previous steps and uses the theta calculations
    to give an estimate of soil moisture. 

    Soil moisture can be estimated with either the desilets or kohli equation. See the crspy-lite functions "desilets_sm_calc()" 
    and "kohli_sm_calc()" functions for more detail on what these are and how they are different.

    Realistic values are provided by constraining soil moisture estimates to physically possible values.
        i.e. A minimum & maximum value are determined (A minimum of 0 and a maximum value based on the porosity at a site).

    The function also provides an estimated depth of measurement using the D86 function from Shcron et al., (2017)

    Gives running averages of measurements using a 12 hour window. To handle missing 
    data a minimum of 6 hours of data per 12 hour window is required, otherwise one
    missing hour could lead to large gaps in 12 hour means. 

    Parameters
    ----------
    df : dataframe 
        dataframe of CRNS data
    
    N0: int
        Theoretical neutron count for dry soil conditions (the calibration value for the site)

    sm_calc_method: str
        Must either be 'desilets' for desilets method or 'kohli' for kohli method.
        
    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.

    default_dir : str
        Default directory for folder creation and outputs.
        

    Returns
    -------
    df : DataFrame
        Final DataFrame for the site. Includes columns for estimated soil moisture, soil moisture error and effective depth. 

        
    References
    ----------
    Desilets et al. (2010) : https://doi.org/10.1029/2009WR008726
    
    Kohli et al (2021) : https://doi.org/10.3389/frwa.2020.544847

    Schron et al (2017) : https://doi.org/10.5194/hess-21-5009-2017
    
    """
    
    df.replace({config_data['general']['noval']: np.nan})
    
    #~~~~~ CONSTANTS ~~~~~#

    #reading in constants
    lw = float(config_data['metadata']['lw'])
    soc = float(config_data['metadata']['soc'])
    bd = float(config_data['metadata']['bd'])
    N0 = float(config_data['metadata']['n0'])              

    #setting maximum value of soil moisture
    if config_data['metadata']['sm_max'] is None:
        
        print("No maximum soil moisture value found in config.json file. Calculating sm_max from bulk density data...")
        sm_max = sm_max_calc(bd=bd, density=float(config_data['general']['density']))
        print("Done.")
        config_file = default_dir + "/inputs/config.json"
        config_data['metadata']['sm_max'] = round(sm_max, (4))
        with open(config_file, "w") as file:
            json.dump(config_data, file, indent=4)

    else:
        
        sm_max = float(config_data['metadata']['sm_max'])
        print("Maximum soil moisture value aquired from metadata.")
        
    # convert SOC to water equivelant (see Hawdon et al., 2014)
    wsom = soc * 0.556
    
    #setting minimum value of soil moisture
    sm_min = 0  
    
    #~~~~~ ESTIMATING ERROR IN NEUTRON COUNTS ~~~~~#
  
    df['MOD_ERR'] = round(mod_error(mod = df['MOD']))
    
    print("Calculating soil moisture with estimated error...")
    
    # Creating MOD count to min and max of error
    df['MOD_CORR_PLUS'] = df['MOD_CORR'] + df['MOD_ERR']
    df['MOD_CORR_MINUS'] = df['MOD_CORR'] - df['MOD_ERR']
    
    #~~~~~ SOIL MOISTURE CALC ~~~~~#

    if sm_calc_method == "desilets":
        
        # Calculating soil moisture for MOD_CORR, +MOD_ERR and -MOD_ERR. Note inverse relationship so use MOD minus for soil moisture positive Error)
        
        df['SM'] = df.apply(lambda row: desilets_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), 
                                        float(config_data['general']['a2']), bd, row['MOD_CORR'], N0, lw, wsom), axis=1)
    
        df['SM_RAW'] = df['SM'] #copying SM column to perform calculations.

        df['SM_PLUS_ERR'] = df.apply(lambda row: desilets_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), 
                            float(config_data['general']['a2']), bd, row['MOD_CORR_MINUS'], N0, lw, wsom), axis=1) 
        
        df['SM_PLUS_ERR'] = abs(df['SM_PLUS_ERR'] - df['SM'])

        df['SM_MINUS_ERR'] = df.apply(lambda row: desilets_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), 
                                float(config_data['general']['a2']), bd, row['MOD_CORR_PLUS'], N0, lw, wsom), axis=1)
        
        df['SM_MINUS_ERR'] = abs(df['SM_MINUS_ERR'] - df['SM'])
    

    elif sm_calc_method == "kohli":

        # Exact same sm and sm error calculation process as the desilets method. The only difference is in the processing equation.
        
        df['SM'] = df.apply(lambda row: kohli_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), 
                                                      float(config_data['general']['a2']), bd, row['MOD_CORR'], N0, 
                                                      lw, wsom), axis=1)
        df['SM_RAW'] = df['SM']

        df['SM_PLUS_ERR'] = df.apply(lambda row: kohli_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), 
                                                               float(config_data['general']['a2']), bd, row['MOD_CORR_MINUS'], N0, lw,
                                                                wsom), axis=1)  
        df['SM_PLUS_ERR'] = df['SM_PLUS_ERR']
        df['SM_PLUS_ERR'] = abs(df['SM_PLUS_ERR'] - df['SM'])

        df['SM_MINUS_ERR'] = df.apply(lambda row: kohli_sm_calc(float(config_data['general']['a0']), float(config_data['general']['a1']), 
                                                                float(config_data['general']['a2']), bd, row['MOD_CORR_PLUS'], N0, lw,
                                                              wsom), axis=1)
        df['SM_MINUS_ERR'] = df['SM_MINUS_ERR']
        df['SM_MINUS_ERR'] = abs(df['SM_MINUS_ERR'] - df['SM'])   

    else:
        KeyError("No other method to calculate soil moisture at this time. Please use either 'desilets' or 'kohli' method.")

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
    df['SM_12h'] = df['SM'].rolling(int(config_data['general']['smwindow']), min_periods=6).mean()

    #~~~~~ EFFECTIVE DEPTH ~~~~~~#
    # Depth calcs - using Schron et al (2017). Depth is given considering radius and bd   
    
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
    df['D86avg_12h'] = df['D86avg'].rolling(window=int(config_data['general']['smwindow']), 
                                            min_periods=6).mean()  
    
    #~~~ FINAL TIDY OF DF ~~~#
    df.fillna(int(config_data['general']['noval']), inplace=True)  # Replace nans with -999
    df = df.round(3)    #round df
    df[['MOD', 'MOD_CORR']] = (df[['MOD', 'MOD_CORR']]).round().astype(int) #neutron counts must be integers
    df['DATETIME'] = df['DT']   #expand DT name back out to DATETIME
    #select & reorder cols
    final_order_of_cols = ['DATETIME', 'MOD', 'PRESS', 'TEMP', 'E_RH', 'I_RH', 'BATT', 'RAIN',     
                           'F_PRESSURE', 'F_HUMIDITY', 'F_INTENSITY', 'F_VEGETATION', 'MOD_CORR',
                           'FLAG', 'SM', 'D86avg', 'MOD_ERR', 'SM_PLUS_ERR', 'SM_MINUS_ERR', 'SM_12h', 'D86avg_12h']
    df = df[final_order_of_cols]       
    
    print("Done")

    return df



