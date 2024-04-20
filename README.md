# **CRSPY_lite**.

Cosmic-Ray neutron Sensor PYthon tool - lite edition.

Joe Wagstaff

**About this package:**

* This code can be used for single site Cosmic-Ray Neutron Sensor (CRNS) data processing. It computes **S**oil **M**oisture (**SM**) estimates from raw neutron counts detected by the sensors. The code requires the input of raw site data and a list of metadata variables.

* The package simplifies the multi-site CRNS data processing tool CRSPY created by Power et al. (2021), for single site CRNS data processing (see> https://github.com/danpower101/crspy).
 
* Functions contain a description of what they do and the parameters they involve. Additional information on the code is added with comments (#).

* The scipts must be run in the following order: 'setup.py', 'raw_to_theta.py' and then 'figures_code' for the graphical outputs.

 **Before running:** 

* Save the three python scripts to a desired file directory. Note that all files and figures will be outputted from this directory.

* Save a copy of the site metadata either as a .csv (comma-delimited) file or .txt file to the working directory and name it **metadata.csv** or **metadata.txt** respectively. Note that both need to follow their respective format given by **metadata_template.csv** and **metadata_template.txt**.

* Full info on the metadata input is given [here](https://github.com/Joe-Wagstaff/CRSPY_lite/wiki/Metadata-Input)

**setup.py**

* This script establishes the working directory, creates the file structure and creates a config.ini file. The config.ini file contains general variables required for the processing, variables from the inputted metadata file and filepaths of the working directory and raw data.

**raw_to_theta.py**

* This script contains the full CRNS data processing procedure from the raw neutron counts detected by the sensor to SM estimates.
* The data is outputted at several key processing stages. These include: the tidyed data, the level 1 (or fully corrected) data, the quality-controlled data and the final dataframe containing the SM estimates.

**figures_code.py**

* This script imports the final dataframe from the previous script and includes code for a range of different figures. Toggle the Boolean statements where the figure functions are called to choose whether to output a particular figure or not.
* The recomended figures to output are **site_features** which includes subplots for temperature, precpitation and SM (including an estimated error) and **typical_year** which averages the data to compute an average year of SM. Note that for an accurate representation, several years of data are required.
* Another optional figure is **crspy_comparison_plt** which plots the CRSPY_lite output against the output from the multi-site CRNS data processing python tool, CRSPY (see> https://github.com/danpower101/crspy). This can be used to compare the two tools.














