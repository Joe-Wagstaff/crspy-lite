# **crspy_lite**.

Cosmic-Ray neutron Sensor PYthon tool - lite edition.

@author: Joe Wagstaff
@institution: University of Bristol

**About this tool:**

* This tool can be used for single site Cosmic-Ray Neutron Sensor (CRNS) data processing. It computes **S**oil **M**oisture (**SM**) estimates from raw neutron counts detected by the sensors.

* The tool has been created from the multi-site CRNS data processing tool *crpsy* Power et al. (2021), for single site CRNS data processing (see> https://github.com/danpower101/crspy).

* User-friendliness was the primary focus behind the creation of *crspy-lite*. It is intended to make CRNS data processing and analysis as easy as possible.

* To run the tool simply run the *full_process_wrapper.py* script. Follow the user prompts in your terminal to process the CRNS data for your site. 

* All data processing functions are stored in *processing_functions.py*. All functions contain a description of what they do and the parameters they involve. References are also given for users 
  who wish to explore the processing steps and understand them further.  Additional information on the code is then added with comments (#).

* *figures.py* can then be run to output some graphical figures. This is expected to be updated in the near future.
  
 **Before running:** 

* Save the python scripts to a desired file directory. Note that all files and figures will be outputted from the working directory.

* Ensure you have the relevent metadata variables for your site. *crspy-lite* will ask for these to be inputted in the terminal when run. Full infomation on the metadata input requirements is 
  given [here](https://github.com/Joe-Wagstaff/CRSPY_lite/wiki/Metadata-Input)


**figures_code.py**

* This script imports the final dataframe from the previous script and includes code for a range of different figures. Toggle the Boolean statements where the figure functions are called to choose whether to output a particular figure or not.
* The recomended figures to output are **site_features** which includes subplots for temperature, precpitation and SM (including an estimated error) and **typical_year** which averages the data to compute an average year of SM. Note that for an accurate representation, several years of data are required.
* Another optional figure is **crspy_comparison_plt** which plots the CRSPY_lite output against the output from the multi-site CRNS data processing python tool, CRSPY (see> https://github.com/danpower101/crspy). This can be used to compare the two tools.














