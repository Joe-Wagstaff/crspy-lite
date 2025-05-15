<img src="assets/logo.png" alt="crspy-lite logo" width="250"/>

Cosmic-Ray neutron Sensing PYthon tool lite edition (**crspy-lite**): A Cosmic-Ray Neutron Sensing (**CRNS**) processing tool for soil moisture estimation.

*@author: Joe Wagstaff*

*@institution: University of Bristol - Department of Civil Engineering*

## Overview

Crspy-lite is a single site data processing and visualisation tool for CRNS and soil moisture estimation. It aims to simplify CRNS data processing so that the technology is more accessible and easy to use. The key development goals of crspy-lite were hence to: 

* Create a Python tool that is quick and easy to use.
* Incorporate data visualisation that allows for more detailed, efficient and site-specific analysis.
* Integrate the most up-to-date processing methods to maintain relavancy to the community and enhance crspy-lite's applications as a research tool. 

For an understanding of [CRNS theory](https://github.com/Joe-Wagstaff/crspy-lite/wiki/CRNS-Theory) please explore the wiki pages attached.

## Installation 

You can install **crspy-lite** on your local machine directly from PyPI using pip. Simply open terminal and type:

```bash
pip install crspy_lite  
```

Or to upgrade to the latest version of the tool:

```bash
pip install --upgrade crspy_lite
```

## Running crspy-lite

To run the code you need only 2 things: (i) Input time-series data from the CRNS, and (ii) a list of metadata variables. For information on the format and requirements of these please go to the wiki pages [Input Timeseries Data](https://github.com/Joe-Wagstaff/crspy-lite/wiki/Input-Timeseries-Data) and [Metadata Input](https://github.com/Joe-Wagstaff/crspy-lite/wiki/Metadata-Input) respectively. To then process and visualise your data simply open terminal and type:

```bash
crspy_lite
```

Or to specify the directory where outputs are stored (as opposed to the working directory that is the default), navigate to your chosen folder and Press Ctrl + Shift + C to copy the full path. Then paste this into the terminal after "crspy_lite" as exemplified below:

```bash
crspy_lite "C:\Users\YourName\Documents\your-folder"
```

If this is the first time the directory has been used, you will be prompted to input all requirements into the terminal. Optional requirements are highlighted and can be left empty. Once this is completed crspy-lite will generate folders and then convert your inputs into a `config.json` file. This can then be accessed by going to `"\your-folder\inputs\config.json"`. To change the inputs simply open, edit and then save the `config.json` file. The code can then be re-run (exactly the same as above) to incorporate the changes. The `config.json` file also contains variables that can be edited to personalise the processing such as different research methodologies, limits and visualisation features. For more information on this please go to the wiki page [Metadata Input](https://github.com/Joe-Wagstaff/crspy-lite/wiki/Metadata-Input).

Once crspy-lite is run, all time-series outputs are saved to `"\your-folder\outputs\data"`, with all graphical outputs saved to `"\your-folder\outputs\figures"`.

### Examples

Please use the links below to see examples of using crspy-lite:

* Test data and metdata (maybe a config file)


## Author

Crspy-lite was developed by Joe Wagstaff as part of a research project at the University of Bristol, under the supervision of Dr Rafael Rosolem. For any issues, feedback or thoughts on the future development of the tool please get in touch at joe.d.wagstaff@gmail.com. The code was originally based on the multi-site processing tool **crspy** by Power et al. (2021) (see> https://github.com/danpower101/crspy), however has been modified to optimise CRNS processing for individual sites, incorporate data visualisation into the tool and improve the user-friendliness.
