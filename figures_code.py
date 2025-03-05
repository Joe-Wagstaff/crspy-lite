
"""
@author: Joe Wagstaff
@institution: University of Bristol

"""
#standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#bokeh imports. bokeh is a plotting library used to create interactive figures in crspy-lite (not required for standard plots).
from bokeh.plotting import figure, show
from bokeh.io import curdoc
from bokeh.models import DatetimeTickFormatter, NumeralTickFormatter
from bokeh.models import HoverTool, BoxZoomTool, PanTool, ResetTool, SaveTool, WheelZoomTool
from bokeh.models import Div, RangeSlider, Spinner
from bokeh.layouts import layout
from bokeh.models import BoxAnnotation
from bokeh.layouts import column, row

#crspy-lite imports
import processing_functions as pf


def standard_plots(df, config_data):
    
    """Function that creates a series of standard plots for the a given CRNS site. Uses the python module MatPlotLib.
    
    Parameters
    ----------
    df : DataFrame  
        Processed data.
        
        
    Returns
    -------
    """

    #Prepare df for plotting.
    df = df.replace(int("-999"), np.nan)  #replace -999s for NaN
    df = df.replace(['0.0', None], pd.NA) #Replace 0s or None with NaN (needed for MOD col)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')
    df_by_day = df.groupby([df['DATETIME'].dt.year, df['DATETIME'].dt.month, df['DATETIME'].dt.day])  #group df by day (converts from hourly intevals to daily).
    day_series = df['DATETIME'].dt.to_period('D') #day series
    day_series = (day_series.drop_duplicates()).reset_index()
    day_series = day_series.astype(str)
    day_col = day_series['DATETIME']
    day_col = pd.to_datetime(day_col) #need to convert this to a datetime 

    #collect columns - averaging for daily resolution.
    raw_counts_daily = df_by_day['MOD'].mean()
    corr_counts_daily = df_by_day['MOD_CORR'].mean()
    fp_daily = df_by_day['F_PRESSURE'].mean()
    fh_daily = df_by_day['F_HUMIDITY'].mean()
    fi_daily = df_by_day['F_INTENSITY'].mean()

    sm_daily = df_by_day['SM'].mean()
    df['SM_PLUS_ERR'] = df['SM'] + abs(df['SM_PLUS_ERR']) # we want sm +/- the sm errors
    df['SM_MINUS_ERR'] = df['SM'] - abs(df['SM_MINUS_ERR']) 
    sm_plus_err_daily = df_by_day['SM_PLUS_ERR'].mean()
    sm_minus_err_daily = df_by_day['SM_MINUS_ERR'].mean()
    effective_depth_daily = df_by_day['D86avg'].mean()

    #~~~ CREATE FIGURE ~~~#
    plt.rcParams['font.size'] = 16
    fig, axs1 = plt.subplots(4, sharex=True, figsize=(15, 12))
    axs1[0].set_title(config_data['metadata']['sitename'] + "_SITE_" + config_data['metadata']['sitenum'] + "Standard Plots")

    #Subplot 1: Uncorrected vs Corrected Neutron Counts
    axs1[0].plot(day_col, raw_counts_daily, lw=0.8, color='black', label='Raw Counts')
    axs1[0].plot(day_col, corr_counts_daily, lw=0.8, color='red', label='Corrected Counts') 
    axs1[0].set_ylabel("Neutron Count (cph)")
    axs1[0].legend()

    #Subplot 2: Correction Factors
    axs1[1].plot(day_col, fp_daily, lw=0.8, color='red', label="Pressure")
    axs1[1].plot(day_col, fh_daily, lw=0.8, color='green', label="Humidity")
    axs1[1].plot(day_col, fi_daily, lw=0.8, color='blue', label="Intensity") 
    axs1[1].set_ylabel("Correction Factor")
    axs1[1].legend()

    #Subplot 3: Soil Moisture
    axs1[2].plot(day_col, sm_daily, lw=0.8, color='black', label="Soil Moisture")
    axs1[2].fill_between(day_col, sm_minus_err_daily, sm_plus_err_daily, alpha=0.5, edgecolor='#CC4F1B', facecolor='firebrick', label='Error')
    axs1[2].set_ylabel("Daily SM (cm$^3$ / cm$^3$)")
    axs1[2].legend()

    #Subplot 4: Sensing Depth
    axs1[3].plot(day_col, effective_depth_daily, lw=0.8, color='orange', label="Depth")
    axs1[3].invert_yaxis()  #want to show the plot as depth (so 0 at the top of the y-axis)
    axs1[3].set_ylabel("Effective Depth (cm)")

    #Format x-axis
    axs1[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.xticks(fontsize=14)

    plt.tight_layout()
    fig.savefig(config_data['filepaths']['default_dir']+"/outputs/figures/standard_plots.png", dpi=350)
    plt.show()   

    return



def plot_page(df):
    
    """ This function outputs an HTML file containing a preset selection of figures for the processed data of 
        a CRNS site.
        
        Parameters
        ----------
        df : DataFrame  
            Processed data.
        
        
        
            
        Returns
        -------
        
        
        """
    
    df.replace(int("-999"), np.nan, inplace=True)

    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')
    df_by_day = df.groupby([df['DATETIME'].dt.year, df['DATETIME'].dt.month, df['DATETIME'].dt.day])  #group df by day (converts from hourly intevals to daily)
   
    day_series = df['DATETIME'].dt.to_period('D')
    day_series = (day_series.drop_duplicates()).reset_index()
    day_series = day_series.astype(str)
    day_col = day_series['DATETIME']
    
    #collect averaged columns
    raw_counts_daily = df_by_day['MOD'].mean()

    corr_counts_daily = df_by_day['MOD_CORR'].mean()

    fp_daily = df_by_day['F_PRESSURE'].mean()
    fh_daily = df_by_day['F_HUMIDITY'].mean()
    fi_daily = df_by_day['F_INTENSITY'].mean()

    sm_daily = df_by_day['SM'].mean()
    sm_plus_err_daily = df_by_day['SM_PLUS_ERR'].mean()
    sm_minus_err_daily = df_by_day['SM_MINUS_ERR'].mean()

    effective_depth_daily = df_by_day['D86avg'].mean()

    #assemble axes 
    x1 = pd.to_datetime(day_col).to_list()   #convert pandas series to lists as this is what bokeh expects 
    y1 = sm_daily.to_list()

    y2 = effective_depth_daily.to_list()

    # apply theme to current document
    curdoc().theme = "caliber"

    #create sm plot
    p1 = figure(
            tools=[HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool(), SaveTool()],
            tooltips="@y",title="Soil Moisture Time Series", 
            x_axis_label='Datetime', x_axis_type="datetime", y_axis_label='Soil Moisture (cm$^3$ / cm$^3$)')
    

    # add a line renderer with legend and line thickness to the plot
    p1.line(x1, y1, legend_label="Soil Moisture.", line_width=2, line_color='cornflowerblue')


    p2 = figure( 
            tools=[HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool(), SaveTool()],
            tooltips="@y",title="Soil Moisture Time Series", 
            x_axis_label='Datetime', x_axis_type="datetime", y_axis_label='Effective Depth (cm)')
    

    # add a line renderer with legend and line thickness to the plot
    p2.line(x1, y2, legend_label="Effective Depth.", line_width=2, line_color='orange')


    #axes ticks
    p1.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
    p2.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")

    #~~~~~~ LEGEND ~~~~~~~#
    # display legend in top left corner (default is top right corner)
    p1.legend.location = "bottom_right"

    # add a title to your legend
    p1.legend.title = "Legend"
    p1.legend.title_text_font_size = "20px"

    # change appearance of legend text
    p1.legend.label_text_font_size = "20px"
    p1.legend.label_text_font = "times"
    p1.legend.label_text_font_style = "italic"
    p1.legend.label_text_color = "black"

    # change border and background of legend
    p1.legend.border_line_width = 1.5
    p1.legend.border_line_color = "black"
    p1.legend.border_line_alpha = 0.8
    p1.legend.background_fill_color = None
    p1.legend.background_fill_alpha = 0.2

    #~~~ TITLE ~~~#
    p1.title.text_font_size = "24px"

    #~~~ BOXANNOTATION ~~~#

    """low_box = BoxAnnotation(top=0, fill_alpha=0.4, fill_color="red")
    mid_box = BoxAnnotation(bottom=0, top=1, fill_alpha=0.4, fill_color="green")
    high_box = BoxAnnotation(bottom=1, fill_alpha=0.4, fill_color="red")

    p.add_layout(low_box)
    p.add_layout(mid_box)
    p.add_layout(high_box)"""

    # adds coloured bands to the y-grid
    ''' p1.ygrid.band_fill_color = "whitesmoke"
    p1.ygrid.band_fill_alpha = 0.2'''

    # define vertical bonds
    p1.xgrid.bounds = (2, 4)    #adds lines every 0.1cm3/cm3 of sm.

    # change the fill colors
    #p.border_fill_color = (0, 0, 0)

    #~~~ TOOLBAR ~~~#
    p1.toolbar.logo = None

    #~~~ CREATE LAYOUT ~~~#
    
    show(column(children=[p1, p2], sizing_mode="stretch_width", height=250))

    return



