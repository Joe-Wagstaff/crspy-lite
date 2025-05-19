
"""
@author: Joe Wagstaff
@institution: University of Bristol

"""

#standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#bokeh imports. bokeh is a plotting library used to create interactive figures in crspy-lite (note only required for "plot_page" function").
from bokeh.plotting import figure, show
from bokeh.io import curdoc
from bokeh.models import HoverTool, BoxZoomTool, PanTool, ResetTool, SaveTool, WheelZoomTool
from bokeh.models import RangeSlider, CustomJS
from bokeh.layouts import layout, gridplot
from bokeh.layouts import column

def standard_plots(df, config_data):
    
    """Function that creates a series of standard plots for the a given CRNS site. Uses the python module MatPlotLib.
    
    Parameters
    ----------
    df : DataFrame  
        Processed data.

    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.
    
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
    #plt.show()   

    return


def typical_year(df, config_data):
    
    """Function that creates a typical year of temperature, rainfall and SM with estimated uncertainty bound.
    This is done by (i) converting these values from hourly to daily resoltion (by averaging) and then (ii) grouping the df by month and day to average by year.
    Note that for step (i) rainfall is summed instead of averaged.

    Parameters
    ----------
    df : DataFrame  
        Processed hourly data.

    config_data : dictionary
        Store of variables required for the data processing. Also stores relevant filepaths.

    """

    # Replace missing values
    df = df.replace([-999, -99], np.nan)  
    df = df.replace(['0.0', None], pd.NA)  
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')

    # Convert to numeric
    df['RAIN'] = pd.to_numeric(df['RAIN'], errors='coerce')
    df['TEMP'] = pd.to_numeric(df['TEMP'], errors='coerce')
    df['SM'] = pd.to_numeric(df['SM'], errors='coerce')

    # Apply limits
    max_rain_per_hour = config_data.get('metadata', {}).get('precip_max', None)
    df['RAIN'] = df['RAIN'].where(df['RAIN'] <= max_rain_per_hour, np.nan)  # Max rainfall per hr to remove erroneous values
    df['TEMP'] = df['TEMP'].where(df['TEMP'].between(-10, 50), np.nan)

    # SM: we want sm +/- the sm errors
    df['SM_PLUS_ERR'] = df['SM'] + abs(df['SM_PLUS_ERR']) 
    df['SM_MINUS_ERR'] = df['SM'] - abs(df['SM_MINUS_ERR'])

    # Convert to daily resolution
    df['DATE'] = df['DATETIME'].dt.date  # Extract date (drops time component)
    daily_df = df.groupby('DATE').agg({
        'RAIN': 'sum',  # Sum rainfall over the day
        'TEMP': 'mean',  # Average temperature
        'SM': 'mean',  # Average soil moisture
        'SM_PLUS_ERR': 'mean',
        'SM_MINUS_ERR': 'mean'
    }).reset_index()

    # Extract year & month-day for typical year calculation
    daily_df['YEAR'] = pd.to_datetime(daily_df['DATE']).dt.year
    daily_df['MONTH_DAY'] = pd.to_datetime(daily_df['DATE']).dt.strftime('%m-%d')

    # Compute typical year
    df_typical = daily_df.groupby('MONTH_DAY').agg({
        'RAIN': 'mean',
        'TEMP': 'mean',
        'SM': 'mean',
        'SM_PLUS_ERR': 'mean',
        'SM_MINUS_ERR': 'mean'
    }).reset_index()

    # Generate date index
    startdate = "2024-01-01"  # Arbitrary non-leap year for plotting
    df_typical['DATE'] = pd.date_range(start=startdate, periods=366, freq='D')[:len(df_typical)]
    df_typical.set_index('DATE', inplace=True)

    # Field capacity & wilting point
    field_capacity = config_data.get('metadata', {}).get('field_capacity', None)
    wilting_point = config_data.get('metadata', {}).get('wilting_point', None)

    # Plotting
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(3, sharex=True, figsize=(15, 9))

    axs[0].plot(df_typical.index, df_typical['TEMP'], lw=1.5, color='orange', label='Temperature')
    axs[0].set_ylabel("Daily Temp ($^o$C)")

    axs[1].plot(df_typical.index, df_typical['RAIN'], lw=1.5, color='blue', label='Precipitation')
    axs[1].set_ylabel("Daily Rainfall (mm)")

    axs[2].plot(df_typical.index, df_typical['SM'], lw=1.5, color='green', label='Soil Moisture')
    axs[2].fill_between(df_typical.index, df_typical['SM_MINUS_ERR'], df_typical['SM_PLUS_ERR'],
                        alpha=0.5, edgecolor='#CC4F1B', facecolor='firebrick', label='Uncertainty')
    if field_capacity is not None:
        axs[2].axhline(y=field_capacity, color='black', linestyle='--', linewidth=1.5, label='Field Capacity')
    if wilting_point is not None:
        axs[2].axhline(y=wilting_point, color='red', linestyle='--', linewidth=1.5, label='Wilting Point')

    axs[2].set_ylabel('Daily SM (cm³/cm³)')
    axs[2].legend()

    # Format x-axis
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(fontsize=16)

    #show plot
    plt.tight_layout()
    fig.savefig(config_data['filepaths']['default_dir']+"/outputs/figures/typical_year.png", dpi=350)
    #plt.show()

    return

def plot_page(df):
    
    """ This function outputs an HTML file containing a preset selection of figures for the processed data of 
        a CRNS site.
        
        Parameters
        ----------
        df : DataFrame  
            Processed data.

        """
    
    df.replace(int("-999"), np.nan, inplace=True)

    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')
    df_by_day = df.groupby([df['DATETIME'].dt.year, df['DATETIME'].dt.month, df['DATETIME'].dt.day])  #group df by day (converts from hourly intevals to daily)
   
    day_series = df['DATETIME'].dt.to_period('D')
    day_series = (day_series.drop_duplicates()).reset_index()
    day_series = day_series.astype(str)
    day_col = day_series['DATETIME']
    
    #collect averaged columns
    raw_counts_daily = df_by_day['MOD'].mean().to_list()
    corr_counts_daily = df_by_day['MOD_CORR'].mean().to_list()

    fp_daily = df_by_day['F_PRESSURE'].mean().to_list()
    fh_daily = df_by_day['F_HUMIDITY'].mean().to_list()
    fi_daily = df_by_day['F_INTENSITY'].mean().to_list()

    sm_daily = df_by_day['SM'].mean().to_list()
    sm_plus_err_daily = df_by_day['SM_PLUS_ERR'].mean().to_list()
    sm_minus_err_daily = df_by_day['SM_MINUS_ERR'].mean().to_list()

    effective_depth_daily = df_by_day['D86avg'].mean().to_list()

    #assemble axes 
    x1 = pd.to_datetime(day_col).to_list()   #convert pandas series to lists as this is what bokeh expects 
    x_min, x_max = x1[0].timestamp() * 1000, x1[-1].timestamp() * 1000  # Convert to milliseconds
    shared_x_range = (x_min, x_max)

    # apply theme to current document
    curdoc().theme = "caliber"

    #create corrections plot
    p3 = figure(
            tools=[HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool(), SaveTool()],
            tooltips="@y",title="Correction Factors Daily Time Series", 
            x_axis_label='Datetime', x_axis_type="datetime", y_axis_label='Correction', x_range=shared_x_range)
    
    p3.line(x1, fp_daily, legend_label="Pressure", line_width=2, line_color='crimson')
    p3.line(x1, fh_daily, legend_label="Humidity.", line_width=2, line_color='green')
    p3.line(x1, fi_daily, legend_label="Intensity.", line_width=2, line_color='deepskyblue')

    #create mod vs mod_corr plot
    p4 = figure(
            tools=[HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool(), SaveTool()],
            tooltips="@y",title="Raw vs Corrected Neutron Counts Daily Time Series", 
            x_axis_label='Datetime', x_axis_type="datetime", y_axis_label='Neutron Count (cph)', x_range=shared_x_range)
    
    p4.line(x1, raw_counts_daily, legend_label="Raw Counts", line_width=2, line_color='black')
    p4.line(x1, corr_counts_daily, legend_label="Corrected Counts.", line_width=2, line_color='red')

    #create sm plot
    p1 = figure(
            tools=[HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool(), SaveTool()],
            tooltips="@y",title="Soil Moisture Daily Time Series", 
            x_axis_label='Datetime', x_axis_type="datetime", y_axis_label='Soil Moisture (cm^3 / cm^3)', x_range=shared_x_range)
    
    # add a line renderer with legend and line thickness to the plot
    p1.line(x1, sm_daily, legend_label="Soil Moisture.", line_width=2, line_color='cornflowerblue')
    
    p2 = figure( 
            tools=[HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool(), SaveTool()],
            tooltips="@y",title="Effective Depth Daily Time Series", 
            x_axis_label='Datetime', x_axis_type="datetime", y_axis_label='Effective Depth (cm)', x_range=shared_x_range)
    

    # add a line renderer with legend and line thickness to the plot
    p2.line(x1, effective_depth_daily, legend_label="Effective Depth.", line_width=2, line_color='orange')

    #~~~~ STYLING ~~~~#
    p4.legend.location = "bottom_right"

    for p in [p1, p2, p3, p4]: 
        
        p.title.text_font_size = "18pt"

        p.xaxis.axis_label_text_font_size = "16pt"  # Set x-axis label font size
        p.yaxis.axis_label_text_font_size = "16pt"  # Set y-axis label font size
        
        p.xaxis.major_label_text_font_size = "14pt"  # Set x-axis tick font size
        p.yaxis.major_label_text_font_size = "14pt"  # Set y-axis tick font size
        
        p.legend.label_text_font_size = "16pt" 
        
    #~~~ SLIDER ~~~#
    # Create a single X-axis range slider
    x_slider = RangeSlider(start=x_min, end=x_max, value=(x_min, x_max), step=86400000, title="Adjust Time Series Length")
    x_slider.show_value = False
    #x_slider.title_text_font_size = "16pt"

    # JavaScript callback to update x-axis range dynamically
    callback = CustomJS(args=dict(plots=[p1, p2, p3, p4], slider=x_slider), code="""
        var start = slider.value[0];
        var end = slider.value[1];
        for (var i = 0; i < plots.length; i++) {
            plots[i].x_range.start = start;
            plots[i].x_range.end = end;
        }
    """)

    # Link the callback to the slider
    x_slider.js_on_change('value', callback)

    plots = gridplot([[p4, p1], [p3, p2]], height=350, width=800, sizing_mode="stretch_width")
    layout = column(plots, x_slider, sizing_mode="stretch_width")  # Slider below the gridplot

    # Set the width of the slider
    x_slider.width = 400  # Set slider width to match the gridplot width

    show(layout)
    
    return
