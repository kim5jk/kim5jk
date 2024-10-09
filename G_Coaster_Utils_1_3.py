####################################################################################
import pandas as pd
import numpy as np
import re
import os
from datetime import timedelta
from datetime import datetime

# from scipy.spatial.transform import Rotation
from scipy import signal
from scipy.signal import firwin, freqz
from scipy.signal import find_peaks
import scipy.ndimage
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore');

####################################################################################
######################### Plotly generation function################################
####################################################################################

def generate_and_save_plots(df, interaction_df, 
                            results_process4,
                            panelist, 
                            html_plots_dir, 
                            time_window_mins,
                           auto_open=False):
    """
    Generates main figure and individual zoomed-in plots for each diary entry and marks drift transitions.
    
    Parameters:
    - df: DataFrame with main sensor data
    - interaction_df: DataFrame with interaction data
    - df_excel: DataFrame with diary entries
    - results_process4: DataFrame containing the drift transition results
    - panelist: Name of the panelist (for labeling and file naming)
    - html_plots_dir: Directory to save the HTML plots
    - time_window_mins: Time window in minutes for visual context around interactions
    """
    if df.empty:
        print(f"No data available to plot for {panelist}.")
        return  # Skip plotting if there is no data
#################
    fig = go.Figure()
    tempmax = df["McuTemperature"].max()
    tempmin = df["McuTemperature"].min()
#################
    fig.update_layout(
        yaxis2=dict(
            title="Temperature", overlaying='y', side='right', 
            showgrid=False, 
            type='linear', 
            fixedrange=True, 
            range=[tempmin, tempmax]  # Sets the range of the axis to be between 10 and 30
        )
    )
#################
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['mean_weight_g'],
                             mode='markers',
                             marker=dict(color="darkblue",size=4, opacity=0.7),
                             name='Original Weight'))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['new_weight'],
                             mode='markers+lines',
                             marker=dict(color="darkgreen",size=4, opacity=0.7),
                             name='Fitted Weight'))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['adjusted_weight'],
                             mode='markers+lines',
                             marker=dict(color="darkorange",size=4),
                             name='Adjusted Weight (Constant Temp)'))

    fig.add_trace(go.Scatter(x=df['datetime'], y=df['detrended_weight'],
                             mode='markers+lines',
                             marker=dict(color="darkcyan",size=4),
                             name='Detrended Weight'))
    fig.add_trace(go.Scatter(x=df['datetime'],y=df['smoothed_temperature'],
                            mode='markers+lines',
                            name='Smoothed Temperature',
                            marker=dict(color='lightblue',size=4, opacity=0.7),
                            yaxis='y2'
                            ))
#################
    # Add error_sigma markers for adjusted_weight
    error_sigma_true = df[df['error_sigma']]
    fig.add_trace(go.Scatter(
        x=error_sigma_true['datetime'],
        y=error_sigma_true['adjusted_weight'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Error Sigma Adjusted Weight'
    ))

#################
    # Mark interactions as shaded areas
    previous_end_time = None
    for idx, row in interaction_df.iterrows():
        if row['Interaction']:
            start_time = row['datetime'] - pd.Timedelta(minutes=time_window_mins)
            end_time = row['datetime'] + pd.Timedelta(minutes=time_window_mins)
            # Avoid overlapping regions by combining close interactions
            if previous_end_time and start_time <= previous_end_time:
                end_time = max(end_time, previous_end_time)
            else:
                if previous_end_time:
                    fig.add_vrect(x0=previous_start_time, x1=previous_end_time, fillcolor='lightpink', opacity=0.6, line_width=0)
                previous_start_time = start_time
            previous_end_time = end_time
    # Add the last interaction rectangle if any
    if previous_end_time:
        fig.add_vrect(x0=previous_start_time, x1=previous_end_time, fillcolor='lightpink', opacity=0.6, line_width=0)
#################
    
    # Mark drift transitions using add_shape for vertical lines
    for _, row in results_process4.iterrows():
        fig.add_shape(type="line",
                      x0=row['datetime'], y0=0, x1=row['datetime'], y1=1,
                      xref='x', yref='paper',
                      line=dict(color="teal", width=0.7, dash="dot"))



    fig.write_html(main_fig_path,auto_open=auto_open)

    print(f"Plot saved to {main_fig_path}")
####################################################################################

    # if apply_tare:
    #     apply_tare_weight_adjustment(df, taresigma_threshold)
def detrend_weight(df, threshold=1):
    
    adjusted_weights = df['adjusted_weight'].copy()

    drift_flags = df['drift_flag'].unique()
    for flag in drift_flags:
        group_data = df[df['drift_flag'] == flag].copy()
        start_weight = adjusted_weights.loc[group_data.index[0]]
        end_weight = adjusted_weights.loc[group_data.index[-1]]
        duration = (group_data['datetime'].iloc[-1] - group_data['datetime'].iloc[0]).total_seconds() / (3600 * 24)  # Convert to days
        
        drift_rate = abs(end_weight - start_weight) / duration
        if drift_rate > threshold:
            diff = end_weight - start_weight
            # Flatten the segment to the start_weight
            adjusted_weights.loc[group_data.index] = start_weight
            
            # Apply the difference to all subsequent weights
            last_datetime = group_data['datetime'].max()
            subsequent_index = df[df['datetime'] > last_datetime].index
            if diff != 0:
                adjusted_weights.loc[subsequent_index] -= diff

    df['detrended_weight'] = adjusted_weights
    return df
# def detrend_weight(df, threshold=1):
#     adjusted_weights = df['adjusted_weight'].copy()
    
#     for group_name, group_data in df.groupby('drift_flag'):
#         start_weight = group_data['adjusted_weight'].iloc[0]
#         end_weight = group_data['adjusted_weight'].iloc[-1]
#         duration = (group_data['datetime'].iloc[-1] - group_data['datetime'].iloc[0]).days
        
#         drift_rate = abs(end_weight - start_weight) / duration
#         if drift_rate > threshold:
#             diff = end_weight - start_weight
#             # Flatten the segment to the start_weight
#             adjusted_weights[group_data.index] = start_weight
            
#             # Shift the subsequent weights
#             if diff != 0:
#                 for i in range(group_data.index[-1] + 1, len(adjusted_weights)):
#                     adjusted_weights[i] -= diff

#     df['detrended_weight'] = adjusted_weights
#     return df



####################################################################################
# def analyze_drift_transitions(df):
#     """
#     Analyzes weight changes at transitions in the drift flag, focusing on consumption metrics,
#     and flags any instances of weight increase.

#     Parameters:
#     - df: DataFrame containing sensor data with 'datetime', 'orig_idx', 'drift_flag', 'adjusted_weight', and 'mean_weight_g'.
    
#     Returns:
#     - DataFrame: Analysis results for each drift transition, including datetime, weight change, and flags for weight increase.
#     """
#     results = []
#     prev_drift_flag = df['drift_flag'].shift(1)
    
#     # Identify transitions by finding where the drift flag changes
#     transition_points = df[df['drift_flag'] != prev_drift_flag]

#     for idx, point in transition_points.iterrows():
#         if idx == 0:
#             continue  # Skip the first row as it has no previous row to compare
#         transition_time = point['datetime']
#         transition_idx = point['orig_idx']
#         drift_flag = point['drift_flag']
        
#         # Calculate the weight change: previous weight - current weight
#         weight_before = df.iloc[idx - 1]['mean_weight_g']
#         weight_after = point['mean_weight_g']
#         weight_change = weight_before - weight_after  # Positive if there's a reduction (consumption)
        
#         # Check for weight increase (negative consumption)
#         flag_weight_increase = weight_change < 0

#         results.append({
#             'orig_idx': transition_idx,
#             'datetime': transition_time,
#             'drift_flag': drift_flag,
#             'weight_before': weight_before,
#             'weight_after': weight_after,
#             'consumption': abs(weight_change),
#             'flag_weight_increase': flag_weight_increase
#         })

#     results_df = pd.DataFrame(results)
#     return results_df

def analyze_drift_transitions(df):
    """
    Analyzes weight changes at transitions in the drift flag, focusing on consumption metrics,
    and flags any instances of weight increase.

    Parameters:
    - df: DataFrame containing sensor data with 'datetime', 'orig_idx', 'drift_flag', 'adjusted_weight', 'detrended_weight', and 'mean_weight_g'.
    
    Returns:
    - DataFrame: Analysis results for each drift transition, including datetime, weight change, and flags for weight increase for mean_weight_g, adjusted_weight, and detrended_weight.
    """
    results = []
    prev_drift_flag = df['drift_flag'].shift(1)
    
    # Identify transitions by finding where the drift flag changes
    transition_points = df[df['drift_flag'] != prev_drift_flag]

    for idx, point in transition_points.iterrows():
        if idx == 0:
            continue  # Skip the first row as it has no previous row to compare
        transition_time = point['datetime']
        transition_idx = point['orig_idx']
        drift_flag = point['drift_flag']
        
        # Calculate the weight changes
        weight_before_mean = df.iloc[idx - 1]['mean_weight_g']
        weight_after_mean = point['mean_weight_g']
        weight_change_mean = weight_before_mean - weight_after_mean  # Positive if there's a reduction (consumption)
        
        weight_before_adjusted = df.iloc[idx - 1]['adjusted_weight']
        weight_after_adjusted = point['adjusted_weight']
        weight_change_adjusted = weight_before_adjusted - weight_after_adjusted
        
        weight_before_detrended = df.iloc[idx - 1]['detrended_weight']
        weight_after_detrended = point['detrended_weight']
        weight_change_detrended = weight_before_detrended - weight_after_detrended
        
        # Check for weight increase (negative consumption)
        flag_weight_increase_mean = weight_change_mean < 0
        flag_weight_increase_adjusted = weight_change_adjusted < 0
        flag_weight_increase_detrended = weight_change_detrended < 0

        results.append({
            'orig_idx': transition_idx,
            'datetime': transition_time,
            'drift_flag': drift_flag,
            'weight_before_mean': weight_before_mean,
            'weight_after_mean': weight_after_mean,
            'consumption_mean': abs(weight_change_mean),
            'flag_weight_increase_mean': flag_weight_increase_mean,
            'weight_before_adjusted': weight_before_adjusted,
            'weight_after_adjusted': weight_after_adjusted,
            'consumption_adjusted': abs(weight_change_adjusted),
            'flag_weight_increase_adjusted': flag_weight_increase_adjusted,
            'weight_before_detrended': weight_before_detrended,
            'weight_after_detrended': weight_after_detrended,
            'consumption_detrended': abs(weight_change_detrended),
            'flag_weight_increase_detrended': flag_weight_increase_detrended
        })

    results_df = pd.DataFrame(results)
    return results_df


def apply_tare_weight_adjustment(df, threshold=2):
    """
    Applies tare weights to the sensor data based on the TareSigma reliability threshold.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the sensor data, including 'TareWeight' and 'TareSigma' columns.
    - threshold (float): TareSigma threshold above which the last valid TareWeight is applied.

    Modifies the DataFrame in place by updating the 'mean_weight_g' column to reflect the net weights after tare adjustment.
    """
    last_valid_tare_weight = np.nan  # Initialize with NaN

    # Ensure TareWeight and TareSigma exist
    if 'TareWeight' not in df.columns or 'TareSigma' not in df.columns:
        raise ValueError("DataFrame must include 'TareWeight' and 'TareSigma' columns.")

    for i in range(len(df)):
        if pd.isna(df.loc[i, 'TareWeight']) or pd.isna(df.loc[i, 'TareSigma']):
            continue

        if df.loc[i, 'TareSigma'] > threshold and not pd.isna(last_valid_tare_weight):
            # Use last valid TareWeight if TareSigma exceeds threshold
            df.loc[i, 'mean_weight_g'] -= last_valid_tare_weight
        else:
            # Update tared_weight_g normally
            df.loc[i, 'mean_weight_g'] -= df.loc[i, 'TareWeight']
            last_valid_tare_weight = df.loc[i, 'TareWeight']  # Update last valid TareWeight


def analyze_sensor_data(df, 
                        sigma_threshold = 5, 
                        taresigma_threshold=5, 
                        # apply_tare=False
                       ):
    """
    Filters and analyzes sensor data for errors based on sigma values and time differences, optionally applying tare weight adjustments.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to process.
    - sigma_threshold (float): Threshold for 'sample_sigma' to flag errors.
    - taresigma_threshold (float): Threshold for 'TareSigma' used in tare weight adjustment.

    Returns:
    - pandas.DataFrame: The processed DataFrame, adjusted and flagged for further analysis.
    """
    df['timediff'] = df['datetime'].diff().dt.total_seconds().div(60)  # time differences in minutes
    df['error_sigma'] = df['sample_sigma'] > sigma_threshold
    
    df = df[df['timediff'] >= 0.099]  # Key filter to only include data with significant time gaps
    df.reset_index(drop=True, inplace=True)
    df.drop(columns='timediff', inplace=True)
    # Recalculate 'timediff'
    df['timediff'] = df['datetime'].diff().dt.total_seconds().div(60)

    median_value=df['timediff'].median()
    lower_threshold = median_value-0.099
    print(f"time diff threshold set to be {lower_threshold}")
    df['Interaction'] = df['timediff'] < lower_threshold
    # Separate DataFrame for interaction data
    interaction_df = df[df['Interaction']].copy()
    interaction_df = interaction_df[['datetime', 'timediff', 'Interaction']]

    return df,interaction_df

def apply_gaussian_smoothing(group, sigma, order=0):
    if len(group) > 1:
        smoothed_values = scipy.ndimage.gaussian_filter(group, sigma=sigma, order=order)
    else:
        smoothed_values = group
    return smoothed_values


def mark_drifts(data, threshold=1):
    if 'drift_flag' not in data.columns:
        data['drift_flag'] = np.nan

    current_drift_id = 1 

    for i in range(len(data)):
        if i == 0:
            # Initialize the first drift
            data.at[i, 'drift_flag'] = current_drift_id
            continue

        diff = abs(data['mean_weight_g'].iloc[i] - data['mean_weight_g'].iloc[i - 1])
        interaction = data['Interaction'].iloc[i]

        if diff >= threshold or  (interaction and diff >= threshold):
            current_drift_id += 1

        # Ensure every row gets a drift_flag
        data.at[i, 'drift_flag'] = current_drift_id

    return data
    
def check_drift(data, drift_length_threshold):
    drift_counts = data['drift_flag'].value_counts()
    short_drift_ids = drift_counts[drift_counts < drift_length_threshold].index
    short_drifts = data[data['drift_flag'].isin(short_drift_ids)]
    return short_drifts
  

