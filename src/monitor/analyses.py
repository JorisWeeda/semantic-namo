import os
os.environ['APPDATA'] = ""

import glob
import pandasgui
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress


def load_data_folder(base_folder):
    data_dict = {}

    for angle_folder in os.listdir(base_folder):
        
        angle_folder_path = os.path.join(base_folder, angle_folder)
        if os.path.isdir(angle_folder_path):
            angle = int(angle_folder)
            data_dict[angle] = {}

            for mass_folder in os.listdir(angle_folder_path):
                mass_folder_path = os.path.join(angle_folder_path, mass_folder)

                mass = float(mass_folder)
                data_dict[angle][mass] = {}

                csv_files = glob.glob(os.path.join(mass_folder_path, "*.csv"))
                for idx, csv_file_path in enumerate(csv_files):

                    df = pd.read_csv(csv_file_path, sep=',')
                    processed_df = process_dataframe(df)
                    data_dict[angle][mass][idx] = processed_df

                    print(f"angle: {angle}, mass: {mass}, index: {idx}, shape: {processed_df.shape}")

    return data_dict


def process_dataframe(df, cut_off=3):
    
    # Add accelerations data frame
    for joint in ['fl', 'fr', 'rl', 'rr']: 
        df[f'joint_acc_{joint}'] = df[f'joint_vel_{joint}'].diff() / df['joint_vel_t'].diff()
    df['joint_acc_t'] = df['joint_vel_t']

    for axis in ['x', 'y', 'z']:
        df[f'rob_lin_acc_{axis}'] = (df[f'rob_lin_vel_{axis}'].diff() / df['rob_lin_vel_t'].diff() +
                                df[f'rob_ang_vel_{axis}'].diff() / df['rob_ang_vel_t'].diff())
    df['rob_lin_acc_t'] = df['rob_lin_vel_t']

    # Add consumed power data frame
    consumed_power = (
        abs(df['wheel_cur_fl'] * df['wheel_vol_fl']) +
        abs(df['wheel_cur_fr'] * df['wheel_vol_fr']) +
        abs(df['wheel_cur_rl'] * df['wheel_vol_rl']) +
        abs(df['wheel_cur_rr'] * df['wheel_vol_rr'])
    )
    
    # Add consumed power DataFrame to df_dict
    df[f'consumed_power_rob'] = consumed_power

    df['consumed_power_rob'] = df['consumed_power_rob'].cumsum()
    df['consumed_power_t'] = df['wheel_vol_t']

    # Add total current per time step
    df['current_rob'] = df[['wheel_cur_fl', 'wheel_cur_fr', 'wheel_cur_rl', 'wheel_cur_rr']].sum(axis=1)
    df['current_t'] = df['wheel_cur_t']

    # Add total voltage per time step
    df['voltage_rob'] = df[['wheel_vol_fl', 'wheel_vol_fr', 'wheel_vol_rl', 'wheel_vol_rr']].sum(axis=1)
    df['voltage_t'] = df['wheel_vol_t']

    return df[:len(df)-cut_off]

if __name__ == "__main__":
    folder_name = "/home/joris/velocity_data/2024-03-28_experiment_1/"
    data_dict = load_data_folder(folder_name)
    
    masses = [0.0, 1.0]

    categories = {
        'joint_pos': ['fl', 'fr', 'rl', 'rr'],
        'joint_vel': ['fl', 'fr', 'rl', 'rr'],
        'joint_eff': ['fl', 'fr', 'rl', 'rr'],
        'joint_acc': ['fl', 'fr', 'rl', 'rr'], 
        'rob_pos': ['x', 'y', 'z'],
        'rob_rot': ['x', 'y', 'z', 'w'],
        'rob_lin_vel': ['x', 'y', 'z'],
        'rob_ang_vel': ['x', 'y', 'z'],
        'rob_lin_acc': ['x', 'y', 'z'], 
        'wheel_cur': ['fl', 'fr', 'rl', 'rr'],
        'wheel_vol': ['fl', 'fr', 'rl', 'rr'],
        'joint_acc': ['fl', 'fr', 'rl', 'rr'],
        'consumed_power': ['rob'],
        'current': ['rob'],
        'voltage': ['rob']
    }

    show_category = {
        'joint_pos': False,
        'joint_vel': False,
        'joint_eff': False,
        'joint_acc': False,
        'rob_pos': False,
        'rob_rot': False,
        'rob_lin_vel': False,
        'rob_ang_vel': False,
        'rob_lin_acc': False,
        'wheel_cur': True,
        'wheel_vol': False,
        'consumed_power': True,
        'current': True,
        'voltage': True
    }

    for category, columns in categories.items():
        if not show_category.get(category, False) or not columns:
            continue
        
        fig, axes = plt.subplots(len(data_dict), len(columns), figsize=(8, 6), sharex=True)
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.4)

        for angle_i, (angle, mass_data) in enumerate(data_dict.items()):
            for mass_i, mass in enumerate(masses):
                data = mass_data.get(mass, None)
                if not data:
                    continue

                dfs = list(data.values())
                sum_df = pd.concat(dfs, axis=0)
                df = sum_df.groupby(level=0).mean()

                for column_i, column in enumerate(columns):
                    x_data = df[f"{category.lower()}_t"]
                    y_data = df[f"{category.lower()}_{column}"]

                    if len(columns) == 1:
                        ax = axes[angle_i]
                    else:
                        ax = axes[angle_i][column_i]

                    ax.plot(x_data, y_data, label=f"{mass} kg")
                    ax.set_title(f"Angle: {angle}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel(f"{category.capitalize()} ({column})")
                    ax.legend()

    # Calculate boxplot figures of linear regression
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Linregression Slopes for Different Masses")

    errors = []
    for angle_i, (angle, mass_data) in enumerate(data_dict.items()):
        for mass_i, mass in enumerate(masses):
            data = mass_data.get(mass, None)
            if not data:
                continue

            slopes = []
            errors = []

            for idx, df in data.items():
                x_data = df['consumed_power_t']
                y_data = df['consumed_power_rob']

                slope, _, _, _, std_e = linregress(x_data, y_data)
                slopes.append(slope)
                errors.append(std_e)

            # Plot slopes in the upper tiles
            axes[0, mass_i].boxplot(slopes, positions=[angle_i], labels=[f"{angle}"])
            axes[0, mass_i].set_title(f"Mass: {mass}")
            axes[0, mass_i].set_ylabel('Slope')

            # Plot errors in the lower tiles
            axes[1, mass_i].boxplot(errors, positions=[angle_i], labels=[f"{angle}"])
            axes[1, mass_i].set_title(f"Mass: {mass}")
            axes[1, mass_i].set_ylabel('Error')

    plt.tight_layout()
    # plt.show()

    flattend_dict = {}
    for angle_key, angle_dict in data_dict.items():
        for mass_key, mass_dict in angle_dict.items():
            for run_key, run_data in mass_dict.items():
                flattend_dict[f"{angle_key}_{mass_key}_{run_key}"] = run_data

    # pandasgui.show(**flattend_dict)

    file_path = '/home/joris/tu_delft_ws/15_msc_thesis/experiments/test_run_5_succes.pkl'
    with open(file_path, 'rb') as f:
        df_dict_reopened = pickle.load(f)

    pandasgui.show(**df_dict_reopened)