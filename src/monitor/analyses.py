import os
import glob
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress


def load_data(base_folder):
    data = {}

    for angle_folder in os.listdir(base_folder):
        
        angle_folder_path = os.path.join(base_folder, angle_folder)
        if os.path.isdir(angle_folder_path):

            angle = int(angle_folder)
            data[angle] = {}

            csv_files = glob.glob(os.path.join(angle_folder_path, "*.csv"))
            for csv_file_path in csv_files:

                mass = pathlib.Path(csv_file_path).stem
                
                df = pd.read_csv(csv_file_path, sep=',')
                processed_df = process_dataframe(df)

                data[angle][mass] = processed_df

    return data

def process_dataframe(df):
    
    # Add accelerations data frame
    for joint in ['fl', 'fr', 'rl', 'rr']: 
        df[f'joint_acc_{joint}'] = df[f'joint_vel_{joint}'].diff() / df['joint_vel_t'].diff()
    df['joint_acc_t'] = df['joint_vel_t']

    for axis in ['x', 'y', 'z']:
        df[f'rob_lin_acc_{axis}'] = (df[f'rob_lin_vel_{axis}'].diff() / df['rob_lin_vel_t'].diff() +
                                df[f'rob_ang_vel_{axis}'].diff() / df['rob_ang_vel_t'].diff())
    df['rob_lin_acc_t'] = df['rob_lin_vel_t']

    # Add consumed power data frame
    df['consumed_power_rob'] = abs(df['wheel_cur_fl'] * df['wheel_vol_fl'] +
                            df['wheel_cur_fr'] * df['wheel_vol_fr'] +
                            df['wheel_cur_rl'] * df['wheel_vol_rl'] +
                            df['wheel_cur_rr'] * df['wheel_vol_rr'])

    df['consumed_power_rob'] = df['consumed_power_rob'].cumsum()
    df['consumed_power_t'] = df['wheel_vol_t']

    # Add total current per time step
    df['current_rob'] = df[['wheel_cur_fl', 'wheel_cur_fr', 'wheel_cur_rl', 'wheel_cur_rr']].sum(axis=1)
    df['current_t'] = df['wheel_cur_t']

    # Add total voltage per time step
    df['voltage_rob'] = df[['wheel_vol_fl', 'wheel_vol_fr', 'wheel_vol_rl', 'wheel_vol_rr']].sum(axis=1)
    df['voltage_t'] = df['wheel_vol_t']

    return df

if __name__ == "main":
    folder_name = "/home/joris/velocity_data/2024-02-22_14-23-23/"
    data_frames = load_data(folder_name)

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

        fig, axes = plt.subplots(len(columns), figsize=(8, 6), sharex=True)
        fig.suptitle(category.capitalize())

        for i, column in enumerate(columns):
            ax = axes[i] if len(columns) > 1 else axes

            for name, df in data_frames.items():
                x_data = df[f"{category.lower()}_t"]
                y_data = df[f"{category.lower()}_{column}"]
                ax.plot(x_data, y_data, label=name)

                slope, intercept, _, _, _ = linregress(x_data, y_data)
                regression_line = slope * x_data + intercept
                ax.plot(x_data, regression_line, '--', label=f'Regression ({name})')

            ax.set_title(column)
            ax.set_ylabel(column)
            ax.set_xlabel("Time")
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()