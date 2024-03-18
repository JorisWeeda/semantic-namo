import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path_1 = "/home/joris/velocity_data/2024-02-22_14-23-23/0.0.csv"
file_path_2 = "/home/joris/velocity_data/2024-02-22_14-23-23/10.0.csv"

df_1 = pd.read_csv(file_path_1, sep=',')
df_2 = pd.read_csv(file_path_2, sep=',')

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
    'consumed_power': ['rob']
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
    'rob_lin_acc': True,
    'wheel_cur': True,
    'wheel_vol': False,
    'consumed_power': True
}

# Add accelerations data frame 1
for joint in ['fl', 'fr', 'rl', 'rr']:
    df_1[f'joint_acc_{joint}'] = df_1[f'joint_vel_{joint}'].diff() / df_1['joint_vel_t'].diff()
df_1['joint_acc_t'] = df_1['joint_vel_t']

for axis in ['x', 'y', 'z']:
    df_1[f'rob_lin_acc_{axis}'] = (df_1[f'rob_lin_vel_{axis}'].diff() / df_1['rob_lin_vel_t'].diff() +
                               df_1[f'rob_ang_vel_{axis}'].diff() / df_1['rob_ang_vel_t'].diff())
df_1['rob_lin_acc_t'] = df_1['rob_lin_vel_t']

# Add accelerations data frame 2
for joint in ['fl', 'fr', 'rl', 'rr']:
    df_2[f'joint_acc_{joint}'] = df_2[f'joint_vel_{joint}'].diff() / df_2['joint_vel_t'].diff()
df_2['joint_acc_t'] = df_2['joint_vel_t']

for axis in ['x', 'y', 'z']:
    df_2[f'rob_lin_acc_{axis}'] = (df_2[f'rob_lin_vel_{axis}'].diff() / df_2['rob_lin_vel_t'].diff() +
                               df_2[f'rob_ang_vel_{axis}'].diff() / df_2['rob_ang_vel_t'].diff())
df_2['rob_lin_acc_t'] = df_2['rob_lin_vel_t']

# Add consumed power data frame 1
df_1['consumed_power_rob'] = abs(df_1['wheel_cur_fl'] * df_1['wheel_vol_fl'] +
                           df_1['wheel_cur_fr'] * df_1['wheel_vol_fr'] +
                           df_1['wheel_cur_rl'] * df_1['wheel_vol_rl'] +
                           df_1['wheel_cur_rr'] * df_1['wheel_vol_rr'])

df_1['consumed_power_rob'] = df_1['consumed_power_rob'].cumsum()
df_1['consumed_power_t'] = df_1['wheel_vol_t']

# Add consumed power data frame 2
df_2['consumed_power_rob'] = abs(df_2['wheel_cur_fl'] * df_2['wheel_vol_fl'] +
                           df_2['wheel_cur_fr'] * df_2['wheel_vol_fr'] +
                           df_2['wheel_cur_rl'] * df_2['wheel_vol_rl'] +
                           df_2['wheel_cur_rr'] * df_2['wheel_vol_rr'])

df_2['consumed_power_rob'] = df_2['consumed_power_rob'].cumsum()
df_2['consumed_power_t'] = df_2['wheel_vol_t']

for category, columns in categories.items():
    if not show_category.get(category, False) or not columns:  # Check if columns list is empty
        continue

    fig, axes = plt.subplots(len(columns), figsize=(8, 6)) 
    fig.suptitle(category.capitalize())

    for i, column in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes  # Adjust for single column plot
        
        x_data_1 = df_1[f"{category.lower()}_t"]
        y_data_1 = df_1[f"{category.lower()}_{column}"]
        ax.plot(x_data_1, y_data_1, label='0 KG', color='blue')

        x_data_2 = df_2[f"{category.lower()}_t"]
        y_data_2 = df_2[f"{category.lower()}_{column}"]
        ax.plot(x_data_2, y_data_2, label='10 KG', color='orange')

        ax.set_title(column)
        ax.set_xlabel('Time')
        ax.set_ylabel(column)
        ax.legend()

    plt.tight_layout()
    plt.show()

