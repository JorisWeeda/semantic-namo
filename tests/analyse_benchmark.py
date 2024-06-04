import os
os.environ['APPDATA'] = ""

import pandasgui

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == "__main__":

    csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/benchmark_global_planner/2024-05-22_19-39-08.csv'
    
    df = pd.read_csv(csv_file)

    # Plotting boxplots
    plt.figure(figsize=(12, 6))

    # Boxplot for run_time
    plt.subplot(1, 2, 1)
    sns.boxplot(x='stationary_area', y='run_srm_time', data=df, hue='run_srm_result')
    plt.title('Boxplot of Run Time vs Stationary Area')
    plt.xlabel('Stationary Area (%)')
    plt.ylabel('Run Time')
    plt.legend(title='Result')

    # Boxplot for run_force
    plt.subplot(1, 2, 2)
    sns.boxplot(x='stationary_area', y='run_srm_force', data=df, hue='run_srm_result')
    plt.title('Boxplot of Run Force vs Stationary Area')
    plt.xlabel('Stationary Area (%)')
    plt.ylabel('Run Force')
    plt.legend(title='Result')

    plt.tight_layout()
    plt.show()

    # pandasgui.show(df)
