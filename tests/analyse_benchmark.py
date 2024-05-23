import os
os.environ['APPDATA'] = ""

import pandasgui

import pandas as pd


if __name__ == "__main__":

    csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/benchmark_global_planner/2024-05-22_19-39-08.csv'

    df = pd.read_csv(csv_file)

    pandasgui.show(df)
