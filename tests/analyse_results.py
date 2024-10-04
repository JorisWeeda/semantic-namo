import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker


def calculate_success_percentages(data, result_column):
    success_counts = data.groupby(
        ['setup', 'planner', result_column]).size().unstack(fill_value=0).reset_index()
    success_counts['total'] = success_counts[True] + success_counts[False]
    success_counts[f'{result_column}_success_percentage'] = (
        success_counts[True] / success_counts['total']) * 100
    return success_counts[['setup', 'planner', f'{result_column}_success_percentage']]


def plot_planner_results(data, unique_planners, palette=None):
    planner_success_percentages = calculate_success_percentages(
        data, 'planner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner,
                    color in zip(unique_planners, palette)}

    plt.figure(figsize=(16, 8))
    sns.barplot(data=planner_success_percentages, x='setup', y='planner_result_success_percentage', hue='planner',
                palette=palette_dict, dodge=True, errorbar=None, hue_order=unique_planners)

    plt.xlabel('Setup', fontsize=16)
    plt.ylabel('Planner Success Percentage (%)', fontsize=16)
    plt.title('Planner Success Percentages per Setup', fontsize=18)
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=16, title_fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points', fontsize=16, color='black')
    plt.show()


def plot_runner_results(data, unique_planners, palette=None):
    runner_success_percentages = calculate_success_percentages(
        data, 'runner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner,
                    color in zip(unique_planners, palette)}

    plt.figure(figsize=(16, 8))
    sns.barplot(data=runner_success_percentages, x='setup', y='runner_result_success_percentage', hue='planner',
                palette=palette_dict, dodge=True, errorbar=None, hue_order=unique_planners)

    plt.xlabel('Setup', fontsize=16)
    plt.ylabel('Runner Success Percentage (%)', fontsize=16)
    plt.title('Runner Success Percentages Per Setup', fontsize=18)
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=16, title_fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points', fontsize=16, color='black')
    plt.show()


def plot_combined_planner_runner_results(data, unique_planners, palette=None):
    planner_success_percentages = calculate_success_percentages(
        data, 'planner_result')
    runner_success_percentages = calculate_success_percentages(
        data, 'runner_result')

    planner_success_percentages['type'] = 'Planner'
    runner_success_percentages['type'] = 'Runner'

    planner_success_percentages = planner_success_percentages.rename(
        columns={'planner_result_success_percentage': 'success_percentage'})
    runner_success_percentages = runner_success_percentages.rename(
        columns={'runner_result_success_percentage': 'success_percentage'})

    combined_data = pd.concat([planner_success_percentages[['setup', 'planner', 'success_percentage', 'type']],
                               runner_success_percentages[['setup', 'planner', 'success_percentage', 'type']]])

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))

    g = sns.FacetGrid(combined_data, col="setup",
                      height=6, aspect=1.2, sharey=True)
    g.map_dataframe(
        sns.barplot,
        x='planner',
        y='success_percentage',
        hue='type',
        palette=palette,
        dodge=True,
        errorbar=None,
    )

    g.set_axis_labels("Planner", "Success Percentage (%)", fontsize=16)
    g.set_titles(col_template="Setup {col_name}", size=16)
    g.add_legend(title='Type', title_fontsize=16,
                 fontsize=16, loc='upper right')

    for ax in g.axes.flat:
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points', fontsize=16, color='black')

    plt.show()


def filter_successful_runs(data, result_column):
    return data[data[result_column] == True]


def plot_planner_time(data, unique_planners, palette=None):
    successful_data = filter_successful_runs(data, 'planner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner,
                    color in zip(unique_planners, palette)}

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=successful_data,
        x='setup',
        y='planner_time',
        hue='planner',
        palette=palette_dict,
        errorbar='se',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Setup', fontsize=16, labelpad=10)
    plt.ylabel('Planner Time (ms)', fontsize=16, labelpad=10)
    plt.title('Planner Time by Setup and Planner', fontsize=18, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=16, title_fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_runner_force(data, unique_planners, palette=None):
    # Filter successful runs from the data
    successful_data = filter_successful_runs(data, 'runner_result')

    # Set color palette if not provided
    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner,
                    color in zip(unique_planners, palette)}

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=successful_data,
        x='setup',
        y='runner_force',
        hue='planner',
        palette=palette_dict,
        errorbar='se',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Setup', fontsize=16, labelpad=10)
    plt.ylabel('Runner Force (N)', fontsize=16, labelpad=10)
    plt.title('Runner Force by Setup and Planner', fontsize=18, pad=15)

    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)

    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))

    plt.legend(title='Planner', loc='upper right',
               fontsize=16, title_fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

    plt.show()


def plot_runner_time(data, unique_planners, palette=None):
    successful_data = filter_successful_runs(data, 'runner_result')
    successful_data['avg_force_per_time'] = successful_data['runner_force'] / \
        successful_data['runner_time']

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner,
                    color in zip(unique_planners, palette)}

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=successful_data,
        x='setup',
        y='runner_time',
        hue='planner',
        palette=palette_dict,
        errorbar='se',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Setup', fontsize=16, labelpad=10)
    plt.ylabel('Runner Time (s)', fontsize=16, labelpad=10)
    plt.title('Runner Time by Setup and Planner', fontsize=18, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=16, title_fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def create_times_table(data):
    # Filter successful runs from the data
    successful_planner_data = filter_successful_runs(data, 'planner_result')
    successful_runner_data = filter_successful_runs(data, 'runner_result')

    # Get a set of planners that were successful in either planning or running phases (using union)
    successful_planners = set(successful_planner_data['planner'].unique()).union(
        successful_runner_data['planner'].unique()
    )

    # Filter data to only include successful planners
    successful_planner_data = successful_planner_data[successful_planner_data['planner'].isin(successful_planners)]
    successful_runner_data = successful_runner_data[successful_runner_data['planner'].isin(successful_planners)]

    # Total runs for planning and running
    total_planner_runs = data.groupby(['setup', 'planner'])['planner_result'].count().reset_index(name='total_planner_runs')
    total_runner_runs = data.groupby(['setup', 'planner'])['runner_result'].count().reset_index(name='total_runner_runs')

    # Successful runs for planning and running
    successful_planner_counts = successful_planner_data.groupby(['setup', 'planner'])['planner_result'].count().reset_index(name='successful_planner_runs')
    successful_runner_counts = successful_runner_data.groupby(['setup', 'planner'])['runner_result'].count().reset_index(name='successful_runner_runs')

    # Merge success and total counts for planning and running
    count_summary = pd.merge(total_planner_runs, total_runner_runs, on=['setup', 'planner'], how='outer')
    count_summary = pd.merge(count_summary, successful_planner_counts, on=['setup', 'planner'], how='outer')
    count_summary = pd.merge(count_summary, successful_runner_counts, on=['setup', 'planner'], how='outer')

    # Fill NaN for planners that didn't have successful runs
    count_summary.fillna(0, inplace=True)

    # Add columns for successful/total counts
    count_summary['planner_success_ratio'] = count_summary['successful_planner_runs'].astype(int).astype(str) + '/' + count_summary['total_planner_runs'].astype(int).astype(str)
    count_summary['runner_success_ratio'] = count_summary['successful_runner_runs'].astype(int).astype(str) + '/' + count_summary['total_runner_runs'].astype(int).astype(str)

    # Summary for planner times
    planner_summary = successful_planner_data.groupby(['setup', 'planner'])['planner_time'].agg(
        mean='mean', std='std', count='count').reset_index()
    planner_summary['planner_time_se'] = planner_summary['std'] / \
        np.sqrt(planner_summary['count'])
    planner_summary = planner_summary.drop(columns=['std', 'count'])
    planner_summary.columns = ['Setup', 'Planner',
                               'Planner Time Mean (ms)', 'Planner Time SE (ms)']

    # Summary for runner times
    runner_summary = successful_runner_data.groupby(['setup', 'planner'])['runner_time'].agg(
        mean='mean', std='std', count='count').reset_index()
    runner_summary['runner_time_se'] = runner_summary['std'] / \
        np.sqrt(runner_summary['count'])
    runner_summary = runner_summary.drop(columns=['std', 'count'])
    runner_summary.columns = ['Setup', 'Planner',
                              'Runner Time Mean (s)', 'Runner Time SE (s)']

    # Summary for runner force
    runner_force_summary = successful_runner_data.groupby(['setup', 'planner'])['runner_force'].agg(
        mean='mean', std='std', count='count').reset_index()
    runner_force_summary['runner_force_se'] = runner_force_summary['std'] / \
        np.sqrt(runner_force_summary['count'])
    runner_force_summary = runner_force_summary.drop(columns=['std', 'count'])
    runner_force_summary.columns = [
        'Setup', 'Planner', 'Runner Force Mean (N)', 'Runner Force SE (N)']

    # Merge all summaries into one table, using outer merge to include all planners
    summary_table = pd.merge(planner_summary, runner_summary, on=['Setup', 'Planner'], how='outer')
    summary_table = pd.merge(summary_table, runner_force_summary, on=['Setup', 'Planner'], how='outer')

    # Merge the success ratios with the summary table
    summary_table = pd.merge(summary_table, count_summary[['setup', 'planner', 'planner_success_ratio', 'runner_success_ratio']],
                             left_on=['Setup', 'Planner'], right_on=['setup', 'planner'], how='outer')

    # Drop unnecessary 'setup' and 'planner' columns from count summary
    summary_table.drop(columns=['setup', 'planner'], inplace=True)

    # Reorder columns to have success ratios at the end
    summary_table = summary_table[['Setup', 'Planner', 'Planner Time Mean (ms)', 'Planner Time SE (ms)', 
                                   'Runner Time Mean (s)', 'Runner Time SE (s)', 
                                   'Runner Force Mean (N)', 'Runner Force SE (N)',
                                   'planner_success_ratio', 'runner_success_ratio']]

    return summary_table



def generate_percentage_table(data):
    # Filter successful runs from the data
    successful_planner_data = filter_successful_runs(data, 'planner_result')
    successful_runner_data = filter_successful_runs(data, 'runner_result')

    # Get a set of planners that were successful in either planning or running phases (using union)
    successful_planners = set(successful_planner_data['planner'].unique()).union(
        successful_runner_data['planner'].unique()
    )

    # Use only successful planners in percentage table
    filtered_data = data[data['planner'].isin(successful_planners)]

    # Calculate success percentages
    planner_success_percentages = calculate_success_percentages(
        filtered_data, 'planner_result')
    runner_success_percentages = calculate_success_percentages(
        filtered_data, 'runner_result')

    # Merge planner and runner success percentages
    merged_data = pd.merge(planner_success_percentages, runner_success_percentages,
                           on=['setup', 'planner'], how='outer', suffixes=('_planner', '_runner'))

    merged_data.fillna(0, inplace=True)

    summary_table = merged_data[[
        'setup', 'planner', 'planner_result_success_percentage', 'runner_result_success_percentage']]
    summary_table.columns = [
        'Setup', 'Planner', 'Path Planning Success (%)', 'Execution to Goal Success (%)']

    return summary_table


def load_and_merge_data(files):
    data = []
    for file in files:
        data.append(pd.read_csv(file))

    combined_data = pd.concat(data, ignore_index=True)
    return combined_data


def increase_brightness(color, factor):
    return [min(1, c + factor) for c in color]


def create_real_world_movement_sequence():
    folder_path = f'/home/joris/tu_delft_ws/15_msc_thesis/Thesis images/deprecated/frames/'
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    image_files.sort()

    num_images = len(image_files)
    num_cols = 2
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(
        15, num_rows * 3), constrained_layout=True)
    axes = axes.flatten()

    for index, (ax, image_file) in enumerate(zip(axes, image_files)):
        img_path = os.path.join(folder_path, image_file)
        image = mpimg.imread(img_path)
        ax.imshow(image)
        ax.set_title(f'Stage {index + 1}', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


if __name__ == "__main__":
    experiment_2_File = '/home/joris/tu_delft_ws/15_msc_thesis/experiment_2/experiment_2_2024-06-16.csv'
    setup_1_csv_file_1 = '/home/joris/tu_delft_ws/15_msc_thesis/setup_1/setup_1_movability_2024-09-12.csv'
    setup_2_csv_file_1 = '/home/joris/tu_delft_ws/15_msc_thesis/setup_2/setup_2_movability_2024-09-12.csv'
    setup_2_csv_file_2 = '/home/joris/tu_delft_ws/15_msc_thesis/setup_2/setup_2_movability_2024-09-13.csv'

    setup_files = [setup_2_csv_file_1, setup_2_csv_file_2]

    data = load_and_merge_data(setup_files)

    unique_planners = ['nvg', 'bvg', 'svg', 'rrt']

    palette_1 = sns.color_palette("Set1", 4)
    palette_2 = sns.color_palette("Set2", 4)[::-1]

    print(generate_percentage_table(data))
    print(create_times_table(data))

    # plot_combined_planner_runner_results(data, unique_planners, palette=palette_2)

    # plot_planner_results(data, unique_planners, palette=[increase_brightness(color, 0.0) for color in palette_1])
    # plot_runner_results(data, unique_planners, palette=[increase_brightness(color, 0.2) for color in palette_1])
    # plot_planner_time(data, unique_planners, palette=[increase_brightness(color, 0.0) for color in palette_1])
    # plot_runner_time(data, unique_planners, palette=[increase_brightness(color, 0.2) for color in palette_1])
    # plot_runner_force(data, unique_planners, palette=[increase_brightness(color, 0.1) for color in palette_1])

    # create_real_world_movement_sequence()
