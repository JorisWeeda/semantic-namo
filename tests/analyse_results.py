import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_merge_data(file_1, file_2):
    data_1 = pd.read_csv(file_1)
    data_2 = pd.read_csv(file_2)
    combined_data = pd.concat([data_1, data_2], ignore_index=True)
    return combined_data

def increase_brightness(color, factor):
    return [min(1, c + factor) for c in color]

def filter_successful_runs(data, result_column):
    return data[data[result_column] == True]

def calculate_success_percentages(data, result_column):
    success_counts = data.groupby(['experiment', 'planner', result_column]).size().unstack(fill_value=0).reset_index()
    success_counts['total'] = success_counts[True] + success_counts[False]
    success_counts[f'{result_column}_success_percentage'] = (success_counts[True] / success_counts['total']) * 100
    return success_counts[['experiment', 'planner', f'{result_column}_success_percentage']]

def plot_planner_results(data, unique_planners, palette=None):
    planner_success_percentages = calculate_success_percentages(data, 'planner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(16, 8))
    sns.barplot(data=planner_success_percentages, x='experiment', y='planner_result_success_percentage', hue='planner',
                palette=palette_dict, dodge=True, errorbar=None, hue_order=unique_planners)

    plt.xlabel('Experiment', fontsize=22)
    plt.ylabel('Path Planning Success Percentage (%)', fontsize=22)
    plt.title('Path Planning Success Percentages by Experiment', fontsize=26)
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22, title_fontsize=22)
    plt.xticks(rotation=0, ha='center', fontsize=22)
    plt.yticks(fontsize=22)
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
                        textcoords='offset points', fontsize=22, color='black')
    plt.show()

def plot_runner_results(data, unique_planners, palette=None):
    runner_success_percentages = calculate_success_percentages(data, 'runner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(16, 8))
    sns.barplot(data=runner_success_percentages, x='experiment', y='runner_result_success_percentage', hue='planner',
                palette=palette_dict, dodge=True, errorbar=None, hue_order=unique_planners)

    plt.xlabel('Experiment', fontsize=22)
    plt.ylabel('Execution to Goal Success Percentage (%)', fontsize=22)
    plt.title('Execution to Goal Success Percentages by Experiment', fontsize=26)
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22, title_fontsize=22)
    plt.xticks(rotation=0, ha='center', fontsize=22)
    plt.yticks(fontsize=22)
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
                        textcoords='offset points', fontsize=22, color='black')
    plt.show()


def plot_planner_time(data, unique_planners, palette=None):
    successful_data = filter_successful_runs(data, 'planner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=successful_data, 
        x='experiment', 
        y='planner_time', 
        hue='planner', 
        palette=palette_dict, 
        errorbar='sd',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Experiment', fontsize=22, labelpad=10)
    plt.ylabel('Planner Time (ms)', fontsize=22, labelpad=10)
    plt.title('Planner Time by Experiment and Planner', fontsize=26, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(bottom=0)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22, title_fontsize=22)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_runner_time(data, unique_planners, palette=None):
    successful_data = filter_successful_runs(data, 'runner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=successful_data, 
        x='experiment', 
        y='runner_time', 
        hue='planner', 
        palette=palette_dict, 
        errorbar='sd',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Experiment', fontsize=22, labelpad=10)
    plt.ylabel('Runner Time (s)', fontsize=22, labelpad=10)
    plt.title('Runner Time by Experiment and Planner', fontsize=26, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(bottom=0)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22, title_fontsize=22)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_runner_force_per_time(data, unique_planners, palette=None):
    successful_data = filter_successful_runs(data, 'runner_result')
    
    # Calculate average force per time for each row
    successful_data['avg_force_per_time'] = successful_data['runner_force'] / successful_data['runner_time']
    
    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=successful_data, 
        x='experiment', 
        y='avg_force_per_time', 
        hue='planner', 
        palette=palette_dict,
        hue_order=unique_planners
    )

    plt.xlabel('Setup', fontsize=22, labelpad=10)
    plt.ylabel('Average Force per Time (N/s)', fontsize=22, labelpad=10)
    plt.title('Average Force per Time by Setup and Planner', fontsize=26, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=22)
    plt.yticks(fontsize=22)
    
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22, title_fontsize=22)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()






def create_times_table(data):
    successful_planner_data = filter_successful_runs(data, 'planner_result')
    successful_runner_data = filter_successful_runs(data, 'runner_result')

    planner_summary = successful_planner_data.groupby(['experiment', 'planner'])['planner_time'].agg(['mean', 'std']).reset_index()
    planner_summary.columns = ['Experiment', 'Planner', 'Planner Time Mean (ms)', 'Planner Time Std (ms)']

    runner_summary = successful_runner_data.groupby(['experiment', 'planner'])['runner_time'].agg(['mean', 'std']).reset_index()
    runner_summary.columns = ['Experiment', 'Planner', 'Runner Time Mean (s)', 'Runner Time Std (s)']

    summary_table = pd.merge(planner_summary, runner_summary, on=['Experiment', 'Planner'])

    return summary_table


def generate_percentage_table(data):
    planner_success_percentages = calculate_success_percentages(data, 'planner_result')
    runner_success_percentages = calculate_success_percentages(data, 'runner_result')
    
    merged_data = pd.merge(planner_success_percentages, runner_success_percentages,
                           on=['experiment', 'planner'], how='outer', suffixes=('_planner', '_runner'))
    
    merged_data.fillna(0, inplace=True)
    
    summary_table = merged_data[['experiment', 'planner', 'planner_result_success_percentage', 'runner_result_success_percentage']]
    summary_table.columns = ['Experiment', 'Planner', 'Path Planning Success (%)', 'Execution to Goal Success (%)']
    
    return summary_table

# Function to calculate avg_force_per_time and its summary
def calculate_avg_force_summary(data):
    successful_data = data[data['runner_result']]
    successful_data['avg_force_per_time'] = successful_data['runner_force'] / successful_data['runner_time']
    avg_force_summary = successful_data.groupby(['experiment', 'planner'])['avg_force_per_time'].agg(['mean', 'std']).reset_index()
    return avg_force_summary



if __name__ == "__main__":
    setup_1_csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/setup_1/setup_1_2024-06-15.csv'
    setup_2_csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/setup_2/setup_2_2024-06-15.csv'

    # setup_1_csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/setup_1/setup_1_2024-07-30.csv'
    # setup_2_csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/setup_2/setup_2_2024-07-30.csv'

    data = load_and_merge_data(setup_1_csv_file, setup_2_csv_file)

    unique_planners = ['VG', 'RRT', 'SVG']

    palette_1 = sns.color_palette("Set1", 3)
    palette_2 = sns.color_palette("Set2", 3)[::-1]

    plot_planner_results(data, unique_planners, palette=[increase_brightness(color, 0.0) for color in palette_1])
    plot_runner_results(data, unique_planners, palette=[increase_brightness(color, 0.1) for color in palette_1])

    plot_planner_time(data, unique_planners, palette=[increase_brightness(color, 0.0) for color in palette_2])
    plot_runner_time(data, unique_planners, palette=[increase_brightness(color, 0.1) for color in palette_2])

    print(create_times_table(data))
    print(generate_percentage_table(data))
    print(calculate_avg_force_summary(data))

    plot_runner_force_per_time(data, unique_planners, palette=[increase_brightness(color, 0) for color in palette_2])
