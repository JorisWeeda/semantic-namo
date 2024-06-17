import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    plt.xlabel('Experiment', fontsize=16)
    plt.ylabel('Planner Success Percentage (%)', fontsize=16)
    plt.title('Planner Success Percentages by Experiment', fontsize=18)
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)
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
                        textcoords='offset points', fontsize=14, color='black')
    plt.show()

def plot_runner_results(data, unique_planners, palette=None):
    runner_success_percentages = calculate_success_percentages(data, 'runner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(16, 8))
    sns.barplot(data=runner_success_percentages, x='experiment', y='runner_result_success_percentage', hue='planner',
                palette=palette_dict, dodge=True, errorbar=None, hue_order=unique_planners)

    plt.xlabel('Experiment', fontsize=16)
    plt.ylabel('Runner Success Percentage (%)', fontsize=16)
    plt.title('Runner Success Percentages by Experiment', fontsize=18)
    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)
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
                        textcoords='offset points', fontsize=14, color='black')
    plt.show()


def filter_successful_runs(data, result_column):
    return data[data[result_column] == True]

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
        errorbar='se',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Experiment', fontsize=16, labelpad=10)
    plt.ylabel('Planner Time (ms)', fontsize=16, labelpad=10)
    plt.title('Planner Time by Experiment and Planner', fontsize=18, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_runner_force(data, unique_planners, palette=None):
    successful_data = filter_successful_runs(data, 'runner_result')

    if palette is None:
        palette = sns.color_palette("Set1", n_colors=len(unique_planners))
    palette_dict = {planner: color for planner, color in zip(unique_planners, palette)}

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=successful_data, 
        x='experiment', 
        y='runner_force', 
        hue='planner', 
        palette=palette_dict, 
        errorbar='se',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Experiment', fontsize=16, labelpad=10)
    plt.ylabel('Runner Force (N)', fontsize=16, labelpad=10)
    plt.title('Runner Force by Experiment and Planner', fontsize=18, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
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
        errorbar='se',
        capsize=.1,
        hue_order=unique_planners
    )

    plt.xlabel('Experiment', fontsize=16, labelpad=10)
    plt.ylabel('Runner Time (s)', fontsize=16, labelpad=10)
    plt.title('Runner Time by Experiment and Planner', fontsize=18, pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(title='Planner', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def load_and_merge_data(file_1, file_2):
    data_1 = pd.read_csv(file_1)
    data_2 = pd.read_csv(file_2)
    combined_data = pd.concat([data_1, data_2], ignore_index=True)
    return combined_data


def increase_brightness(color, factor):
    return [min(1, c + factor) for c in color]


if __name__ == "__main__":
    experiment_1_csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/experiment_1/experiment_1_2024-06-15.csv'
    experiment_2_csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/experiment_2/experiment_2_2024-06-15.csv'

    data = load_and_merge_data(experiment_1_csv_file, experiment_2_csv_file)

    unique_planners = ['VG', 'RRT', 'SVG']

    palette_1 = sns.color_palette("Set1", 3)
    palette_2 = sns.color_palette("Set2", 3)[::-1]

    plot_planner_results(data, unique_planners, palette=[increase_brightness(color, 0.0) for color in palette_1])
    plot_runner_results(data, unique_planners, palette=[increase_brightness(color, 0.2) for color in palette_1])
    plot_planner_time(data, unique_planners, palette=[increase_brightness(color, 0.0) for color in palette_2])
    plot_runner_force(data, unique_planners, palette=[increase_brightness(color, 0.1) for color in palette_2])
    plot_runner_time(data, unique_planners, palette=[increase_brightness(color, 0.2) for color in palette_2])
