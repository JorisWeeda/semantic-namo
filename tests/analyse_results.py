import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_successful_results_room_sizes(data, room_sizes, methods, method_colors):
    result_counts = pd.DataFrame(
        columns=['room_area', 'nvg_result', 'svg_result', 'rrt_result'])
    for room in room_sizes:
        subset = data[data['room_area'] == room]
        result_counts = pd.concat([result_counts, pd.DataFrame({
            'room_area': [room],
            'nvg_result': [subset['nvg_result'].sum()],
            'svg_result': [subset['svg_result'].sum()],
            'rrt_result': [subset['rrt_result'].sum()]
        })], ignore_index=True)

    _, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.2
    bar_positions = np.arange(len(room_sizes)) * (len(methods) + 1) * bar_width

    for i, method in enumerate(methods):
        bar_values = result_counts[method].values
        bar_offset = i * bar_width
        ax.bar(bar_positions + bar_offset, bar_values, width=bar_width,
               color=method_colors[i], label=method.split("_")[0].upper())

    ax.set_title('Number of Successful Results by Room Area and Planner',
                 fontsize=18, fontweight='bold')
    ax.set_xlabel('Room Area (m$^2$)', fontsize=14)
    ax.set_ylabel('Number of Successful Runs', fontsize=14)
    ax.set_xticks(bar_positions + bar_width)
    ax.set_xticklabels([f'Room {room}' for room in room_sizes], fontsize=12)

    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Planner', fontsize=12, title_fontsize=12, loc='upper left')


def plot_successful_time_boxplots_room_sizes(data, room_sizes, methods, method_times, method_colors):
    _, axes = plt.subplots(1, len(room_sizes), figsize=(14, 8))

    for idx, room in enumerate(room_sizes):
        subset = data[data['room_area'] == room]
        successful_subset = subset[(subset['nvg_result']) | (subset['svg_result']) | (subset['rrt_result'])]
        time_data = []
        for method_time, method in zip(method_times, methods):
            method_data = successful_subset[successful_subset[method]][method_time]
            time_data.append(method_data)

        sns.boxplot(data=time_data, palette=method_colors, showfliers=False, ax=axes[idx])
        axes[idx].set_title(f'Room Area: {room}', fontsize=14)

        if idx == 0:
            axes[idx].set_ylabel('Time (s)', fontsize=12)

        axes[idx].set_xticks(np.arange(len(methods)))
        axes[idx].set_xticklabels([method.split('_')[0].upper() for method in methods], fontsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)

    plt.suptitle('Boxplot of Time for Successful Trials by Room Area and Method', fontsize=18, fontweight='bold')


def plot_run_time_boxplots(data, room_sizes, methods, method_times, method_colors):
    _, axes = plt.subplots(1, len(room_sizes), figsize=(14, 8), sharey=True)

    for idx, room in enumerate(room_sizes):
        subset = data[data['room_area'] == room]
        time_data = []
        for method_time, method in zip(method_times, methods):
            successful_times = subset[subset[method]][method_time]
            time_data.append(successful_times)

        sns.boxplot(data=time_data, palette=method_colors, showfliers=False, ax=axes[idx])
        axes[idx].set_title(f'Room Area: {room}', fontsize=14)
        if idx == 0:
            axes[idx].set_ylabel('Time (s)', fontsize=12)
        axes[idx].set_xticks(np.arange(len(methods)))
        axes[idx].set_xticklabels([method.split('_')[0].upper() for method in methods], fontsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)

    plt.suptitle('Boxplot of Run Time for Successful Trials by Room Area and Method', fontsize=18, fontweight='bold')


def plot_run_force_boxplots(data, room_sizes, methods, method_forces, method_colors):
    _, axes = plt.subplots(1, len(room_sizes), figsize=(14, 8), sharey=True)

    for idx, room in enumerate(room_sizes):
        subset = data[data['room_area'] == room]
        force_data = []
        for method_force, method in zip(method_forces, methods):
            successful_forces = subset[subset[method]][method_force]
            force_data.append(successful_forces)

        sns.boxplot(data=force_data, palette=method_colors, showfliers=False, ax=axes[idx])
        axes[idx].set_title(f'Room Area: {room}', fontsize=14)
        if idx == 0:
            axes[idx].set_ylabel('Force (N)', fontsize=12)
        axes[idx].set_xticks(np.arange(len(methods)))
        axes[idx].set_xticklabels([method.split('_')[0].upper() for method in methods], fontsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)

    plt.suptitle('Boxplot of Run Force for Successful Trials by Room Area and Method', fontsize=18, fontweight='bold')


if __name__ == "__main__":
    csv_file = '/home/joris/tu_delft_ws/15_msc_thesis/benchmark_global_planner/2024-06-08_16-24-21.csv'
    data = pd.read_csv(csv_file)

    methods = ['nvg_result', 'svg_result', 'rrt_result']
    method_times = ['nvg_time', 'svg_time', 'rrt_time']

    run_times = ['nvg_run_time', 'svg_run_time', 'rrt_run_time']
    run_forces = ['nvg_run_force', 'svg_run_force', 'rrt_run_force']

    method_colors = sns.color_palette("magma", len(methods))

    room_sizes = data['room_area'].unique()

    plot_successful_results_room_sizes(data, room_sizes, methods, method_colors)
    plot_successful_time_boxplots_room_sizes(data, room_sizes, methods, method_times, method_colors)

    if all(data.get(key, None) for key in run_times):
        plot_run_time_boxplots(data, room_sizes, methods, run_times, method_colors)

    if all(data.get(key, None) for key in run_times):
        plot_run_force_boxplots(data, room_sizes, methods, run_forces, method_colors)

    plt.show()
