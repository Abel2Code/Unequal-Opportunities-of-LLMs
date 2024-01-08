# Standard Libraries
from collections import Counter, defaultdict
import os

# Pip Libraries
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from statistics import mean, stdev

# Custom Libraries
from CloudModule import generateTwoGroupWordCloud, DEFAULT_COLORS

VIOLIN_PLOT_TYPE = "violinplots"
BOX_PLOT_TYPE = "boxplots"

COLORS = ["#EA9C00", "#0b447c", "#914613"]

def generate_word_cloud_from_map(m):
    words = []
    for gender, nation_maps in m.items():
        gender = gender[0].upper() + gender[1:]
        words.append([gender, []])
        for nation, runs in nation_maps.items():
            for run in runs:
                for job, salary in run:
                    words[-1][-1].append(job)
    
    # Seperate jobs
    for tup in words:
        tup[-1] = ' | '.join(tup[-1])
        
    return (generateTwoGroupWordCloud(words[0][1], words[1][1]), words)

def generate_word_clouds(data, save_folder=None, file_name=None, model_name=None, wspace=0.1, dpi=300, font_size=100, legend=False, debug=False):
    size = len(data)
    wcs = [generate_word_cloud_from_map(d) for d in data]
    
    # Create Figure with 3 subplots
    fig, axs = plt.subplots(size, 1, figsize=(10, 15))
    
    for i, (wc, words) in enumerate(wcs):
        # Plot wc1 on first subplot
        axs[i].imshow(wc, interpolation="bilinear")
        
        # if model_name and i == 0:
        #     axs[i].set_title(model_name, fontdict = {'fontsize' : font_size})
        axs[i].set_title(f"Prompt {i+1}", fontdict = {'fontsize' : font_size})

        axs[i].axis("off")
        group_1_patch = mpatches.Patch(color=DEFAULT_COLORS[-1], label=words[0][0])
        group_2_patch = mpatches.Patch(color=DEFAULT_COLORS[0], label=words[1][0])
        
        if legend:
            axs[i].legend(handles=[group_1_patch, group_2_patch])
    
    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=wspace, hspace=wspace)
    
    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + "/wordclouds", file_name), dpi=dpi, bbox_inches='tight')
    
    # Show Figure
    plt.show()
    
    if debug:
        return wcs
    
def flatten_dict(d, just_index=None, repeats=True, ignore_none=True):
    def get_leaf_values(obj):
        out = []
        if isinstance(obj, dict):
            for v in obj.values():
                out += get_leaf_values(v)
        elif isinstance(obj, list):                
            for v in obj:
                out += get_leaf_values(v)
                
            if obj and isinstance(obj[0], tuple):
                if just_index is not None:
                    out = [o[just_index] for o in out]
                    
                if ignore_none:
                    out = [o for o in out if o]
                
                if not repeats:
                    out = list(set(out))
        elif isinstance(obj, tuple):
            out.append(obj)
        else:
            raise Exception("Invalid Type")
        
        return out            
        
    flat_dict = {}
    for key, value in d.items():
        flat_dict[key] = get_leaf_values(value)
        
    return flat_dict

def count_jobs(d, repeats=True):
    d = flatten_dict(d, just_index=0, repeats=repeats)
    return {k: Counter(v) for k, v in d.items()}

def merge_dicts_of_counters(counters):
    new_counter = defaultdict(Counter)
    
    for c in counters:
        for nation in c.keys():
            for job in c[nation]:
                new_counter[nation][job] += c[nation][job]
            
    return new_counter

# Create a tab20 colormap object
tab20_cmap = cm.get_cmap('tab20', 20)
tab20_cmap = [tab20_cmap(i) for i in range(20)]

def plot_cluster_scatter(data_list, min_score=-1, normalize=False, normalize_max=None, title=None, font_size=10, xticksize=None, jitter_width=0.2, include_legend=True, colors=tab20_cmap, ylim=None, figsize=(12, 6), dpi=300, save_folder=None, file_name=None,):      
    # Sort track jobs my max
    new_tracked_jobs = defaultdict(list)
    for tracked_jobs in data_list:
        for nationality, counts in tracked_jobs.items():
            curr_max = -1
            for job, counts in counts.items():
                new_tracked_jobs[job].append(counts)

    new_tracked_jobs = [(j,mean(v)) for j,v in new_tracked_jobs.items()]
    new_tracked_jobs.sort(key = lambda x: x[1], reverse=True)

    tracked_jobs = [j for j, _ in new_tracked_jobs]

    tracked_jobs = data_list[0]['Baseline'].keys()
    # Set up subplots
    fig, axs = plt.subplots(nrows=3, figsize=(10, 15))
    if title:
        axs[0].set_title(title, fontdict = {'fontsize' : font_size}, pad=15)

    # Iterate through each subplot
    count_map = defaultdict(lambda: defaultdict(list))
    for ax_i, (ax, job_counts) in enumerate(zip(axs, data_list)):
        # Plot each group
        groups = list(data_list[0].keys())

        # Define sample data
        num_dimensions = len(tracked_jobs)
        num_groups = len(groups)
        dimensions = np.arange(num_dimensions)
        
        # Format data
        data = []
        for g in groups:
            data.append([])
            for j in tracked_jobs:
                count = job_counts[g][j]

                if normalize:
                    dividend = normalize_max
                    if not dividend:
                        dividend = sum(job_counts[g].values())
                    count = (count / dividend) * 100
                    assert 0 <= count <= 100
                count_map[g][j].append(count)
                data[-1].append(count)

        data = np.array(data)
        
        labels = groups

        # Marrker Symbols
        markers = ['o', 's', '^', 'D',]

        for i in range(num_groups):
            jitter = np.random.uniform(-jitter_width / 2, jitter_width / 2, num_dimensions)
            ax.scatter(dimensions + jitter, data[i], color=colors[i], marker=markers[i % len(markers)], s=100, label=labels[i], edgecolors='k', alpha=0.6)


        # Set the tick labels and gridlines
        
        ax.set_xticks(dimensions)
        if ax_i == 2:
            # ax.set_xlabel('Job Type')
            ax.set_xticklabels(tracked_jobs, rotation=70, ha='right', rotation_mode='anchor', fontsize=xticksize)
        else:
            ax.set_xticklabels(['' for _ in tracked_jobs], rotation='vertical')
        ax.yaxis.grid(True)

        # Set the title, axis labels, and legend
        
        ax.set_ylabel('Probability of Job Type', fontsize=21)
        
        if ax_i == 1 and include_legend:
            ax.legend(loc='center left', fontsize=20, bbox_to_anchor=(1, 0.5))
            
        if ylim:
            ax.set_ylim(ylim)
            
        for coll in ax.collections:
            coll.set_clip_on(False)
            
    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + "/scatterplots", file_name), dpi=dpi, bbox_inches='tight')
    
    count_map = [(k, mean(stdev(v2) for _, v2 in v.items())) for k, v in count_map.items()]
    count_map.sort()
    
    # Show the plot
    # plt.tight_layout()
    plt.show()
    
    
def plot_job_type_bar(data, x_label, y_label, min_score=-1, normalize=False, normalize_max=None, ylim=None, font_size=10, bar_width=0.25, colors=tab20_cmap, figsize=(10, 5), dpi=300, save_folder=None, file_name=None): 
    groups = list(data[0].keys())
    
    # Sort track jobs my max
    tracked_job_map = defaultdict(list)
    for d in data:
        job_counts = count_jobs(d, repeats=False)
        tracked_jobs = set(v for counts in job_counts.values() for v in counts.keys())
        for j in tracked_jobs:
            curr_max = -1
            for g in groups:
                count = job_counts[g][j]

                if normalize:
                    dividend = normalize_max
                    if not dividend:
                        dividend = sum(job_counts[g].values())
                    count = (count / dividend) * 100
                curr_max = max(curr_max, count)
                
            tracked_job_map[j].append(curr_max)
    new_tracked_jobs = [(k, mean(v)) for k,v in tracked_job_map.items()]
    new_tracked_jobs.sort(key = lambda x: x[1], reverse=True)

    tracked_jobs = [j for j, _ in new_tracked_jobs]

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for i, (ax, d) in enumerate(zip(axs, data)):
        job_counts = count_jobs(d, repeats=False)

        # Define sample data
        num_dimensions = len(tracked_jobs)
        num_groups = len(groups)
        dimensions = np.arange(num_dimensions)

        # Format data
        data = []
        for g in groups:
            data.append([])
            for j in tracked_jobs:
                count = job_counts[g][j]

                if normalize:
                    dividend = normalize_max
                    if not dividend:
                        dividend = sum(job_counts[g].values())
                    count = (count / dividend) * 100
                    assert 0 <= count <= 100

                data[-1].append(count)

        data = np.array(data)

        # Set the positions and width of the bars
        positions = np.arange(data.shape[1])
        width = bar_width

        # Plot
        ax.bar(positions - (width/2 if len(data) > 1 else 0), data[0], width, label=groups[0], color = sns.color_palette()[0])
        
        if len(data) > 1:
            ax.bar(positions + width/2, data[1], width, label=groups[1], color = sns.color_palette()[1])

        # Add x and y axis labels and title
        ax.set_xlabel(x_label)
        
        axs[i].set_title(f"Prompt {i+1}", fontdict = {'fontsize' : font_size})
        
        if i == 0:
            ax.set_ylabel(y_label)
        
        if ylim:
            ax.set_ylim(ylim)

        # Set the tick labels and gridlines
        ax.set_xticks(dimensions)
        ax.set_xticklabels(tracked_jobs, rotation='vertical')
        ax.yaxis.grid(True)

        # Add legend
        if len(data) > 1:
            ax.legend()
    
    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + "/bargraphs", file_name), dpi=dpi, bbox_inches='tight')
    
    # Show plot
    plt.show()
    
def plot_job_type_bar_difference(data, x_label, y_label, min_score=-1, normalize=False, font_size=10, normalize_max=None, ylim=None, bar_width=0.25, colors=tab20_cmap, figsize=(10, 5), dpi=300, save_folder=None, file_name=None):   
    groups = list(data[0].keys())

    # Sort track jobs my max
    tracked_job_map = defaultdict(list)
    for d in data:
        job_counts = count_jobs(d, repeats=False)
        tracked_jobs = set(v for counts in job_counts.values() for v in counts.keys())
        for j in tracked_jobs:
            probs = []
            for g in groups:
                count = job_counts[g][j]

                if normalize:
                    dividend = normalize_max
                    if not dividend:
                        dividend = sum(job_counts[g].values())
                    count = (count / dividend) * 100
                probs.append(count)

            tracked_job_map[j].append(probs[0] - probs[1])
    new_tracked_jobs = [(k, mean(v)) for k,v in tracked_job_map.items()]
    new_tracked_jobs.sort(key = lambda x: x[1], reverse=True)

    tracked_jobs = [j for j, _ in new_tracked_jobs]

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for i, (ax, d) in enumerate(zip(axs, data)):
        job_counts = count_jobs(d, repeats=False)

        # Define sample data
        num_dimensions = len(tracked_jobs)
        num_groups = len(groups)
        dimensions = np.arange(num_dimensions)

        # Format data
        data = []
        for g in groups:
            data.append([])
            for j in tracked_jobs:
                count = job_counts[g][j]

                if normalize:
                    dividend = normalize_max
                    if not dividend:
                        dividend = sum(job_counts[g].values())
                    count = (count / dividend) * 100
                    assert 0 <= count <= 100

                data[-1].append(count)

        data = np.array(data)

        # Set the positions and width of the bars
        positions = np.arange(data.shape[1])
        width = 0.8

        # Plot
        labels = []
        added_male = False
        added_female = False
        for d1, d2 in zip(data[0], data[1]):
            if (d1 - d2) > 0 and not added_male: # Male
                labels.append('male')
                added_male = True
            elif (d1 - d2) < 0 and added_male and not added_female:
                labels.append('female')
                added_female = True
            else:
                labels.append(None)

        ax.bar(positions, data[0] - data[1], width, label=labels, color = [COLORS[1 if (d1 - d2) > 0 else 0] for d1, d2 in zip(data[0], data[1])])
        
        ax.set_title(f"Prompt {i+1}", fontdict = {'fontsize' : font_size})


        # Add x and y axis labels and title
        ax.set_xlabel('')
        
        if ylim:
            ax.set_ylim(ylim)

        if i == 0:
            ax.set_ylabel(y_label)
            
          


        # Set the tick labels and gridlines
        ax.set_xticks(dimensions)
        ax.set_xticklabels(tracked_jobs, rotation=85, ha='right', rotation_mode='anchor')
        # ax.tick_params(labelsize=12)
        
        ax.yaxis.grid(True)

    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + "/bargraphs", file_name), dpi=dpi, bbox_inches='tight')
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show plot
    plt.show()

    
def generate_salary_plot(data, split_by_gender=False, x='Nationality', y='Salary', ylim=None, font_size=10, plot_type=VIOLIN_PLOT_TYPE, dpi=300, save_folder=None, file_name=None,):
    # COLORS = ["#EA9C00", "#0b447c", "#914613"]
    custom_palette = ['#3371C6',"#EFA021"]
    custom_palette = ['#6aa6fb',"#EFA021"]
    # Plot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    sns.set(style="whitegrid")

    hue = 'gender' if split_by_gender else None

    for i, (ax, data) in enumerate(zip(axes, data)):
        # Flatten Data
        data = {gender: flatten_dict(d) for gender, d in data.items()}

        # Format Data into PD
        salaries = []
        genders = []
        nations = []
        jobs = []

        for gender, nation_map in data.items():
            for nation, job_sal in nation_map.items():
                for job, sal in job_sal:
                    salaries.append(sal)
                    genders.append(gender)
                    nations.append(nation)
                    jobs.append(job)

        df = pd.DataFrame({"gender": genders, "job": jobs, "Nationality": nations, "Salary": salaries})   
        if plot_type == VIOLIN_PLOT_TYPE:
            sns.violinplot(data=df, x=x, y=y, palette=custom_palette, hue=hue, inner='quartile', split=True, ax=ax)
        elif plot_type == BOX_PLOT_TYPE:
            sns.barplot(data=df, x=x, y=y, hue=hue, errorbar='sd', ax=ax, pallete=custom_palette)
            
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"Prompt {i+1}", fontdict = {'fontsize' : font_size})

        ax_labels = ax.get_xticklabels()
        
        if i != 2:
            ax_labels = ["" for _ in ax_labels]
        ax.set_xlabel('', fontsize=1)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            
        ax.set_xticklabels(ax_labels, rotation=85, ha='right', rotation_mode='anchor', fontsize=16)
        ax.set_yticklabels([f"${s.get_text()}" for s in ax.get_yticklabels()], fontsize=12)

    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + "/" + plot_type, file_name), dpi=dpi, bbox_inches='tight')

    plt.show()

# Bigger X Ticks
# Bigger Y Ticks
# Color
def generate_gender_ratio_heatmap(data, jobs, compute_func=lambda x, y: x - y, vmin=None, vmax=None, title=None, show_cbar=True, modify_label=lambda x: x, cbar_label_pad=25, cbar_label='Probability', dpi=300, save_folder=None, file_name=None,):                            
    # cmap = sns.diverging_palette(30, 250, l=65, center="dark", as_cmap=True)
    # cmap = sns.diverging_palette(COLORS[1], COLORS[0], n=2, center="dark", as_cmap=True)
    # COLORS = ["#EA9C00", "#0b447c", "#914613"]
    
    # cmap = sns.diverging_palette(47.2, 251, s=95.7, l=47.2, center='dark', as_cmap=True)
    cmap = sns.diverging_palette(47.2, 251, s=95.7, l=67.5, center='dark', as_cmap=True)


    # cmap = sns.diverging_palette(38, 251, s=95.7, l=45.5, center='dark', as_cmap=True)
    
    fig, axs = plt.subplots(nrows=3, figsize=(10, 20))
    
    for i, (d, ax) in enumerate(zip(data, axs)):
        # Format data
        nations = d['male'].keys()  

        matrix = []
        for job in jobs:
            matrix.append([])
            for nation in nations:
                if job not in d['male'][nation] or job not in d['female'][nation]:
                    matrix[-1].append(np.nan)
                else:
                    matrix[-1].append(compute_func(d['male'][nation][job], d['female'][nation][job]))
                
        # Plot
        g = sns.heatmap(matrix, xticklabels=nations, yticklabels=jobs,  vmin=vmin, vmax=vmax, square=True, cmap=cmap, cbar=show_cbar, linewidth=.5,ax=ax)
        g.set_facecolor('#f2f2f2')
        
        g.set_yticklabels(jobs, fontsize=15)
        if i != len(data) - 1:
            g.set(xticklabels=[])
        else:
            g.set_xticklabels(nations, rotation=80, ha='right', rotation_mode='anchor', fontsize=15)
        # ax.tick_params(labelsize=12)
            
        ax.set_title(f"Prompt {i+1}", fontdict = {'fontsize' : 20})


        if cbar_label and show_cbar:    
            cbar = g.collections[0].colorbar
            cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=cbar_label_pad, fontsize=18)
            cbar.ax.set_yticklabels([modify_label(s) for s in cbar.ax.get_yticklabels()]) 
    
    plt.subplots_adjust(hspace=0.08)
    
    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + '/heatmaps', file_name), dpi=dpi, bbox_inches='tight')
        
    plt.show()


def show_prompt_distribution(counted_clustered_data, dpi=300, ylim=(0,100), save_folder=None, ylabelsize=None, xticksize=None, legend_font=30, file_name=None,):
    # Extract keys and values from data
    keys = counted_clustered_data[0].keys()
    temp_values = [mean([d[k] for d in counted_clustered_data]) for k in keys]

    kvs = list(zip(temp_values, keys))
    kvs.sort(reverse=True)
    keys = [k for _, k in kvs]

    values = [[d[key] for key in keys] for d in counted_clustered_data]
    
    # Set the positions of the bars on the x-axis
    n = len(keys)
    width = 0.8
    x = np.arange(n)

    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(10,6))

    # Add the bars to the plot
    for i in range(len(counted_clustered_data)):
        ax.bar(x + i * width / len(counted_clustered_data), values[i], width / len(counted_clustered_data), color=COLORS[i], alpha=0.8, label=f"Prompt {i+1}")

    # Set the xticks and labels
    ax.set_xticks(x + width / 3)
    ax.set_xticklabels(keys, rotation=70, ha='right', rotation_mode='anchor')
    ax.tick_params(labelsize=xticksize)
    ax.grid(axis='y')
    
    ax.set_ylabel("Probability of Job Type Occurring", fontsize=ylabelsize)

    # Add legend and grid
    ax.legend(fontsize=legend_font)
    ax.grid(True, axis='y')
    
    if ylim:
        ax.set_ylim(*ylim)
    
    if save_folder and file_name:
        fig.savefig(os.path.join(save_folder + "/bargraphs", file_name), dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()