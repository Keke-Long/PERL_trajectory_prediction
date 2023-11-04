import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from brokenaxes import brokenaxes
import matplotlib.gridspec as gridspec

from MSE_data import data_sizes, models, metrics, r0, r1, r2


plt.rcParams['font.size'] = 18
tick_font_size = 16

def process_data(r):
    # Compute averages
    averages = {}
    lower_percentiles = {}
    upper_percentiles = {}
    for model in models:
        averages[model] = {}
        lower_percentiles[model] = {}
        upper_percentiles[model] = {}
        for size in data_sizes:
            if r[model][size]:  # Check if there are records for this model and size
                averages[model][size] = np.mean(r[model][size], axis=0)
                lower_percentiles[model][size] = np.percentile(r[model][size], 25, axis=0)
                upper_percentiles[model][size] = np.percentile(r[model][size], 75, axis=0)
            else:
                averages[model][size] = [0] * len(metrics)  # Fill with None if no records
                lower_percentiles[model][size] = averages[model][size]-0.05
                upper_percentiles[model][size] = averages[model][size]+0.05
    return averages, lower_percentiles, upper_percentiles


def plot_content(ax, k, averages, lower_percentiles, upper_percentiles):
    for model, color, linestyle, marker in zip(['Physics', 'Data', 'PINN', 'PERL'],
                                               ["#505050", "#ff7700", "#7A00CC", "#0059b3"],
                                               [':', ':', '--', '-'],
                                               ['d', 's', '^', 'o']):
        scaled_data_sizes = np.array(data_sizes) * 0.6  # The trianing dataset ratio
        ax.plot(scaled_data_sizes, [averages[model][size][k] for size in data_sizes],
                label=model, linestyle=linestyle, color=color,
                marker=marker, markersize=5, markerfacecolor='none', linewidth=2)

        # Add the shaded area for the standard deviation
        ax.fill_between(scaled_data_sizes,
                        [lower_percentiles[model][size][k] for size in data_sizes],
                        [upper_percentiles[model][size][k] for size in data_sizes],
                        color=color, alpha=0.3)

    ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=tick_font_size)
    ax.tick_params(axis='y', labelsize=tick_font_size)

    ax.text(0.047, -0.03, "300", transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
            color='black', fontsize=tick_font_size)

    fig.text(0.58, 0.03, 'Training Sample Size', ha='center', va='center')
    if k in [0, 1]:
        fig.text(0.04, 0.55, 'MSE of $a$ $(m^2/s^4)$', ha='center', va='center', rotation='vertical')
    else:
        fig.text(0.04, 0.55, 'MSE of $v$ $(m^2/s^2)$', ha='center', va='center', rotation='vertical')

    plt.subplots_adjust(left=0.22, right=0.95, bottom=0.17, top=0.95)


def plot_content2(ax1, ax2, k, averages, lower_percentiles, upper_percentiles):
    for model, color, linestyle, marker in zip(['Physics', 'Data', 'PINN', 'PERL'],
                                               ["#505050", "#ff7700", "#7A00CC", "#0059b3"],
                                               [':', ':', '--', '-'],
                                               ['d', 's', '^', 'o']):
        scaled_data_sizes = np.array(data_sizes) * 0.6
        ax1.plot(scaled_data_sizes, [averages[model][size][k] for size in data_sizes],
                 label=model, linestyle=linestyle, color=color,
                 marker=marker, markersize=5, markerfacecolor='none', linewidth=2)
        ax2.plot(scaled_data_sizes, [averages[model][size][k] for size in data_sizes],
                 linestyle=linestyle, color=color,
                 marker=marker, markersize=5, markerfacecolor='none', linewidth=2)

        # Add the shaded area for the standard deviation
        ax1.fill_between(scaled_data_sizes,
                        [lower_percentiles[model][size][k] for size in data_sizes],
                        [upper_percentiles[model][size][k] for size in data_sizes],
                        color=color, alpha=0.3)
        ax2.fill_between(scaled_data_sizes,
                        [lower_percentiles[model][size][k] for size in data_sizes],
                        [upper_percentiles[model][size][k] for size in data_sizes],
                        color=color, alpha=0.3)

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.tick_params(axis='x', labelsize=tick_font_size)
    ax2.tick_params(axis='x', labelsize=tick_font_size)
    ax1.tick_params(axis='y', labelsize=tick_font_size)
    ax2.tick_params(axis='y', labelsize=tick_font_size)

    ax1.text(0.95, 0.44, "300", transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
            color='black', fontsize=tick_font_size)

    fig.text(0.58, 0.03, 'Training Sample Size', ha='center', va='center')
    if k in [0, 1]:
        fig.text(0.04, 0.55, 'MSE of $a$ $(m^2/s^4)$', ha='center', va='center', rotation='vertical')
    else:
        fig.text(0.04, 0.55, 'MSE of $v$ $(m^2/s^2)$', ha='center', va='center', rotation='vertical')

    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.11, top=0.95)


def apply_axis_adjustments(ax1, ax2):
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False)

    # Hide x-axis labels for the top subplot
    ax1.set_xticks([])
    ax1.xaxis.set_visible(False)

    ax1.set_xscale("log")
    ax2.set_xscale("log")

    d = .015
    kwargs = dict(color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), transform=ax1.transAxes, **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)


img_size = (4, 4)
img_dpi = 350

# r0 #################
k = 0 #multi-a
fig, ax = plt.subplots(figsize=img_size)
plot_content(ax, k, process_data(r0)[0], process_data(r0)[1], process_data(r0)[2])
plt.yscale("symlog")
formatter = ticker.ScalarFormatter() # 使用 ScalarFormatter 来防止使用科学计数法
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim([0, 2])
ax.set_yticks([0, 0.5, 1, 2, 3])
plt.savefig("r0_multi-step a.png", dpi=img_dpi)
plt.close(fig)


# r0
k = 1
fig, ax = plt.subplots(figsize=img_size)
plot_content(ax, k, process_data(r0)[0], process_data(r0)[1], process_data(r0)[2])
plt.yscale("symlog")
formatter = ticker.ScalarFormatter() # 使用 ScalarFormatter 来防止使用科学计数法
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim([0, 3])
ax.set_yticks([0, 0.5, 1, 2, 3])
plt.savefig("r0_one-step a.png", dpi=img_dpi)
plt.close(fig)


# r0
k = 2
fig = plt.figure(figsize=img_size)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 0.05])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
bs = 0.04  # bottom split
ts = 0.17  # top split
ax2.set_ylim(0, bs)
ax1.set_ylim(ts, 0.23)
ax1.set_yticks(np.linspace(ts, 0.23, 3))
ax2.set_yticks(np.linspace(0, bs, 5))
apply_axis_adjustments(ax1, ax2)
plot_content2(ax1, ax2, k, process_data(r0)[0], process_data(r0)[1], process_data(r0)[2])
plt.savefig("r0_multi-step v.png", dpi=img_dpi)
plt.close()


# r0
k = 3
fig = plt.figure(figsize=img_size)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 0.05])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
bs = 0.04  # bottom split
ts = 0.07  # top split
ax2.set_ylim(0, bs)
ax1.set_ylim(ts, 0.09)
ax1.set_yticks(np.linspace(ts, 0.09, 3))
ax2.set_yticks(np.linspace(0, bs, 5))
apply_axis_adjustments(ax1, ax2)
plot_content2(ax1, ax2, k, process_data(r0)[0], process_data(r0)[1], process_data(r0)[2])
plt.savefig("r0_one-step v.png", dpi=img_dpi)
plt.close()



# r1 #################
k = 0
fig, ax = plt.subplots(figsize=img_size)
plot_content(ax, k, process_data(r1)[0], process_data(r1)[1], process_data(r1)[2])
plt.yscale("symlog")
formatter = ticker.ScalarFormatter() # 使用 ScalarFormatter 来防止使用科学计数法
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim([0, 4])
ax.set_yticks([0, 0.5, 1, 2, 3,4])
plt.savefig("r1_multi-step a.png", dpi=img_dpi)
plt.close(fig)


# r1
k = 1
fig, ax = plt.subplots(figsize=img_size)
plot_content(ax, k, process_data(r1)[0], process_data(r1)[1], process_data(r1)[2])
plt.yscale("symlog")
formatter = ticker.ScalarFormatter() # 使用 ScalarFormatter 来防止使用科学计数法
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim([0, 1.4])
ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2,1.4])
plt.savefig("r1_one-step a.png", dpi=img_dpi)
plt.close(fig)


# r1
k = 2
fig = plt.figure(figsize=img_size)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 0.05])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
bs = 0.05  # bottom split
ts = 0.17  # top split
ax2.set_ylim(0, bs)
ax1.set_ylim(ts, 0.23)
ax1.set_yticks(np.linspace(ts, 0.23, 3))
ax2.set_yticks(np.linspace(0, bs, 6))
apply_axis_adjustments(ax1, ax2)
plot_content2(ax1, ax2, k, process_data(r1)[0], process_data(r1)[1], process_data(r1)[2])
plt.savefig("r1_multi-step v.png", dpi=img_dpi)
plt.close()


# r1
k = 3
fig = plt.figure(figsize=img_size)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 0.05])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
bs = 0.05  # bottom split
ts = 0.08  # top split
ax2.set_ylim(0, bs)
ax1.set_ylim(ts, 0.09)
ax1.set_yticks(np.linspace(ts, 0.09, 2))
ax2.set_yticks(np.linspace(0, bs, 6))
apply_axis_adjustments(ax1, ax2)
plot_content2(ax1, ax2, k, process_data(r1)[0], process_data(r1)[1], process_data(r1)[2])
plt.savefig("r1_one-step v.png", dpi=img_dpi)
plt.close()



# r2 #################
k = 0
fig, ax = plt.subplots(figsize=img_size)
plot_content(ax, k, process_data(r2)[0], process_data(r2)[1], process_data(r2)[2])
plt.yscale("symlog")
formatter = ticker.ScalarFormatter() # 使用 ScalarFormatter 来防止使用科学计数法
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim([0, 2])
ax.set_yticks([0, 0.5, 1, 2, 3])
plt.savefig("r2_multi-step a.png", dpi=img_dpi)
plt.close(fig)


# r2
k = 1
fig, ax = plt.subplots(figsize=img_size)
plot_content(ax, k, process_data(r2)[0], process_data(r2)[1], process_data(r2)[2])
plt.yscale("symlog")
formatter = ticker.ScalarFormatter() # 使用 ScalarFormatter 来防止使用科学计数法
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim([0, 1.9])
ax.set_yticks([0, 0.5, 1, 1.5])
plt.savefig("r2_one-step a.png", dpi=img_dpi)
plt.close(fig)


# r2
k = 2
fig = plt.figure(figsize=img_size)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 0.05])  # The last value is for a small spacing between the two plots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
bs = 0.04  # bottom split
ts = 0.17  # top split
ax2.set_ylim(0, bs)
ax1.set_ylim(ts, 0.23)
ax1.set_yticks(np.linspace(ts, 0.23, 3))
ax2.set_yticks(np.linspace(0, bs, 5))
apply_axis_adjustments(ax1, ax2)
plot_content2(ax1, ax2, k, process_data(r2)[0], process_data(r2)[1], process_data(r2)[2])
plt.savefig("r2_multi-step v.png", dpi=img_dpi)
plt.close()


# r2
k = 3
fig = plt.figure(figsize=img_size)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0.05])  # The last value is for a small spacing between the two plots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
bs = 0.04  # bottom split
ts = 0.08  # top split
ax2.set_ylim(0, bs)
ax1.set_ylim(ts, 0.09)
ax1.set_yticks(np.linspace(ts, 0.09, 2))
ax2.set_yticks(np.linspace(0, bs, 5))
apply_axis_adjustments(ax1, ax2)
plot_content2(ax1, ax2, k, process_data(r2)[0], process_data(r2)[1], process_data(r2)[2])
plt.savefig("r2_one-step v.png", dpi=img_dpi)

