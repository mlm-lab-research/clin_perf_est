from pathlib import Path
import sys
root = Path().resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap_b = plt.get_cmap('Blues')  # same hue, different lightness
cmap_r = plt.get_cmap('Reds')
import os
import matplotlib.transforms as transforms
import string

def add_letters(fig, ax, dx=-35/72., dy=15/72.):
    
    letterkwargs = dict(weight='bold', va='top', ha='left')

    offset = transforms.ScaledTranslation(
            dx, dy, fig.dpi_scale_trans)

    for idx in range(len(ax)):
        ax[idx].text(0, 1, string.ascii_lowercase[idx], transform=ax[idx].transAxes + offset, 
                    **letterkwargs)
        

##############################
#### MICCAI PAPER FIGURES ####
##############################
def plot_controlled_shift(data_df, save_folder, optional_name='', FIGURE_WIDTH=4.803, shift_type='label', methods_to_plot=None, metrics=None):   
    color_mapping = {'test': 'k',
                'validation': 'grey',
                'CBPE': 'g',
                'ATC': cmap_b(0.55),
                'CMATC': cmap_b(0.9),
                'DoC': cmap_r(0.55),
                'CMDoC': cmap_r(0.9),
                }
    methods_name_map = {'CMATC': 'CM-ATC', 'CMDoC': 'CM-DoC', 'test':'realized'}
    if metrics is not None:
        keys=metrics
    else:
        keys=['bal_accuracy','recall', 'specificity', 'auc', 
                                    'accuracy', 'precision', 'f1_score', 'ACE / RBS', ]
    if methods_to_plot is not None:
        color_mapping = {key: color_mapping[key] for key in methods_to_plot if key in color_mapping}
    else:
        methods_to_plot = list(color_mapping.keys())

    with plt.style.context('../config/plot_style.txt'):  # type: ignore # Use the custom style
        mpl.rcParams.update({'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5,
                             'legend.fontsize': 5, 'figure.titlesize': 6, 'axes.titlepad': 2, 'axes.labelpad': 2,})
        if metrics is None:
            m = 3
            FIG_HEIGHT = 0.6 * FIGURE_WIDTH
        else:
            m = 2
            FIG_HEIGHT = 0.4 * FIGURE_WIDTH
        fig, axs = plt.subplots(m, len(keys) // m, figsize=(FIGURE_WIDTH, FIG_HEIGHT), sharey=False, sharex=True,layout='constrained')
        axs = axs.flatten()
            
        # add_letters(fig, axs, dx = -8/72, dy=15/72)
        x = data_df['shift']
        x = pd.to_numeric(data_df['shift'], errors='coerce').values

        ax = axs
        add_letters(fig, ax, dx=-4/72, dy=6/72)
        
        overall_min_lower = np.inf
        overall_min_lower_row2 = np.inf
        overall_min_lower_row3 = np.inf
        for n, metric_name in enumerate(keys):
            for column in data_df.columns:
                # Check if the column is in the methods to plot
                if methods_to_plot is not None and column.split('_')[0] not in methods_to_plot:
                    continue
                if metric_name in column and 'mean' in column:
                    if metric_name == 'accuracy' and 'bal' in column:
                        continue               
                    color = color_mapping[column.split('_')[0]]
                    if metric_name == 'bal_accuracy':
                        metric_name_ = 'bal. acc'
                    elif metric_name == 'f1_score':
                        metric_name_ = 'F1-Score'
                    elif metric_name == 'auc':
                        metric_name_ = 'AUC'
                    elif metric_name == 'precision':
                        metric_name_ = 'PPV'
                    elif metric_name in ['TPr', 'FPr', 'TNr', 'FNr']:
                        metric_name_ = metric_name.replace('r', 'f ').replace('r', 'f ')
                    else:
                        metric_name_ = metric_name
                    
                    # ax[n].set_title(metric_name_, pad=10, fontsize=7)
                    ax[n].set_ylabel(metric_name_,)
                    if 'new' in column.lower():
                        ax[n].plot(x, data_df[column], color=color, alpha=1, linestyle='--', label=methods_name_map.get(column.split('_')[0], column.split('_')[0]))
                    else:
                        ax[n].plot(x, data_df[column], color=color, alpha=1, label=methods_name_map.get(column.split('_')[0], column.split('_')[0]))
                    mean_col = pd.to_numeric(data_df[column]).values
                    std_col = pd.to_numeric(data_df[column.replace('mean', 'std')]).values
                    lower = mean_col - std_col
                    upper = mean_col + std_col
                    # if 'new' in column.lower():
                    #     axs[n].fill_between(x, lower, upper, color=color, alpha=0.3, hatch='xxx')  # <-- added hatch
                    # else:
                    #     axs[n].fill_between(x, lower, upper, color=color, alpha=0.3)
                    # ax[n].fill_between(x, lower, upper, color=color, alpha=0.3)
                    ax[n].set_xlim(0, 1)
        
                    if np.min(lower) < overall_min_lower and n//4==0:
                        overall_min_lower = np.min(lower)
                    elif np.min(lower) < overall_min_lower_row2 and n//4==1:
                        overall_min_lower_row2 = np.min(lower)
                    elif np.min(lower) < overall_min_lower_row3 and n//4==2:    
                        overall_min_lower_row3 = np.min(lower)
                    
                # ECE RBS
                if ('ece' in column and 'mean' in column or 'rbs' in column and 'mean' in column) and metric_name == 'ACE / RBS':
                    color = 'orange' if 'rbs' in column else 'grey'
                    label_name = 'RBS' if 'rbs' in column else 'ACE'
                    # ax[n].set_title('Calibration', pad=10, fontsize=7)
                    ax[n].set_ylabel('Calibration')
                    #ax[n].set_ylabel('ACE / RBS')
 
                    if 'new' in column.lower():
                        ax[n].plot(x, data_df[column], color=color, alpha=1, label=label_name, linestyle='--')
                    else:
                        ax[n].plot(x, data_df[column], color=color, alpha=1, label=label_name)
                    mean_col = pd.to_numeric(data_df[column]).values
                    std_col = pd.to_numeric(data_df[column.replace('mean', 'std')]).values
                    lower = mean_col - std_col
                    upper = mean_col + std_col
                    # if 'new' in column.lower():
                    #     axs[n].fill_between(x, lower, upper, color=color, alpha=0.3, hatch='xxx')  # <-- added hatch
                    # else:
                    #     axs[n].fill_between(x, lower, upper, color=color, alpha=0.3)
                    # ax[n].fill_between(x, lower, upper, color=color, alpha=0.3)                    
                    ax[n].legend(loc='upper center', columnspacing=1,ncol=2,  # Adjust the spacing between columns
                    handlelength=0.7, # Adjust the length of the legend handles
                    #handleheight=2,  # Adjust the height of the legend handles
                    frameon=False,
                    fontsize=5)
                    axs[n].set_ylim(0, 1.03)  
                    axs[n].yaxis.tick_right()
                    axs[n].spines["left"].set_visible(False)
                    axs[n].spines["right"].set_visible(True)
            ax[n].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
            ax[n].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax[n].tick_params(axis='y', labelsize=5)  # or any font size you want
            ax[n].tick_params(axis='x', labelsize=5)  # or any font size you want
        for ax in axs[1:]:
            # if ax == axs[len(keys)//m] or ax==axs[7] or ax ==axs[8]:
            #     continue
            if ax == axs[4] or ax==axs[-1]:
                continue
            ax.set_yticklabels([])  # Remove y-tick labels for the last subplot
        for ax in axs[:3]:
            ax.set_ylim(overall_min_lower - 0.1, 1.03)  # Set y-limits to [0, 1.03] for all subplots
        for ax in axs[3:-2]:
            ax.set_ylim(overall_min_lower_row2- 0.1, 1.03)  # Set y-limits to [0, 1.03] for all subplots
        # for ax in axs[:7:-1]:
        #     ax.set_ylim(overall_min_lower_row3- 0.1, 1.03)  # Set y-limits to [0, 1.03] for all subplots
        # axs[-1].set_yticks(np.arange(0, 0.4, 0.8))  # Set y-ticks for the last subplot
        # axs[-1].set_yticklabels(np.arange(0, 0.4, 0.8), fontsize=5)  # Set y-tick labels for the last subplot   
        
        # Add legend
        # Get every handle/label created on the first axes --
        # this captures all artists because each new line was added to ax[0]..ax[N]
        handles, labels = axs[0].get_legend_handles_labels()
        labels = labels[:1]+labels[3::2]+labels[1:2]+labels[2::2]
        handles = handles[:1]+handles[3::2]+handles[1:2]+handles[2::2]


        fig.legend(handles, labels, ncols=len(labels), bbox_to_anchor=(0.5, 1.07), loc='upper center',
            columnspacing=1,  # Adjust the spacing between columns
            handlelength=1.3,  # Adjust the length of the legend handles
            #handleheight=2,  # Adjust the height of the legend handles
            frameon=False,
            fontsize=6)
        # # Split the legend entries
        # first_three  = handles[:3], labels[:3]
        # second_three = handles[3:6], labels[3:6]

        # Put the legends where you want them
        # ax[0].legend(*first_three,  loc='lower left', handlelength=1,  # Adjust the length of the legend handles
        #     frameon=False,
        #     fontsize=4.5)
        # ax[1].legend(*second_three, loc='lower left', handlelength=1,  # Adjust the height of the legend handles
        #     frameon=False,
        #     fontsize=4.5)
        #put legend in last ax
        # handles = []
        # labels = []
        # methods = [i.split('_')[0] for i in set(data_df.columns)] # Get unique methods from the columns
        # for label, color in color_mapping.items():
        #     if label not in methods:
        #         continue
        #     handle = mlines.Line2D([0], [0], color=color, lw=2, label=label)
        #     handles.append(handle)
        #     labels.append(label)
        # # handles.append(mlines.Line2D([0], [0], color='brown', lw=2, label='RBS'))
        # # handles.append(mlines.Line2D([0], [0], color='grey', lw=2, label='ACE'))
        # # labels.append('RBS')
        # # labels.append('ACE')
        # # Add the labels to the legend
        # labels = pd.Series(labels)
        # labels.replace({'ATCnew': 'CM_ATC', 'DoCnew': 'CM_DoC', 'test':'realized'}, inplace=True)


        # axs[-1].legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.97),
        #     columnspacing=0.6,  # Adjust the spacing between columns
        #     handlelength=1.2,  # Adjust the length of the legend handles
        #     # handleheight=2,  # Adjust the height of the legend handles
        #     frameon=False,
        #     fontsize=5)
        #put legend in last ax

        if shift_type == 'label':
            fig.text(0.5, -0.04, 'Positive Class Prevalence', ha='center') 
        elif shift_type == 'covariate':
            fig.text(0.5, -0.03, 'Majority Fraction', ha='center')
        # fig.text(-0.02, 0.5, 'realized / estimated', va='center', rotation='vertical')
        # fig.savefig(os.path.join(save_folder, f'prevalence_shift_simulation_{optional_name}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(save_folder, f'{shift_type}_shift_simulation_{optional_name}.pdf'), bbox_inches='tight')
