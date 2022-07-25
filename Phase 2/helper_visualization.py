import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as st
import datetime as dt
pd.options.mode.chained_assignment = None

intensity_c = {'Sedentary': '#a8d0e3', # lightblue
               'Light': '#f6b79b', # peach
               'Moderate / Vigorous': '#DE6454' # red
              }

intensity_legend = {'Sedentary': '#a8d0e3', # lightblue
               'Light': '#f6b79b', # peach
               'Moderate /\n Vigorous': '#DE6454' # red
              }


def set_intensity(row, col_name):
    if row[col_name]<1.5:
        return 'Sedentary'
    elif 1.5<=row[col_name]<3:
        return 'Light'
    elif row[col_name]>=3:
        return 'Moderate / Vigorous'
    

def inlab_preprocess(inlab):
    inlab = inlab[['Participant', 'Activity','MET (GoogleFit)','MET (Freedson)','MET (VM3)', 'MET (MetCart)', 'estimation']]
    inlab['Intensity'] = ''
    inlab['Intensity'] = inlab.apply(set_intensity, col_name='MET (MetCart)', axis=1)
    inlab['avgMetCartEst'] = inlab[['MET (MetCart)', 'estimation']].mean(axis=1)
    inlab['avgMetCartVM3'] = inlab[['MET (MetCart)', 'MET (VM3)']].mean(axis=1)
    inlab['avgVM3Est'] = inlab[['MET (VM3)', 'estimation']].mean(axis=1)

    inlab['MetCart-Est'] = inlab.apply(lambda x: x['MET (MetCart)'] - x['estimation'], axis=1)
    inlab['MetCart-VM3'] = inlab.apply(lambda x: x['MET (MetCart)'] - x['MET (VM3)'], axis=1)
    inlab['MetCart-Freedson'] = inlab.apply(lambda x: x['MET (MetCart)'] - x['MET (Freedson)'], axis=1)
    inlab['MetCart-GoogleFit'] = inlab.apply(lambda x: x['MET (MetCart)'] - x['MET (GoogleFit)'], axis=1)
    inlab['VM3-Est'] = inlab.apply(lambda x: x['MET (VM3)'] - x['estimation'], axis=1)
    return(inlab)


def plot_subplot(ax, xdata, ydata, df, xlabel, ylabel, title, color_dic, color_by):
    i=3
    for key in color_dic:
        temp = df.loc[df[color_by]==key]
        ax.scatter(temp[xdata], temp[ydata], c = color_dic[key], s=20, zorder=i)
        i-=1
        
    # x and y label
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    l = df[ydata].tolist()
    md = df[ydata].mean()
    sd = df[ydata].std()
    
    # calculating 95%CI around the mean, mean+-1.96s
    ci_mean = st.t.interval(0.95, len(l)-1, loc=np.mean(l), scale=st.sem(l))
    ci_plus1z = st.t.interval(0.95, len(l)-1, loc=np.mean(l) + 1.96*np.std(l), scale=(3*np.std(l)**2/len(l))**0.5)
    ci_minus1z = st.t.interval(0.95, len(l)-1, loc=np.mean(l) - 1.96*np.std(l), scale=(3*np.std(l)**2/len(l))**0.5)

    # 3 lines at mean, mean+1.96s, mean-1.96s
    ax.axhline(md,           color='black', linestyle='-', zorder=20)
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--', zorder=20)
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--', zorder=20)

    # x and y lim
    xmin, xmax = ax.get_xlim()
    xrange = xmax-xmin
    ax.set_xlim([xmin, xmax+xrange*0.21])
    ymin, ymax = ax.get_ylim()
    yrange = ymax-ymin
    
    textpos = yrange*0.02
    numpos = yrange*0.06
    rightmargin = xrange*0.55

    #text on the lines
    ax.text(xmax+rightmargin, md+textpos,'Mean', ha='right', zorder=20)
    ax.text(xmax+rightmargin, md-numpos,round(md,2), ha='right', zorder=20)
    ax.text(xmax+rightmargin, md+textpos+1.96*sd,'+1.96 SD', ha='right', zorder=20)
    ax.text(xmax+rightmargin, md-numpos+1.96*sd,round(md + 1.96*sd,2), ha='right', zorder=20)
    ax.text(xmax+rightmargin, md+textpos-1.96*sd,'-1.96 SD', ha='right', zorder=20)
    ax.text(xmax+rightmargin, md-numpos-1.96*sd,round(md - 1.96*sd,2), ha='right', zorder=20)
    
    #shaded bands for 95% CI
    ax.axhspan(ci_mean[0], ci_mean[1], alpha=0.5, color='lightgray', zorder=1)
    ax.axhspan(ci_plus1z[0], ci_plus1z[1], alpha=0.5, color='lightgray', zorder=1)
    ax.axhspan(ci_minus1z[0], ci_minus1z[1], alpha=0.5, color='lightgray', zorder=1)
    
    ax.set_xticks((2, 4, 6, 8))
    
    ax.set_ylim([-5, 5])
    
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(intensity_legend.values(), intensity_legend.keys())]
    ax.legend(handles=patches, labels=[label for _, label in zip(intensity_legend.values(), intensity_legend.keys())], loc='upper left', ncol = 1)
    return


def blandAltman_lab(df, filename, color_dic, color_by):
        
    fig = plt.figure(figsize = (12,10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3, sharey=ax1)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1 = plot_subplot(ax1, 'MET (MetCart)', 'MetCart-Est', df,
                       'MetCart METs \n\n(a)', 'MetCart METs - WRIST METs',
                       'MetCart METs vs WRIST METs',
                       color_dic, color_by)

    ax2 = plot_subplot(ax2, 'MET (MetCart)', 'MetCart-VM3', df,
                       'MetCart METs \n\n(b)', 'MetCart METs - ActiGraph VM3 METs', 
                       'MetCart METs vs ActiGraph VM3 METs',
                       color_dic, color_by)

    ax3 = plot_subplot(ax3, 'MET (MetCart)', 'MetCart-Freedson', df, 
                       'MetCart METs \n\n(c)', 'MetCart METs - Freedson METs', 
                       'MetCart METs vs Freedson METs',
                       color_dic, color_by)
    ax4 = plot_subplot(ax4, 'MET (MetCart)', 'MetCart-GoogleFit', df, 
                       'MetCart METs \n\n(c)', 'MetCart METs - GoogleFit METs', 
                       'MetCart METs vs GoogleFit METs',
                       color_dic, color_by)
    
    fig.tight_layout(pad=4)
    plt.suptitle('Leave-One-Subject-Out Bland-Altman Plot')
    plt.show()
    #fig.savefig(f'{filename}.png')

#blandAltman_lab(df_inlab, 'inlab', intensity_c, 'Intensity')