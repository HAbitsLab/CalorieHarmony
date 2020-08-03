import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as st
import datetime as dt


intensity_c = {'Sedentary': '#a8d0e3', # lightblue
               'Light': '#f6b79b', # peach
               'Moderate / Vigorous': '#DE6454' # red
              }


def set_intensity(row, col_name):
    if row[col_name]<1.5:
        return 'Sedentary'
    elif 1.5<=row[col_name]<3:
        return 'Light'
    elif row[col_name]>=3:
        return 'Moderate / Vigorous'


def inlab_preprocess(inlab):
	inlab = inlab[['Participant', 'Activity', 'MET (Ainsworth)', 'MET (Google Fit)', 'MET (VM3)', 'estimation']]
	inlab['Intensity'] = ''
	inlab['Intensity'] = inlab.apply(set_intensity, col_name='MET (Ainsworth)', axis=1)

	inlab['avgAinsworthEst'] = inlab[['MET (Ainsworth)', 'estimation']].mean(axis=1)
	inlab['avgAinsworthVM3'] = inlab[['MET (Ainsworth)', 'MET (VM3)']].mean(axis=1)
	inlab['avgVM3Est'] = inlab[['MET (VM3)', 'estimation']].mean(axis=1)
	inlab['avgAinsworthGF'] = inlab[['MET (Ainsworth)', 'MET (Google Fit)']].mean(axis=1)

	inlab['Ainsworth-Est'] = inlab.apply(lambda x: x['MET (Ainsworth)'] - x['estimation'], axis=1)
	inlab['Ainsworth-VM3'] = inlab.apply(lambda x: x['MET (Ainsworth)'] - x['MET (VM3)'], axis=1)
	inlab['VM3-Est'] = inlab.apply(lambda x: x['MET (VM3)'] - x['estimation'], axis=1)
	inlab['Ainsworth-GF'] = inlab.apply(lambda x: x['MET (Ainsworth)'] - x['MET (Google Fit)'], axis=1)

	return inlab


def inwild_preprocess(inwild):
	inwild = inwild[['Participant', 'Datetime', 'MET (VM3)', 'estimation']]
	inwild['Intensity'] = ''
	inwild['Intensity'] = inwild.apply(set_intensity, col_name='MET (VM3)', axis=1)

	inwild['avgVM3Est'] = inwild[['MET (VM3)', 'estimation']].mean(axis=1)

	inwild['VM3-Est'] = inwild.apply(lambda x: x['MET (VM3)'] - x['estimation'], axis=1)

	return inwild


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
	# ax.axhline(0, color='gray', linestyle=':', zorder=20) # 0 line

    # x and y lim
    xmin, xmax = ax.get_xlim()
    xrange = xmax-xmin
    ax.set_xlim([xmin, xmax+xrange*0.21])
    ymin, ymax = ax.get_ylim()
    yrange = ymax-ymin
    
    textpos = yrange*0.02
    numpos = yrange*0.06
    rightmargin = xrange*0.19

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
    return


def blandAltman_lab(df, filename, color_dic, color_by):
    fig = plt.figure(figsize = (10.5,10.5))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 4, sharey=ax1)
    ax4 = fig.add_subplot(2, 2, 3)

    ax1 = plot_subplot(ax1, 'MET (Ainsworth)', 'Ainsworth-Est', df,
                       'Compendium METs \n\n(a)', 'Compendium METs - WRIST',
                       'Compendium METs vs WRIST',
                       color_dic, color_by)

    ax2 = plot_subplot(ax2, 'MET (Ainsworth)', 'Ainsworth-VM3', df,
                       'Compendium METs \n\n(b)', 'Compendium METs - ActiGraph VM3 METs', 
                       'Compendium METs vs ActiGraph VM3 METs',
                       color_dic, color_by)

    ax3 = plot_subplot(ax3, 'MET (Ainsworth)', 'VM3-Est', df, 
                       'Compendium METs \n\n(d)', 'ActiGraph VM3 METs - WRIST', 
                       'ActiGraph VM3 METs vs WRIST',
                       color_dic, color_by)

    ax4 = plot_subplot(ax4, 'MET (Ainsworth)', 'Ainsworth-GF', df,
                       'Compendium METs\n\n(c)', 'Compendium METs - Google Fit METs', 
                       'Compendium METs vs Google Fit METs',
                       color_dic, color_by)
    
    fig.tight_layout(pad=4)

    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(intensity_c.values(), intensity_c.keys())]
    plt.legend(handles=patches, labels=[label for _, label in zip(intensity_c.values(), intensity_c.keys())], loc='lower center', bbox_to_anchor=(1, -0.38), ncol = 3)
    fig.savefig(f'{filename}.png')


def blandAltman_wild(df, filename, color_dic, color_by):
	fig = plt.figure(figsize = (6,5))

	ax1 = fig.add_subplot(1, 1, 1)
	plot_subplot(ax1, 'MET (VM3)', 'VM3-Est', inwild, 
	                       'ActiGraph VM3 METs', 'ActiGraph VM3 METs - WRIST',
	                       '', intensity_c, 'Intensity')
	
	# legend
	patches = [mpatches.Patch(color=color, label=label) for color, label in zip(intensity_c.values(), intensity_c.keys())]
	plt.legend(handles=patches, labels=[label for _, label in zip(intensity_c.values(), intensity_c.keys())], loc='lower center', bbox_to_anchor=(0.49, -0.4), ncol = 3)

	fig.savefig('inwild.png')



if __name__ == '__main__':


	inlab = pd.read_csv('Data/df_lab.csv')
	inlab = inlab_preprocess(inlab)

	blandAltman_lab(inlab, 'inlab', intensity_c, 'Intensity')

	inwild = pd.read_csv('Data/df_wild.csv')
	inwild = inwild_preprocess(inwild)

	blandAltman_wild(inwild, 'inwild', intensity_c, 'Intensity')

