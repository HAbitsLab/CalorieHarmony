import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as st
from collections import Counter 
import datetime as dt
pd.options.mode.chained_assignment = None


count_unique = lambda x: len(list(np.unique(x)))
count_unique.__name__ = 'count_unique'
counter = lambda x: Counter(list(x))
counter.__name__ = 'counter'

def set_intensity(row, col_name):
    if row[col_name]<1.5:
        return 'Sedentary'
    elif 1.5<=row[col_name]<3:
        return 'Light'
    elif row[col_name]>=3:
        return 'Moderate / Vigorous'


def get_outliers(inwild):
  inwild = inwild[['Participant', 'Datetime', 'MET (VM3)', 'estimation']]
  inwild['Intensity'] = ''
  inwild['Intensity'] = inwild.apply(set_intensity, col_name='MET (VM3)', axis=1)

  inwild['avgVM3Est'] = inwild[['MET (VM3)', 'estimation']].mean(axis=1)

  inwild['VM3-Est'] = inwild.apply(lambda x: x['MET (VM3)'] - x['estimation'], axis=1)

  l = inwild['VM3-Est']
  m_plus_196s = np.mean(l) + 1.96*np.std(l)
  m_minus_196s = np.mean(l) - 1.96*np.std(l)

  inwild['Classification'] = ''
  inwild.loc[inwild['VM3-Est']>m_plus_196s, 'Classification'] = '>m + 1.96s'
  inwild.loc[inwild['VM3-Est']<m_minus_196s, 'Classification'] = '<m - 1.96s'

  outliers = inwild.loc[(inwild['VM3-Est']<m_minus_196s) | (inwild['VM3-Est']>m_plus_196s)]

  return outliers


def preprocess_outliers(outliers):
  outliers['Datetime'] = outliers['Datetime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
  outliers['Date'] = outliers['Datetime'].apply(lambda x: x.date())
  outliers['Hour'] = outliers['Datetime'].apply(lambda x: x.hour)
  outliers['Minute'] = outliers['Datetime'].apply(lambda x: x.minute)

  outliers.to_csv('Data/IW_outlier_details.csv', index=False)
  outliers_grouped = outliers.groupby(['Participant', 'Date', 'Hour']).agg({'Minute':lambda x: list(x),
                                                                            'Minute':lambda x: len(list(x))})

  outliers_grouped.reset_index().to_csv('Data/IW_outlier_grouped.csv', index=False)


def check_keyword_count(df):
    l = []
    for keyword in keywords:
        dic ={}
        keyword = keyword.lower()
        keyword_df = df[df['Labels'].str.contains(keyword)]
        
        dic['Keyword'] = keyword
        dic['No. of instances'] = keyword_df.shape[0]
        dic['No. of participants'] = len(keyword_df['Participant'].unique())
        
        l.append(dic)
        
    return l 


def approach1(df, title):
  df = pd.DataFrame(check_keyword_count(df))
  df = df.sort_values('No. of instances', ascending=False).reset_index(drop=True)
  df.to_csv(f'{title} split by keyword.csv', index=False)


def approach2(df, title):
  df = df.groupby('Labels').agg({'Participant': [count_unique, counter], 
                               'Classification': 'count'}).reset_index()
  df.columns = [' '.join(col).strip() for col in df.columns.values]

  df = df[['Labels', 'Classification count', 'Participant count_unique', 'Participant counter']]
  df.rename(columns={'Classification count': 'No. of instances', 
                     'Participant count_unique': 'No. of unique particpants',
                     'Participant counter': 'Details',}, inplace=True)
  df = df.sort_values('No. of instances', ascending=False).reset_index(drop=True)
  df.to_csv(f'Data/{title} predicted.csv', index=False)



keywords = ['Fidgeting', 'Sitting', 'Computer', 'Public_transit', 'Driving', 'Phone',
            'Walking', 'Transition', 'Bathroom', 'Shopping', 'Drinking', 'Eating',
            'Washing_dishes', 'Cleaning', 'Stairs', 'TV', 'Smoking', 'Camera',
            'Sweeping', 'Carrying', 'Writing', 'Talking', 'Others', 'Cooking']

if __name__ == '__main__':

  inwild = pd.read_csv('Data/df_wild.csv')
  outliers = get_outliers(inwild)
  outliers = preprocess_outliers(outliers)
  labelled_outliers = pd.read_csv('Data/InWild outliers labeled.csv')

  labelled_outliers = labelled_outliers.loc[labelled_outliers['Participant']!='P418']
  labelled_outliers = labelled_outliers.loc[~(labelled_outliers['Remove']=='1')]
  labelled_outliers.reset_index(drop=True, inplace=True)

  under = labelled_outliers.loc[labelled_outliers['Classification']=='>m + 1.96s']
  over = labelled_outliers.loc[labelled_outliers['Classification']=='<m - 1.96s']

  # approach1(under, 'Under')
  # approach1(over, 'Over')

  approach2(under, 'Under')
  approach2(over, 'Over')

