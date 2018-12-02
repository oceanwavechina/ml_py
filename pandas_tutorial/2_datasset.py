'''
Created on Dec 1, 2018

@author: liuyanan
'''

import quandl
import pandas as pd


def quandl_data():
    api_key = open('quandlapikey.txt', 'r').read().strip('\n')
    df = quandl.get('FMAC/HPI_AK', authtoken=api_key)
    print(df.head())

    fiddy_states = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")
    print(fiddy_states)
    for state in fiddy_states:
        print(state, '\n')
    print(fiddy_states[0][1])

    for abbv in fiddy_states[0][1][1:]:
        print('FMAC/HPI_' + abbv)


def concatenating_data():
    df1 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thounsands': [50, 55, 65, 55]},
                       index=[2001, 2001, 2003, 2004])

    df2 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thounsands': [50, 55, 65, 55]},
                       index=[2005, 2006, 2007, 2008])

    df3 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'Low_tier_HPI': [50, 52, 50, 53]},
                       index=[2001, 2002, 2003, 2004])

    concat = pd.concat([df1, df2])
    concat = pd.concat([df2, df3])
    concat = pd.concat([df1, df2, df3])
    print(concat)

    print(df1.append(df2))

    s = pd.Series([80, 2, 50], index=['HPI', 'Int_rate', 'US_GDP_Thounsands'])
    df4 = df1.merge(s, ignore_index=True)
    print(df1)
    print(df4)


def joining_and_merging():
    df1 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thounsands': [50, 55, 65, 55]},
                       index=[2001, 2001, 2003, 2004])

    df2 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thounsands': [50, 55, 65, 55]},
                       index=[2005, 2006, 2007, 2008])

    df3 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Unemployemnt': [7, 8, 9, 6],
                        'Low_tier_HPI': [50, 52, 50, 53]},
                       index=[2001, 2002, 2003, 2004])

    # on 是共享哪一列
#     print(pd.merge(df1, df2, on='HPI'))
#     print(pd.merge(df1, df2, on=['HPI', 'Int_rate']))

    df1.set_index('HPI', inplace=True)
    df3.set_index('HPI', inplace=True)

    joined = df1.join(df3)
    print(joined)

    df1 = pd.DataFrame({'Year': [2001, 2002, 2003, 2004],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thounsands': [50, 55, 65, 55]},
                       )

    df3 = pd.DataFrame({'Year': [2001, 2003, 2004, 2005],
                        'Unemployemnt': [7, 8, 9, 6],
                        'Low_tier_HPI': [50, 52, 50, 53]},
                       )
    merged = pd.merge(df1, df3, on='Year', how='outer')
    merged.set_index('Year', inplace=True)
    print(merged)


if __name__ == '__main__':
    # quandl_data()
    # concatenating_data()
    joining_and_merging()
