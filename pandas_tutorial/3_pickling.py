'''
Created on Dec 1, 2018

@author: liuyanan
'''
import quandl
import pandas as pd
import pickle
import numpy as np

api_key = open('quandlapikey.txt', 'r').read().strip('\n')


def state_list():
    fiddy_states = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")
    return fiddy_states[0][1][1:]


def grab_initial_state_data():

    #     df = quandl.get('FMAC/HPI_AK', authtoken=api_key)
    #     print(df.head())

    main_df = pd.DataFrame()

    for abbv in state_list():
        query = 'FMAC/HPI_' + str(abbv)
        df = quandl.get(query, authtoken=api_key)

        # 这里要让每个column的值唯一
        df.rename(columns={'NSA Value': str(abbv) + '_NSA_Value', 'SA Value': str(abbv) + '_SA_Value'}, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    print(main_df)

    pickle_out = open('fiddy_states.pickles', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()


def from_pickle():
    pickle_in = open('fiddy_states.pickles', 'rb')
    HPI_data = pickle.load(pickle_in)
    print(HPI_data)

    HPI_data.to_pickle('pickle.pickle')
    HPI_data2 = pd.read_pickle('pickle.pickle')
    print(HPI_data2)
    # print(HPI_data2 == HPI_data)


if __name__ == '__main__':
    grab_initial_state_data()
    from_pickle()
