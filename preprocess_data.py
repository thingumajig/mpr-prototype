# -*- coding: utf-8 -*-
import streamlit as st

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

Input_file = 'data.14-09-2019.xlsx'
def load_ideas_data(nrows)-> pd.DataFrame :
    df = pd.read_excel(Input_file)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pandas.to_datetime(data[DATE_COLUMN])
    
    df.fillna(0, inplace=True)

    df.rename(columns={'Комплекс / Отдел':'Комплекс/Отдел',
        'Направление / Функция': 'Направление/Функция', 
        'Производство / Подразделение': 'Производство/Подразделение',
    }, inplace=True)
    print( df.dtypes)
    # df['Категория'] = df['Категория'].astype('category')
    #df['Направление / Функция'] = df['Направление / Функция'].astype('category')
    df = df[['Название', 'Дата создания',  'Предприятие', 'Автор инициативы', 
         'Категория', 
        'Направление/Функция', 'Производство/Подразделение', 'Комплекс/Отдел', 'Установка',
        'Описание ситуации', 'Проблема', 'Решение', 'Ожидаемый эффект',
        # 'Количество лайков', 'Количество комментариев'
        ]]

    # df.set_index('ID')    
    df['Дата создания'] = pd.to_datetime(df['Дата создания'], infer_datetime_format=True)
    
    return df

def clean_column(df: pd.DataFrame, colname):
    return df[colname].str.replace(r'\[\d+\]\s*', '',regex=True)

def extract_id_and_value(df: pd.DataFrame, colname):
    return df[colname].str.extract(r'\[(\d+)\]\s*(.+)')

def clean_columns(df: pd.DataFrame):
    def cl(colname):
        df[colname]=clean_column(df, colname)
    cl('Предприятие')
    cl('Направление/Функция')
    cl('Автор инициативы')
    cl('Производство/Подразделение')
    cl('Комплекс/Отдел')
    cl('Установка')


def load_emb():
    df = pd.read_hdf('emb.h5', key='emb')
    return df

def load_ideas():
    df = pd.read_hdf('ideas.h5', key='ideas')
    return df

def get_integrated_data():
    ideas_df = load_ideas()
    emb_df = load_emb()
    cdf = pd.concat([ideas_df, emb_df], axis=1)
    cdf.rename(columns={
        0:'Название_emb',
        1:'Описание ситуации_emb',
        2:'Проблема_emb',
        3:'Решение_emb',
        4:'Ожидаемый эффект_emb',
                      }, inplace=True)
    return cdf


if __name__ == "__main__":
    # ideas_df:pd.DataFrame  = load_ideas_data(1000)
    # clean_columns(ideas_df)
    # ideas_df.to_hdf('ideas.h5', key='ideas')

    # users_df:pd.DataFrame = pd.DataFrame(columns=['ID', ])
    idf = get_integrated_data()
    idf.to_hdf('ideas_emb.h5', key='ideas_emb')

    print(idf.columns)
    print(idf)