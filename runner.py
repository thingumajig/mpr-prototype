# -*- coding: utf-8 -*-
# coding=utf-8
import streamlit as st

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import preprocess_data as ppd


# sys.stdout.reconfigure(encoding='utf-8')
# st.set_option()

# import tfhub_utils as tfh
# embedder = tfh.TFHubContext()

def show_row(row):
    st.markdown(f'''
| **{row['Название']}** | создано: {row['Дата создания']} | автор: {row['Автор инициативы']}
| ---: | --- | --- | 
| {row['Предприятие']} |{row['Категория']} |{row['Направление/Функция']} |{row['Производство/Подразделение']} |{row['Комплекс/Отдел']} |{row['Установка']} |
    ''')

query = st.sidebar.text_input('запрос:', value='')

# st.title(str("Очередной русский текст".encode('windows-1251'), 'utf-8'))

progress_bar = st.sidebar.progress(0)
# cb_vectors = st.sidebar.checkbox('Вектора', value=False)
# cb_diag_3d = st.sidebar.checkbox('t-SNE Точечная диаграмма 3D', value=False)
# cb_diag_2d = st.sidebar.checkbox('t-SNE Точечная диаграмма 2D', value=False)

# rez_count = st.sidebar.slider('кол-во результатов', 1, 1000, 10)
total = st.sidebar.empty()

st.sidebar.title("Фасеты")
# status_text = st.sidebar.empty()

facet_names = ['Категория', 'Предприятие', 'Направление/Функция', 'Автор инициативы', 'Производство/Подразделение','Комплекс/Отдел', 'Установка']

color_facet = st.sidebar.selectbox("Схема цветов:", facet_names)
enabled_facets = st.sidebar.multiselect('Выбор фасетов', facet_names)
work_columns = ['Название', 'Предприятие', 'Автор инициативы',
                'Категория',
                'Направление/Функция', 'Производство/Подразделение', 'Комплекс/Отдел', 'Установка',
                'Описание ситуации', 'Проблема', 'Решение', 'Ожидаемый эффект',
                # 'Количество лайков', 'Количество комментариев'
                ]


@st.cache
def load_data():
    # df = pd.read_hdf('ideas_emb.h5', key='ideas_emb')
    df = pd.read_hdf('emb.h5', key='emb')

    df = df[['Название', 'Дата создания',  'Предприятие', 'Автор инициативы',
         'Категория',
        'Направление/Функция', 'Производство/Подразделение', 'Комплекс/Отдел', 'Установка',
        'Описание ситуации', 'Проблема', 'Решение', 'Ожидаемый эффект',
        # 'Количество лайков', 'Количество комментариев'
        # 'all_x', 'all_y', 'all_z',
        'short_x','short_y','short_z',
             # 'short_emb'
        ]]

    return df

data = load_data()
progress_bar.progress(30)

# st.subheader('Сырые данные')

NONE = '-None-'
selected_list = []


@st.cache
def col_value_counts(data, colName):
  return data[colName].value_counts(sort=True)

def radio_for_col(data, colName):
    global NONE, enabled_facets,  selected_list

    valList = col_value_counts(data, colName)
    catsList = [f'{v}:\t[{c}]' for v, c in dict(valList).items()]
    cat = st.sidebar.radio(colName, catsList)
    cleaned_cat = re.sub(r":\s*\[\d+\]",'', str(cat))
    selected_list.append(cleaned_cat)
    return cleaned_cat

def checkboxs_for_col(data, colName):
    global NONE, enabled_facets, selected_list

    valList = col_value_counts(data, colName)

    selected_values=[]

    for c, v in dict(valList).items():
      if st.sidebar.checkbox(f'{c}:\t[{c}]'):
        selected_list.append(c)
        selected_values.append(c)

    return selected_values

def multiselect_for_col(data, colName):
  global NONE, enabled_facets, selected_list

  valList = col_value_counts(data, colName)
  catsList = [f'{v}:\t[{c}]' for v, c in dict(valList).items()]

  cats = st.sidebar.multiselect(colName, catsList)
  selected_values = []

  for c in cats:
    cleaned_cat = re.sub(r":\s*\[\d+\]", '', str(c))
    selected_values.append(cleaned_cat)
    selected_list.append(cleaned_cat)
  return selected_values


@st.cache
def data_filter(dt: pd.DataFrame, col, val):
    return dt[dt[col]==val]

@st.cache
def data_filter_isin(dt: pd.DataFrame, col, values):
    return dt[dt[col].isin(values)]

@st.cache
def find_subtext(df, txt):
    global work_columns
    df = df[work_columns]
    contains = df.stack().str.contains(txt).unstack()
    return contains[contains.any(1)].idxmax(1)

@st.cache
def query_filter(dt: pd.DataFrame, query):
    if query:
        t = dt.assign(found_cols=find_subtext(dt, query))
        return t[t['found_cols'].notnull()]
    return dt


from scipy.spatial.distance import cosine

emb_columns = ['Название_emb', 'Описание ситуации_emb', 'Проблема_emb', 'Решение_emb', 'Ожидаемый эффект_emb']

# @st.cache
# def query_filter_emb(dt:pd.DataFrame, query):
#   def calc_score(row, e):
#     return max([1. - cosine(row[col], e) for col in emb_columns])
#
#   if query:
#     e = embedder.get_embedding([query])[0]
#     dt['score'] = dt.apply(lambda row: calc_score(row, e), axis=1, result_type='expand').apply(pd.Series)
#     return dt[dt['score']>0.6]
#   return dt


def radio_list1(dt:pd.DataFrame, cols:list) -> pd.DataFrame:
    for col in cols:
        val = radio_for_col(dt, col)
        if val != NONE:
            dt = data_filter(dt, col, val)
    return dt

def multiselect_list(dt:pd.DataFrame, cols:list) -> pd.DataFrame:
    for col in cols:
        selected_list = multiselect_for_col(dt, col)
        dt = data_filter_isin(dt, col, selected_list)
    return dt


@st.cache
def prepare_vis(final_df):
  c_names = []
  dict = {'x': [i for i in range(0, 512)]}
  emb_results = []
  full_names = []

  for index, row in final_df.iterrows():
       # cname = f"[{index}] {str(row['Название'])[:20]}"
       cname = f"{index} {str(row['Название'])[:20]}"
       c_names.append(cname)
       full_names.append(str(row['Название']))
       emb = list(row['Название_emb'])
       dict[cname] = emb
       emb_results.append(emb)

  return {'c_names': c_names, 'dict': dict, 'emb_results': emb_results, 'full_names': full_names}

# print(dict.keys())
# print(c_names)
# print(len(emb_results))

import altair as alt
def visualize_emb(vis_dict):
  dict = vis_dict['dict']
  c_names = vis_dict['c_names']
  emb_vis_data = pd.DataFrame(dict)
  step = 20
  overlap = 1
  emb_chart = alt.Chart(emb_vis_data).transform_fold(
      c_names,
      as_=['embedding', 'lv']
    ).mark_area(
      interpolate='monotone',
      fillOpacity=0.8,
      stroke='lightgray',
      strokeWidth=0.2
    ).encode(
      # x='x',
      # y='lv:Q',
      # alt.Color('embedding:N'),
      alt.X('x:Q', title=None,
            scale=alt.Scale(domain=[0,512], range=[0,1500])),
      alt.Y(
          'lv:Q',
          title="",
          scale=alt.Scale(rangeStep=40),
          # scale=alt.Scale(range=[step, -step * overlap]),
          axis=None
      ),
      alt.Fill(
          'embedding:N',
          legend=None,
          scale=alt.Scale(scheme='redyellowblue')
      ),
      row=alt.Row(
           'embedding:N',
           title=None,
           header=alt.Header(labelAngle=360)
       )
   ).properties(
       bounds='flush', title='Вектор статьи', height=step, width=1200
  ).configure_facet(
      spacing=0
  ).configure_view(
      stroke=None
  ).configure_title(
      anchor='middle'
  )
  st.altair_chart(emb_chart, width=-1)




from sklearn.manifold import TSNE
import plotly.express as px


def vis_tsne3d_m(tsne_data: pd.DataFrame, color_facet):

  fig = px.scatter_3d(tsne_data,x='short_x', y='short_y',z='short_z',color=str(color_facet), hover_name="Название",
                      hover_data=[ 'Предприятие', 'Автор инициативы',
                                   'Категория', 'Направление/Функция',
                                   'Производство/Подразделение', 'Комплекс/Отдел', 'Установка' ])
  # fig.show()

  st.plotly_chart(fig, height=1000)

def vis_tsne_3d(vis_dict):
  tsne_data = tsne_3d(vis_dict)

  fig = px.scatter_3d(tsne_data,x='x', y='y',z='z',color='сокр. назв.', hover_data=["название"])
  # fig.show()

  st.plotly_chart(fig, height=1000)

@st.cache
def tsne_3d(vis_dict):
  c_names = vis_dict['c_names']
  emb_results = vis_dict['emb_results']
  full_names = vis_dict['full_names']
  X_embedded = TSNE(n_components=3, init='random',
                    random_state=0, perplexity=50).fit_transform(emb_results)
  tsne_data = pd.DataFrame(X_embedded, columns=['x', 'y', 'z'])
  # tsne_data.index.name = 'embedding'
  tsne_data['сокр. назв.'] = c_names
  tsne_data['название'] = full_names
  return tsne_data


def vis_tsne_2d(vis_dict):
  tsne_data = tsne_2d(vis_dict)

  fig = px.scatter(tsne_data,x='x', y='y',color='сокр. назв.', hover_data=["название"])
  # fig.show()

  st.plotly_chart(fig, height=1000)

@st.cache
def tsne_2d(vis_dict):
  c_names = vis_dict['c_names']
  emb_results = vis_dict['emb_results']
  full_names = vis_dict['full_names']
  X_embedded = TSNE(n_components=2, init='random',
                    random_state=0, perplexity=50).fit_transform(emb_results)
  tsne_data = pd.DataFrame(X_embedded, columns=['x', 'y'])
  # tsne_data.index.name = 'embedding'
  tsne_data['сокр. назв.'] = c_names
  tsne_data['название'] = full_names
  return tsne_data


###=====================================================


dt = query_filter(data, query)

if 'found_cols' in dt.columns:
    dt = dt.drop(columns=['found_cols'])
# dt = radio_list1(dt, list(enabled_facets))
dt = multiselect_list(dt, list(enabled_facets))
progress_bar.progress(60)
options = st.multiselect('Выбрано/Контекст:', selected_list, selected_list)


progress_bar.progress(80)


# st.title("Результаты")
# st.text(f'Rows:{len(dt.index)}')
st.markdown(f'''| #Результаты | всего:{len(dt.index)} |
| --- | --- |
''', unsafe_allow_html=True)

total.text(f"всего:{len(dt.index)}")

# final_df = dt.head(rez_count)
final_df = dt

# vis_dict = prepare_vis(final_df)
# if cb_diag_3d:
vis_tsne3d_m(final_df, color_facet)

# if cb_diag_2d:
#   vis_tsne_2d(vis_dict)
#
# if cb_vectors:
#   visualize_emb(vis_dict)


# st.table(final_df[work_columns])
st.table(final_df.head(10)[work_columns])

# for index, row in dt.iterrows():
#     show_row(row)







progress_bar.progress(100)






