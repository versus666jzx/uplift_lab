import streamlit as st

import tools
from time import sleep

norm_columns = ['age', 'children', 'gender', 'main_format', 'months_from_register', 'response_sms', 'response_viber']
dataset = tools.get_data()

st.title('Uplift lab')

st.write('Какие данные выбрать для рассылки?')
st.write(dataset.data[norm_columns].head())
columns = st.multiselect(options=norm_columns, label='Выберите признак')
age = st.select_slider(label='', options=range(1, 101), value=[18, 100])
st.write(dataset.data[dataset.data['age'].isin(age)][norm_columns])
