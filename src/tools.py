from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom
from catboost import CatBoostClassifier
import sklearn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.experimental_memo
def get_data() -> tuple[Any, Any, Any]:
	# получаем датасет
	dataset = fetch_hillstrom(target_col='visit')
	dataset, target, treatment = dataset['data'], dataset['target'], dataset['treatment']
	# выбираем два сегмента
	dataset = dataset[treatment != 'Mens E-Mail']
	target = target[treatment != 'Mens E-Mail']
	treatment = treatment[treatment != 'Mens E-Mail'].map({
		'Womens E-Mail': 1,
		'No E-Mail':     0
	})

	return dataset, target, treatment


@st.experimental_memo
def data_split(data, treatment, target) -> tuple[Any, Any, Any, Any, Any, Any]:
	# склеиваем threatment и target для дальнейшей стратификации по ним
	stratify_cols = pd.concat([treatment, target], axis=1)
	# сплитим датасет
	X_train, X_val, trmnt_train, trmnt_val, y_train, y_val = train_test_split(
		data,
		treatment,
		target,
		stratify=stratify_cols,
		test_size=0.3,
		random_state=42
	)
	return X_train, X_val, trmnt_train, trmnt_val, y_train, y_val


def get_newbie_plot(data):
	fig = px.histogram(
		data['newbie'],
		color=data['newbie'],
		title='Распределение клиентов по флагу newbie'
	)

	fig.update_xaxes(
		title='',
		ticktext=['"Старые" клиенты', 'Новые клиенты'],
		tickvals=[0, 1]
	)

	fig.update_yaxes(
		title='Количество   клиентов'
	)

	fig.update_layout(
		showlegend=False,
		bargap=0.3,
		margin=dict(l=20, r=10, t=80, b=10)
	)

	fig.update_traces(hovertemplate="Количество клиентов: %{y}")

	return fig


def get_zipcode_plot(data):
	fig = px.histogram(
		data['zip_code'],
		color=data['newbie'],
		title='Распределение клиентов по почтовым индексам'
	)

	fig.update_xaxes(
		title='',
		categoryorder='total descending'
	)

	fig.update_yaxes(
		title='Количество   клиентов'
	)

	fig.update_layout(
		showlegend=True,
		legend_orientation="h",
		legend=dict(x=.66, y=.99, title='Новый клиент'),
		margin=dict(l=20, r=10, t=80, b=10),
		hovermode="x",
		bargap=0.3
	)

	fig.update_traces(hovertemplate="Количество клиентов: %{y}")

	return fig


def get_channel_plot(data):
	fig = px.histogram(
		data['channel'],
		color=data['newbie'],
		title='Распределение клиентов по каналам покупки товаров'
	)

	fig.update_xaxes(
		title='',
		categoryorder='total descending'
	)

	fig.update_yaxes(
		title='Количество   клиентов'
	)

	fig.update_layout(
		showlegend=True,
		legend_orientation="h",
		legend=dict(x=.66, y=.99, title='Новый клиент'),
		margin=dict(l=20, r=10, t=80, b=10),
		hovermode="x",
		bargap=0.3
	)

	fig.update_traces(hovertemplate="Количество клиентов: %{y}")

	return fig


def get_history_segment_plot(data):
	fig = px.histogram(
		data['history_segment'],
		color=data['history_segment'],
		title='Распределение клиентов по количеству $, потраченных в прошлом году'
	)

	fig.update_xaxes(
		title='',
		categoryorder='total descending',
		tickangle=45
	)

	fig.update_yaxes(
		title='Количество   клиентов'
	)

	fig.update_layout(
		showlegend=False,
		bargap=0.3,
		margin=dict(l=20, r=10, t=80, b=10)
	)

	fig.update_traces(hovertemplate="Количество клиентов: %{y}")

	return fig


def get_recency_plot(data):
	fig = px.histogram(
		data['recency'],
		color=data['newbie'],
		title='Распределение клиентов по количеству месяцев с последней покупки'
	)

	fig.update_xaxes(
		title='Месяцев  после  покупки'
	)

	fig.update_yaxes(
		title='Количество  клиентов'
	)

	fig.update_layout(
		showlegend=True,
		legend_orientation="h",
		legend=dict(x=.66, y=.99, title='Новый клиент'),
		margin=dict(l=20, r=10, t=80, b=10),
		hovermode="x",
		bargap=0.3
	)

	fig.update_traces(hovertemplate="<br>".join(
			[
				"Месяцев: %{x}",
				"Клиентов: %{y}"
			]
		)
	)

	return fig


def get_history_plot(data):
	fig = px.histogram(
		data['history'],
		color=data['newbie'],
		title='Распределение клиентов по количеству месяцев с последней покупки'
	)

	fig.update_xaxes(
		title='Месяцев  после  покупки'
	)

	fig.update_yaxes(
		title='Количество  клиентов'
	)

	fig.update_layout(
		showlegend=True,
		legend_orientation="h",
		legend=dict(x=.66, y=.99, title='Новый клиент'),
		margin=dict(l=20, r=10, t=80, b=10),
		hovermode="x",
		bargap=0.3
	)

	fig.update_traces(hovertemplate="<br>".join(
			[
				'Совершено покупок на: $%{x}',
				'Количество клиентов: %{y}'
			]
		)
	)

	return fig
