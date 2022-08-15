import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom
from sklift.metrics import weighted_average_uplift
import streamlit as st
import plotly.express as px


def test():
	return 'Test'


@st.experimental_memo
def get_data():
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
def data_split(data: pd.DataFrame, treatment: pd.DataFrame, target: pd.DataFrame):
	# склеиваем threatment и target для дальнейшей стратификации по ним
	stratify_cols = pd.concat([treatment, target], axis=1)
	# сплитим датасет
	X_train, X_val, trmnt_train, trmnt_val, y_train, y_val = train_test_split(
		data,
		treatment,
		target,
		stratify=stratify_cols,
		test_size=0.5,
		random_state=42
	)
	return X_train, X_val, trmnt_train, trmnt_val, y_train, y_val


def filter_by_newbie(data: pd.DataFrame, newbie_filter: str) -> pd.DataFrame:
	if newbie_filter == 'Все':
		return data
	elif newbie_filter == 'Только новые':
		return data[data['newbie'] == 1]
	elif newbie_filter == 'Только старые':
		return data[data['newbie'] == 0]


def filter_by_channel(data: pd.DataFrame, channel_filter: str) -> pd.DataFrame:
	if channel_filter == 'Все':
		return data
	if channel_filter == 'Phone':
		return data[data['channel'] == channel_filter]
	if channel_filter == 'Web':
		return data[data['channel'] == channel_filter]
	if channel_filter == 'Multichannel':
		return data[data['channel'] == channel_filter]


def filter_by_mens(data: pd.DataFrame, mens_filter: str) -> pd.DataFrame:
	if mens_filter == 'Любые':
		return data
	if mens_filter == 'Мужские':
		return data[data['mens'] == 1]
	if mens_filter == 'Женские':
		return data[data['womens'] == 1]


def filter_by_history_segments(data: pd.DataFrame, history_segments_filter: dict) -> pd.DataFrame:
	filtered_indexes = set()
	if history_segments_filter.get('1) $0 - $100'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '1) $0 - $100'].index)
	if history_segments_filter.get('2) $100 - $200'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '2) $100 - $200'].index)
	if history_segments_filter.get('3) $200 - $350'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '3) $200 - $350'].index)
	if history_segments_filter.get('4) $350 - $500'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '4) $350 - $500'].index)
	if history_segments_filter.get('5) $500 - $750'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '5) $500 - $750'].index)
	if history_segments_filter.get('6) $750 - $1,000'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '6) $750 - $1,000'].index)
	if history_segments_filter.get('7) $1,000 +'):
		filtered_indexes = filtered_indexes.union(data[data['history_segment'] == '7) $1,000 +'].index)

	return data.loc[list(filtered_indexes)]


def filter_by_zip_code(data: pd.DataFrame, zip_code_filter: dict) -> pd.DataFrame:
	filterd_indexes = set()
	if zip_code_filter.get('surburban'):
		filterd_indexes = filterd_indexes.union(data[data['zip_code'] == 'Surburban'].index)
	if zip_code_filter.get('urban'):
		filterd_indexes = filterd_indexes.union(data[data['zip_code'] == 'Urban'].index)
	if zip_code_filter.get('rural'):
		filterd_indexes = filterd_indexes.union(data[data['zip_code'] == 'Rural'].index)

	return data.loc[list(filterd_indexes)]


def filter_by_recency(data: pd.DataFrame, recency_filter: list) -> pd.DataFrame:
	return data[(data['recency'] >= recency_filter[0]) & (data['recency'] <= recency_filter[1])]


def filter_data(data: pd.DataFrame, filters: dict) -> pd.DataFrame or None:
	"""
	Filter data by user filters

	:param data: filtered data
	:param filters: dict of filters
	:return: filtered data
	"""
	data = filter_by_newbie(data, filters['newbie_filter'])
	if data.shape[0] == 0:
		return None
	data = filter_by_channel(data, filters['channel_filter'])
	if data.shape[0] == 0:
		return None
	data = filter_by_mens(data, filters['mens_filter'])
	if data.shape[0] == 0:
		return None
	data = filter_by_history_segments(data, filters['history_segments'])
	if data.shape[0] == 0:
		return None
	data = filter_by_zip_code(data, filters['zip_code'])
	if data.shape[0] == 0:
		return None
	data = filter_by_recency(data, filters['recency'])
	if data.shape[0] == 0:
		return None
	return data


def uplift_by_percentile():
	pass


def get_weighted_average_uplift(target_test: pd.DataFrame, uplift, treatment_test: pd.DataFrame):
	return weighted_average_uplift(target_test, uplift, treatment_test)


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
