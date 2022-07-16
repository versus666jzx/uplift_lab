import pandas as pd

import streamlit as st

import tools


dataset, target, treatment = tools.get_data()

st.title('Uplift lab')

st.markdown(
	"""
	#### Рассмотрим пример применения одного из подходов прогнозирования _uplift_.
	
	Данные для примера взяты из [_The MineThatData E-Mail Analytics And Data Mining Challenge_](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
	
	Этот набор данных содержит 42 693 строк с данными клиентов, которые в последний раз совершали покупки в течение двенадцати месяцев.
	
	Среди клиентов была проведена рекламная кампания с помощью _email_ рассылки:
	- 1/2 клиентов были выбраны случайным образом для получения электронного письма, рекламирующего женскую продукцию;
	- С оставшейся 1/2 коммуникацию не проводили.
	
	Для каждого клиента из выборки замерили факт перехода по ссылке в письме, факт совершения покупки и сумму трат за
	две недели, следующими после получения письма.
	
	Пример данных приведен ниже.
	"""
)
refresh = st.button('Обновить выборку')
title_subsample = dataset.sample(7)
if refresh:
	title_subsample = dataset.sample(7)
st.dataframe(title_subsample, width=700)
st.write( f"Всего записей: {dataset.shape[0]}")

st.write('Описание данных')
st.markdown(
	"""
		| Колонка           | Обозначение                                                            |
		|-------------------|------------------------------------------------------------------------|
		| _recency_         | Месяцев с момента последней покупки                                    |
		| _history_segment_ | Классификация клиентов в долларах, потраченных в прошлом году          |
		| _history_         | Фактическая стоимость в долларах, потраченная в прошлом году           |
		| _mens_            | Флаг 1/0, 1 = клиент приобрел мужские товары в прошлом году            |
		| _womens_          | Флаг 1/0, 1 = клиент приобрел женские товары в прошлом году            |
		| _zip_code_        | Классифицирует почтовый индекс как городской, пригородный или сельский |
		| _newbie_          | Флаг 1/0, 1 = Новый клиент за последние двенадцать месяцев             |
		| _channel_         | Описывает каналы, через которые клиент приобрел тоовар в прошлом году  |

		---
	"""
)
st.write("Для того, чтобы лучше понять на какую аудиторию лучше запустить рекламную кампанию, проведем небольшой \
		  анализ данных")

with st.expander('Развернуть блок анализа данных'):

	st.plotly_chart(tools.get_newbie_plot(dataset), use_container_width=True)
	st.write(f'В данных примерно одинаковое количество новых и "старых клиентов". '
			 f'Отношение новых клиентов к старым: {(dataset["newbie"] == 1).sum() / (dataset["newbie"] == 0).sum():.2f}')

	st.plotly_chart(tools.get_zipcode_plot(dataset), use_container_width=True)
	tmp_res = dataset.zip_code.value_counts(normalize=True) * 100
	st.write(f'Большинство клиентов из пригорода: {tmp_res["Surburban"]:.2f}%, из города: {tmp_res["Urban"]:.2f}% и из села: {tmp_res["Rural"]:.2f}%')

	tmp_res = dataset.channel.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_channel_plot(dataset), use_container_width=True)
	st.write(f'В прошлом году почти одинаковое количество клиентов покупало товары через телефон и сайт, {tmp_res["Phone"]:.2f}% и {tmp_res["Web"]:.2f}% соответственно,'
	         f' а {tmp_res["Multichannel"]:.2f}% клиентов покупали товары воспользовавшись двумя платформами.')

	tmp_res = dataset.history_segment.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_history_segment_plot(dataset), use_container_width=True)
	st.write(f'Как мы видим, большинство пользователей относится к сегменту \$0-\$100 ({tmp_res[0]:.2f}%), второй и '
	         f'третий по количеству пользователей сегменты \$100-\$200 ({tmp_res[1]:.2f}%) и \$200-\$350 ({tmp_res[2]:.2f}%).')
	st.write(f'К сегментам \$350-\$500 и \$500-\$750 относится {tmp_res[3]:.2f}% и {tmp_res[4]:.2f}% пользователей соответственно.')
	st.write(f'Меньше всего пользователей в сегментах \$750-\$1.000 ({tmp_res[-2]:.2f}%) и \$1.000+ ({tmp_res[-1]:.2f}%).')

	tmp_res = list(dataset.recency.value_counts(normalize=True) * 100)
	st.plotly_chart(tools.get_recency_plot(dataset), use_container_width=True)
	st.write(f'Большинство клиентов являются активными клиентами платформы, и совершали покупки в течение месяца ({tmp_res[0]:.2f}%)')
	st.write('Также заметно, что 9 и 10 месяцев назад, много клиентов совершали покупки. Это может свидетельствовать о проведении'
	         'рекламной кампании в это время или чего-то еще.')
	st.write('Также интересно понаблюдать за долями новых клиентов в данном распределении.')

	st.plotly_chart(tools.get_history_plot(dataset), use_container_width=True)
	st.markdown('_График интерактивный_')
	st.write('Абсолютное большинство клиентов тратят \$25-\$35 на покупки, но есть и малая доля тех, кто тратит более \$3.000')
	st.write('Интересный факт: все покупки более \$500 совершают только новые клиенты')

st.subheader('Какие данные возьмем для проведения рекламной кампании?')
clients_filter = st.radio('На каких клиентов будем воздействовать?', options=['Все', 'Только на новых', 'Только на старых'])


st.write('Выберите класс клиентов, по объему денег, потраченных в прошлом году (history segments).')
first_group = st.checkbox('$0-$100', value=True)
second_group = st.checkbox('$100-$200', value=True)
third_group = st.checkbox('$200-$350', value=True)
fourth_group = st.checkbox('$350-$500', value=True)
fifth_group = st.checkbox('$500-$750', value=True)
sixth_group = st.checkbox('$750-$1.000', value=True)
seventh_group = st.checkbox('$1.000+', value=True)

st.write('Каких пользователей по почтовому коду выберем')
surburban = st.checkbox('Surburban', value=True)
urban = st.checkbox('Urban', value=True)
rural = st.checkbox('Rural', value=True)

recency = st.slider(label='Месяцев с момента покупки', min_value=int(dataset.recency.min()), max_value=int(dataset.recency.max()), value=(int(dataset.recency.min()), int(dataset.recency.max())))

if not first_group and not second_group and not third_group and not fourth_group and not fifth_group and not sixth_group and not seventh_group:
	pass
else:
	st.button('Отфильтровать')
