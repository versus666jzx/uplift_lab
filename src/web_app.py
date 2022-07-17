import pandas as pd

import streamlit as st

import tools

dataset, target, treatment = tools.get_data()

data_train, data_test, treatment_train, treatment_test, target_train, target_test = tools.data_split(dataset, treatment, target)

if 'filter_data' not in st.session_state.keys():
	st.session_state.filter_data = True


st.title('Uplift lab')

st.markdown(
	"""
	#### Рассмотрим пример применения одного из подходов прогнозирования _uplift_.
	
	Данные для примера взяты из [_The MineThatData E-Mail Analytics And Data Mining Challenge_](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
	
	Этот набор данных содержит 42 693 строк с данными клиентов, которые в последний раз совершали покупки в течение двенадцати месяцев.
	
	Из данных уже отделена тестовая выборка в виде 30% записей клиентов, так что данных в предоставленной выборке будет меньше.
	
	Среди клиентов была проведена рекламная кампания с помощью _email_ рассылки:
	- 1/2 клиентов были выбраны случайным образом для получения электронного письма, рекламирующего женскую продукцию;
	- С оставшейся 1/2 коммуникацию не проводили.
	
	Для каждого клиента из выборки замерили факт перехода по ссылке в письме, факт совершения покупки и сумму трат за
	две недели, следующими после получения письма.
	
	Пример данных приведен ниже.
	"""
)
refresh = st.button('Обновить выборку')
title_subsample = data_test.sample(7)
if refresh:
	title_subsample = data_test.sample(7)
st.dataframe(title_subsample, width=700)
st.write(f"Всего записей: {data_test.shape[0]}")

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

	st.plotly_chart(tools.get_newbie_plot(data_test), use_container_width=True)
	st.write(f'В данных примерно одинаковое количество новых и "старых клиентов". '
			 f'Отношение новых клиентов к старым: {(data_test["newbie"] == 1).sum() / (data_test["newbie"] == 0).sum():.2f}')

	st.plotly_chart(tools.get_zipcode_plot(data_test), use_container_width=True)
	tmp_res = data_test.zip_code.value_counts(normalize=True) * 100
	st.write(f'Большинство клиентов из пригорода: {tmp_res["Surburban"]:.2f}%, из города: {tmp_res["Urban"]:.2f}% и из села: {tmp_res["Rural"]:.2f}%')

	tmp_res = data_test.channel.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_channel_plot(data_test), use_container_width=True)
	st.write(f'В прошлом году почти одинаковое количество клиентов покупало товары через телефон и сайт, {tmp_res["Phone"]:.2f}% и {tmp_res["Web"]:.2f}% соответственно,'
	         f' а {tmp_res["Multichannel"]:.2f}% клиентов покупали товары воспользовавшись двумя платформами.')

	tmp_res = data_test.history_segment.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_history_segment_plot(data_test), use_container_width=True)
	st.write(f'Как мы видим, большинство пользователей относится к сегменту \$0-\$100 ({tmp_res[0]:.2f}%), второй и '
	         f'третий по количеству пользователей сегменты \$100-\$200 ({tmp_res[1]:.2f}%) и \$200-\$350 ({tmp_res[2]:.2f}%).')
	st.write(f'К сегментам \$350-\$500 и \$500-\$750 относится {tmp_res[3]:.2f}% и {tmp_res[4]:.2f}% пользователей соответственно.')
	st.write(f'Меньше всего пользователей в сегментах \$750-\$1.000 ({tmp_res[-2]:.2f}%) и \$1.000+ ({tmp_res[-1]:.2f}%).')

	tmp_res = list(data_test.recency.value_counts(normalize=True) * 100)
	st.plotly_chart(tools.get_recency_plot(data_test), use_container_width=True)
	st.write(f'Большинство клиентов являются активными клиентами платформы, и совершали покупки в течение месяца ({tmp_res[0]:.2f}%)')
	st.write('Также заметно, что 9 и 10 месяцев назад, много клиентов совершали покупки. Это может свидетельствовать о проведении'
	         'рекламной кампании в это время или чего-то еще.')
	st.write('Также интересно понаблюдать за долями новых клиентов в данном распределении.')

	st.plotly_chart(tools.get_history_plot(data_test), use_container_width=True)
	st.markdown('_График интерактивный. Двойной клик вернет в начальное состояние._')
	st.write('Абсолютное большинство клиентов тратят \$25-\$35 на покупки, но есть и малая доля тех, кто тратит более \$3.000')
	st.write('Интересный факт: все покупки более \$500 совершают только новые клиенты')

filters = {}

st.subheader('Выберем клиентов, которым отправим рекламу.')
newbie_filter = st.radio('Каким клиентам отправим рекламу?', options=['Всем', 'Только новым', 'Только старым'])
filters['newbie_filter'] = newbie_filter

channel_filter = st.radio('Канал, по которому клиент покупал в прошлом году', options=['Всем', 'Phone', 'Web', 'Multichannel'])
filters['channel_filter'] = channel_filter

mens_filter = st.radio('Клиенты, приобретавшие', options=['Любые товары', 'Мужские', 'Женские'])
filters['mens_filter'] = mens_filter

st.write('Выберите класс клиентов, по объему денег, потраченных в прошлом году (history segments)')
filters['history_segments'] = {}
first_group = st.checkbox('$0-$100', value=True)
if first_group:
	filters['history_segments']['1) $0 - $100'] = True
second_group = st.checkbox('$100-$200', value=True)
if second_group:
	filters['history_segments']['2) $100 - $200'] = True
third_group = st.checkbox('$200-$350', value=True)
if third_group:
	filters['history_segments']['3) $200 - $350'] = True
fourth_group = st.checkbox('$350-$500', value=True)
if fourth_group:
	filters['history_segments']['4) $350 - $500'] = True
fifth_group = st.checkbox('$500-$750', value=True)
if fifth_group:
	filters['history_segments']['5) $500 - $750'] = True
sixth_group = st.checkbox('$750-$1.000', value=True)
if sixth_group:
	filters['history_segments']['6) $750 - $1,000'] = True
seventh_group = st.checkbox('$1.000+', value=True)
if seventh_group:
	filters['history_segments']['7) $1,000 +'] = True

st.write('Каких пользователей по почтовому коду выберем')
filters['zip_code'] = {}
surburban = st.checkbox('Surburban', value=True)
if surburban:
	filters['zip_code']['surburban'] = True
urban = st.checkbox('Urban', value=True)
if urban:
	filters['zip_code']['urban'] = True
rural = st.checkbox('Rural', value=True)
if rural:
	filters['zip_code']['rural'] = True

recency = st.slider(label='Месяцев с момента покупки', min_value=int(data_test.recency.min()), max_value=int(data_test.recency.max()), value=(int(data_test.recency.min()), int(data_test.recency.max())))
filters['recency'] = recency

disabled = False
if not first_group and not second_group and not third_group and not fourth_group and not fifth_group and not sixth_group and not seventh_group:
	st.error('Необходимо выбрать хотя бы один класс')
	disabled = True
elif not surburban and not urban and not rural:
	st.error('Необходимо выбрать хотя бы один почтовый индекс')
	disabled = True


if not disabled:
	filtered_dataset = tools.filter_data(data_test, filters)
	# значение uplift для записей тех клиентов, который выбрал пользователь равен 1
	import numpy as np
	uplift = pd.DataFrame(
		data=[np.random.random() for _ in filtered_dataset.index],
		index=filtered_dataset.index
		)
	target_filtered =target_test.loc[filtered_dataset.index]
	treatment_filtered = treatment_test.loc[filtered_dataset.index]
	sample_size = 7 if filtered_dataset.shape[0] >= 7 else filtered_dataset.shape[0]
	example = filtered_dataset.sample(sample_size)
	st.write('Пример пользователей, которым будет отправлена реклама')
	st.dataframe(example)
	st.info(f'Количество клиентов, которым реклама будет отправлена: _**{filtered_dataset.shape[0]}**_ ({filtered_dataset.shape[0] / data_train.shape[0] * 100 :.2f}% от всех клиентов)')


send_promo = st.button('Отправить рекламу и посмотреть результат', disabled=disabled)
if send_promo:
	from sklift.metrics import uplift_by_percentile, uplift_at_k
	st.write(uplift_by_percentile(y_true=target_filtered, uplift=uplift, treatment=treatment_filtered))
	st.write(uplift_at_k(y_true=target_filtered, uplift=uplift, treatment=treatment_filtered, strategy='by_group', k=0.3))
	# st.write(tools.get_weighted_average_uplift(target_filtered, uplift, treatment_filtered))

# st.write('Если известно, на какой процент пользователей необходимо воздействовать, укажите это ниже')
# st.slider(label='Процент пользователей', min_value=0, max_value=100, value=100)

