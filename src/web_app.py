import catboost
import pandas as pd
import os
from sklift.metrics import uplift_at_k, uplift_by_percentile, qini_auc_score, qini_curve
from sklift.viz import plot_qini_curve, plot_uplift_curve
from sklift.models import SoloModel, TwoModels, ClassTransformation
import streamlit as st
import catboost

import tools

# загрузим датасет
dataset, target, treatment = tools.get_data()

# загрузим предикты моделей
ct_cbc = pd.read_csv('src/model_predictions/ct_cbc.csv', index_col='Unnamed: 0')
sm_cbc = pd.read_csv('src/model_predictions/sm_cbc.csv', index_col='Unnamed: 0')
tm_dependend_cbc = pd.read_csv('src/model_predictions/tm_dependend_cbc.csv', index_col='Unnamed: 0')
tm_independend_cbc = pd.read_csv('src/model_predictions/tm_independend_cbc.csv', index_col='Unnamed: 0')

# загрузим данные
data_train_index = pd.read_csv('data/data_train_index.csv')
data_test_index = pd.read_csv('data/data_test_index.csv')
treatment_train_index = pd.read_csv('data/treatment_train_index.csv')
treatment_test_index = pd.read_csv('data/treatment_test_index.csv')
target_train_index = pd.read_csv('data/target_train_index.csv')
target_test_index = pd.read_csv('data/target_test_index.csv')

# фиксируем выборки, чтобы результат работы ML был предсказуем
data_train = dataset.loc[data_train_index['0']]
data_test = dataset.loc[data_test_index['0']]
treatment_train = treatment.loc[treatment_train_index['0']]
treatment_test = treatment.loc[treatment_test_index['0']]
target_train = target.loc[target_train_index['0']]
target_test = target.loc[target_test_index['0']]

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
title_subsample = data_train.sample(7)
if refresh:
	title_subsample = data_train.sample(7)
st.dataframe(title_subsample, width=700)
st.write(f"Всего записей: {data_train.shape[0]}")

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

	st.plotly_chart(tools.get_newbie_plot(data_train), use_container_width=True)
	st.write(f'В данных примерно одинаковое количество новых и "старых клиентов". '
			 f'Отношение новых клиентов к старым: {(data_train["newbie"] == 1).sum() / (data_train["newbie"] == 0).sum():.2f}')

	st.plotly_chart(tools.get_zipcode_plot(data_train), use_container_width=True)
	tmp_res = data_train.zip_code.value_counts(normalize=True) * 100
	st.write(f'Большинство клиентов из пригорода: {tmp_res["Surburban"]:.2f}%, из города: {tmp_res["Urban"]:.2f}% и из села: {tmp_res["Rural"]:.2f}%')

	tmp_res = data_train.channel.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_channel_plot(data_train), use_container_width=True)
	st.write(f'В прошлом году почти одинаковое количество клиентов покупало товары через телефон и сайт, {tmp_res["Phone"]:.2f}% и {tmp_res["Web"]:.2f}% соответственно,'
			 f' а {tmp_res["Multichannel"]:.2f}% клиентов покупали товары воспользовавшись двумя платформами.')

	tmp_res = data_train.history_segment.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_history_segment_plot(data_train), use_container_width=True)
	st.write(f'Как мы видим, большинство пользователей относится к сегменту \$0-\$100 ({tmp_res[0]:.2f}%), второй и '
			 f'третий по количеству пользователей сегменты \$100-\$200 ({tmp_res[1]:.2f}%) и \$200-\$350 ({tmp_res[2]:.2f}%).')
	st.write(f'К сегментам \$350-\$500 и \$500-\$750 относится {tmp_res[3]:.2f}% и {tmp_res[4]:.2f}% пользователей соответственно.')
	st.write(f'Меньше всего пользователей в сегментах \$750-\$1.000 ({tmp_res[-2]:.2f}%) и \$1.000+ ({tmp_res[-1]:.2f}%).')

	tmp_res = list(data_train.recency.value_counts(normalize=True) * 100)
	st.plotly_chart(tools.get_recency_plot(data_train), use_container_width=True)
	st.write(f'Большинство клиентов являются активными клиентами платформы, и совершали покупки в течение месяца ({tmp_res[0]:.2f}%)')
	st.write('Также заметно, что 9 и 10 месяцев назад, много клиентов совершали покупки. Это может свидетельствовать о проведении'
			 'рекламной кампании в это время или чего-то еще.')
	st.write('Также интересно понаблюдать за долями новых клиентов в данном распределении.')

	st.plotly_chart(tools.get_history_plot(data_train), use_container_width=True)
	st.markdown('_График интерактивный. Двойной клик вернет в начальное состояние._')
	st.write('Абсолютное большинство клиентов тратят \$25-\$35 на покупки, но есть и малая доля тех, кто тратит более \$3.000')
	st.write('Интересный факт: все покупки более \$500 совершают только новые клиенты')

filters = {}

# блок фильтров
with st.form(key='filter-clients'):
	st.subheader('Выберем клиентов, которым отправим рекламу.')

	col1, col2, col3 = st.columns(3)

	channel_filter = col1.radio('Канал покупки прошлом году', options=['Все', 'Phone', 'Web', 'Multichannel'])
	filters['channel_filter'] = channel_filter

	newbie_filter = col2.radio('Тип клиента', options=['Все', 'Только новые', 'Только старые'])
	filters['newbie_filter'] = newbie_filter

	mens_filter = col3.radio('Клиенты, приобретавшие товары', options=['Любые', 'Мужские', 'Женские'])
	filters['mens_filter'] = mens_filter

	filters['history_segments'] = {}

	col1, col2 = st.columns(2)

	with col1:
		st.write('Класс клиентов по объему денег, потраченных в прошлом году (history segments)')
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

	with col2:
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

	st.write('Если известно на какой процент аудитории необходимо повлиять, измените значение')
	k = st.slider(label='Процент аудитории', min_value=1, max_value=100, value=100)

	filter_form_submit_button = st.form_submit_button('Применить фильтр')

# проверка корректности заполнения форм
if not first_group and not second_group and not third_group and not fourth_group and not fifth_group and not sixth_group and not seventh_group:
	st.error('Необходимо выбрать хотя бы один класс')
	st.stop()
elif not surburban and not urban and not rural:
	st.error('Необходимо выбрать хотя бы один почтовый индекс')
	st.stop()

# фильтруем тестовые данные по пользовательскому выбору
filtered_dataset = tools.filter_data(data_test, filters)

# проверяем, что данные отфильтровались
if filtered_dataset is None:
	st.error('Не найдено пользователей для данных фильтров. Попробуйте изменить фильтры.')
	st.stop()

# значение uplift для записей тех клиентов, который выбрал пользователь равен 1
uplift = [1 for _ in filtered_dataset.index]
target_filtered = target_test.loc[filtered_dataset.index]
treatment_filtered = treatment_test.loc[filtered_dataset.index]

# блок с демонстрацией отфильтрованных данных
with st.expander(label='Посмотреть пример пользователей, которым будет отправлена реклама'):
	sample_size = 7 if filtered_dataset.shape[0] >= 7 else filtered_dataset.shape[0]
	example = filtered_dataset.sample(sample_size)
	st.dataframe(example)
	st.info(f'Количество пользователей, попавших в выборку: {filtered_dataset.shape[0]} ({filtered_dataset.shape[0] / data_test.shape[0] * 100 :.2f}%)')
	res = st.button('Обновить')

with st.expander('Результаты ручной фильтрации', expanded=True):
	# считаем метрики для пользователя
	user_metric_uplift_at_k = uplift_at_k(target_filtered, uplift, treatment_filtered, strategy='overall', k=k)
	user_metric_uplift_by_percentile = uplift_by_percentile(target_filtered, uplift, treatment_filtered)
	user_metric_qini_auc_score = qini_auc_score(target_filtered, uplift, treatment_filtered)
	user_metric_weighted_average_uplift = tools.get_weighted_average_uplift(target_filtered, uplift, treatment_filtered)
	qini_curve_user_score = qini_curve(target_filtered, uplift, treatment_filtered)
	# отображаем метрики
	col1, col2, col3 = st.columns(3)
	col1.metric(label=f'Uplift для {k}% пользователей', value=f'{user_metric_uplift_at_k:.4f}')
	col2.metric(label=f'Qini AUC score', value=f'{user_metric_qini_auc_score:.4f}', help='Всегда будет 0 для пользователя')
	col3.metric(label=f'Weighted average uplift', value=f'{user_metric_weighted_average_uplift:.4f}')

	st.write('Uplift по процентилям')
	st.write(user_metric_uplift_by_percentile)

show_ml_reasons = st.checkbox('Показать решения с помощью ML')
if show_ml_reasons:
	with st.expander('Решение с помощью CatBoost'):
		with st.form(key='catboost_metricks'):

			final_uplift = sm_cbc.loc[filtered_dataset.index]['0']

			# считаем метрики для ML
			catboost_uplift_at_k = uplift_at_k(target_filtered, final_uplift, treatment_filtered, strategy='overall', k=k)
			catboost_uplift_by_percentile = uplift_by_percentile(target_filtered, final_uplift, treatment_filtered)
			catboost_qini_auc_score = qini_auc_score(target_filtered, final_uplift, treatment_filtered)
			catboost_weighted_average_uplift = tools.get_weighted_average_uplift(target_filtered, final_uplift, treatment_filtered)
			qini_curve_score = qini_curve(target_filtered, final_uplift, treatment_filtered)
			# отображаем метрики
			col1, col2, col3 = st.columns(3)
			col1.metric(label=f'Uplift для {k}% пользователей', value=f'{catboost_uplift_at_k:.4f}', delta=f'{catboost_uplift_at_k - user_metric_uplift_at_k:.4f}')
			col2.metric(label=f'Qini AUC score', value=f'{catboost_qini_auc_score:.4f}', help='Всегда будет 0 для пользователя', delta=f'{catboost_qini_auc_score - user_metric_qini_auc_score:.4f}')
			col3.metric(label=f'Weighted average uplift', value=f'{catboost_weighted_average_uplift:.4f}', delta=f'{catboost_weighted_average_uplift - user_metric_weighted_average_uplift:.4f}')

			st.write('Uplift по процентилям')
			st.write(catboost_uplift_by_percentile)
			st.form_submit_button('Обновить графики', help='При изменении флагов')
			perfect_qini = st.checkbox('Отрисовать идеальную метрику qini')
			# получаем координаты пользовательской метрики для точки на графике
			x, y = qini_curve_user_score[0][1], qini_curve_user_score[1][1]
			# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
			fig = plot_qini_curve(target_test, sm_cbc['0'], treatment_test, perfect=perfect_qini)
			# добавляем пользовательскую метрику на оси графика
			fig.ax_.plot(x, y, 'ro', markersize=3)
			st.pyplot(fig.figure_)
			prefect_uplift = st.checkbox('Отрисовать идеальную метрику uplift')
			st.pyplot(plot_uplift_curve(target_test, sm_cbc['0'], treatment_test, perfect=prefect_uplift).figure_)
