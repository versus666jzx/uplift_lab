from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_lenta
from catboost import CatBoostClassifier
import sklearn
import streamlit as st


@st.experimental_memo
def get_data() -> sklearn.utils._bunch.Bunch:

	treat_dict = {
		'test':    1,
		'control': 0
	}
	# получаем датасет
	dataset = fetch_lenta()
	# преобразуем строковые значения колонки в числовыые значения
	dataset.treatment = dataset.treatment.map(treat_dict)
	# заполняем пропуски
	dataset.data['gender'] = dataset.data['gender'].fillna(value='Не определен')
	dataset.data['children'] = dataset.data['children'].fillna(0).astype('int')
	dataset.data['age'] = dataset.data['age'].fillna(0).astype('int')
	dataset.data['months_from_register'] = dataset.data['months_from_register'].fillna(0).astype('int')
	return dataset


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
