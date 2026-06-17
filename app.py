import utils.data_funcs as ut
from utils.filters import create_all_filters, apply_filters, create_search_filter, apply_search_filter 
from utils.stats import render_statistical_analysis_tab
from utils.ml import render_ml_tab
import streamlit as st
import numpy as np
from utils.eda import plot_quantitative_distribution, plot_categorical_distribution, show_descriptive_stats, show_correlation_analysis

st.set_page_config(
    page_title="Анализ данных ЦМГН",
    page_icon="☺",
    layout="centered",
)

st.title("Метаданные корпуса французской драмы DraCor")

df = ut.load_data()

# ФИЛЬТРАЦИЯ (ЛЕВАЯ ПАНЕЛЬ)

# Создаем фильтры в левой панели
st.sidebar.header("Фильтры")
filters = create_all_filters(df, sidebar=True)

# Фильтры
filtered_df = apply_filters(df, filters)

# Поиск по таблице
search_query, search_columns = create_search_filter(filtered_df, sidebar=True)
filtered_df = apply_search_filter(filtered_df, search_query, search_columns)

# Отображаем результаты
st.markdown(f"### Найдено пьес: **{len(filtered_df)}**")

# Если есть поисковый запрос — показываем, что именно искали
if search_query:
    st.info(f"Поиск: **\"{search_query}\"** по столбцам: {', '.join(search_columns)}")

st.dataframe(filtered_df)


# АНАЛИЗ (ОТФИЛЬТРОВАННЫХ ДАННЫХ)
tab1, tab2, tab3 = st.tabs(["EDA", "Статистический анализ", "Машинное обучение"])

# EDA
with tab1:
    st.subheader("Информация о датасете")
    
    if not filtered_df.empty:
        st.markdown("#### Распределение количественных признаков")
        plot_quantitative_distribution(filtered_df)
        
        st.markdown("#### Распределение категориальных признаков")
        plot_categorical_distribution(filtered_df)
        
        st.markdown("#### Описательные статистики")
        show_descriptive_stats(filtered_df)
        
        st.markdown("#### Корреляционный анализ")
        show_correlation_analysis(filtered_df)
    else:
        st.warning("Нет данных для выбранных фильтров.")

# Статистический анализ

with tab2:
    render_statistical_analysis_tab(filtered_df)

# Машинное обучение
with tab3:
    render_ml_tab(filtered_df)

    