import streamlit as st
import pandas as pd
import plotly.express as px


def plot_quantitative_distribution(df):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if not num_cols:
        st.info("Нет количественных признаков для отображения.")
        return
    col = st.selectbox(
        "Выберите признак:", num_cols,
        index=num_cols.index('yearNormalized') if 'yearNormalized' in num_cols else 0,
        key="num_dist_col"
    )
    bins = st.slider("Количество бинов:", 1, 500, 30, key="num_dist_bins")
    fig = px.histogram(df, x=col, nbins=bins)
    st.plotly_chart(fig)


def plot_categorical_distribution(df):
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    cat_cols.remove('title')
    if not cat_cols:
        st.info("Нет категориальных признаков для отображения.")
        return
    col = st.selectbox("Выберите признак:", cat_cols, key="cat_dist_col")
    st.bar_chart(df[col].value_counts(), horizontal=True, sort=True, color='red')


def show_descriptive_stats(df):
    """Описательные статистики для выбранного признака."""
    col = st.selectbox("Выберите признак:", df.columns, key="desc_stats_col")
    series = df[col]
    
    if pd.api.types.is_numeric_dtype(series) and col != 'libretto':
        stats = series.describe()
        stats.name = "Значение"
        st.dataframe(stats)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Среднее", f"{stats['mean']:.2f}")
        c2.metric("Медиана", f"{stats['50%']:.2f}")
        c3.metric("Ст. отклонение", f"{stats['std']:.2f}")
        c4.metric("Пропуски", series.isna().sum())
    else:
        stats = series.describe()
        st.dataframe(stats)
        st.write("**Топ-5 значений:**")
        st.dataframe(series.value_counts().head(5))


def show_correlation_analysis(df):
    """Корреляционный анализ с кодированием категорий."""
    # Кодируем категориальные признаки
    df_enc = df.copy()
    for col in df_enc.select_dtypes(exclude='number').columns:
        df_enc[col] = df_enc[col].astype('category').cat.codes
    
    num_cols = df_enc.select_dtypes(include='number').columns.tolist()
    
    if len(num_cols) < 2:
        st.info("Нужно минимум 2 признака для корреляционного анализа.")
        return
    
    selected = st.multiselect("Признаки:", num_cols, default=num_cols[:3], key="corr_features")
    
    if len(selected) < 2:
        st.warning("Выберите хотя бы 2 признака.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Метод:", ["pearson", "spearman", "kendall"],
            format_func=lambda x: {"pearson": "Пирсон", "spearman": "Спирмен", "kendall": "Кендалл"}[x],
            key="corr_method"
        )
    with col2:
        view = st.selectbox("Вид:", ["Тепловая карта", "Диаграмма рассеяния"], key="corr_view")
    
    corr = df_enc[selected].corr(method=method).round(2)
    
    if view == "Тепловая карта":
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
    else:
        fig = px.scatter_matrix(df_enc, dimensions=selected, opacity=0.6)
    
    st.plotly_chart(fig)