import streamlit as st
import pandas as pd
import scipy.stats
import statsmodels.api as sm


def render_normality_test(df, alpha):
    st.subheader("Критерии согласия (нормальность)")
    st.markdown("**$H_0$**: Признак распределен нормально.")
    st.markdown("**$H_1$**: Признак не распределен нормально.")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if not num_cols:
        st.warning("В данных нет числовых признаков.")
        return

    col = st.selectbox("Числовой признак:", num_cols, key="norm_col")
    method = st.radio("Критерий:", ["Шапиро-Уилка", "Колмогорова-Смирнова"], horizontal=True, key="norm_method")
    
    data = df[col].dropna()
    if len(data) <= 3:
        st.warning("Недостаточно данных для теста.")
        return

    if method == "Шапиро-Уилка":
        if len(data) > 5000:
            st.warning("Внимание! Размер выборки > 5000. p-value может быть неинтерпретируемым.")
        stat, p = scipy.stats.shapiro(data)
    else:
        stat, p = scipy.stats.kstest(data, 'norm')
    
    c1, c2 = st.columns(2)
    c1.metric("Статистика", f"{stat:.4f}")
    c2.metric("p-value", f"{p:.4e}")
    
    if p < alpha:
        st.error(f"p-value ({p:.4e}) < α ({alpha}). **Отвергаем $H_0$**: данные не распределены нормально.")
    else:
        st.success(f"p-value ({p:.4e}) ≥ α ({alpha}). **Нет оснований отвергать $H_0$**: данные распределены нормально.")
    
    with st.expander("Показать Q-Q plot"):
        fig = sm.qqplot(data, line='s')
        st.pyplot(fig)


def render_two_groups_test(df, alpha):
    st.subheader("Критерии сдвига / t-критерий")
    st.markdown("**$H_0$**: Распределения признака в двух группах совпадают.")
    st.markdown("**$H_1$**: Распределения различаются.")
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    if not num_cols or not cat_cols:
        st.warning("Нужен хотя бы один числовой и один категориальный признак.")
        return

    col_num = st.selectbox("Числовой признак:", num_cols, key="num_shift")
    col_cat = st.selectbox("Признак для группировки:", cat_cols, key="cat_shift")
    
    groups = df[col_cat].dropna().unique()
    if len(groups) != 2:
        st.warning(f"Внимание! Признак '{col_cat}' имеет {len(groups)} уникальных значений. Для этих тестов нужно ровно 2 группы.")
        return

    g1, g2 = groups
    sample1 = df[df[col_cat] == g1][col_num].dropna()
    sample2 = df[df[col_cat] == g2][col_num].dropna()
    
    method = st.radio("Критерий:", ["Манна-Уитни (непараметрический)", "Стьюдента (параметрический)"], horizontal=True, key="shift_method")
    
    if "Манна" in method:
        stat, p = scipy.stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
    else:
        stat, p = scipy.stats.ttest_ind(sample1, sample2)
    
    st.info(f"**Группы:** {g1} (n={len(sample1)}) vs {g2} (n={len(sample2)})")
    c1, c2 = st.columns(2)
    c1.metric("Статистика", f"{stat:.4f}")
    c2.metric("p-value", f"{p:.4e}")
    
    if p < alpha:
        st.error(f"p-value ({p:.4e}) < α ({alpha}). **Отвергаем $H_0$**: между группами есть статистически значимые различия.")
    else:
        st.success(f"p-value ({p:.4e}) ≥ α ({alpha}). **Нет оснований отвергать $H_0$**: группы статистически не различаются.")


def render_chi_square_test(df, alpha):
    st.subheader("Критерий Хи-квадрат Пирсона (таблицы сопряженности)")
    st.markdown("**$H_0$**: Признаки независимы.")
    st.markdown("**$H_1$**: Между признаками есть связь.")
    
    # Формируем список категориальных признаков, гарантируя наличие 'libretto' без дубликатов
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    if 'libretto' not in cat_cols:
        cat_cols.append('libretto')
    
    if len(cat_cols) < 2:
        st.warning("Нужно как минимум два категориальных признака.")
        return

    col1_cat = st.selectbox("Признак 1:", cat_cols, key="cat1_chi")
    col2_cat = st.selectbox("Признак 2:", cat_cols, index=min(1, len(cat_cols)-1), key="cat2_chi")
    
    if col1_cat == col2_cat:
        st.warning("Выберите два разных признака.")
        return

    crosstab = pd.crosstab(df[col1_cat], df[col2_cat])
    small_freq = (crosstab < 5).sum().sum()
    total_cells = crosstab.size
    
    st.write("**Таблица сопряженности:**")
    st.dataframe(crosstab)
    
    if small_freq / total_cells > 0.25:
        st.warning(f"В {small_freq} из {total_cells} ячеек частота < 5. Результаты могут быть некорректными.")
    
    # expected не используется, поэтому заменяем на _
    chi2, p, dof, _ = scipy.stats.chi2_contingency(crosstab)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Хи-квадрат", f"{chi2:.4f}")
    c2.metric("p-value", f"{p:.4e}")
    c3.metric("Степени свободы (dof)", f"{dof}")
    
    if p < alpha:
        st.error(f"p-value ({p:.4e}) < α ({alpha}). **Отвергаем $H_0$**: между признаками есть статистически значимая связь.")
    else:
        st.success(f"p-value ({p:.4e}) ≥ α ({alpha}). **Нет оснований отвергать $H_0$**: признаки независимы.")


def render_statistical_analysis_tab(filtered_df):
    """Главная функция для отрисовки вкладки статистического анализа."""
    st.header("Проверка статистических гипотез")
    st.markdown("Выберите критерий для проверки статистических гипотез на отфильтрованных данных.")
    
    alpha = st.selectbox("Уровень значимости (α):", [0.05, 0.01, 0.10], index=0, key="stats_alpha")
    
    test_type = st.selectbox(
        "Тип теста:",
        ["1. Проверка на нормальность (Шапиро-Уилка / Колмогорова-Смирнова)", 
         "2. Сравнение двух групп (Манна-Уитни / Стьюдента)", 
         "3. Связь категориальных признаков (Хи-квадрат Пирсона)"],
        key="stats_test_type"
    )
    
    if "1." in test_type:
        render_normality_test(filtered_df, alpha)
    elif "2." in test_type:
        render_two_groups_test(filtered_df, alpha)
    elif "3." in test_type:
        render_chi_square_test(filtered_df, alpha)