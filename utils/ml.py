import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix, silhouette_score
)


def render_regression_tab(df):
    st.subheader("Линейная регрессия")
    st.markdown("Модель: $y = \\langle w, x \\rangle + w_0$")
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) < 2:
        st.warning("Нужно минимум 2 числовых признака.")
        return
        
    target = st.selectbox("Целевой признак (y):", num_cols, 
                          index=num_cols.index('yearNormalized') if 'yearNormalized' in num_cols else 0, 
                          key="reg_target")
    features = st.multiselect("Признаки (X):", [c for c in num_cols if c != target], 
                              default=[c for c in num_cols if c != target][:3], 
                              key="reg_features")
    
    if not features:
        st.warning("Выберите хотя бы один признак.")
        return
        
    df_reg = df[[target] + features].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df_reg[features].values, df_reg[target].values, test_size=0.2, random_state=42
    )
    
    scaler_type = st.selectbox("Масштабирование:", ["Нет", "StandardScaler", "MinMaxScaler"], key="reg_scaler")
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    c3.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
    
    st.write("**Коэффициенты модели:**")
    coef_df = pd.DataFrame({'Признак': features, 'Вес (w)': model.coef_})
    st.dataframe(coef_df)
    st.write(f"Свободный коэффициент (w₀): **{model.intercept_:.2f}**")
    
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Факт', 'y': 'Предсказание'}, title="Факт vs Предсказание")
    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), 
                  line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)


def render_classification_tab(df):
    st.subheader("Классификация")
    
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    if not cat_cols:
        st.warning("Нет категориальных признаков для таргета.")
        return
        
    target = st.selectbox("Целевой признак (класс):", cat_cols, key="clf_target")
    features = st.multiselect("Числовые признаки:", num_cols, 
                              default=num_cols[:min(3, len(num_cols))], key="clf_features")
    
    if not features:
        st.warning("Выберите хотя бы один числовой признак.")
        return
        
    # Берем сырые данные
    X_raw = df[features].values
    y_raw = df[target].values
    
    # Оставляем только строки, где известен целевой класс (иначе мы не сможем обучаться)
    valid_mask = ~pd.isna(y_raw)
    X_raw = X_raw[valid_mask]
    y_raw = y_raw[valid_mask]
    
    classes = np.unique(y_raw)
    st.info(f"Классы: {list(classes)}")
    
    # Заполняем пропуски в признаках медианой (чтобы не терять данные, как в лекциях)
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)
    
    # Теперь X и y_raw синхронизированы (оба отфильтрованы по valid_mask)
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model_type = st.radio("Модель:", ["Логистическая регрессия", "kNN"], horizontal=True, key="clf_model")
    
    if model_type == "Логистическая регрессия":
        if len(classes) > 2:
            st.warning("Логистическая регрессия рассматривалась для бинарной классификации. Выберите признак с 2 классами.")
            return
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    else:
        k = st.slider("Количество соседей (k):", 1, 30, 5, key="clf_k")
        model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    
    # Вывод метрик и матрицы ошибок
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    st.write("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))
    
    # Указываем labels явно, чтобы порядок классов в матрице не зависел от тестовой выборки
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig_cm, ax = plt.subplots()
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Факт')
    st.pyplot(fig_cm)


def render_clustering_tab(df):
    st.subheader("Кластеризация")
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    features = st.multiselect("Признаки для кластеризации:", num_cols, 
                              default=num_cols[:min(4, len(num_cols))], key="clust_features")
    
    if len(features) < 2:
        st.warning("Выберите минимум 2 признака для PCA-визуализации.")
        return
        
    df_clust = df[features].dropna()
    X = StandardScaler().fit_transform(df_clust.values)
    
    algo = st.radio("Алгоритм:", ["k-Means", "DBSCAN"], horizontal=True, key="clust_algo")
    
    if algo == "k-Means":
        k = st.slider("Количество кластеров (k):", 2, 20, 4, key="clust_k")
        labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X)
        
        st.metric("Silhouette Score", f"{silhouette_score(X, labels):.3f}")
        
        inertias = []
        K_range = range(2, min(20, max(3, len(df_clust) // 2)))
        for kk in K_range:
            inertias.append(KMeans(n_clusters=kk, random_state=42, n_init='auto').fit(X).inertia_)
        
        fig_elbow = px.line(x=list(K_range), y=inertias, markers=True, title="Метод локтя", labels={'x': 'k', 'y': 'Inertia'})
        st.plotly_chart(fig_elbow, use_container_width=True)
    else:
        c1, c2 = st.columns(2)
        eps = c1.slider("ε (радиус):", 0.1, 3.0, 0.8, 0.1, key="clust_eps")
        min_samples = c2.slider("min_samples:", 2, 50, 20, key="clust_min_samples")
        
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Кластеров найдено", n_clusters)
        c2.metric("Шумовых точек", n_noise)
        if n_clusters >= 2:
            c3.metric("Silhouette Score", f"{silhouette_score(X, labels):.3f}")
            
    st.subheader("Визуализация кластеров (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df_viz = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Кластер': labels.astype(str)})
    
    fig = px.scatter(df_viz, x='PC1', y='PC2', color='Кластер', title=f"PCA-проекция ({algo})")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"**Объяснённая дисперсия:** PC1 = {pca.explained_variance_ratio_[0]:.1%}, PC2 = {pca.explained_variance_ratio_[1]:.1%}")


def render_ml_tab(filtered_df):
    """Главная функция для отрисовки вкладки машинного обучения."""
    st.header("Машинное обучение")
    
    task = st.radio(
        "Тип задачи:",
        ["Регрессия (предсказание числа)", "Классификация (предсказание класса)", "Кластеризация (поиск скрытых групп)"],
        horizontal=True,
        key="ml_task"
    )
    
    if "Регрессия" in task:
        render_regression_tab(filtered_df)
    elif "Классификация" in task:
        render_classification_tab(filtered_df)
    else:
        render_clustering_tab(filtered_df)