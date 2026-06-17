# filters.py
import streamlit as st
import pandas as pd
import numpy as np


def create_genre_filter(df, sidebar=True):
    location = st.sidebar if sidebar else st
    df['normalizedGenre_filled'] = df['normalizedGenre'].fillna("Не указан")
    genres = sorted(df['normalizedGenre_filled'].unique().tolist())

    selected_genres = location.multiselect(
        "Жанр пьесы", 
        genres, 
        default=genres,
        help="Можно выбрать несколько жанров одновременно"
    )
    return selected_genres


def create_year_filter(df, sidebar=True):
    location = st.sidebar if sidebar else st
    year_min = int(df['yearNormalized'].min())
    year_max = int(df['yearNormalized'].max())
    selected_years = location.slider(
        "Годы",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
        help="Примерный год, к которому можно отнести пьесу (yearNormalized), посчитанный" \
        " авторами FreDraCor на основе года написания, года публикации и года премьеры."
    )
    return selected_years


def create_word_count_filter(df, sidebar=True):
    location = st.sidebar if sidebar else st
    text_min = int(df['wordCountText'].min())
    text_max = int(df['wordCountText'].max())
    selected_word_count = location.slider(
        "Количество слов в пьесе:",
        min_value=text_min,
        max_value=text_max,
        value=(text_min, text_max),
        step=100
    )
    return selected_word_count


def create_structure_filter(df, sidebar=True):
    location = st.sidebar if sidebar else st
    location.subheader("Структура пьесы")
    acts_min, acts_max = location.slider(
        "Количество актов:",
        min_value=int(df['numOfActs'].min()),
        max_value=int(df['numOfActs'].max()),
        value=(int(df['numOfActs'].min()), int(df['numOfActs'].max())),
        step=1
    )
    scenes_min, scenes_max = location.slider(
        "Количество сцен:",
        min_value=int(df['numOfScenes'].min()),
        max_value=int(df['numOfScenes'].max()),
        value=(int(df['numOfScenes'].min()), int(df['numOfScenes'].max())),
        step=1
    )
    return {
        'acts_range': (acts_min, acts_max),
        'scenes_range': (scenes_min, scenes_max)
    }


def create_speakers_filter(df, sidebar=True):
    """Создает фильтр по персонажам"""
    location = st.sidebar if sidebar else st
    
    location.subheader("Персонажи")
    
    speakers_min, speakers_max = location.slider(
        "Количество персонажей:",
        min_value=int(df['numOfSpeakers'].min()),
        max_value=int(df['numOfSpeakers'].max()),
        value=(int(df['numOfSpeakers'].min()), int(df['numOfSpeakers'].max())),
        step=1
    )
    
    
    return {
        'speakers_range': (speakers_min, speakers_max)
    } # тут должен был быть фильтр еще по персонажам разного пола

# тут фильтруем просто обычным для pandas способом
def apply_filters(df, filters):
    filtered_df = df.copy()
    
    if 'selected_genres' in filters and filters['selected_genres']:
        filtered_df = filtered_df[filtered_df['normalizedGenre_filled'].isin(filters['selected_genres'])]

    if 'selected_years' in filters and filters['selected_years']:
        filtered_df = filtered_df[
            (filtered_df['yearNormalized'] >= filters['selected_years'][0]) & 
            (filtered_df['yearNormalized'] <= filters['selected_years'][1])
        ]

    if 'selected_word_count' in filters and filters['selected_word_count']:
        filtered_df = filtered_df[
            (filtered_df['wordCountText'] >= filters['selected_word_count'][0]) & 
            (filtered_df['wordCountText'] <= filters['selected_word_count'][1])
        ]

    if 'structure' in filters:
        struct = filters['structure']
        if struct.get('classic_5_acts'):
            filtered_df = filtered_df[filtered_df['numOfActs'] == 5]
        
        filtered_df = filtered_df[
            (filtered_df['numOfActs'] >= struct['acts_range'][0]) & 
            (filtered_df['numOfActs'] <= struct['acts_range'][1])
        ]
        filtered_df = filtered_df[
            (filtered_df['numOfScenes'] >= struct['scenes_range'][0]) & 
            (filtered_df['numOfScenes'] <= struct['scenes_range'][1])
        ]

    if 'speakers' in filters:
        sp = filters['speakers']
        filtered_df = filtered_df[
            (filtered_df['numOfSpeakers'] >= sp['speakers_range'][0]) & 
            (filtered_df['numOfSpeakers'] <= sp['speakers_range'][1])
        ]
    
    return filtered_df

def create_search_filter(df, sidebar=True):
    location = st.sidebar if sidebar else st
    location.markdown("---")
    location.subheader(" Поиск по таблице")
    text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    default_search_cols = [col for col in ['title', 'normalizedGenre'] if col in text_columns]
    
    search_columns = location.multiselect(
        "Поиск в столбцах:",
        text_columns,
        default=default_search_cols
    )
    
    # поле ввода поискового запроса
    search_query = location.text_input(
        "Введите запрос:",
        placeholder="Например: Molière, Tragedy, 1670...",
        help="Поиск не чувствителен к регистру. Для сброса оставьте поле пустым и нажмите Enter."
    )
    
    return search_query, search_columns

# эта функция создает все фильтры и возвращает параметры фильтров (и еще кнопку сброса сразу создает)
def create_all_filters(df, sidebar=True):
    filters = {}
    
    filters['selected_genres'] = create_genre_filter(df, sidebar)
    filters['selected_years'] = create_year_filter(df, sidebar)
    filters['selected_word_count'] = create_word_count_filter(df, sidebar)
    filters['structure'] = create_structure_filter(df, sidebar)
    filters['speakers'] = create_speakers_filter(df, sidebar)
    
    # Кнопка сброса
    location = st.sidebar if sidebar else st
    if location.button("Сбросить все фильтры"):
        st.rerun()
    
    return filters


def apply_search_filter(df, search_query, search_columns):
    if not search_query or not search_columns:
        return df
    mask = pd.Series(False, index=df.index)
    for col in search_columns:
            col_mask = df[col].astype(str).str.contains(
                search_query, 
                case=False, 
                na=False
            )
            mask = mask | col_mask
    
    return df[mask]