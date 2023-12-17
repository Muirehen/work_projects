'''
Модуль состоит из нескольких частей:
- импорты;
- общие функции;
- класс SqlConnector;
- класс CreateSparse;
- класс Matching
'''

# импорты
# ______________________________________________________________________________________________________________________
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.types import INTEGER

import pandas as pd
from scipy import sparse
from scipy.spatial import distance
from scipy.sparse import hstack

import re
import pickle
from lingua import (
    Language,
    LanguageDetectorBuilder
)
from transliterate import get_translit_function
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
pd.options.mode.chained_assignment = None

# общие функции
# ______________________________________________________________________________________________________________________
'''
функция для фильтрации пустых названий, либо названий из одного символа
'''
def to_none(x):
    if len(x) == 0 or x == len(x) * x[0]:
        return False
    else:
        return True


'''
функция приводит к нижнему регистру и оставляет только русские символы,
запятые и символы i из белорусского и украинского языков. Это сделано для того, 
чтобы оставить названия, использующие часть русского алфавита. Если установлен
флаг table, то разделяет str по запятым и фильтрует
'''
def ru_text(string, table=True):
    if string is None:
        return
    string = re.sub(r'[^А-Яа-я,іi]', '', string.lower())
    if table:
        string = string.split(',')
        string = list(filter(to_none, string))
    return string


'''
функция приводит к нижнему регистру и оставляет только латинские символы.
Если значение table = False, то вдобавок приводит к ascii формату
'''
def en_text(string, table=True):
    if string is None:
        return
    if table:
        string = re.sub(r'[^A-Za-z]', '', string.lower())
        return string
    string = str(
        unicodedata.normalize('NFD', string)
        .encode('ascii', 'ignore'))[2:-1]
    string = re.sub(r'[^A-Za-z]', '', string.lower())
    return string


'''
функция трансформирует список в формат (elm_1, elm_2...) для
использования в запросах SQL. Если установлен флаг quotes, то формат
будет следующим: ('elm_1', 'elm_2'...)
'''
def transform(input_, quotes=False):
    string = ''''''
    if quotes:
        for i in input_:
            string = string + "'" + str(i) + "'" + ','
        return "(" + string[:-1] + ')'
    for i in input_:
        string = string + str(i) + ','
    return '(' + string[:-1] + ')'


'''
функция для транслитерации с английского на русский 
'''
translit_ru = get_translit_function('ru')
def transliterate(string):
    return [translit_ru(string.lower()).translate(string).lower()]


'''
Функция для добавления выбранных столбцов в разреженную матрицу векторов
'''
def stacking(sparse_matrix, columns, sql_name, engine, corpus):
    # добавляем столбцы
    for i in columns:
        sparse_matrix = hstack((sparse_matrix, corpus[i].to_numpy()[:, None])).tocsr()

    # загружаем в PostgreSQL
    pd.DataFrame.sparse.from_spmatrix(sparse_matrix).to_sql(
        sql_name,
        con=engine,
        dtype=INTEGER,
        if_exists='replace',
        index=False)

    return sparse_matrix


# класс SqlConnector
# ______________________________________________________________________________________________________________________
'''
Класс служит для присоединения к базе SQL и скачивания оттуда
матриц векторов, а также для загрузки предобученных CountVectorizer
'''
class SqlConnector:
    # при инициализации класса необходимо указать данные базы SQL
    def __init__(self, database):
        self.database = database
        self.engine = create_engine(URL(**database))

    # загрузка разреженной матрицы из базы SQL по заданному пользователем имени
    def download_sparse(self, sparse_name):
        query = 'SELECT * FROM ' + sparse_name
        temp_df = pd.read_sql_query(query, con=self.engine)
        return sparse.coo_matrix(temp_df).tocsr()

    # загрузка модели
    @staticmethod
    def download_vec(vectorizer):
        return pickle.load(open(vectorizer, "rb"))


# класс CreateSparse
# ______________________________________________________________________________________________________________________
'''
Класс служит для создания векторизованных представлений названий населенных пунктов. Также
в классе можно настроить страны, по которым будет производиться векторизация (поиск). Методы класса для
внутреннего пользования начинаются с нижнего подчеркивания
'''
class CreateSparse:
    # при инициализации класса необходимо указывать engine для связи с базой SQL и
    # таблицу с городами, вспомогательные таблицы прописаны по умолчанию
    # страны СНГ, Грузия, Сербия и Турция прописаны по умолчанию
    def __init__(self, engine, main_table, regions="admin1CodesASCII", country_code="countryInfo"):
        self.admin = regions
        self.main_table = main_table
        self.country_info = country_code
        self.engine = engine
        self.countries = ['AZ', 'AM', 'BY',
                          'KG', 'KZ', 'MD',
                          'RU', 'TJ', 'UZ',
                          'GE', 'RS', 'TR']
        print('Модуль готов!')

    # добавление стран по коду ISO
    def add(self, countries: list[str]):
        self.countries.extend(
            [i.upper() for i in countries if i.upper() not in self.countries])
        print(f'''Добавлены следующие страны: {countries}\nПоиск по странам: {self.countries}''')

    # удаление стран по коду ISO
    def remove(self, countries: list[str]):
        countries = [i.upper() for i in countries]
        self.countries = [i for i in self.countries if i not in countries]
        print(f'''Убраны следующие страны: {countries}\nПоиск по странам: {self.countries}''')

    # полная очистка списка стран
    def clear(self):
        self.countries.clear()
        print('Список стран очищен')

    # проверка стран поиска
    def check(self):
        print('Поиск по странам:', self.countries)

    # добавление стран по коду ISO
    def _merging(self):
        sql_query = f'''SELECT * FROM {self.main_table} WHERE "country code" IN {transform(
            input_=self.countries,
            quotes=True)}'''
        cities = pd.read_sql_query(sql_query, con=self.engine)

        # создание ключа для присоединения таблиц
        cities['code'] = cities['country code'] + '.' + cities['admin1 code']

        sql_query = f'''SELECT * FROM "{self.admin}"'''
        admin = pd.read_sql_query(sql_query, con=self.engine)

        sql_query = f'''SELECT * FROM "{self.country_info}"'''
        country_info = pd.read_sql_query(sql_query, con=self.engine)

        cities = cities.merge(
            right=admin,
            left_on=['code'],
            right_on=['code'],
            how='inner',
        )

        cities = cities.merge(
            right=country_info,
            left_on=['country code'],
            right_on=['ISO'],
            how='inner'
        )

        # пропуски в ключевых столбцах удаляются
        return cities.dropna(subset=['geonameid_x', 'asciiname', 'geonameid_y'])

    # после настройки списка стран вызывается метод, который векторизует созданную таблицу
    def vectorize(self):
        cities = self._merging()
        corpus = cities[['geonameid_x',
                         'alternatenames',
                         'asciiname',
                         'population',
                         'geonameid_y']]

        # обработка русского текста
        corpus.loc[:, 'alternatenames'] = corpus.loc[:, 'alternatenames']\
            .apply(ru_text)

        # замена пустых списков на None
        corpus.loc[~corpus['alternatenames'].isna(), 'alternatenames'] = \
            corpus.loc[~corpus['alternatenames'].isna(), 'alternatenames']\
            .apply(lambda x: None if len(x) == 0 else x)

        # обработка английского текста
        corpus['en_name'] = corpus['asciiname'].apply(en_text)

        # добавление перевода там, где не было русских названий
        corpus.loc[corpus['alternatenames'].isna(), 'alternatenames'] = \
            corpus.loc[corpus['alternatenames'].isna(), 'asciiname'] \
            .apply(transliterate)
        # раскрытие списков и расширение таблицы
        temp = pd.DataFrame([*corpus['alternatenames'].values], corpus.index) \
            .stack().reset_index(-1, name='ru_name')
        corpus = corpus[['geonameid_x',
                         'alternatenames',
                         'population',
                         'en_name',
                         'geonameid_y']].join(temp)
        corpus.drop(
            ['alternatenames', 'level_1'],
            inplace=True,
            axis=1)
        corpus.reset_index(inplace=True, drop=True)

        # векторизация русского и английского корпуса
        ru_vectorizer = CountVectorizer(analyzer='char',
                                        ngram_range=(1, 2),
                                        decode_error='ignore')
        en_vectorizer = CountVectorizer(analyzer='char',
                                        ngram_range=(1, 2),
                                        decode_error='ignore')
        ru_sparse = ru_vectorizer.fit_transform(corpus['ru_name'])
        en_sparse = en_vectorizer.fit_transform(corpus['en_name'])

        # добавление в матрицу с векторами дополнительных столбцов с айди и населением
        ru_id = stacking(ru_sparse,
                         ['geonameid_x', 'population', 'geonameid_y'],
                         'ru_sparse', engine=self.engine, corpus=corpus)

        en_id = stacking(en_sparse,
                         ['geonameid_x', 'population', 'geonameid_y'],
                         'en_sparse', engine=self.engine, corpus=corpus)
        # сохранение моделей
        pickle.dump(en_vectorizer, open("en_vector", "wb"))
        pickle.dump(ru_vectorizer, open("ru_vector", "wb"))

        print('Векторизация завершена.')

        # возврат разреженных матриц с дополнительыми столбцами
        return ru_id, en_id


# класс Matching
# ______________________________________________________________________________________________________________________
'''
Класс служит для поиска ближайших названий к введенному запросу пользователя. Языки ввода - русский, 
либо английский. Близость определяется с помощью косинусного расстояния и расстояния Минковского. При вводе 
на русском языке используется русский корпус, при вводе на английском - английский
'''
class Matching:
    # при инициализации класса необходимо указать engine для связи с базой SQL, разреженные матрицы
    # русского и английского корпуса с дополнительными столбцами, а также предобученные модели
    def __init__(self, ru_sparse, en_sparse, ru_vectorizer, en_vectorizer, engine):
        languages = [
            Language.RUSSIAN,
            Language.ENGLISH]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        self.ru_sparse = ru_sparse
        self.en_sparse = en_sparse
        self.ru_vectorizer = ru_vectorizer
        self.en_vectorizer = en_vectorizer
        self.engine = engine
        print('Модуль готов!')
        return

    # Поиск похожих названий, по умолчанию количество выводимых городов равно трём
    def find(self, user_query: str, number=3):
        # определение языка ввода
        if self.detector.detect_language_of(user_query).name != 'RUSSIAN':
            query_vector = en_text(user_query, table=False)
            sparse_matrix = self.en_sparse
            vectorizer = self.en_vectorizer
        else:
            query_vector = ru_text(user_query, table=False)
            sparse_matrix = self.ru_sparse
            vectorizer = self.ru_vectorizer

        # трансформация запроса
        query_vector = vectorizer.transform([query_vector]).toarray()[0]

        # создание словаря с пустыми списками, в них будут записываться результаты сравнения
        temp_dict = {}
        dist = ['cosine', 'minkowski']
        for i in dist:
            temp_dict[i] = []

        # сравнение запроса с векторами разреженной матрицы и добавление в словарь
        for i in range(sparse_matrix.shape[0]):
            temp = sparse_matrix.getrow(i).toarray()[0][:-3]
            temp_dict['cosine'].append(1 - distance.cosine(query_vector, temp))
            # 0.001 для избежания нулевых значений при полном совпадении
            temp_dict['minkowski'].append(distance.minkowski(query_vector, temp, p=3) + 0.001)

        # создание временного датафрейма с результатами сравнения
        temp_df = pd.DataFrame(
            {'id': sparse_matrix.getcol(-3).toarray().flatten(),
             'population': sparse_matrix.getcol(-2).toarray().flatten(),
             'region_id': sparse_matrix.getcol(-1).toarray().flatten()})

        # добавление во временный датафрейм результатов и их стандартизация
        for i in temp_dict.keys():
            temp_df[i] = temp_dict[i]
            if i not in ['cosine']:
                temp_df[i] = min(temp_df[i]) / temp_df[i]
            else:
                temp_df[i] = temp_df[i] / max(temp_df[i])

        # суммирование косинусного расстояния и расстояния Минковского, группировка по geonameid, так как
        # у одного и того же города может быть несколько названий на русском/английском
        temp_df['summary'] = temp_df.iloc[:, -2:].sum(axis=1)
        temp_df = temp_df.groupby('id').agg('max').sort_values(by=['summary', 'population'])[-number:]

        # присоединение к временному датафрейму названий регионов по region_id
        sql_query = f'''SELECT geonameid, name AS region
                    FROM "admin1CodesASCII"
                    WHERE geonameid IN {transform(set(temp_df['region_id']))}'''
        region = pd.read_sql_query(sql_query, con=self.engine)
        temp_df = temp_df.merge(
            how='left',
            left_on='region_id',
            right=region[['region', 'geonameid']],
            right_on='geonameid').set_index(temp_df.index)

        # присоединение названий стран
        sql_query = f'''SELECT a.geonameid, a.name, c."Country" 
                    FROM cities500 AS a
                    JOIN "countryInfo" AS c ON a."country code" = c."ISO"
                    WHERE a.geonameid IN {transform(temp_df.index)}'''
        country = pd.read_sql_query(sql_query, con=self.engine)
        country = pd.merge(
            left=country,
            right=temp_df[['region', 'cosine']],
            left_on='geonameid',
            right_index=True)\
            .sort_values(by='cosine', ascending=False)

        # вывод словарей
        return [country.to_dict('index')]
