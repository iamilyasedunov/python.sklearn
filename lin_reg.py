import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv') 

data_train['FullDescription'] = data_train['FullDescription'].str.lower()         #переводим весь текст в нижни регистр
data_train['LocationNormalized'] = data_train['LocationNormalized'].str.lower()

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)  #всё кроме букв и цифр заменяется пробелами, 
data_train['LocationNormalized'] = data_train['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True) #для простоты деления текста на слова

TF_IDF = TfidfVectorizer()
TfidfVectorizer(data_train['FullDescription'], min_df = 5)        # Оставляем только те слова, которые встречаются хотя бы в 5 объектах
TfidfVectorizer(data_train['LocationNormalized'], min_df = 5)
TfidfVectorizer(data_train['ContractTime'], min_df = 5)

data_train['LocationNormalized'].fillna('nan', inplace=True)      #пропуски в данных столбцах = nan
data_train['ContractTime'].fillna('nan', inplace=True)

tfidf = TF_IDF.fit_transform(data_train['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records')) #one-hot кодирование для
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))       #LocationNormalized и ContractTime
X_train = hstack([tfidf, X_train_categ])                #Объединяем все полученные признаки в одну матрицу "объекты-признаки"

y_train = data_train['SalaryNormalized']                #Целевая переменная
clf = Ridge(alpha = 1, random_state = 241)              #Обучение гребневой регрессии
clf.fit(X_train, y_train)

tfidf_test =  TF_IDF.transform(data_test['FullDescription']) #Тестовые данные для проверки в системе
X_test = hstack([tfidf_test, X_test_categ])
predicted = clf.predict(X_test)                          #SalaryNormalized для дыух тестовых наборов(строк)
predicted
