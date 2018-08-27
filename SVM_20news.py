import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn import metrics 

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target

TF_IDА = TfidfVectorizer()                               #строим числовое представление для текстовых данных


X_scaled = TF_IDА.fit_transform(X)                       #с помощью метода fit_transform() оуществляем преобразование тестовой выборки


feature_mapping = TF_IDА.get_feature_names()             #массив объектов(слов), веса этих объектов находятся в clf.coef_

grid = {'C': np.power(10.0, np.arange(-5, 6))}          #диапазон в котором ищем C
cv = KFold(n_splits=5, random_state=241)                #кросс-валидация по пяти блокам
clf = SVC(C = 10, kernel='linear',random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)#осуществим подбор оптимального параметра C с помощью функции GridSearch
                                                        #он будет равен 10, поэтому сразу подставим значение в SVC
gs.fit(X_scaled, y)                                     
print(gs.best_params_)                                 #выводим лучший из параметров

clf = clf.fit(X_scaled, y)                              #обучаем SVC

y_pred = clf.predict(X_scaled)                          #оценка качества алгоритма
accuracy  = metrics.accuracy_score(y, y_pred)
precision = metrics.precision_score(y, y_pred)
recall = metrics.recall_score(y, y_pred)
print('accuracy = ',accuracy, 
      'precision =  ',precision,
      'recall = ', recall)                              
                                                        #отнесем заданный текст к одному из двух классов, пример текста: биография Стивена Хокинга

Sample = ['Stephen Hawking (January 8, 1942 to March 14, 2018) was a British scientist, professor and author who performed groundbreaking work in physics and cosmology, and whose books helped to make science accessible to everyone. At age 21, while studying cosmology at the University of Cambridge, he was diagnosed with amyotrophic lateral sclerosis (ALS). Part of his life story was depicted in the 2014 film The Theory of Everything.']
X_new = [element.lower() for element in Sample]
X_new_tfidf = TF_IDF.transform(X_new)
predicted = clf.predict(X_new_tfidf)

print(predicted)

#далее: извлекаем 10 слов с наибольшим абсолютным значением веса
#сортируем их в лексиграфическом порядке

ind = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
print(ind)

words = []
for i in range(10):
    words.append(feature_mapping[ind[i]])
sorted(words)
#[17318  3868 14050  4420  8481 14543 12268  3864 10085 1
