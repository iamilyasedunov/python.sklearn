import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB

newsgroups_train = datasets.fetch_20newsgroups(
                    subset='train', 
                    categories=['alt.atheism', 'sci.space','talk.politics.guns','soc.religion.christian']
             )

newsgroups_test = datasets.fetch_20newsgroups(
                    subset='test', 
                    categories=['alt.atheism', 'sci.space','talk.politics.guns','soc.religion.christian']
             )

X = newsgroups_train.data
y = newsgroups_train.target

TF_IDF = TfidfVectorizer()                               #строим числовое представление для текстовых данных

X_scaled = TF_IDF.fit_transform(X)                       #с помощью метода fit_transform() оуществляем преобразование тестовой выборки

feature_mapping = TF_IDF.get_feature_names()             #массив объектов(слов), веса этих объектов находятся в clf.coef_

clf = MultinomialNB()
  
clf = clf.fit(X_scaled, y)                               #обучаем MultinomialNB
                                                                                  
X_test = newsgroups_test.data                            #Далее: оценка точности модели
X_test = TF_IDF.transform(X_test)
y_test = clf.predict(X_test)
score = clf.score(X_test, y_test)

print(score)

metrics.confusion_matrix(newsgroups_test.target, y_test) #метрика ошибок, показывающая какие классы clf путает 
