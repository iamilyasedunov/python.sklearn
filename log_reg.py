import pandas as pd
import numpy as np
from scipy.spatial import distance 
from sklearn.metrics import roc_auc_score

data = pd.read_csv('data-logistic.csv', names = ['y[i]', 'x1[i]', 'x2[i]'] )
x1 = data['x1[i]']
x2 = data['x2[i]']
y = data['y[i]']

k, w1, w2, w1_pr, w2_pr, w1_new, w2_new = 0.1, 0, 0, 0, 0, 0, 0
eps = 0.00001
C = 10
l = len(y) #205
for r in range(10000):
    S1 = 0
    S2 = 0
    for i in range(l):
        S1 = S1 + y[i]*x1[i]*(1 - 1/(1 + np.exp(-y[i] * (w1*x1[i] + w2*x2[i])))) 
        S2 = S2 + y[i]*x2[i]*(1 - 1/(1 + np.exp(-y[i] * (w1*x1[i] + w2*x2[i]))))     
    S1 = k * 1/l * S1 - k*C*w1 #для L-2 с регуляризацией
    S2 = k * 1/l * S2 - k*C*w2 
    
    #S1 = k * 1/l * S1          #для L-2 без регуляризации
    #S2 = k * 1/l * S2 
    
    w1 = w1 + S1
    w2 = w2 + S2
    
    eps1 = np.linalg.norm(w1-w1_pr) #Эвклидово расстояние
    eps2 = np.linalg.norm(w2-w2_pr)
    print('iter = ',r, eps1, eps2,'w1 = ', w1, 'w2 = ', w2)   
    
    if (eps1 <= eps or eps2 <= eps): #Условие сходимости: Эвклидово расстояние между векторами на соседних итерациях не больше 0.00001
        print('wrong on',r,'w1 = ', w1, 'w2 = ', w2,
              )
        break    
    w1_pr = w1
    w2_pr = w2
    
#wrong on 227 w1 =  0.2876729392144029 w2 =  0.09211154846441766 результат без регуляризации
#wrong on 7 w1 =  0.028558754546234223 w2 =  0.024780137249735563 результат с регуляризацией

y_noreg = []                         #строим предсказание по весам, которые вывчислили без регуляризации
w1 =  0.2876729392144029 
w2 =  0.09211154846441766
for i in range(l):
    y_noreg.append(1 / (1 + np.exp(-w1*x1[i] - w2*x2[i]))) #sigmoid

y_reg = []
w1 = 0.028558754546234223
w2 = 0.024780137249735563
for i in range(l):
    y_reg.append(1 / (1 + np.exp(-w1*x1[i] - w2*x2[i])))

print(roc_auc_score(y, y_noreg))
print(roc_auc_score(y, y_reg))
