#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from sklearn.metrics import precision_score, accuracy_score

y_pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 1]
y = [1,0,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1]
y = [1,1,1]
y_pred = [0, 0, 1]
precision = accuracy_score(y, y_pred)
print(precision)