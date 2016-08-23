#the columns had data leakage issue, redundant info,missing value,null value already removed
#converted text varaibles to categorical variables, then to dummy variables already
#our goal for this model is to predict loan_status

import pandas as pd
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, KFold

filtered_loans=pd.read_csv('filtered_loans.csv',delimiter=',')
filtered_loans=filtered_loans.drop("pymnt_plan",axis=1)
#print(filtered_loans.info())


#using logistic regression
#penalty of ten if missclassifying 0 and penalty 1 if missclassifying 1
penalty={ 0:10, 1:1}
lr = LogisticRegression(class_weight=penalty)

#error metric - loan_status is 0s and 1s


# Predict that all loans will be paid off on time- fill in predictions array with all ones
predictions = pd.Series(numpy.ones(filtered_loans.shape[0]))

#find number of true negative
tn_filter=(predictions==0)&(filtered_loans['loan_status']==0)
tn = len(predictions[tn_filter])


#find number of true positive
tp_filter=(predictions==1)&(filtered_loans['loan_status']==1)
tp = len(predictions[tp_filter])


#find number of false positive
fp_filter=(predictions==1)&(filtered_loans['loan_status']==0)
fp = len(predictions[fp_filter])


#4.find number of false negative
fn_filter=(predictions==0)&(filtered_loans['loan_status']==1)
fn = len(predictions[fn_filter])

#to reduce class imbalance
#conservative assumption: focus more on false positive than false negative
#optimize for high recall(TP) and low fall-out(FP)

#metrix:using tpr and fpr
tpr=tp/(tp+fn) #correctly identify the true number
fpr=fp/(fp+tn) #probability of falsely reject the null hyoothesis (type I)
#print(tpr)
#print(fpr)

#remove target column

features=filtered_loans.drop("loan_status",axis=1)

target=filtered_loans["loan_status"]

#use fit mehthod
lr.fit(features,target)
predictions=lr.predict(features)

#use cross validation to generate predictions

kf = KFold(features.shape[0], random_state=1)

predictions = cross_val_predict(lr, features, target, cv=kf)

#convert preditions to pandas series
#-if does not do this,FPR and TPR won't work
predictions = pd.Series(predictions)


# Predict that all loans will be paid off on time- fill in predictions array with all ones
predictions = pd.Series(numpy.ones(filtered_loans.shape[0]))

#find number of true negative
tn_filter=(predictions==0)&(filtered_loans['loan_status']==0)
tn = len(predictions[tn_filter])


#find number of true positive
tp_filter=(predictions==1)&(filtered_loans['loan_status']==1)
tp = len(predictions[tp_filter])


#find number of false positive
fp_filter=(predictions==1)&(filtered_loans['loan_status']==0)
fp = len(predictions[fp_filter])


#4.find number of false negative
fn_filter=(predictions==0)&(filtered_loans['loan_status']==1)
fn = len(predictions[fn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)









