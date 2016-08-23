#cleaned the columns had data leakage issue, redundant info ,missing value,null value
#converted text varaibles to categorical variables, then to dummy variables

import pandas as pd
import os

#do not read row one, low_memory=False insetad of specify dtype option
#drop the columns that has less than half of column full
loans=pd.read_csv('Loan.csv',skiprows=1,low_memory=False)
half_count=len(loans)*0.5
loans=loans.drop(['desc', 'url'],axis=1) 
loans=loans.dropna(thresh=half_count, axis=1)
loans=loans.drop_duplicates()

#test
#print(loans.head())
#print a whole row
#print(loans.ix[1])
#print(loans.shape)


#Break columns into groups and drop unnecessary columns in groups
loans=loans.drop(['id','member_id','funded_amnt','funded_amnt_inv','grade','sub_grade','emp_title','issue_d'],axis=1)
loans=loans.drop(['zip_code','out_prncp','out_prncp_inv','total_pymnt'],axis=1)
loans=loans.drop(['total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt'],axis=1)

#print(loans.ix[1])
#print(loans.shape)

#for whether or nor the loan is paid or not paid, we use loan_status
#figure out the unique values inside loan_status column
#print(loans["loan_status"].value_counts())

#only predict the loans will be fully paid will or be charged off: using binary classfication
#remove the rows with other values
loans=loans[(loans.loan_status=="Fully Paid")|(loans.loan_status=="Charged Off")]
#print(loans["loan_status"].value_counts())
#use the dataframe method (dictionary) to replace these values with zero and 1
replace_dic={
    "loan_status":{
        "Fully Paid":1,
        "Charged Off":0
        }
}
loans=loans.replace(replace_dic)
#print(loans["loan_status"].value_counts())

#remove columns with one single value except null
drop_columns=[]
for col in loans.columns:
    #drop null value in each single row for that column
    non_null=loans[col].dropna()
    #print(non_null.shape)
    #non-null unique value
    new_value=non_null.unique()
    #get the number of unique value
    uni_len=len(new_value)
    #append columns to drop list if only have one non null value
    if uni_len<=1:drop_columns.append(col)
    
loans=loans.drop(drop_columns,axis=1)
#print(drop_columns)
#print(loans.shape)
#print(loans.ix[1])

#return number of null value in the dataframe
null_counts={}
for col in loans.columns:
    count=loans[col].isnull().sum()
    if count>0: null_counts[col]=count
#print the key value pairs of null value
#print(null_counts)    

#drop the pub_rec_bankruptcies column from loans (too many nulls)
#axis=1 drop columns, axis=0 drop rows
loans=loans.drop("pub_rec_bankruptcies",axis=1)

#remove rows from other columns with null value
loans=loans.dropna(axis=0)
#print(loans.isnull().sum())
#print(loans.head())

#select columns with data loan datatypes
textData=loans.select_dtypes(include=["object"])
#print(textData.ix[1])

#for categorical columns

cols = ['home_ownership', 'verification_status',
        'emp_length', 'term', 'addr_state']

#display unique value counts:
unique_counts={}
for col in cols:
    count=textData[col].unique()
    unique_counts[col]=count
#print(unique_counts)
    
#display unique vaoue for purpose & title
purpose_value=textData['purpose'].value_counts()
title_value=textData['title'].value_counts()
#print(purpose_value)
#print(title_value)

#remove the last_credit_pull_d, addr_state, title, and earliest_cr_line
lists=['last_credit_pull_d', 'addr_state','title','earliest_cr_line']
loans=loans.drop(lists,axis=1)

#convert int_rate & revol_util columns to float:
loans['int_rate']=loans['int_rate'].str.rstrip('%').astype(float)
#print(loans['int_rate'].head())
loans['revol_util']=loans['revol_util'].str.rstrip('%').astype(float)

#clean emp_length
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans=loans.replace(mapping_dict)
#print(loans['emp_length'].head())

#make home_ownership, verification_status, title, and term into dummy variables
covert_list=['home_ownership', 'verification_status','purpose','term']
#convert into category first
for item in covert_list:
    loans[item]=loans[item].astype('category')
#covert them into dummy variables
dummy_df = pd.get_dummies(loans[covert_list])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(covert_list, axis=1)
#print(loans.ix[1])

path_d = 'C:\\Users\\Best Trader\\Desktop'
loans.to_csv(os.path.join(path_d,'fitered_loans.csv'))


