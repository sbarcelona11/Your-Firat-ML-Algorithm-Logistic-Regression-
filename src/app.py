import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", delimiter=';')
df

df.shape
print(f'The dataset have {df.shape[0]} rows and {df.shape[1]} columns.')
df.head(10)
df.info()

df_to_process = df.copy()
df_to_process.describe()
print(f'The dataset have {df_to_process.duplicated().sum()} duplicated rows.')

df_to_process.drop_duplicates(inplace=True)
df_to_process.shape

columns = ['job', 'marital','education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']

for column in columns:
    print (f'Value count for column - {column}')
    print(df[column].value_counts())
    print('\n\n')

df_to_process = df_to_process.replace('unknown', np.nan)
df_to_process

# for categorical variables replace NaN with the most frequent value
for var in df_to_process.columns[df_to_process.dtypes == 'object']:
    df_to_process[var] = df_to_process[var].fillna(df_to_process[var].mode()[0])

# for numerical variables replace NaN with the most frequent value
for var in df_to_process.columns[df_to_process.dtypes == 'int64']:
    df_to_process[var] = df_to_process[var].fillna(df_to_process[var].mean())

df_to_process.info()
df_to_process['age'].describe()

fig = px.box(df_to_process, y="age")
fig.show()

def remove_outliers(df_to_process, column):
    IQR = df_to_process[column].describe()['75%'] - df_to_process[column].describe()['25%']
    upper = df_to_process[column].describe()['75%'] + 1.5 * IQR
    print('The upper bound for suspected outliers for {} feature is {}.'.format(column, upper))

    # remove observations above this value
    df_to_process = df_to_process.drop(df_to_process[df_to_process[column] > upper].index)
    fig = px.box(df_to_process, y=column)
    fig.show()      

columns = ['age', 'duration', 'campaign']
for column in columns:
    remove_outliers(df_to_process, column)

fig = px.box(df_to_process, y=df_to_process.loc[df["pdays"] < 999, "pdays"])
fig.show()

df_to_process['education'] = df_to_process['education'].replace({'basic.9y': 'middle_school', 'basic.6y': 'middle_school', 'basic.4y': 'middle_school'})
df_to_process['education'] 

df_to_process['age_bins'] = pd.cut(x=df_to_process['age'], bins=[10,20,30,40,50,60,70,80,90,100])

df_to_process[['age_bins','age']].head()

def countplot_features(feature, title):
    sns.set_palette("Set2")
    plot=sns.countplot(x=feature,data=df_to_process).set(title=title)
    plt.show()


countplot_features("age_bins", "Age bins")

df_to_process = df_to_process.drop('age',axis=1)
df_to_process
values = df_to_process['y'].value_counts()
print('% of clients that did not suscribe a term desposit:',round((values[0] / len(df_to_process)) *100, 2))
print('% of clients that suscribed a term desposit:',round((values[1] / len(df_to_process)) *100, 2))
print('Umbalanced data')

df_to_process = pd.get_dummies(df_to_process, columns=['y','age_bins','job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'], drop_first=True)
df_to_process.describe()

df_to_process = df_to_process.drop(['duration','pdays'], axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaler = scaler.fit(df_to_process[['campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
df_to_process[['campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = df_scaler.transform(df_to_process[['campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])

df_processed = df_to_process.copy()
df_processed.to_csv('../data/processed/df_processed.csv')

# # Logistic Regresion
df_raw = pd.read_csv('../data/processed/df_processed.csv', sep=',')
df_raw.columns

features = ['campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'age_bins_(20, 30]', 'age_bins_(30, 40]', 'age_bins_(40, 50]', 'age_bins_(50, 60]', 'age_bins_(60, 70]', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'marital_married', 'marital_single', 'education_middle_school', 'education_professional.course', 'education_university.degree', 'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone', 'month_aug','poutcome_nonexistent', 'poutcome_success']

X = df_raw[features]
y = df_raw['y_yes']

y.value_counts(normalize=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train) 

filename = '../models/original_model.sav'
pickle.dump(model, open(filename, 'wb'))

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

cross_tab = pd.crosstab(np.array(y_test), np.array(y_pred), rownames=['actual'], colnames=['predicted'])
print(f'{cross_tab[0][1]+ cross_tab[1][1]} Wrong predictions on observations.')

df_pred = pd.DataFrame({'Actual': np.array(y_test), 'Prediction': np.array(y_pred)})
df_pred

print(classification_report(df_pred['Actual'], df_pred['Prediction']))

# optimize model
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2'] # regularización
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

optimized_model = LogisticRegression(C= 0.1, penalty='l2', solver= 'lbfgs')

optimized_model.fit(X_train, y_train) 

y_pred = optimized_model.predict(X_test)
y_pred

accuracy_score(y_pred, y_test)

cross_tab = pd.crosstab(np.array(y_test), np.array(y_pred), rownames=['actual'], colnames=['predicted'])
print(f'{cross_tab[0][1]+ cross_tab[1][1]} Wrong predictions on observations.')

df_pred = pd.DataFrame({'Actual': np.array(y_test), 'Prediction': np.array(y_pred)})
df_pred

print(classification_report(df_pred['Actual'], df_pred['Prediction']))

filename = '../models/optimized_model.sav'
pickle.dump(optimized_model, open(filename, 'wb'))


