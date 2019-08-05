#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics  
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[2]:


import warnings
warnings.filterwarnings("ignore") # Don't want to see the warnings in the notebook


# In[64]:


df = pd.read_csv (r'C:\Users\vivobook\Desktop\HeartDisease.csv')
df.head()


# In[65]:


df.info ()


# In[66]:


df.drop(['patient_id'], inplace=True, axis=1)


# In[67]:


df.head()


# In[68]:


df.isnull().any()


# In[69]:


df.dtypes


# In[70]:


df.nunique()


# In[71]:


df.describe()


# # Data Exploration

# In[72]:


print('# heart_disease_present   = {}'.format(len(df[df['heart_disease_present'] == 1])))
print('# heart_disease_not_present = {}'.format(len(df[df['heart_disease_present'] == 0])))  
print('% heart_disease_present   = {}%'.format(round(float(len(df[df['heart_disease_present'] == 1])) / len(df) * 100), 3))


# In[73]:


sns.set()
sns.countplot(x='heart_disease_present', data=df)


# In[74]:


sns.countplot(x='resting_ekg_results', data=df)


# In[75]:


sns.countplot(x='serum_cholesterol_mg_per_dl', data=df)


# In[76]:


sns.countplot(x='sex', data=df)


# In[77]:


sns.countplot(x='age', data=df)


# In[78]:


sns.countplot(x='max_heart_rate_achieved', data=df)


# In[21]:


sns.countplot(x='exercise_induced_angina', data=df)


# In[79]:


sns.countplot(x='thal', data=df)


# In[80]:


table=pd.crosstab(df.max_heart_rate_achieved, df.heart_disease_present)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(8,6), stacked=True)
plt.title('max_heart_rate_achieved vs heart_disease_present')
plt.xlabel('max_heart_rate_achieved')
plt.ylabel('chest_pain_type')
plt.show()


# In[81]:


df.groupby('thal').mean()


# In[30]:


sns.set(font_scale = 1.3)
correlation = df.corr()  
plt.figure(figsize=(10, 10))  
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='seismic') 


# In[82]:


df_sample = df.sample(frac=0.1) 
cols_to_use = ['thal','resting_blood_pressure', 'max_heart_rate_achieved', 'chest_pain_type','heart_disease_present']
sns.set(font_scale=1.5)
pplot = sns.pairplot(df_sample[cols_to_use], hue="heart_disease_present")  


# In[84]:


proj_sat = df2.groupby(['thal','chest_pain_type']).size().unstack()
proj_sat


# In[85]:


sns.heatmap(proj_sat,cmap='coolwarm',linecolor='white', linewidths=1)


# ## Bivariate and Multivariate Analysis

# ### Correlation heatmap

# In[119]:


sns.set(font_scale = 1.3)
correlation = df.corr()  
plt.figure(figsize=(10, 10))  
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='seismic') 


# In[120]:


sns.jointplot(x='chest_pain_type', y='max_heart_rate_achieved', data=df, size=8)


# In[87]:


seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)


# In[88]:


df= pd.get_dummies(df, drop_first = True) 
df.head()


# In[ ]:





# In[89]:


y = df.heart_disease_present
cols_used = ['thal_normal', 'resting_blood_pressure', 'chest_pain_type', 'resting_ekg_results',
             'serum_cholesterol_mg_per_dl', 'sex', 'age', 'max_heart_rate_achieved', 'exercise_induced_angina']
X = df[cols_used]


# # Performance with the Logistic Regression (default) model

# In[90]:


lr = LogisticRegression(penalty = 'l1', C = 1.0, random_state = seed)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print( 'Training accuracy :',(accuracy_score(y_train, lr.predict(X_train))) )
print( 'Test     accuracy :',(accuracy_score(y_test, predictions)) ,"\n")

# Classification report and confusion matrix
print(classification_report(y_test, lr.predict(X_test)))
y_pred = lr.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f', xticklabels = ["Left", "Stayed"], yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')


# # Performance with the Logistic Regression model (tuned via GridSearchCV)

# In[91]:


lr_reg = LogisticRegression(random_state=seed)

penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)

hyperparameters = dict(C=C, penalty=penalty)

kfold = model_selection.KFold(n_splits = 10, random_state = seed)
clf = GridSearchCV(lr_reg, hyperparameters, cv = kfold, verbose = 0)

best_model = clf.fit(X_train, y_train)

print('Best regularization method :', best_model.best_estimator_.get_params()['penalty'])
print('Best penalty parameter (C) :', best_model.best_estimator_.get_params()['C'])


# In[92]:


lr = LogisticRegression(penalty = 'l1', C = 60, random_state = seed)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print( 'Training accuracy :',(accuracy_score(y_train, lr.predict(X_train))) )
print( 'Test     accuracy :',(accuracy_score(y_test, predictions)) ,"\n")

print(classification_report(y_test, lr.predict(X_test)))
y_pred = lr.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')


# # SVM

# In[94]:


scalerX = preprocessing.StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)


# In[95]:


import time
param_grid = [{'kernel': ['linear'], 'C': [0.1, 1, 10]}]
kfold = model_selection.KFold(n_splits = 10, random_state=seed)
CV_svm = model_selection.GridSearchCV(estimator=SVC(random_state=seed), param_grid=param_grid, cv=kfold, n_jobs=-1)
t_start = time.clock()
CV_svm.fit(X_train, y_train)
t_end = time.clock()
print('Best parameters    :', CV_svm.best_params_, '\n')
print('Classifier used    :', CV_svm.best_estimator_,'\n')
print('Average CV accuracy:', np.mean(CV_svm.cv_results_['mean_test_score']))
print('Time elapsed in sec:', t_end-t_start)


# In[96]:


svm_model = SVC(kernel='linear', C=0.1, random_state=seed)
svm_model.fit(X_train, y_train)

print( 'Training   accuracy :',(svm_model.score(X_train, y_train)) )
print( 'Test       accuracy :',(svm_model.score(X_test, y_test)) )


# In[97]:


param_grid = [ {'kernel': ['rbf'], 'gamma': [1, 10, 50], 'C': [1, 5, 10]} ]

kfold = model_selection.KFold(n_splits = 10, random_state=seed)
CV_svm = model_selection.GridSearchCV(estimator=SVC(random_state=seed), param_grid=param_grid, cv=kfold)
t_start = time.clock()
CV_svm.fit(X_train, y_train)
t_end = time.clock()
print('Best parameters    :', CV_svm.best_params_, '\n')
print('Classifier used    :', CV_svm.best_estimator_,'\n')
print('Average CV accuracy:', np.mean(CV_svm.cv_results_['mean_test_score']))
print('Time elapsed in sec:', t_end-t_start)


# In[98]:


svm_model = SVC(kernel='rbf', C=5, gamma=10, random_state=seed)
svm_model.fit(X_train, y_train)

print( 'Training   accuracy :',(svm_model.score(X_train, y_train)) )
print( 'Test       accuracy :',(svm_model.score(X_test, y_test)) )


# In[99]:


print(classification_report(y_test, svm_model.predict(X_test)))
y_pred = svm_model.predict(X_test)
svm_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(svm_cm, annot=True, fmt='.2f', xticklabels = ["Left", "Stayed"], yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')


# #  Random Forest

# In[100]:


y = np.array(df['heart_disease_present'])
X = np.array(df.drop(['heart_disease_present'], axis=1))
print('X and y:', X.shape, y.shape)


# In[101]:


from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#
seed = 40 #for reproducibility
#--- Training and test portions of data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
print('Training and Test sizes:',len(X_train),len(X_test) ) #Dimension of the training and test sets,


# # Outcome of the default model:

# In[102]:


from sklearn.ensemble import RandomForestClassifier as RF
Clf_def = RF(random_state=seed)
kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_results = model_selection.cross_val_score(Clf_def, X_train, y_train, cv=kfold, scoring='accuracy')
Clf_def.fit(X_train, y_train)
predictions = Clf_def.predict(X_test)
print('Classifier used:\n', Clf_def, '\n')
print( 'Training accuracy       :',(accuracy_score(y_train, Clf_def.predict(X_train))) )
print( 'Classification accuracy :',(accuracy_score(y_test, predictions)))


# # Search for a better model via hyperparameter tuning:

# In[103]:


seed = 40
clf = RF(random_state=seed)
n_estimators = [20, 50, 100, 200, 400, 600, 800, 1000, 1500]
cv_scores = [] ; train_scores = [] ; exectime = []
print('num-of-trees  CV-mean-score  Train-score')

for i in n_estimators:
    clf.set_params(n_estimators = i)
    
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean())
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_train, clf.predict(X_train))
    train_scores.append(acc)
   
    print('%4d          %5.3f          %5.3f' %(i, scores.mean(), acc ))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(n_estimators, cv_scores, label='Validation Acc')
plt.plot(n_estimators, train_scores, label='Training Acc')
plt.xlabel('Number of trees')
plt.ylabel('Validation score')
plt.legend() ; plt.ylim(0.9,1.05)
plt.subplot(122)
plt.xlabel('Number of trees')
plt.legend()
plt.show()


# Maximum tree depth vs Validation accuracy  with num_trees = 100

# In[104]:


seed = 40
num_tree = 100
clf = RF(n_estimators = num_tree, random_state=seed)
max_depth = [2, 4, 6, 8, 10, 12, 15, 20]
cv_scores = [] ; train_scores = [] ; exectime = []
print('Number of trees used:', num_tree)
print('max-depth  CV-mean-score  Train-score')

for i in max_depth:
    clf.set_params(max_depth = i)
   
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean())
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_train, clf.predict(X_train))
    train_scores.append(acc)
   
    print('%2d         %5.3f          %5.3f' %(i, scores.mean(), acc ))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(max_depth, cv_scores, label='Validation Acc')
plt.plot(max_depth, train_scores, label='Training Acc')
plt.xlabel('Max depth')
plt.ylabel('Validation score')
plt.legend() ; plt.ylim(0.9,1.05)
plt.subplot(122)

plt.xlabel('Max depth')

plt.legend()
plt.show()


# Maximum features vs Validation accuracy  with num_trees = 100 and max_depth = 4

# In[105]:


seed = 40
num_tree = 100
maxdepth = 3
clf = RF(n_estimators = num_tree, max_depth=maxdepth, random_state=seed)
max_features = [2, 4, 6, 8, 10]
cv_scores = [] ; train_scores = [] ; exectime = []
print('Number of trees used:', num_tree, '    Max depth:', maxdepth)
print('max-features  CV-mean-score  Train-score')

for i in max_features:
    clf.set_params(max_features = i)
  
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean())
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_train, clf.predict(X_train))
    train_scores.append(acc)
   
    print('%2d            %5.3f          %5.3f' %(i, scores.mean(), acc ))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(max_features, cv_scores, label='Validation Acc')
plt.plot(max_features, train_scores, label='Training Acc')
plt.xlabel('Max features')
plt.ylabel('Validation score')
plt.legend() ; plt.ylim(0.9,1.05)
plt.subplot(122)

plt.xlabel('Max features')

plt.legend()
plt.show()


# # RandomizedSearchCV

# In[106]:


criterion = ['entropy', 'gini']
max_features = [3, 4, 5, 10]
max_depth = [5, 10, 15, 20, 25, 30]
min_samples_split = [2, 3, 5, 10, 12, 15]
min_samples_leaf = [1, 2, 3, 4, 5]
n_estimators = [50, 100, 200]
random_grid = {'criterion'        : criterion,
               'max_features'     : max_features,
               'max_depth'        : max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
               'n_estimators'     : n_estimators}
print(random_grid)


# In[107]:


import time
import numpy as np
import pandas as pd
import seaborn as sns
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[108]:


rfc = RF(random_state = seed)
kfolds = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
randomGrid = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, cv = kfolds, 
                                random_state=seed, n_jobs = -1)
randomGrid.fit(X_train, y_train)
print()


# #Best hyperparameters

# In[110]:


randomGrid.best_params_


# In[111]:


# All parameters used in the classifier
randomGrid.best_estimator_


# In[112]:


print('Avg CV scores in each fold:\n', randomGrid.cv_results_['mean_test_score'], '\n')
print('Best CV score in all folds:\n', randomGrid.best_score_, '\n')
print('Average CV score:\n', np.mean(randomGrid.cv_results_['mean_test_score']))


# Using GridSearchCV

# In[113]:


param_grid = {"criterion"        : ['gini'],
              "min_samples_split": [5,8,10,12,15],
              "max_features"     : [3,4, 5],
              "max_depth"        : [15,18,20,22,25],
              "min_samples_leaf" : [1,2],
              "n_estimators"     : [95,100,105]
              }

kfolds = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
t_start = time.clock()
grid_search = GridSearchCV(estimator = RF(random_state = seed), param_grid = param_grid, 
                           cv = kfolds, n_jobs = -1)
grid_search.fit(X_train, y_train)
t_end = time.clock()


# In[114]:


print('Time elapsed :', t_end-t_start, 'sec\n')
print('Best parameters:\n', grid_search.best_params_,'\n')
print('Average CV accuracy:', np.mean(grid_search.cv_results_['mean_test_score']))


# # Train with the chosen `GridSearchCV` hyperparameters and compute the test accuracy

# In[115]:


Clf = RF(criterion = 'gini', n_estimators = 100, max_depth = 15, min_samples_leaf = 2, 
         min_samples_split = 12, max_features = 5, random_state = seed)
Clf.fit(X_train, y_train)
predictions = Clf.predict(X_test)
print( 'RF Training accuracy       :',(accuracy_score(y_train, Clf.predict(X_train))) )
print( 'RF Classification accuracy :',(accuracy_score(y_test, predictions)) ,"\n")


# # Confusion matrix and classification report

# In[116]:


def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted, [1,0] )
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = ["1", "0"] , 
                yticklabels = ["1", "0"] )
    plt.ylabel('ACTUAL')
    plt.xlabel('PREDICTED')
    plt.show()

print( "Confusion Matrix:" )
draw_cm( y_test, predictions ) #function defined previously

print( "Classification Report:" )
print( classification_report(y_test, predictions) )


# # Feature importance

# In[117]:


importances = Clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in Clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

feature_names = df.columns.tolist()
# Print the feature ranking
print("Feature ranking:")
feature_list = []
for f in range(X.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    feature_list.append(feature_names[indices[f]])
    print("%2d. %15s %2d (%f)" % (f + 1, feature_names[indices[f]], indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")

#plt.xticks(range(X.shape[1]), indices, rotation='vertical')
plt.xticks(range(X.shape[1]), feature_list, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()


# In[142]:


df_test = pd.read_csv (r'C:\Users\vivobook\Desktop\HeartDisease-test.csv')
df_test.head()


# In[ ]:





# In[136]:


df= pd.get_dummies(df, drop_first = True) 
df.head()


# In[138]:


features = ['resting_blood_pressure','chest_pain_type','resting_ekg_results','serum_cholesterol_mg_per_dl','sex','age','max_heart_rate_achieved','exercise_induced_angina','thal_normal','thal_reversible_defect']


# In[127]:





# In[139]:


predictions = clf.predict(df_test[features])


# In[129]:


predictions


# In[ ]:


submission = pd.DataFrame({'patient_id':df_test['patient_id'],'heart_disease_present':predictions})
submission.head()


# In[ ]:




