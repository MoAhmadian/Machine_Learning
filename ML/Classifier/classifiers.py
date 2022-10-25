"""
All Classifiers in Machine Learning - Step by Step
1. Decision Tree
2. Random Forest
3. Naive Bayes
4. Gradient Boosting
5. K-Nearest Neighbor
6. Logistic Regression
7. Support Vector Machine(SVM)
"""
#import the neccessary modules
import pandas as pd
import numpy as np
import seaborn as sb
#Read the dataset into a dataframe
dataPath = r'\data\titanic.csv'
df = pd.read_csv(dataPath, sep='\t', engine='python')

#Drop some columns which is not relevant to the analysis (they are not numeric)
cols_to_drop = ['Name', 'Ticket', 'Cabin']
df=df.drop(cols_to_drop, axis=1)
sb.heatmap(df.isnull())
import matplotlib.pyplot as plt
plt.show()

# To replace missirng values with interpolated values
df['Age'] = df['Age'].interpolate()
sb.heatmap(df.isnull())
plt.show()

# Drop rows with missing values
df = df.dropna()

# To do that create dumny columns for the columns you want to convert, concatenate it with the dataframe, then drop the existinc columns
EmbarkedColumnDummy = pd.get_dummies(df['Embarked'])
SexColumnDummy = pd.get_dummies(df['Sex'])
df= pd.concat((df, EmbarkedColumnDummy, SexColumnDummy), axis=1)
#Check that the columns were concatenated

df = df.drop(['Sex', 'Embarked'], axis=1)

print(df.info())
print(df.head())

# Seperate the dat aframe ibto x and y data
X = df.values
y = df['Survived'].values
# Detete the Survived column from X
X = np.delete(X, 1, axis=1)
# Split the dataset into 70% Training and 30% Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# Build Decision Tree Classifier
#Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. 
# Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth 
# of the tree are necessary to avoid this problem.
from sklearn import tree
Dt_clf = tree.DecisionTreeClassifier(max_depth=5)
Dt_clf.fit(x_train, y_train) # Train
y_pred = Dt_clf.predict(x_test)
print(f'decision tree score:{Dt_clf.score(x_test, y_test)}') # Make prediction
# predict for a given observation
print(f'Prediction for a 38 years old female by DT : {Dt_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female
from sklearn.metrics import confusion_matrix
print(f'confusion matrix of DT: {confusion_matrix(y_test, y_pred)}')

# Build Random Forest Classifier
from sklearn import ensemble
# n_estimators : number of trees involved in max voteing process
Rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
Rf_clf.fit(x_train, y_train)
print(f'random forest score: {Rf_clf.score(x_test, y_test)}')
y_pred = Rf_clf.predict(x_test)
print(f'confusion matrix of RF: {confusion_matrix(y_test, y_pred)}')
# predict for a given observation
print(f'Prediction for a 38 years old female RF: {Rf_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female

# Build Gradient Boosting Classifier
Gb_clf = ensemble.GradientBoostingClassifier()
Gb_clf.fit(x_train, y_train)
print(f'Gradient Boosting score: {Gb_clf.score(x_test, y_test)}')
# We can tune GB to imporve a little bit
Gb_clf = ensemble.GradientBoostingClassifier(n_estimators = 50)
Gb_clf.fit(x_train, y_train)
print(f'Tuned Gradient Boosting score: {Gb_clf.score(x_test, y_test)}')
y_pred = Gb_clf.predict(x_test)
print(f'confusion matrix of GB: {confusion_matrix(y_test, y_pred)}')
print(f'Prediction for a 38 years old female GB: {Gb_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female

#Build a Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
Nb_clf =  GaussianNB()
Nb_clf.fit(x_train, y_train)
Nb_clf.score(x_test, y_test)
print(f'Naive bayes score: {Nb_clf.score(x_test, y_test)}')
y_pred = Nb_clf.predict(x_test)
print(f'confusion matrix of NB: {confusion_matrix(y_test, y_pred)}')
print(f'Prediction for a 38 years old female by NB: {Nb_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female

#Build a K-Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
Knn_clf= KNeighborsClassifier(n_neighbors=3)
Knn_clf.fit(x_train, y_train)
print(f'K-Nearest Neighbor Classifier score: {Knn_clf.score(x_test, y_test)}')
y_pred = Knn_clf.predict(x_test)
print(f'confusion matrix of Knn: {confusion_matrix(y_test, y_pred)}')
print(f'Prediction for a 38 years old female by KNN: {Knn_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female

# Build a Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
Lr_clf = LogisticRegression()
Lr_clf.fit(x_train, y_train)
Lr_clf.score(x_test, y_test)
print(f'Logistic Regression score: {Lr_clf.score(x_test, y_test)}')
y_pred = Lr_clf.predict(x_test)
print(f'confusion matrix of LR: {confusion_matrix(y_test, y_pred)}')
print(f'Prediction for a 38 years old female by LR: {Lr_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female

#Build an SVM Classifier
from sklearn.svm import SVC
Sv_clf = SVC(probability=True, kernel='linear')# let's SVC choose default kernel instead of kernel='linear'
Sv_clf.fit(x_test, y_test)
print(f'SVM score: {Sv_clf.score(x_test, y_test)}')
y_pred = Sv_clf.predict(x_test)
print(f'confusion matrix of SV: {confusion_matrix(y_test, y_pred)}')
print(f'Prediction for a 38 years old female by SVM: {Sv_clf.predict([[1000,3,38,0,0,73.5,0,0,1,1,0]])}') # 38 years old female


# Prediction Probabilities
r_probs = [0 for _ in range(len(y_test))]
rf_probs = Rf_clf.predict_proba(x_test)
nb_probs = Nb_clf.predict_proba(x_test)
dt_probs = Dt_clf.predict_proba(x_test)
gb_probs = Gb_clf.predict_proba(x_test)
knn_probs = Knn_clf.predict_proba(x_test)
lr_probs = Lr_clf.predict_proba(x_test)
sv_probs = Sv_clf.predict_proba(x_test)

#Probabilities for the positive outeome is kept
rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]
dt_probs = dt_probs[:, 1]
gb_probs = gb_probs[:, 1]
knn_probs = knn_probs[:, 1]
lr_probs = lr_probs[:, 1]
sv_probs = sv_probs[:, 1]

#Compute the AUROCc Values
from sklearn.metrics import roc_curve, roc_auc_score
r_auc   = roc_auc_score(y_test, r_probs)
rf_auc  = roc_auc_score(y_test, rf_probs)
nb_auc  = roc_auc_score(y_test, nb_probs)
dt_auc  = roc_auc_score(y_test, dt_probs)
gb_auc  = roc_auc_score(y_test, gb_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
lr_auc  = roc_auc_score(y_test, lr_probs)
sv_auc  = roc_auc_score(y_test, sv_probs)

#Display the AUROc Scores
print("Random Prediction : AUROC %.3f" %(r_auc))
print("Random Forest: AUROC = %.3f" %(rf_auc))
print("Naive Bayes: AUROC = %.3f" %(nb_auc))
print("Decistion Tree Prediction: AUROC %.3f" %(dt_auc))
print("Gradient Boosting Prediction : AUROC= %.3f" % (gb_auc ))
print("KNearest Neighbors Prediction: AUROC= %.3f" % (knn_auc))
print("Logistic Regresssion : AUROC %.3f" % (lr_auc ))
print("Support vector Machine: AUROC %.3f" %(sv_auc))

#Calculate the ROC Curve
r_fpr, r_tpr,_   = roc_curve(y_test, r_probs)
rf_fpr, rf_tpr,_ = roc_curve(y_test, rf_probs)
nb_fpr, nb_tpr,_ = roc_curve(y_test, nb_probs)
dt_fpr, dt_tpr,_ = roc_curve(y_test, dt_probs)
gb_fpr, gb_tpr,_ = roc_curve(y_test, gb_probs)
knn_fpr, knn_tpr,_ = roc_curve(y_test, knn_probs)
lr_fpr, lr_tpr,_ = roc_curve(y_test, lr_probs)
sv_fpr, sv_tpr,_ = roc_curve(y_test, sv_probs)


#Plot the ROC Curve
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plt.plot(r_fpr, r_tpr, linestyle='solid', label=' Random prediction (AUC = %0.3f)' %r_auc)
plt.plot(rf_fpr, rf_tpr, linestyle='--', label=' Random Forest (AUC= %0.3f)'%rf_auc)
plt.plot(nb_fpr, nb_tpr, linestyle='--', label='Naive Bayes (AUC = %0.3f)' %nb_auc)
plt.plot(dt_fpr, dt_tpr, linestyle='--', label='Decision Tree (AUC = %0.3f)' %dt_auc)
plt.plot(gb_fpr, gb_tpr, linestyle='--', label='Gradient Boosting (AUC = %0.3f)' %gb_auc)
plt.plot(knn_fpr, knn_tpr, linestyle='--', label='K-Nearest Neighbors (AUC = %0.3f)' %knn_auc)
plt.plot(lr_fpr, lr_tpr, linestyle='--', label= 'LogistiC Regression (AUC = %0.3f)' %lr_auc)
plt.plot(sv_fpr, sv_tpr, linestyle='dotted', label='Support Vector Machine (AUC = %0.3f)' %sv_auc)
#Title
plt.title(' ROC Plot')
#Axis Labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#Show Legend
plt.legend()
plt.show()

import tensorflow as tf






