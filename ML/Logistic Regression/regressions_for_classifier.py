from matplotlib.cbook import to_filehandle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
regression = 2
if regression == 1:

    # Goal: show linear regression is not good idea for classification
    # Plasma score is my independent variable and plasma higher than 140 is labeled as diabete positive
    data = [[100, 0],[80,0],[75,0],[60,0],[150, 1],[165,1],[180,1],[170,1]]
    df = pd.DataFrame(data,columns=['plasmascore','diabetes'])
    print(df.head(10))

    plt.scatter(df.plasmascore, df.diabetes)
    plt.title('Plasma Score vs Has_Diabetes')
    plt.xlabel('Plasma Score')
    plt.ylabel('Diabetes - Yes or No')
    plt.show()

    x = df.iloc[:, :-1].values # all columns except last column of all rows 
    y = df.iloc[:, -1].values  # last column of all rows

    linearRegressor = LinearRegression()
    linearRegressor.fit(x, y)

    plt.scatter(x, y, color = 'red')
    plt.plot(x, linearRegressor.predict(x), color = 'blue')
    plt.title('Plasma Score vs Has_Diabetes')
    plt.xlabel('Plasma Score')
    plt.ylabel('Diabetes - Yes or No')
    plt.show()

    print(linearRegressor.predict(np.array(125).reshape(1,-1)))
    print(linearRegressor.coef_)
    # Let's add two extreme points
    x_new = np.append(x, [500])
    x_new = np.append(x_new, [450])
    y_new = np.append(y, [1])
    y_new = np.append(y_new, [1])

    plt.scatter(x_new, y_new)
    plt.title('Plasma Score vs Has_Diabetes')
    plt.xlabel('Plasma Score')
    plt.ylabel('Diabetes - Yes or No')
    plt.show()

    lr = LinearRegression()
    lr_new = lr.fit(x_new.reshape(-1, 1), y_new.reshape(-1, 1))

    plt.scatter(x_new, y_new, color = 'red')
    plt.plot(x_new, lr_new.predict(x_new.reshape(-1, 1)), color = 'blue')
    plt.title('Plasma Score vs Has_Diabetes')
    plt.xlabel('Plasma Score')
    plt.ylabel('Diabetes - Yes or No')
    plt.show()

    print(f'coef of new data: {lr_new.coef_}')
    print(f' Decrease of the coefficient or slope of fitted line: {lr_new.coef_ - linearRegressor.coef_}')
    print(f'{lr_new.predict(np.array(160).reshape(1,-1))}')
    # 1- Value of probability 0.5 was 125 in the first LR model but in new model it is 165 which this varience is not acceptable.
    # 2- The values for y is sometime is larger than 1 and smaller than 0 which defetes purpose of classification.
    # That is why LR fails for classification task.

#Predicting if a person would buy life insurnace based on his age using logistic regression
elif regression == 2: # logistic regression
    """
    y = m.x + c for linear regression
    sigmoid(x) = 1/(1+e^(-x))
    apply sigmoid function on straight line
    sigmoid(y) = sigmoid(m.x + c)

    This is true for a dataset with n number of features:
    y = w_n.x_n + w_n-1.x_n-1 + ... + w_2.x_2 + w_1.x_1 + b Note: w are parameters also called weights and x are our features
    Now we can apply sigmoid over this equation and we can make prediction for any data point with given features.
    sigmoid(y) = sigmoid (w_n.x_n + w_n-1.x_n-1 + ... + w_2.x_2 + w_1.x_1 + b)
    we can also represent this equation in form of matrix.
    y = w^T. x + b note: W^T: represents transposed of w of (n, 1) dimension
    w=[w_n  ]
      |w_n-1|
        .
        .
       |w_1 |

       x is a matrix of (n,1) dimension
       y^ = sigmoid(w^T. x + b)

       if y^ >= 0.5 then 1
       if y^ <0 then 0
    """
    insurance_data = [[22,0],[25,0],[47,1],[52,0],[46,1],[56,1],[55,0],[60,1],[62,1],[61,1],
    [18,0],[28,0],[27,0],[29,0],[49,1],[55,1],[25,1],[58,1],[19,0],[18,0],
    [21,0],[26,0],[40,1],[45,1],[50,1],[54,1],[23,0]]

    df = pd.DataFrame(insurance_data,columns=['age','bought_insurance'])
    print(df.head(27))

    plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
    plt.show()

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)

    print(x_test)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    model.predict_proba(x_test)
    model.score(x_test,y_test)
    print(y_predicted)
    #model.coef_ indicates value of m in y=m*x + b equation
    print(model.coef_)
    #model.intercept_ indicates value of b in y=m*x + b equation
    print(model.intercept_)

    # Lets defined sigmoid function now and do the math with hand
    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def prediction_function(age):
        z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
        y = sigmoid(z)
        return y
    age = 35
    print(f'prediction for age =30: {prediction_function(age)}')    
    #0.485 is less than 0.5 which means person with 35 age will not buy insurance

    age = 43
    print(f'prediction for age =43: {prediction_function(age)}')
    # 0.56 is more than 0.5 which means person with 43 will buy the insurance

# In this tutorial we will see how to use logistic regression for multiclass classification.
elif regression == 3: # logistic regression multiclass
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    """
    This dataset is made up of 1797 8x8 images. Each image, like the one shown below, is of a hand-written digit. In order to 
    utilize an 8x8 figure like this, we'd have to first transform it into a feature vector with length 64.
    See here for more information about this dataset.
    """
    digits = load_digits() 
    #plt.gray() 
    # for i in range(10):
    #     plt.matshow(digits.images[i])
    
    fig = plt.figure(figsize=(4, 4))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))
    plt.show()

    print(dir(digits))
    print(digits.data[0])

    #Create and train logistic regression model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)
    model.fit(x_train, y_train)
    #Measure accuracy of our model
    print(f' accuracy of our model:  {model.score(x_test, y_test)}')
    print(f'model prediction 1 - 5 :{model.predict(digits.data[0:5])}')

    #Confusion Matrix
    y_predicted = model.predict(x_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_predicted)
    print(f'confusion matrix: {cm}')
    
    import seaborn as sn
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

elif regression == 4: # Multivariate Logistic Regression python
    dataPath = r'\data\diabetes.csv'
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset
    data = pd.read_csv(dataPath, header=None, names=col_names)
    print(data.head())

    # Selecting Feature:  divide the given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).
    #split dataset in features and target variable
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
    x = data[feature_cols] # Features
    y = data.label # Target variable

    # Splitting Data
    # split X and y into training and testing sets
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
    
    # Model Development and Prediction
    # import the class
    from sklearn.linear_model import LogisticRegression

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()

    # fit the model with data
    logreg.fit(x_train,y_train)

    y_pred=logreg.predict(x_test)
    
    x = [7,158,25,75,33.6,19,50]
    x= np.array(x)    
    print(f'Prediction of {x} is {logreg.predict(x.reshape(1, -1))}')

    #Model Evaluation using Confusion Matrix
    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

    # we have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. 
    # In the output, 119 and 36 are actual predictions, and 26 and 11 are incorrect predictions.

    #Visualizing Confusion Matrix using Heatmap
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.Text(0.5,257.44,'Predicted label')
    plt.show()

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

    # Well, you got a classification rate of 80%, considered as good accuracy.
    #Precision: Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often it is correct. 
    # In our prediction case, when your Logistic Regression model predicted patients are going to suffer from diabetes, that patients have 76% of the time.

    #Recall: If there are patients who have diabetes in the test set and your Logistic Regression model can identify it 58% of the time.

    #ROC Curve
    #Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity.

    y_pred_proba = logreg.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    # AUC score for the case is 0.86. AUC score 1 represents perfect classifier, and 0.5 represents a worthless classifier.
elif regression == 5: # Second example: Multivariate Logistic Regression python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
                'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
                'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
                'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
                }

    df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
    X = df[['gmat', 'gpa','work_experience']]
    y = df['admitted']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  #in this case, you may choose to set the test_size=0. You should get the same prediction here
    logistic_regression= LogisticRegression()
    logistic_regression.fit(X_train,y_train)
    new_candidates = {'gmat': [590,740,680,610,710],
                    'gpa': [2,3.7,3.3,2.3,3],
                    'work_experience': [3,4,6,1,5]
                    }

    df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience'])
    y_pred=logistic_regression.predict(df2)
    print (df2)
    print (y_pred)

elif regression == 6: # Logistic regression using tensorflow
    
    candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
                'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
                'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
                'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
                }
    df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
    X = df[['gmat', 'gpa','work_experience']]
    Y = df[['admitted']]

    import tensorflow as tf
    from tensorflow import keras

    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense

    model = Sequential()
    model.add(Dense(1, input_dim= X.shape[1], activation=tf.keras.activations.sigmoid))
    model.compile(loss = 'binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'] )
    model.fit(x=X, y=Y, epochs = 2048, verbose = 1)

    # Check to see how good is the model
    print(Y[:10].T)
    prediction = model.predict(X)
    print(prediction[:10].T)
    model.save(r'venv\model\OneNeuron.h5')
    reconstructed_model = keras.models.load_model(r'venv\model\OneNeuron.h5')
    # Let's check:
    np.testing.assert_allclose(
        model.predict(X), reconstructed_model.predict(X)
    )



    











