# Feature Selection-Information gain - mutual information In Regression Problem Statements
#data is train.csv from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
import pandas as pd
dataPath = r'\data\housing_data.csv'
housing_df=pd.read_csv(dataPath)
print(housing_df.info())
print(housing_df.isnull().sum())
numeric_lst=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_cols = list(housing_df.select_dtypes(include=numeric_lst).columns)

housing_df=housing_df[numerical_cols]
housing_df=housing_df.drop("Id",axis=1)

#overfitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(housing_df.drop(labels=['SalePrice'], axis=1),
    housing_df['SalePrice'],
    test_size=0.3,
    random_state=0)
print(X_train.isnull().sum())
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
# determine the mutual information
mutual_info = mutual_info_regression(X_train.fillna(0), y_train)
print(mutual_info)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending=False))
mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5))
plt.show()
from sklearn.feature_selection import SelectPercentile
## Selecting the top 20 percentile
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X_train.fillna(0), y_train)
selected_top_columns.get_support()
print(X_train.columns[selected_top_columns.get_support()])
