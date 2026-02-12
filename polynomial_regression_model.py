import os
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df1 = pd.read_csv(os.path.join(BASE_DIR, "insurance.csv"))
x = df1.drop("charges" , axis=1)
y = df1["charges"]

num_cols = ['age' ,  'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

xtrain , xtest , ytrain , ytest = train_test_split(x,y , test_size=0.2 , random_state=42)

preprocess = ColumnTransformer([("num" , StandardScaler() , num_cols) , ("cat" , OneHotEncoder(drop="first" , handle_unknown="ignore") , cat_cols)])

pipeline = Pipeline([("preprcess" , preprocess) , ("poly" , PolynomialFeatures(degree=2 , include_bias=False)) ,("elastic" , ElasticNet(max_iter=10000))])

param_grid = {"poly__degree" : [1,2],  "elastic__alpha" : [0.01 , 0.1 , 1 , 10 , 100] , "elastic__l1_ratio" : [0.2,0.5,0.8]}

grid = GridSearchCV(pipeline , param_grid , cv=5)

grid.fit(xtrain , ytrain)
model = grid.best_estimator_
ypred = model.predict(xtest)
print(ypred)