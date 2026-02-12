import os
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df1 = pd.read_csv(os.path.join(BASE_DIR, "insurance.csv"))
x = df1.drop("charges" , axis=1)
y = df1["charges"]

num_cols = ['age' ,  'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']
xtrain , xtest , ytrain , ytest = train_test_split(x,y , test_size=0.2 , random_state=42)
preprocess = ColumnTransformer([("num" , StandardScaler() , num_cols) , ("cat" , OneHotEncoder(drop="first" , handle_unknown="ignore") , cat_cols)])

pipeline = Pipeline([("preprcess" , preprocess) , ("linear" , LinearRegression())])

model = pipeline.fit(xtrain , ytrain)

ypred = model.predict(xtest)
residual = ytest - ypred


# --------------------
# Residual Plot
# -------------------
plt.scatter(ypred , residual)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid()
plt.title("Residual Plot")
plt.show()

# -----------------
# Conclusion:
# The residual plot shows a non-random pattern, indicating possible non-linearity.
# Polynomial Regression (and regularization methods like ElasticNet) may improve performance.
# -----------------
