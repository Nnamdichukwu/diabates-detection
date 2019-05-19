import numpy as np 
import pandas as pd 
# import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv("/home/hotelsng/Downloads/diabetes.csv")
print(df.head())
print(df.isnull().sum())
pregnancies = tf.feature_column.numeric_column("Pregnancies")
glucose = tf.feature_column.numeric_column("Glucose")
bp = tf.feature_column.numeric_column("BloodPressure")
skinthick = tf.feature_column.numeric_column("SkinThickness")
insulin = tf.feature_column.numeric_column("Insulin") 
bmi = tf.feature_column.numeric_column("BMI")
diabetes_func = tf.feature_column.numeric_column("DiabetesPedigreeFunction")
age = tf.feature_column.numeric_column("Age")
my_col = [pregnancies,glucose,bp, skinthick , insulin , bmi , diabetes_func, age]
x =  df.drop("Outcome", axis=1)
y = df["Outcome"]
x_train , x_test, y_train , y_test = train_test_split(x , y, test_size =0.3)
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train , y=y_train , num_epochs= 100, batch_size= 10,  shuffle = True )
model = tf.estimator.LinearClassifier(feature_columns = my_col, n_classes = 2)
model.train(input_fn= input_func , steps = 100)
results = model.evaluate(tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=100, shuffle=False))
print(results)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
result = model.evaluate(eval_input_func)
print(result)
pred_input_func= tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
y_pred= [d['logits'] for d in predictions]

x, y, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc =metrics.auc(x , y)
print(roc_auc)

