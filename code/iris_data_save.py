from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first two features (sepal)
Xl = X.tolist()
y = iris.target
dict = {0: "setosa",1: "versicolor", 2: "virginica"}
species = [dict[i] for i in y]  

# Sepal features
df = pd.DataFrame(X,columns=['lenght','width'])
df['species'] = species

# print(df)

df.to_parquet('iris.parquet')