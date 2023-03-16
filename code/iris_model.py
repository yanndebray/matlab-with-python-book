from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def prep(iris):
    X = iris.data[:, :2]  # we only take the first two features (sepal)
    y = iris.target
    return X,y

def train(X,y):
    model = LogisticRegression()
    model.fit(X,y)
    return model

def predict(model,X):
    return model.predict(X)

if __name__ == '__main__':
    
    iris = datasets.load_iris()
    X,y = prep(iris)
    mdl = train(X,y)
    print(sum(predict(mdl,X)==y)/len(X))


