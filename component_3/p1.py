import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic_project.csv")
df = df.drop(columns=["age", "pclass"])
age  = df["sex"]
alive = df["survived"]
X_train = age.iloc[0:800].to_numpy().reshape(-1,1)

X_test = age.iloc[800:].to_numpy().reshape(-1,1)

y_train = alive[0:800].to_numpy()

y_test = alive[800:].to_numpy()

model = SGDClassifier(loss="log_loss", learning_rate="constant", max_iter=50000, eta0=0.001)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_pred=y_pred, y_true=y_test)
print(score)