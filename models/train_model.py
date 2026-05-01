from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

df = pd.read_csv("../data/airline_passenger_data.csv")

X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

joblib.dump(model, "../model.pkl")
