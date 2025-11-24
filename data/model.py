import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    df = pd.read_csv("data/water.csv")

    X = df[['pH','DO','Conductivity','BOD','Nitrogen']]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    with open("water_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
