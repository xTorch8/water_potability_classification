import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from joblib import dump

def decision_tree_model(X_train, y_train, X_val, y_val, file_path = "../src/models/saved_models/decision_tree_model.pkl"):
    try:
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_val, pd.DataFrame):
            raise TypeError("X is not a valid Pandas DataFrame")
        
        if not isinstance(y_train, pd.Series) or not isinstance(y_val, pd.Series):
            raise TypeError("y is not a valid Pandas Series")
        
        print("Starting training Decision Tree model process...")

        clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        print("Training Decision Tree model process completed.")

        print("Classification Report: ")
        print(classification_report(y_val, y_pred))

        dump(clf, file_path)
        
    except Exception as e:
        print(f"[ERR] An error occured when training model: {e}")