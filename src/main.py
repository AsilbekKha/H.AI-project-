import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error

clients_data = pd.read_csv(r"C:\Users\user\PycharmProjects\H_Ai_2025\H.AI-project-\tables\clients.csv")
visits_data = pd.read_csv(r"C:\Users\user\PycharmProjects\H_Ai_2025\H.AI-project-\tables\visits.csv")

merged_data = pd.merge(clients_data, visits_data, left_on='id', right_on='Client ID')
encoder = LabelEncoder()

merged_data['gender'] = encoder.fit_transform(merged_data['gender'])
merged_data['Visit Day'] = encoder.fit_transform(merged_data['Visit Day'])
merged_data['Visit Purpose'] = encoder.fit_transform(merged_data['Visit Purpose'])

X = merged_data[['age', 'gender', 'Visit Day', 'Entry Time', 'Exit Time']]
y_class = merged_data['Visit Purpose']
y_reg = merged_data['Duration (minutes)']

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

clf = CatBoostClassifier()
clf.fit(X_train, y_class_train)

y_class_pred = clf.predict(X_test)
class_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"Visit Purpose Classifier Accuracy: {class_accuracy:.2f}")

reg = CatBoostRegressor(verbose=0)
reg.fit(X_train, y_reg_train)

y_reg_pred = reg.predict(X_test)
reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Duration Regressor MSE: {reg_mse:.2f}")

plt.figure(figsize=(8, 6))
feature_importance = clf.get_feature_importance()
sns.barplot(x=feature_importance, y=X.columns)
plt.title("Feature Importance (Visit Purpose Classifier)")
plt.show()


