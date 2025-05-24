import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

clients_data = pd.read_csv(r"C:\Users\user\PycharmProjects\H_Ai_2025\H.AI-project-\tables\clients.csv")
# visits_data = pd.read_csv()

encoder = LabelEncoder()

clients_data['name'] = encoder.fit_transform(clients_data['name'])
clients_data['gender'] = encoder.fit_transform(clients_data['gender'])
clients_data['age'] = encoder.fit_transform(clients_data['age'])

X = clients_data[['age', 'gender']]
# clients_data.columns = clients_data.str.strip()

# print(clients_data.head())
# print(clients_data.shape)
# print(clients_data.info())
# print(clients_data)

# plt.figure(figsize=(6,6))
# sns.distplot(clients_data['age'])
# plt.show()

X = clients_data[['']]