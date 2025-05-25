from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Загрузка данных
visits = pd.read_csv("tables/visits.csv")
clients = pd.read_csv("tables/clients.csv")
purchases = pd.read_csv("tables/purchases.csv")  # Пока не используется, но можно подключить при расширении модели

# Вспомогательная функция
def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

# Предобработка
visits['Entry_Minutes'] = visits['Entry_Time'].apply(time_to_seconds)
visits['Exit_Minutes'] = visits['Exit_Time'].apply(time_to_seconds)
visits['Visit_Duration'] = visits['Exit_Minutes'] - visits['Entry_Minutes']
visits = visits[visits['Visit_Duration'] > 0]

data = visits.merge(clients, left_on='Client_ID', right_on='ClientID')

visit_counts = data.groupby('Client_ID')['Visit_ID'].count().reset_index()
visit_counts['Repeat_Visit'] = (visit_counts['Visit_ID'] > 1).astype(int)
data = data.merge(visit_counts[['Client_ID', 'Repeat_Visit']], on='Client_ID')

# Признаки
X = data.drop(columns=[
    'Visit_Duration', 'Visit_ID', 'Client_ID', 'ClientID', 'Name',
    'Entry_Time', 'Exit_Time', 'Date', 'Visit_Purpose', 'PhoneNumber', 'Repeat_Visit'
])

categorical_features = ['Gender', 'Weekday']
cat_features_indices = [X.columns.get_loc(col) for col in categorical_features]

# ===================== РЕГРЕССИЯ =====================
y_reg = data['Visit_Duration']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

train_pool_reg = Pool(X_train_reg, y_train_reg, cat_features=cat_features_indices)
test_pool_reg = Pool(X_test_reg, y_test_reg, cat_features=cat_features_indices)

model_reg = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, random_seed=42, verbose=100)
model_reg.fit(train_pool_reg)

print(model_reg.feature_names_)

# Сохранение модели регрессии
joblib.dump(model_reg, 'models/model_reg.pkl')

# Оценка
y_pred_reg = model_reg.predict(test_pool_reg)
print("\n--- Регрессия (продолжительность визита) ---")
print(f'MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}')
print(f'MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.4f}')
print(f'R^2: {r2_score(y_test_reg, y_pred_reg):.4f}')

# ===================== КЛАССИФИКАЦИЯ =====================
y_clf = data['Repeat_Visit']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

train_pool_clf = Pool(X_train_clf, y_train_clf, cat_features=cat_features_indices)
test_pool_clf = Pool(X_test_clf, y_test_clf, cat_features=cat_features_indices)

model_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=100,
    auto_class_weights='Balanced',
    eval_metric='AUC'
)

model_clf.fit(train_pool_clf, eval_set=test_pool_clf, use_best_model=True)

joblib.dump(model_clf, 'models/model_clf.pkl')

y_pred_proba_clf = model_clf.predict_proba(test_pool_clf)[:, 1]
y_pred_clf = (y_pred_proba_clf >= 0.5).astype(int)

print("\n--- Классификация (повторный визит) ---")
print(f'Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.4f}')
print(f'ROC AUC: {roc_auc_score(y_test_clf, y_pred_proba_clf):.4f}')
print(classification_report(y_test_clf, y_pred_clf, zero_division=0))

print("\n--- Распределение классов ---")
print(data['Repeat_Visit'].value_counts(normalize=True))
