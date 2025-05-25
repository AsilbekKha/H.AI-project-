import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, \
    classification_report


def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s


@st.cache_data
def load_data():
    visits = pd.read_csv("tables/visits.csv")
    clients = pd.read_csv("tables/clients.csv")
    purchases = pd.read_csv("tables/purchases.csv")
    return visits, clients, purchases


def preprocess_data(visits, clients):
    visits['Entry_Minutes'] = visits['Entry_Time'].apply(time_to_seconds)
    visits['Exit_Minutes'] = visits['Exit_Time'].apply(time_to_seconds)
    visits['Visit_Duration'] = visits['Exit_Minutes'] - visits['Entry_Minutes']
    visits = visits[visits['Visit_Duration'] > 0]

    data = visits.merge(clients, left_on='Client_ID', right_on='ClientID')

    visit_counts = data.groupby('Client_ID')['Visit_ID'].count().reset_index()
    visit_counts['Repeat_Visit'] = (visit_counts['Visit_ID'] > 1).astype(int)
    data = data.merge(visit_counts[['Client_ID', 'Repeat_Visit']], on='Client_ID')

    X = data.drop(columns=[
        'Visit_Duration', 'Visit_ID', 'Client_ID', 'ClientID', 'Name',
        'Entry_Time', 'Exit_Time', 'Date', 'Visit_Purpose', 'PhoneNumber', 'Repeat_Visit'
    ])

    categorical_features = ['Gender', 'Weekday']
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_features]

    return data, X, cat_features_indices


def run_regression(data, X, cat_features_indices):
    y_reg = data['Visit_Duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_features_indices)

    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
    model.fit(train_pool)

    y_pred = model.predict(test_pool)

    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }


def run_classification(data, X, cat_features_indices):
    y_clf = data['Repeat_Visit']
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_features_indices)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',
        eval_metric='AUC'
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_proba = model.predict_proba(test_pool)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'report': classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        'distribution': y_clf.value_counts(normalize=True)
    }

st.title("Анализ визитов клиентов")

visits, clients, purchases = load_data()
data, X, cat_features_indices = preprocess_data(visits, clients)

tab1, tab2 = st.tabs(["📈 Регрессия", "📊 Классификация"])

with tab1:
    st.subheader("Предсказание продолжительности визита (регрессия)")
    reg_metrics = run_regression(data, X, cat_features_indices)
    st.metric("Mean Squared Error", f"{reg_metrics['MSE']:.2f}")
    st.metric("Mean Absolute Error", f"{reg_metrics['MAE']:.2f}")
    st.metric("R² Score", f"{reg_metrics['R2']:.2f}")

with tab2:
    st.subheader("Предсказание повторного визита (классификация)")
    clf_metrics = run_classification(data, X, cat_features_indices)
    st.metric("Accuracy", f"{clf_metrics['accuracy']:.2f}")
    st.metric("ROC AUC", f"{clf_metrics['roc_auc']:.2f}")
    st.write("Распределение классов:")
    st.bar_chart(clf_metrics['distribution'])
    st.write("Classification Report:")
    st.json(clf_metrics['report'])
