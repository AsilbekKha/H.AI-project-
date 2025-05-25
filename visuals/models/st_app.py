import streamlit as st
import joblib
import pandas as pd
from catboost import Pool

# Загрузка моделей (проверьте пути)
model_reg = joblib.load("model_reg.pkl")
model_clf = joblib.load("model_clf.pkl")

# Категориальные признаки
cat_features = ['Gender', 'Weekday']

# Порядок признаков, как в обучении модели
feature_order = ['Weekday', 'Entry_Minutes', 'Exit_Minutes', 'Age', 'Gender']

st.title("📊 Предсказание поведения клиента в салоне")

st.markdown("Введите данные клиента, чтобы предсказать продолжительность визита и вероятность повторного визита.")

# Ввод данных
gender = st.selectbox("Пол клиента", ["Male", "Female"])
weekday = st.selectbox("День недели визита", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

entry_hour = st.slider("Час входа (0-23)", 0, 23, 10)
entry_minute = st.slider("Минуты входа (0-59)", 0, 59, 30)
exit_hour = st.slider("Час выхода (0-23)", 0, 23, 11)
exit_minute = st.slider("Минуты выхода (0-59)", 0, 59, 0)

age = st.slider("Возраст клиента", 18, 70, 30)

# Переводим время в минуты от начала суток
entry_minutes = entry_hour * 60 + entry_minute
exit_minutes = exit_hour * 60 + exit_minute

# Проверка логики времени
if exit_minutes < entry_minutes:
    st.error("Ошибка: Время выхода не может быть раньше времени входа.")
else:
    if st.button("Предсказать"):

        input_data = {
            'Weekday': weekday,
            'Entry_Minutes': entry_minutes,
            'Exit_Minutes': exit_minutes,
            'Age': age,
            'Gender': gender
        }
        X_input = pd.DataFrame([input_data])
        X_input = X_input[feature_order]

        for cat_col in cat_features:
            X_input[cat_col] = X_input[cat_col].astype('category')

        input_pool = Pool(X_input, cat_features=cat_features)

        # Регрессия: предсказание продолжительности визита
        duration_pred = model_reg.predict(input_pool)[0]

        # Классификация: предсказание повторного визита
        repeat_visit_proba = model_clf.predict_proba(input_pool)[0][1]
        repeat_visit_pred = int(repeat_visit_proba >= 0.5)

        st.subheader("🕒 Предсказанная продолжительность визита (в минутах):")
        st.write(f"{duration_pred / 60:.1f} минут")  # Перевод из секунд в минуты

        st.subheader("🔁 Вернётся ли клиент в будущем?")
        st.write("**Да**" if repeat_visit_pred == 1 else "**Нет**")
        st.write(f"Вероятность того, что вернётся: {repeat_visit_proba:.2%}")
