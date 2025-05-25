import streamlit as st
import sys
import os
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from main import load_data, preprocess_data

st.set_page_config(page_title="Страница 2: Визуализация данных", layout="wide")
st.title("Страница 2: Визуализация данных")

visits, clients, _ = load_data()
data = preprocess_data(visits, clients)

st.subheader("Продолжительность визитов по дням")
visit_duration = data[['Date', 'Visit_Duration']].sort_values('Date')
visit_duration['Date'] = pd.to_datetime(visit_duration['Date'])
daily_duration = visit_duration.groupby('Date')['Visit_Duration'].mean()
st.line_chart(daily_duration)

st.subheader("Количество визитов по дням недели")
visits_per_weekday = data['Weekday'].value_counts().sort_index()
st.bar_chart(visits_per_weekday)

st.subheader("Средняя продолжительность визита по полу (в минутах)")
avg_duration_gender = data.groupby('Gender')['Visit_Duration'].mean() / 60
st.bar_chart(avg_duration_gender)

st.subheader("Распределение возраста клиентов (Altair)")

hist_chart = alt.Chart(data).mark_bar(
    color="#4C78A8",
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    alt.X("Age:Q", bin=alt.Bin(maxbins=30), title="Возраст"),
    alt.Y("count()", title="Количество клиентов"),
    tooltip=["count()"]
).properties(
    width=600,
    height=400
).interactive()

st.altair_chart(hist_chart, use_container_width=True)
