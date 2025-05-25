import streamlit as st

main_page = st.Page("visuals/pages/main_page.py", title="Предсказывание", icon="🎱")
page_2 = st.Page("visuals/pages/page2.py", title="Аналитика", icon="📈")


pg = st.navigation([main_page, page_2])
pg.run()
