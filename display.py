import streamlit as st

main_page = st.Page("visuals/pages/main_page.py", title="Main Page", icon="🎈")
page_2 = st.Page("visuals/pages/page2.py", title="Page 2", icon="❄️")


pg = st.navigation([main_page, page_2])
pg.run()
