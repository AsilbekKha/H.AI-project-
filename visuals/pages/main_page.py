import streamlit as st
import numpy as np
import pandas as pd
import time

st.markdown(
    """
    <h2 style='text-align: center; color: #4CAF50; font-family: Arial;'>
        Important map data is provided belove!
    </h2>
    """,
    unsafe_allow_html=True
)
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [10, 10] + [41.3111, 69.2797],
    columns=['lat', 'lon']
)s
if st.button("Celebrate"):
    st.balloons()
st.map(map_data)