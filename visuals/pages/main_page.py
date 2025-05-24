import streamlit as st
import numpy as np
import pandas as pd
import time


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [10, 10] + [41.3111, 69.2797],
    columns=['lat', 'lon']
)

st.map(map_data)