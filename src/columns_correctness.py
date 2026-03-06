import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
import streamlit as st


def test_columns_correctness(cols, df):
    if set(cols).issubset(df.columns):
        return True
    else:
        missing = list(set(cols) - set(df.columns))
        st.write(f"所上傳的檔案缺少 {missing} 欄位")