import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Title of the app

st.markdown('''
# **Exploratory Data Analysis web application** 
''')

# creating option for user to upload the file
with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])
    df = sns.load_dataset("titanic")
    st.sidebar.markdown("[Example CSV file](https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv)")

# profiling report for pandas 
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header(" **Input DF**")
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)

else:
    st.info("Awaiting for file upload...")
    if st.button("Press to use example dataset"):
        @st.cache_data 
        def load_example_data():
            a = pd.DataFrame(np.random.rand(100,5),
                         columns=['A','B','C','D','E'])
            return a
        df = load_example_data()
        pr = ProfileReport(df, explorative=True)
        st.header(" **Input DF**")
        st.write(df)
        st.write('---')
        st.header('**Profiling report with pandas**')
        st_profile_report(pr)
