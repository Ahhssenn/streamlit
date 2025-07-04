import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the Iris flower type!
""")

st.sidebar.header('User Input Pramaters')

def user_input_feature():
    sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)
    data = {'sepal_width': sepal_width,
            'sepal_length': sepal_length,
            'petal_length': petal_length,
            'petal_width' : petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_feature()
st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

predictions = clf.predict(df)
predictions_proba = clf.predict_proba(df)

st.subheader('Class probabilities and their corresponding index numbers')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[predictions])

st.subheader('Prediction Probability')
st.write(predictions_proba)


