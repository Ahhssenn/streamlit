import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# make containers 
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Titanic app")
    st.text("In this app we will work on the Titanic dataset")

with datasets:
    st.header("Datasets")
    st.text("Loading the titanic dataset")
    # import data
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head(10))
    st.bar_chart(df["sex"].value_counts())

    # other plot
    st.subheader("Difference by class")
    st.bar_chart(df["class"].value_counts())
    
    # barplot
    st.subheader("Barplot of the age")
    st.bar_chart(df["age"].sample(10)) # or head(10)

with features:
    st.header("These are our app features")
    st.text("List of different important features")
    st.markdown('1. **Feature 1**: This is the first feature')
    st.markdown('2. **Feature 2**: This is the second feature')
    st.markdown('3. **Feature 3**: This is the third feature')

with model_training:
    st.header("Model Training")
    st.text("We will use different parameters here")

    # making columns
    input, display = st.columns(2)

    # making slider but there must be some values in the slider
    max_depth = input.slider('How many people do you know?', min_value= 10, max_value=100, value=20, step=5)

n_estimators = input.selectbox("How many tree should be there in a RF?", options=[50,100,200,300,'No Limit'])

# adding list of features
input.write(df.columns)

# input features from user 
input_features = input.text_input("which feature we should use ?")    

# machine learning model
model = RandomForestRegressor(max_depth= max_depth, n_estimators=n_estimators)

# if condition for no limit
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

# define x and y 
X = df[[input_features]]
y = df[['fare']]

# fit our model
model.fit(X,y)
pred = model.predict(X)

# display metrices 
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R square score of the model is: ")
display.write(r2_score(y,pred))