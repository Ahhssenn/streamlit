# import libraries
import streamlit as st
import plotly.express as px
import pandas as pd


# import dataset 
st.title("Using plotly and streamlit to create a simple dashboard")
df = px.data.gapminder()
st.write(df)
st.write(df.columns)

# summary stat
st.write(df.describe())

# data management

year_option = df['year'].unique().tolist()
year = st.selectbox("year to plot", year_option[0])
# df = df[df["year"] == year]

# Plotting 
fig = px.scatter(df, x= 'gdpPercap', y= 'lifeExp', size='pop', color= 'continent', hover_name='continent',
                 log_x=True, size_max=55, range_x=[100,100000], range_y=[20,90],
                 animation_frame='year', animation_group='country')

fig.update_layout(width = 800, height = 600)
st.write(fig)