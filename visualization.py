import pandas as pd
import plotly.express as px

df = pd.read_csv('embeddings/DSC_embeddings_4518998.csv')

fig = px.scatter(df, x='x', y='y', color='title', hover_name='title')
fig.show()
