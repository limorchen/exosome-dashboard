
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, dash_table

df_raw = pd.read_excel('Company DB for partnerships.xlsx', sheet_name='Sheet 1')
df = df_raw.iloc[2:].reset_index(drop=True)
df.columns = df_raw.iloc[1]
df.columns.name = None

def extract_country(location):
    if pd.isna(location):
        return "Unknown"
    parts = location.split(',')
    return parts[-1].strip()

def split_sectors(entry):
    if pd.isna(entry):
        return []
    return [s.strip() for s in entry.split(',')]

df['Country'] = df['Location'].apply(extract_country)
df_sectors = df.explode('Business Area')
df_sectors['Business Area'] = df_sectors['Business Area'].apply(split_sectors)
df_sectors = df_sectors.explode('Business Area')

cns_keywords = ['CNS', 'neuro', 'brain', 'Parkinson', 'Alzheimer', 'neurology']
def contains_cns(text):
    if pd.isna(text):
        return False
    return any(kw.lower() in text.lower() for kw in cns_keywords)

df['CNS Related'] = df[['Product Types', 'Regulatory Status', 'Notable Partnerships/Deals', 'Recent News']].apply(
    lambda row: any(contains_cns(val) for val in row), axis=1)
df_cns = df[df['CNS Related'] == True]

import re
import numpy as np

def estimate_funding(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r'\$?~?\$?([\d\.]+)[ -\u2013]?([\d\.]*)[Mm]?', value)
    if match:
        # Check if the matched group contains only a decimal point
        if match.group(1) == '.':  
            return np.nan  # Or any other appropriate handling
        low = float(match.group(1))
        high = float(match.group(2)) if match.group(2) else low
        return (low + high) / 2
    return np.nan

df['Funding Estimate ($M)'] = df['Est. Market Cap/funding'].apply(estimate_funding)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Exosome Therapeutics Company Dashboard"),
    html.H3("Companies by Business Area"),
    dcc.Graph(figure=px.histogram(df_sectors, y='Business Area', color='Business Area',
                                  title='Number of Companies by Business Area')),
    html.H3("Companies by Country"),
    dcc.Graph(figure=px.histogram(df, y='Country', color='Country',
                                  title='Number of Companies by Country')),
    html.H3("CNS-Relevant Companies"),
    dash_table.DataTable(
        data=df_cns[['Company Name', 'Product Types', 'Recent News']].to_dict('records'),
        columns=[{'name': col, 'id': col} for col in ['Company Name', 'Product Types', 'Recent News']],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    ),
    html.H3("Funding Estimate Distribution"),
    dcc.Graph(figure=px.histogram(df, x='Funding Estimate ($M)', nbins=15,
                                  title='Funding Estimate Distribution',
                                  labels={'Funding Estimate ($M)': 'Estimated Funding ($M)'})),
    html.H3("Companies with Active Clinical Programs"),
    dash_table.DataTable(
        data=df[df['Regulatory Status'].str.contains("IND|Phase|FDA|Orphan", case=False, na=False)][['Company Name', 'Regulatory Status', 'Location']].to_dict('records'),
        columns=[{'name': col, 'id': col} for col in ['Company Name', 'Regulatory Status', 'Location']],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    )
])

server = app.server

if __name__ == "__main__":
    test_values = [
        "$50M",
        "~$45M",
        "$40–60M",
        "$30 - 50M",
        "Not a number",
        "$.M",
        None
    ]
    for val in test_values:
        result = estimate_funding(val)
        print(f"Input: {val}, Estimated Funding: {result}")