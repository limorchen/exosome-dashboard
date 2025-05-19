import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Set Streamlit config
st.set_page_config(layout="wide", page_title="Company Insights Dashboard")
st.title("Company Insights Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your company data CSV", type=["csv"])

# Function to parse market cap values
def parse_market_cap(value):
    try:
        value = str(value).strip().replace(",", "").replace("$", "")
        if pd.isna(value) or value.lower() == 'nan':
            return None
        if value.startswith("<") or value.startswith(">"):
            value = value[1:]
        if "M" in value:
            return float(value.replace("M", "")) * 1e6
        elif "B" in value:
            return float(value.replace("B", "")) * 1e9
        else:
            return float(value)
    except Exception as e:
        print(f"Could not parse market cap value: {value} — {e}")
        return None

# Function to reduce category counts
def reduce_categories(series, top_n=7, other_label="Other"):
    counts = series.value_counts()
    top_categories = counts.nlargest(top_n).index
    return series.apply(lambda x: x if x in top_categories else other_label)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Strip column names
    df.columns = df.columns.str.strip()

    # Parse Market Cap
    if 'Market Cap' in df.columns:
        df['Market Cap'] = df['Market Cap'].apply(parse_market_cap)

    # Group and normalize similar rows except Company Name and Market Cap
    cat_cols = [col for col in df.columns if col not in ['Company Name', 'Market Cap']]
    if len(cat_cols) > 0:
        grouped = (
            df.groupby(cat_cols, dropna=False)
              .agg({
                  'Company Name': lambda x: ', '.join(sorted(set(x.dropna()))),
                  'Market Cap': 'mean'
              })
              .reset_index()
        )
        df = grouped[['Company Name', 'Market Cap'] + cat_cols]

        # Normalize Business Area names
        business_area_map = {
            'Exosome Therapy': 'Exosome-Based Therapy',
            'Exosome Therapeutics': 'Exosome-Based Therapy',
            'Extracellular Vesicles': 'Exosome-Based Therapy',
            'Cell Therapy': 'Cell-Based Therapy',
            'CAR-T Therapy': 'Cell-Based Therapy',
            'Gene Therapy': 'Gene & Nucleic Acid Therapies',
            'mRNA Therapeutics': 'Gene & Nucleic Acid Therapies',
            'siRNA': 'Gene & Nucleic Acid Therapies',
            'Rejuvenation': 'Longevity & Anti-aging',
            'Longevity': 'Longevity & Anti-aging',
        }

        if 'Business Area' in df.columns:
            df['Business Area'] = df['Business Area'].replace(business_area_map)

        # 🔽 Step: Reduce category counts across all other categorical columns
        for col in df.columns:
            if col not in ['Company Name', 'Market Cap'] and df[col].dtype == 'object':
                df[col] = reduce_categories(df[col], top_n=5)

    # Show filtered data
    st.subheader("Filtered Data")
    st.dataframe(df)

    # Sidebar Filters
    with st.sidebar:
        st.header("Filters")
        business_area = st.multiselect("Business Area", options=df['Business Area'].dropna().unique() if 'Business Area' in df.columns else [])
        location = st.multiselect("Location", options=df['Location'].dropna().unique() if 'Location' in df.columns else [])

    # Apply filters
    filtered_df = df.copy()
    if business_area:
        filtered_df = filtered_df[filtered_df['Business Area'].isin(business_area)]
    if location:
        filtered_df = filtered_df[filtered_df['Location'].isin(location)]

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Companies", len(filtered_df))
    if 'Market Cap' in filtered_df.columns:
        avg_market_cap = filtered_df['Market Cap'].dropna().mean()
        col2.metric("Avg. Market Cap", f"${avg_market_cap:,.0f}")

    # Charts
    if 'Business Area' in filtered_df.columns:
        st.subheader("Company Count by Business Area")
        ba_counts = filtered_df['Business Area'].value_counts().reset_index()
        ba_counts.columns = ['Business Area', 'Count']
        fig1 = px.bar(ba_counts, x='Business Area', y='Count')
        st.plotly_chart(fig1, use_container_width=True)

    if 'Location' in filtered_df.columns and 'Market Cap' in filtered_df.columns:
        st.subheader("Market Cap by Location")
        fig2 = px.box(filtered_df, x="Location", y="Market Cap")
        st.plotly_chart(fig2, use_container_width=True)

    if 'Stage of development' in filtered_df.columns and 'Market Cap' in filtered_df.columns:
        st.subheader("Market Cap by Stage of Development")
        fig3 = px.scatter(
            filtered_df.dropna(subset=['Market Cap']),
            x="Stage of development",
            y="Market Cap",
            size="Market Cap",
            color="Stage of development",
            hover_data=["Company Name"] if "Company Name" in df.columns else None
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Pie chart for Product
    if 'product' in filtered_df.columns:
        st.subheader("Company Distribution by Product")
        product_counts = filtered_df['product'].value_counts().reset_index()
        product_counts.columns = ['product', 'Count']
        if product_counts.empty:
            st.write("No data available for product column.")
        else:
            fig_pie = px.pie(
                product_counts,
                names='product',
                values='Count',
                title='Companies by product',
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Optional ML Model
    st.subheader("Predict Market Cap (Simple Model)")
    if 'Market Cap' in df.columns:
        model_df = df.dropna(subset=['Market Cap'])
        required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
        if all(col in model_df.columns for col in required_cols):
            X = model_df[required_cols].fillna("Unknown")
            y = model_df['Market Cap']

            encoder = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns)],
                remainder='drop'
            )

            pipeline = Pipeline(steps=[
                ('preprocessor', encoder),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            st.write(f"Sample prediction result: ${y_pred[0]:,.0f}")
        else:
            st.warning("One or more required columns are missing for prediction.")





