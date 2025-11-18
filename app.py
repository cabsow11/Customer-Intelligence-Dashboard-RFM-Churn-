# app.py – RFM + Churn Dashboard (100% working on Streamlit Cloud – Nov 2025)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# --------------------- CHARGEMENT ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_customer_data.csv")
    df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'])
    return df

df = load_data()
CURRENT_DATE = datetime(2025, 11, 17)

# --------------------- RFM ---------------------
rfm = df.copy()
rfm['Recency'] = (CURRENT_DATE - rfm['Last_Purchase_Date']).dt.days
rfm['Frequency'] = rfm['Frequency']
rfm['Monetary'] = rfm['Monetary_Spendings']

# Calcul des quantiles proprement
quantiles = rfm.quantile(q=[0.2, 0.4, 0.6, 0.8])

def R_score(x):
    if x <= quantiles.loc[0.2, 'Recency']: return 5
    elif x <= quantiles.loc[0.4, 'Recency']: return 4
    elif x <= quantiles.loc[0.6, 'Recency']: return 3
    elif x <= quantiles.loc[0.8, 'Recency']: return 2
    else: return 1

def FM_score(x, col):
    if x <= quantiles.loc[0.2, col]: return 1
    elif x <= quantiles.loc[0.4, col]: return 2
    elif x <= quantiles.loc[0.6, col]: return 3
    elif x <= quantiles.loc[0.8, col]: return 4
    else: return 5

rfm['R'] = rfm['Recency'].apply(R_score)
rfm['F'] = rfm['Frequency'].apply(lambda x: FM_score(x, 'Frequency'))
rfm['M'] = rfm['Monetary'].apply(lambda x: FM_score(x, 'Monetary'))

# --------------------- SEGMENTATION ---------------------
def segment(row):
    if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4: return "Champions"
    elif row['R'] >= 3 and row['F'] >= 3: return "Loyal Customers"
    elif row['R'] >= 4 and row['F'] <= 2: return "New Customers"
    elif row['R'] <= 2 and row['F'] >= 4: return "At Risk – High Value"
    elif row['R'] <= 2 and row['F'] <= 2: return "Lost Customers"
    elif row['R'] <= 2 and row['M'] >= 4: return "Lost Big Spenders"
    elif row['M'] >= 4: return "Big Spenders"
    else: return "Hibernating"

rfm['Segment'] = rfm.apply(segment, axis=1)

colors = {
    "Champions": "#00D4AA", "Loyal Customers": "#55E6C1", "Potential Loyalists": "#A29BFE",
    "New Customers": "#6C5CE7", "Big Spenders": "#FDCB6E", "At Risk – High Value": "#FF6B6B",
    "Lost Big Spenders": "#E17055", "Lost Customers": "#636E72", "Hibernating": "#B2BEC3"
}

# --------------------- CHURN MODEL ---------------------
X = rfm[['Recency', 'Frequency', 'Monetary', 'Behavior_Score']]
y = rfm['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

rfm['Churn_Probability'] = model.predict_proba(X)[:, 1]
rfm['Risk'] = pd.cut(rfm['Churn_Probability'], bins=[0, 0.3, 0.6, 0.8, 1], 
                     labels=['Low', 'Medium', 'High', 'Critical'])

# --------------------- DASHBOARD ---------------------
st.set_page_config(page_title="RFM + Churn", layout="wide")
st.title("Customer Intelligence Dashboard – RFM + AI Churn Prediction")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clients", f"{len(rfm):,}")
col2.metric("Revenue", f"${rfm['Monetary'].sum():,.0f}")
col3.metric("Churn Rate", f"{rfm['Churn'].mean():.1%}")
col4.metric("AUC Score", f"{auc:.3f}")

c1, c2 = st.columns([7, 5])
with c1:
    st.plotly_chart(px.treemap(rfm, path=['Segment'], values='Monetary', color='Segment',
                               color_discrete_map=colors, title="Revenue by Segment"), use_container_width=True)
with c2:
    st.plotly_chart(px.bar(rfm['Segment'].value_counts().reset_index(), x='Segment', y='count',
                           color='Segment', color_discrete_map=colors, title="Clients per Segment"), use_container_width=True)

st.plotly_chart(px.scatter(rfm, x='Recency', y='Monetary', size='Frequency', color='Segment',
                           color_discrete_map=colors, hover_data=['Client_ID', 'Churn_Probability'],
                           title="RFM Bubble Chart"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), showlegend=False))
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color='#00D4AA', width=5), name=f'AUC = {auc:.3f}'))
    fig_roc.update_layout(title="ROC Curve", template="plotly_dark")
    st.plotly_chart(fig_roc, use_container_width=True)

with c4:
    st.plotly_chart(px.histogram(rfm, x='Churn_Probability', color='Risk', nbins=50,
                                 title="Churn Risk Distribution"), use_container_width=True)

st.success("Live & Ready – Perfect Upwork Portfolio Demo")
