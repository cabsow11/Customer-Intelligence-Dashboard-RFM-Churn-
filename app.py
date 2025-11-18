# app.py – Dashboard RFM + Churn (prêt pour Streamlit Cloud)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --------------------- CHARGEMENT DES DONNÉES ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_customer_data.csv")
    df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'])
    return df

df = load_data()
CURRENT_DATE = datetime(2025, 11, 17)

# --------------------- CALCUL RFM ---------------------
rfm = df.copy()
rfm['Recency'] = (CURRENT_DATE - rfm['Last_Purchase_Date']).dt.days
rfm['Frequency'] = rfm['Frequency']
rfm['Monetary'] = rfm['Monetary_Spendings']

q = rfm[['Recency', 'Frequency', 'Monetary']].quantile([0.2, 0.4, 0.6, 0.8])

# Calcul des scores RFM
def r_score(x):
    return pd.cut(x, [-1] + q['Recency'].tolist() + [9999], labels=[5, 4, 3, 2, 1]).astype(int)

def fm_score(x, col):
    return pd.cut(x, [-1] + q[col].tolist() + [9999], labels=[1, 2, 3, 4, 5]).astype(int)

rfm['R'] = r_score(rfm['Recency'])
rfm['F'] = fm_score(rfm['Frequency'], 'Frequency')
rfm['M'] = fm_score(rfm['Monetary'], 'Monetary')

# --------------------- SEGMENTATION ---------------------
def segment(row):
    if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4:
        return "Champions"
    elif row['R'] >= 3 and row['F'] >= 3:
        return "Loyal Customers"
    elif row['R'] >= 4 and row['F'] <= 2:
        return "New Customers"
    elif row['R'] <= 2 and row['F'] >= 4:
        return "At Risk – High Value"
    elif row['R'] <= 2 and row['F'] <= 2:
        return "Lost Customers"
    elif row['R'] <= 2 and row['M'] >= 4:
        return "Lost Big Spenders"
    elif row['M'] >= 4:
        return "Big Spenders"
    else:
        return "Hibernating"

rfm['Segment'] = rfm.apply(segment, axis=1)

colors = {
    "Champions": "#00D4AA", "Loyal Customers": "#55E6C1", "New Customers": "#6C5CE7",
    "Big Spenders": "#FDCB6E", "At Risk – High Value": "#FF6B6B", "Lost Big Spenders": "#E17055",
    "Lost Customers": "#636E72", "Hibernating": "#B2BEC3"
}

# --------------------- MODÈLE CHURN ---------------------
X = rfm[['Recency', 'Frequency', 'Monetary', 'Behavior_Score']]
y = rfm['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Calcul des scores et de l'AUC
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

rfm['Churn_Probability'] = model.predict_proba(X)[:, 1]
rfm['Risk'] = pd.cut(rfm['Churn_Probability'], bins=[0, 0.3, 0.6, 0.8, 1], 
                     labels=['Low', 'Medium', 'High', 'Critical'])

# --------------------- DASHBOARD STREAMLIT ---------------------
st.set_page_config(page_title="RFM + Churn Dashboard", layout="wide")
st.title("Customer Intelligence Dashboard – RFM + AI Churn Prediction 2025")

# First row metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Clients", f"{len(rfm):,}")
c2.metric("Total Revenue", f"${rfm.Monetary.sum():,.0f}")
c3.metric("Churn Rate", f"{rfm.Churn.mean():.1%}")
c4.metric("Model AUC", f"{auc:.3f}")

# Second row - charts
col1, col2 = st.columns([7, 5])

with col1:
    st.plotly_chart(px.treemap(rfm, path=['Segment'], values='Monetary', color='Segment',
                               color_discrete_map=colors, title="Revenue by Segment"), use_container_width=True)

with col2:
    st.plotly_chart(px.bar(rfm.Segment.value_counts().reset_index(), x='index', y='count',
                           color='index', color_discrete_map=colors, title="Customers per Segment"), use_container_width=True)

# Third row - scatter plot
st.plotly_chart(px.scatter(rfm, x='Recency', y='Monetary', size='Frequency', color='Segment',
                           color_discrete_map=colors, hover_data=['Client_ID', 'Churn_Probability'],
                           title="RFM Bubble Chart"), use_container_width=True)

# Fourth row - ROC Curve and Risk Distribution
col3, col4 = st.columns(2)
with col3:
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'), showlegend=False),
                     go.Scatter(x=fpr, y=tpr, line=dict(color='#00D4AA', width=4), name=f'AUC = {auc:.3f}')])
    fig.update_layout(title="ROC Curve – Churn Model", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.plotly_chart(px.histogram(rfm, x='Churn_Probability', color='Risk', nbins=50,
                                 title="Churn Risk Distribution"), use_container_width=True)

# Success message
st.success("Live dashboard – 100% ready for Upwork portfolio")
st.success("Live & Ready – Perfect Upwork Portfolio Demo")
