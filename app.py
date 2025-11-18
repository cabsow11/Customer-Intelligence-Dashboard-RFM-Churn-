# app.py – FINAL VERSION – 18 Nov 2025
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Chargement données
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_customer_data.csv")
    df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'])
    return df

df = load_data()
CURRENT_DATE = datetime(2025, 11, 17)

# RFM
rfm = df.copy()
rfm['Recency'] = (CURRENT_DATE - rfm['Last_Purchase_Date']).dt.days
rfm['Frequency'] = rfm['Frequency']
rfm['Monetary'] = rfm['Monetary_Spendings']

# Quantiles
q = rfm[['Recency','Frequency','Monetary']].quantile(q=[0.2,0.4,0.6,0.8])

# Fonctions de scoring SANS astype(int) direct → on convertit en int après .cat.codes
def r_score(x):
    return pd.cut(x, bins=[-1] + q['Recency'].tolist() + [9999], labels=[5,4,3,2,1])

def fm_score(x, col):
    return pd.cut(x, bins=[-1] + q[col].tolist() + [9999], labels=[1,2,3,4,5])

rfm['R'] = r_score(rfm['Recency']).cat.codes + 1
rfm['F'] = fm_score(rfm['Frequency'], 'Frequency').cat.codes + 1
rfm['M'] = fm_score(rfm['Monetary'], 'Monetary').cat.codes + 1

# Segmentation
def segment(row):
    r, f, m = row['R'], row['F'], row['M']
    if r >= 4 and f >= 4 and m >= 4: return "Champions"
    elif r >= 3 and f >= 3: return "Loyal Customers"
    elif r >= 4 and f <= 2: return "New Customers"
    elif r <= 2 and f >= 4: return "At Risk – High Value"
    elif r <= 2 and f <= 2: return "Lost Customers"
    elif r <= 2 and m >= 4: return "Lost Big Spenders"
    elif m >= 4: return "Big Spenders"
    else: return "Hibernating"

rfm['Segment'] = rfm.apply(segment, axis=1)

colors = {"Champions":"#00D4AA","Loyal Customers":"#55E6C1","New Customers":"#6C5CE7",
          "Big Spenders":"#FDCB6E","At Risk – High Value":"#FF6B6B","Lost Big Spenders":"#E17055",
          "Lost Customers":"#636E72","Hibernating":"#B2BEC3"}

# Modèle Churn
X = rfm[['Recency','Frequency','Monetary','Behavior_Score']]
y = rfm['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
rfm['Churn_Probability'] = model.predict_proba(X)[:,1]
rfm['Risk'] = pd.cut(rfm['Churn_Probability'], bins=[0,0.3,0.6,0.8,1], labels=['Low','Medium','High','Critical'])

# Dashboard
st.set_page_config(page_title="RFM + Churn Dashboard", layout="wide")
st.title("Customer Intelligence Dashboard – RFM + AI Churn Prediction")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Clients", f"{len(rfm):,}")
c2.metric("Revenue", f"${rfm['Monetary'].sum():,.0f}")
c3.metric("Churn Rate", f"{rfm['Churn'].mean():.1%}")
c4.metric("AUC", f"{auc:.3f}")

a, b = st.columns([7,5])
with a:
    st.plotly_chart(px.treemap(rfm, path=['Segment'], values='Monetary', color='Segment',
                               color_discrete_map=colors, title="Revenue by Segment"), use_container_width=True)
with b:
    st.plotly_chart(px.bar(rfm['Segment'].value_counts().reset_index(), x='Segment', y='count',
                           color='Segment', color_discrete_map=colors, title="Clients per Segment"), use_container_width=True)

st.plotly_chart(px.scatter(rfm, x='Recency', y='Monetary', size='Frequency', color='Segment',
                           color_discrete_map=colors, hover_data=['Client_ID','Churn_Probability'],
                           title="RFM Bubble Chart"), use_container_width=True)

d, e = st.columns(2)
with d:
    fig = go.Figure([go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), showlegend=False),
                     go.Scatter(x=fpr, y=tpr, line=dict(color='#00D4AA', width=5), name=f'AUC = {auc:.3f}')])
    fig.update_layout(title="ROC Curve", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
with e:
    st.plotly_chart(px.histogram(rfm, x='Churn_Probability', color='Risk', nbins=50,
                                 title="Churn Risk Distribution"), use_container_width=True)

st.success("LIVE & READY ")
