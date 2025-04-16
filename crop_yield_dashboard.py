import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Set plot style for dark theme
plt.style.use('dark_background')
sns.set_theme(style='darkgrid')

# Custom CSS for compact styling
css = """
<style>
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-size: 14px;
    }
    .data-panel {
        border: 1px solid #333;
        background-color: #1e1e1e;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 0.6rem;
        text-transform: uppercase;
        opacity: 0.7;
    }
    .metric-value {
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #0088ff;
        color: white;
        border-radius: 0.3rem;
        padding: 0.2rem 0.5rem;
    }
    .stSelectbox, .stNumberInput {
        background-color: #252525;
        color: #e0e0e0;
        font-size: 12px;
    }
    .stTabs > div > button {
        padding: 0.2rem 0.5rem;
        font-size: 12px;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield_train.csv")
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()
    for col in ['State', 'District', 'Crop', 'Season']:
        df[col] = df[col].str.strip()  # Clean categorical columns
    if 'Log_Production' not in df.columns:
        df['Log_Production'] = np.log1p(df['Production'])
    return df

# Load encoders and model with caching
@st.cache_resource
def load_encoders_and_model():
    label_encoder_state = joblib.load('label_encoder_state.pkl')
    label_encoder_district = joblib.load('label_encoder_district.pkl')
    label_encoder_crop = joblib.load('label_encoder_crop.pkl')
    label_encoder_season = joblib.load('label_encoder_season.pkl')
    model_clf = joblib.load('random_forest_classifier_full_feature_set.pkl')
    return label_encoder_state, label_encoder_district, label_encoder_crop, label_encoder_season, model_clf

# Load data and model
df = load_data()
label_encoder_state, label_encoder_district, label_encoder_crop, label_encoder_season, model_clf = load_encoders_and_model()

# Header
st.markdown("<h1 style='color: #00ff88; font-size: 1.5rem; margin-bottom: 0.5rem;'>Crop Yield Prediction</h1>", unsafe_allow_html=True)

# Prepare data for model performance
X = df[['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area', 'Log_Production']].copy()
state_mapping = {label: code for label, code in zip(label_encoder_state.classes_, label_encoder_state.transform(label_encoder_state.classes_))}
X['State'] = X['State'].map(state_mapping).fillna(-1).astype(int)
district_mapping = {label: code for label, code in zip(label_encoder_district.classes_, label_encoder_district.transform(label_encoder_district.classes_))}
X['District'] = X['District'].map(district_mapping).fillna(-1).astype(int)
crop_mapping = {label: code for label, code in zip(label_encoder_crop.classes_, label_encoder_crop.transform(label_encoder_crop.classes_))}
X['Crop'] = X['Crop'].map(crop_mapping).fillna(-1).astype(int)
season_mapping = {label: code for label, code in zip(label_encoder_season.classes_, label_encoder_season.transform(label_encoder_season.classes_))}
X['Season'] = X['Season'].map(season_mapping).fillna(-1).astype(int)
y = (df['Yield'] > df['Yield'].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model_clf.predict(X_test)
y_prob = model_clf.predict_proba(X_test)[:, 1]

# Tabs for compact layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Viz", "Perf", "Pred", "Tests"])

# Tab 1: Data Overview
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='data-panel'><span class='metric-title'>Rows</span><br><span class='metric-value'>{df.shape[0]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-panel'><span class='metric-title'>Cols</span><br><span class='metric-value'>{df.shape[1]}</span></div>", unsafe_allow_html=True)
    with col2:
        st.dataframe(df.head(), height=100)

# Tab 2: Visualizations
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(df['Yield'], bins=20, ax=ax, color='#00ff88')
        ax.set_title("Yield Dist", fontsize=8)
        ax.tick_params(labelsize=6)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 2))
        avg_yield = df.groupby('Crop_Year')['Yield'].mean()
        ax.plot(avg_yield.index, avg_yield.values, color='#9b5de5')
        ax.set_title("Avg Yield/Year", fontsize=8)
        ax.tick_params(labelsize=6)
        st.pyplot(fig)

# Tab 3: Model Performance
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown(f"<div class='data-panel'><span class='metric-title'>Acc</span><br><span class='metric-value'>{accuracy:.2f}</span></div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 6})
        ax.set_title("Conf Matrix", fontsize=8)
        ax.tick_params(labelsize=6)
        st.pyplot(fig)
    with col2:
        auc_score = roc_auc_score(y_test, y_prob)
        st.markdown(f"<div class='data-panel'><span class='metric-title'>AUC</span><br><span class='metric-value'>{auc_score:.2f}</span></div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(fpr, tpr, color='#00ff88', label=f'AUC={auc_score:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("ROC", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        st.pyplot(fig)

# Tab 4: Prediction
with tab4:
    states = df['State'].unique()
    state_to_districts = df.groupby('State')['District'].unique().to_dict()
    crops = df['Crop'].unique()
    seasons = df['Season'].unique()
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", states, key="state")
        crop = st.selectbox("Crop", crops, key="crop")
        year = st.number_input("Year", 1900, 2100, 2020, key="year")
    with col2:
        district = st.selectbox("District", state_to_districts.get(state, []), key="district")
        season = st.selectbox("Season", seasons, key="season")
        area = st.number_input("Area", 0.0, value=1000.0, key="area")
    production = st.number_input("Prod (t)", 0.0, value=5000.0, key="prod")
    def encode_input(label, encoder):
        if label in encoder.classes_:
            return encoder.transform([label])[0]
        return -1
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'State': [encode_input(state, label_encoder_state)],
            'District': [encode_input(district, label_encoder_district)],
            'Crop': [encode_input(crop, label_encoder_crop)],
            'Crop_Year': [year],
            'Season': [encode_input(season, label_encoder_season)],
            'Area': [area],
            'Log_Production': [np.log1p(production)]
        })
        pred = model_clf.predict(input_data)[0]
        prob = model_clf.predict_proba(input_data)[0][1]
        category = "High" if pred == 1 else "Low"
        st.markdown(f"<div class='data-panel'><span class='metric-value'>{category}</span> (Prob: {prob:.2f})</div>", unsafe_allow_html=True)

# Tab 5: Test Cases
with tab5:
    test_cases = [
        {'State': 'Punjab', 'District': 'LUDHIANA', 'Crop': 'Wheat', 'Crop_Year': 2018, 'Season': 'Rabi', 'Area': 1000.0, 'Production': 4000.0},
        {'State': 'Maharashtra', 'District': 'PUNE', 'Crop': 'Sugarcane', 'Crop_Year': 2017, 'Season': 'Whole Year', 'Area': 5000.0, 'Production': 350000.0}
    ]
    col1, col2 = st.columns(2)
    for i, case in enumerate(test_cases):
        with col1 if i == 0 else col2:
            st.markdown(f"<div class='data-panel'><span class='metric-title'>Test {i+1}</span><br>{case}</div>", unsafe_allow_html=True)
            input_data = pd.DataFrame({
                'State': [encode_input(case['State'], label_encoder_state)],
                'District': [encode_input(case['District'], label_encoder_district)],
                'Crop': [encode_input(case['Crop'], label_encoder_crop)],
                'Crop_Year': [case['Crop_Year']],
                'Season': [encode_input(case['Season'], label_encoder_season)],
                'Area': [case['Area']],
                'Log_Production': [np.log1p(case['Production'])]
            })
            pred = model_clf.predict(input_data)[0]
            prob = model_clf.predict_proba(input_data)[0][1]
            category = "High" if pred == 1 else "Low"
            st.markdown(f"<span class='metric-value'>{category}</span> (Prob: {prob:.2f})", unsafe_allow_html=True)