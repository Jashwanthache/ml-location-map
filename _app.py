import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Location Classification Map", layout="wide")
st.title("📍 Location-Based Classification with XGBoost")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# 2. Train Model
X = df[["lat", "lon", "feature1", "feature2"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{acc:.2f}")

# 3. Create Map
st.write("### Interactive Map")
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=6)

marker_cluster = MarkerCluster().add_to(m)

for i, row in df.iterrows():
    color = "red" if row["target"] == 1 else "blue"
    popup_text = f"Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}<br>Class: {row['target']}"
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=popup_text,
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background:white; padding: 10px; border:2px solid grey;">
<h4>Legend</h4>
<p><i class="fa fa-circle" style="color:red"></i> Class 1</p>
<p><i class="fa fa-circle" style="color:blue"></i> Class 0</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=700, height=500)
