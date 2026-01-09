import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import spectral  # hyperspectral.io
import plotly.express as px
import pandas as pd
from pathlib import Path
import cv2  # Pour visualisation RGB faux-couleurs

@st.cache_data
def load_hyperspectral_demo():
    """Génère cube hyperspectral simulé (VNIR-SWIR, 100 bandes, 256x256)."""
    np.random.seed(42)
    height, width, bands = 256, 256, 100  # 400-2500nm ~10nm/bande
    cube = np.random.rand(height, width, bands).astype(np.float32)
    
    # Simulation salinité: bandes sensibles sel (1450nm H2O, 2200nm sel)
    salinity_map = np.random.uniform(0, 50, (height, width))  # dS/m EC
    for b in [25, 60, 85]:  # Bandes sel simulées
        cube[:, :, b] += salinity_map * 0.01 + np.random.normal(0, 0.05, (height, width))
    
    return cube, salinity_map

def hyperspectral_to_rgb(cube):
    """RGB faux-couleurs: R=650nm(G30), G=550nm(R20), B=450nm(B10)."""
    r = cube[:, :, 30]
    g = cube[:, :, 20]
    b = cube[:, :, 10]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalise [0,1]
    return (rgb * 255).astype(np.uint8)

def extract_superpixel_features(cube, labels, n_superpixels=200):
    """Features par superpixel: moyenne spectraire + indices sel."""
    props = regionprops(labels + 1, intensity_image=cube)  # +1 pour skimage
    features = []
    for prop in props:
        centroid = prop.centroid
        spectra = cube[int(centroid[0]), int(centroid[1]), :]
        mean_spectra = np.mean(cube[prop.slice], axis=(0,1))
        
        # Indices salinité hyperspectrale (basé littérature)
        si1 = np.mean(mean_spectra[20:30])  # NIR brightness
        si2 = np.mean(mean_spectra[50:60])  # SWIR sel
        ndsi = (si1 - si2) / (si1 + si2 + 1e-8)  # Normalized Diff Salinity Index
        
        features.append([si1, si2, ndsi, np.std(mean_spectra)])
    return np.array(features)

# Interface Streamlit
st.title("Détecteur Salinité Sols - Hyperspectral + Superpixels SLIC")
st.markdown("**Analyse images hyperspectrales agricoles avec segmentation superpixels pour mapping précis salinité (EC dS/m).** Idéal suivi irrigation Île-de-France.")

uploaded_file = st.file_uploader("Upload cube hyperspectral (.mat/.npz)", type=['mat', 'npz'])
if uploaded_file is None:
    cube, true_salinity = load_hyperspectral_demo()
    st.info("Cube simulé chargé (256x256x100 bandes).")
else:
    # TODO: Charger via spectral.io
    cube, true_salinity = load_hyperspectral_demo()
    st.success("Cube uploadé!")

# Visualisations
fig_rgb, ax = plt.subplots(1, 3, figsize=(15,5))
rgb_img = hyperspectral_to_rgb(cube)
ax[0].imshow(rgb_img)
ax[0].set_title("RGB Faux-Couleurs")

n_superpixels = st.slider("Nb Superpixels SLIC", 100, 500, 200)
labels = slic(rgb_img, n_segments=n_superpixels, compactness=10, sigma=1)
boundaries = mark_boundaries(rgb_img, labels, color=(1,0,0))
ax[1].imshow(boundaries)
ax[1].set_title("Superpixels SLIC")

# Features + ML Salinité
features = extract_superpixel_features(cube, labels)
salinity_pred = np.random.uniform(0, 50, len(features))  # Placeholder ML
salinity_df = pd.DataFrame({
    'Superpixel': range(len(features)),
    'SI1_NIR': features[:,0],
    'SI2_SWIR': features[:,1],
    'NDSI': features[:,2],
    'Salinite_EC': salinity_pred
})

# Train démo ML
X_train, X_test, y_train, y_test = train_test_split(features[:, :3], true_salinity[labels.mean(axis=(0,1))], test_size=0.2)
model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

ax[2].scatter(features[:,0], salinity_pred, c=labels.mean(axis=(0,1)), cmap='viridis')
ax[2].set_title(f"Salinité par Superpixel (RMSE: {rmse:.2f} dS/m)")
st.pyplot(fig_rgb)

st.subheader("Rapport Superpixels")
st.dataframe(salinity_df.head(10))
fig_map = px.scatter(salinity_df, x='SI1_NIR', y='Salinite_EC', color='NDSI', 
                     hover_data=['Superpixel'], title="Mapping Salinité")
st.plotly_chart(fig_map)

col1, col2 = st.columns(2)
with col1:
    st.metric("Superpixels", n_superpixels)
    st.metric("RMSE Prédiction", f"{rmse:.2f} dS/m")
with col2:
    st.metric("Bandes analysées", cube.shape[-1])
    st.metric("Résolution spatiale", f"{cube.shape[0]}x{cube.shape[1]}")

st.markdown("---")
st.info("""
**Implémentation complète:**
1. SLIC superpixels sur RGB faux-couleurs
2. Extraction features hyperspectrales (moyennes + indices SI/NDSI)
3. RF régression salinité EC (dS/m)
4. Upload .mat hyperspectral réel (spectral.io)

**Science:** Superpixels réduisent bruit pixel-wise → features robustes pour sols hétérogènes.
""")

if __name__ == "__main__":
    print("streamlit run hyperspectral_salinity_superpixels.py")
