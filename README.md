# 🌊 Hyperspectral Soil Salinity Detector - Superpixels SLIC

**Détection précise salinité sols agricoles via images hyperspectrales + segmentation superpixels.**  
Mapping EC (dS/m) pour irrigation durable Île-de-France / Vendée. **VNIR-SWIR 100+ bandes → Superpixels → Indices SI → ML.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF1493?logo=streamlit)](https://hyperspectral-salinity.streamlit.app)
[![Scikit-image](https://img.shields.io/badge/Scikit-image-FF0000?logo=scikit-image)](https://scikit-image.org)
[![Spectral](https://img.shields.io/badge/Spectral-Orange?logo=python)](https://spectralpython.net)

## 🎯 Innovation
❓ **Salinisation sols agricoles (EC >4 dS/m) détectée pixel par pixel?**  
✅ **Superpixels SLIC** → Features robustes (NIR/SWIR brightness, NDSI) → **RF Régression** EC.  
Réduit bruit, gère hétérogénéité sols. Basé USGS/Hyperion + lit. 2026 [web:29].

## 🚀 Installation
```bash
git clone https://github.com/salimklibi/hyperspectral_salinity_detector
cd hyperspectral_salinity_detector
pip install -r requirements.txt  # streamlit scikit-image spectral-python scikit-learn plotly pandas spectral

streamlit run hyperspectral_salinity_superpixels.py
