import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 1. PAGE SETUP
st.set_page_config(page_title="SBES Professional Processor", layout="wide")

# Custom CSS for a cleaner, professional look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* Styling the metric boxes */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True) # FIXED: Changed to unsafe_allow_html

st.title("🚢 SBES Professional Data Processor")

# 2. SESSION STATE MANAGEMENT
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'last_file' not in st.session_state: st.session_state.last_file = None
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0

def reset_ui():
    st.session_state.reset_key += 1
    st.rerun()

# --- SIDEBAR: ORGANIZED CONTROLS ---
st.sidebar.header("📁 Project Management")
apply_globally = st.sidebar.toggle("Apply Settings Globally", value=False)

if st.sidebar.button("Clear All Data", use_container_width=True):
    st.session_state.uploader_key += 1
    reset_ui()

st.sidebar.divider()
st.sidebar.header("⚓ Vertical Datum")
tide = st.sidebar.number_input("Tide Correction (m)", value=0.00, step=0.01)
draft = st.sidebar.number_input("Transducer Draft (m)", value=0.00, step=0.01)

st.sidebar.divider()
st.sidebar.header("🛠️ Processing Engine")
rk = str(st.session_state.reset_key)

# Filter parameters organized in Expanders
with st.sidebar.expander("Range Filter", expanded=True):
    s_min = st.number_input("Min Depth", value=0.0, key="min"+rk)
    s_max = st.number_input("Max Depth", value=500.0, key="max"+rk)

with st.sidebar.expander("Automated Cleaning (Despiking)", expanded=False):
    s_clean = st.checkbox("Enable Despiking", value=False, key="c"+rk)
    s_cwin = st.slider("Window Size", 3, 31, 7, step=2)
    s_csens = st.slider("Sensitivity (Sigma)", 1.0, 5.0, 3.0)

with st.sidebar.expander("Surface Smoothing", expanded=False):
    s_smooth = st.checkbox("Enable Smoothing", value=False, key="s"+rk)
    s_swin = st.slider("Smoothing Power", 3, 21, 5)

# --- DATA PARSING ENGINE ---
uploaded_files = st.file_uploader("Upload HYPACK .RAW files", type=['RAW'], accept_multiple_files=True, key=str(st.session_state.uploader_key))

def parse_sbes(file):
    data = []
    e, n = None, None
    try:
        content = file.getvalue().decode("utf-8")
        for line in content.splitlines():
            p = line.split()
            if not p: continue
            if p[0] == 'POS': e, n = float(p[3]), float(p[4])
            elif p[0] == 'EC1' and e is not None:
                # Vertical calculation: Depth + Tide + Draft
                data.append({'E': e, 'N': n, 'Z': float(p[3]) + tide + draft})
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(data)

if uploaded_files:
    f_names = [f.name for f in uploaded_files]
    sel_f = st.selectbox("Current Working File", f_names)

    # Individual file reset logic
    if not apply_globally and sel_f != st.session_state.last_file:
        st.session_state.last_file = sel_f
        reset_ui()

    # Process files
    processed_dict = {}
    for f in uploaded_files:
        df = parse_sbes(f)
        if df.empty: continue
        
        df['Proc_Z'] = df['Z'].copy()
        df.loc[(df['Proc_Z'] < s_min) | (df['Proc_Z'] > s_max), 'Proc_Z'] = np.nan
        
        active = apply_globally or (f.name == sel_f)
        if active:
            if s_clean:
                med = df['Proc_Z'].rolling(window=s_cwin, center=True).median()
                std = df['Proc_Z'].rolling(window=s_cwin, center=True).std().replace(0, 0.001)
                df.loc[np.abs(df['Proc_Z'] - med) > (s_csens * std), 'Proc_Z'] = np.nan
                df['Proc_Z'] = df['Proc_Z'].interpolate(method='linear')
            if s_smooth:
                df['Proc_Z'] = df['Proc_Z'].rolling(window=s_swin, center=True).mean().ffill().bfill()
        
        processed_dict[f.name] = df

    # --- TABBED USER INTERFACE ---
    if sel_f in processed_dict:
        curr = processed_dict[sel_f]
        clean_df = curr[['E', 'N', 'Proc_Z']].dropna()

        # Key Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Raw Pings", len(curr))
        m2.metric("Cleaned Pings", len(clean_df))
        health = (len(clean_df)/len(curr)*100) if len(curr)>0 else 0
        m3.metric("Data Quality", f"{health:.1f}%")

        # Organized Tabs
        tab1, tab2, tab3 = st.tabs(["📉 Profile QC", "🗺️ Survey Map", "📄 Data Export"])

        with tab1:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(y=curr['Z'], mode='markers', name='Raw', marker=dict(color='red', size=2, opacity=0.3)))
            fig_p.add_trace(go.Scatter(y=curr['Proc_Z'], mode='lines', name='Cleaned', line=dict(color='#00f2ff', width=2)))
            fig_p.update_layout(yaxis=dict(autorange="reversed", title="Depth (m)"), 
                                xaxis_title="Ping Sequence", template="plotly_dark", height=550)
            st.plotly_chart(fig_p, use_container_width=True)

        with tab2:
            fig_m = px.scatter(clean_df, x='E', y='N', color='Proc_Z',
                               color_continuous_scale='Turbo', labels={'Proc_Z':'Depth'})
            fig_m.update_layout(template="plotly_dark", height=600, coloraxis_colorbar=dict(title="Depth (m)"))
            fig_m.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_m, use_container_width=True)

        with tab3:
            st.subheader("Data Preview & Download")
            st.dataframe(clean_df.head(50), use_container_width=True)
            st.download_button(f"📥 Export {sel_f}.xyz", clean_df.to_csv(index=False, header=False, sep=" "), f"{sel_f}.xyz")

    # Global Export Footer
    if processed_dict:
        st.divider()
        all_final = pd.concat([df[['E', 'N', 'Proc_Z']].dropna() for df in processed_dict.values()])
        st.download_button("💾 SAVE ENTIRE PROJECT (.XYZ)", all_final.to_csv(index=False, header=False, sep=" "), 
                           "Master_Survey_Cleaned.xyz", use_container_width=True)
