import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Crystallinity Calculator Pipeline", layout="wide")

# --------------------------------------------------------------------------
# Core Math Functions (From uploaded files)
# --------------------------------------------------------------------------
def backcor(n, y, ord, s, fct):
    """Mazet background correction"""
    n = np.asarray(n, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    N = len(n)

    i_sort = np.argsort(n)
    n_sorted, y_sorted = n[i_sort], y[i_sort]
    maxy, miny = np.max(y_sorted), np.min(y_sorted)
    dely = (maxy - miny) / 2.0
    if dely == 0: dely = 1.0

    n_resc = 2.0 * (n_sorted - n_sorted[-1]) / (n_sorted[-1] - n_sorted[0]) + 1.0
    y_resc = (y_sorted - maxy) / dely + 1.0

    p = np.arange(ord + 1)
    T = n_resc[:, None] ** p[None, :]
    Tinv = np.linalg.pinv(T.T @ T) @ T.T

    a = Tinv @ y_resc
    z = T @ a
    alpha = 0.99 * 0.5
    it = 0
    zp = np.ones(N)

    max_iter = 5000
    while np.sum((z - zp) ** 2) / max(np.sum(zp ** 2), 1e-300) > 1e-9:
        it += 1
        if it > max_iter: break
        zp = z.copy()
        res = y_resc - z

        if fct == 'sh':
            d = ((res * (2 * alpha - 1)) * (np.abs(res) < s) + (-alpha * 2 * s - res) * (res <= -s) + (alpha * 2 * s - res) * (res >= s))
        elif fct == 'ah':
            d = ((res * (2 * alpha - 1)) * (res < s) + (alpha * 2 * s - res) * (res >= s))
        elif fct == 'stq':
            d = ((res * (2 * alpha - 1)) * (np.abs(res) < s) - res * (np.abs(res) >= s))
        elif fct == 'atq':
            d = ((res * (2 * alpha - 1)) * (res < s) - res * (res >= s))

        a = Tinv @ (y_resc + d)
        z = T @ a

    j_unsort = np.argsort(i_sort)
    z = (z[j_unsort] - 1) * dely + maxy
    return z

def load_spectrum(uploaded_file):
    """Parse uploaded txt files"""
    content = uploaded_file.getvalue().decode("utf-8")
    data = np.loadtxt(io.StringIO(content))
    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    return x[order], y[order]

def get_intensity(x_target, x, y):
    """Interpolate intensity at specific peak"""
    return np.interp(x_target, x, y)

def create_download_link(x, y, filename):
    """Format data for download to match input exactly"""
    df = pd.DataFrame({'x': x, 'y': y})
    csv = df.to_csv(index=False, header=False, sep='\t')
    return csv

# --------------------------------------------------------------------------
# UI Structure
# --------------------------------------------------------------------------
st.title("Spectra Processing & Crystallinity Pipeline")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Baseline Correction", 
    "2. Scale Spectra", 
    "3. Define Formula", 
    "4. Bulk Calculate Unknowns"
])

# --- TAB 1: BASELINE CORRECTION ---
with tab1:
    st.header("Interactive Baseline Correction")
    bl_file = st.file_uploader("Upload a raw spectrum (.txt)", key="bl_file")
    
    if bl_file:
        x, y = load_spectrum(bl_file)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            ord_val = st.slider("Polynomial Order", 0, 10, 4)
            thresh_log = st.slider("Log10(Threshold)", -4.0, 0.0, -2.0)
            fct_val = st.radio("Cost Function", ['sh', 'ah', 'stq', 'atq'], index=3)
            s_val = 10**thresh_log
            
        with col2:
            z = backcor(x, y, ord_val, s_val, fct_val)
            y_corr = y - z
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, label='Raw', color='#3b6ea5', alpha=0.5)
            ax.plot(x, z, label='Baseline', color='#d1495b')
            ax.plot(x, y_corr, label='Corrected', color='#2a9d8f')
            ax.legend()
            st.pyplot(fig)
            
        csv_data = create_download_link(x, y_corr, f"corrected_{bl_file.name}")
        st.download_button("Download Corrected Spectrum", data=csv_data, file_name=f"corrected_{bl_file.name}", mime="text/plain")

# --- TAB 2: SCALE SPECTRA ---
with tab2:
    st.header("Scale Spectra (Manual)")
    col_ref, col_tar = st.columns(2)
    with col_ref:
        ref_file = st.file_uploader("Upload Reference Spectrum", key="ref_file")
    with col_tar:
        tar_file = st.file_uploader("Upload Target Spectrum (to be scaled)", key="tar_file")
        
    if ref_file and tar_file:
        x1, y1 = load_spectrum(ref_file)
        x2, y2 = load_spectrum(tar_file)
        
        scale_val = st.slider("Scale Multiplier (Linear)", 0.01, 5.0, 1.0, step=0.01)
        y2_scaled = y2 * scale_val
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x1, y1, label='Reference', color='#3b6ea5')
        ax.plot(x2, y2_scaled, label=f'Scaled Target (x{scale_val})', color='#d1495b', alpha=0.8)
        ax.legend()
        st.pyplot(fig)
        
        csv_data = create_download_link(x2, y2_scaled, f"scaled_{tar_file.name}")
        st.download_button("Download Scaled Target", data=csv_data, file_name=f"scaled_{tar_file.name}", mime="text/plain")

# --- TAB 3: DEFINE FORMULA ---
with tab3:
    st.header("Crystallinity Formula Generation")
    st.markdown("Define your pure references to generate the formula constants based on the literature derivation.")
    
    col1, col2 = st.columns(2)
    with col1:
        peak_amp = st.number_input("Amorphous Peak Position (ν_amp)", value=1680.0)
        pure_amp_file = st.file_uploader("Upload Pure Amorphous Reference (Baseline Corrected)", key="amp_ref")
    with col2:
        peak_cry = st.number_input("Crystalline Peak Position (ν_cry)", value=1698.0)
        pure_cry_file = st.file_uploader("Upload Pure Crystalline Reference (Baseline Corrected)", key="cry_ref")

    if pure_amp_file and pure_cry_file:
        xa, ya = load_spectrum(pure_amp_file)
        xc, yc = load_spectrum(pure_cry_file)
        
        I_amp_v1 = get_intensity(peak_amp, xa, ya)
        I_amp_v2 = get_intensity(peak_cry, xa, ya)
        
        I_cry_v1 = get_intensity(peak_amp, xc, yc)
        I_cry_v2 = get_intensity(peak_cry, xc, yc)
        
        # Calculate constants based on derivation
        # K_amp ratio = I_amp(v2) / I_amp(v1)
        K_amp_ratio = I_amp_v2 / I_amp_v1 if I_amp_v1 != 0 else 0
        # K_cry ratio = I_cry(v1) / I_cry(v2) -> matching image derivation for inversion
        K_cry_ratio = I_cry_v1 / I_cry_v2 if I_cry_v2 != 0 else 0
        
        st.success("Formula Constants Extracted!")
        st.write(f"**Amorphous Ratio ($K_a^{{\\nu_2}}/K_a^{{\\nu_1}}$):** {K_amp_ratio:.4f}")
        st.write(f"**Crystalline Ratio ($K_c^{{\\nu_1}}/K_c^{{\\nu_2}}$):** {K_cry_ratio:.4f}")
        
        # Save to session state so Tab 4 can use it
        st.session_state['formula'] = {
            'v_amp': peak_amp,
            'v_cry': peak_cry,
            'k_amp_r': K_amp_ratio,
            'k_cry_r': K_cry_ratio
        }

# --- TAB 4: BULK CALCULATE ---
with tab4:
    st.header("Bulk Unknown Sample Calculation")
    
    if 'formula' not in st.session_state:
        st.warning("Please generate a formula in Tab 3 first, or manually enter the constants below.")
        man_v_amp = st.number_input("v_amp", value=1680.0)
        man_v_cry = st.number_input("v_cry", value=1689.0)
        man_const_a = st.number_input("Subtracted Constant (e.g., 0.307)", value=0.307)
        man_const_b = st.number_input("Denominator multiplier (e.g., 0.879)", value=0.879)
        man_const_c = st.number_input("Denominator addition (e.g., 2.222)", value=2.222)
        use_manual = st.checkbox("Use Manual Formula")
    else:
        st.success(f"Using formula generated from Tab 3. (Peaks: {st.session_state['formula']['v_amp']}, {st.session_state['formula']['v_cry']})")
        use_manual = False

    do_baseline = st.checkbox("Apply auto-baseline correction to all unknowns before calculating?")
    if do_baseline:
        st.info("Uses default Mazet parameters: order=4, s=0.01, fct='atq'")

    bulk_files = st.file_uploader("Upload Unknown Samples (.txt)", accept_multiple_files=True)
    
    if bulk_files and st.button("Calculate Bulk Crystallinity"):
        results = []
        for file in bulk_files:
            x, y = load_spectrum(file)
            
            if do_baseline:
                z = backcor(x, y, 4, 0.01, 'atq')
                y = y - z
            
            if use_manual:
                v_amp, v_cry = man_v_amp, man_v_cry
            else:
                v_amp = st.session_state['formula']['v_amp']
                v_cry = st.session_state['formula']['v_cry']
                
            i_amp = get_intensity(v_amp, x, y)
            i_cry = get_intensity(v_cry, x, y)
            
            r = i_cry / i_amp if i_amp != 0 else np.nan
            
            if use_manual:
                xc = (r - man_const_a) / (man_const_b * r + man_const_c)
            else:
                # Custom calculation based on literature derivation standard
                # Adjust this math to fit your specific derived slope/equation
                ka_ratio = st.session_state['formula']['k_amp_r']
                kc_ratio = st.session_state['formula']['k_cry_r']
                xc = (r - ka_ratio) / (kc_ratio * r + 1) # Generalized placeholder
                
            results.append({
                "Filename": file.name,
                f"I({v_amp})": i_amp,
                f"I({v_cry})": i_cry,
                "Ratio (r)": r,
                "Crystallinity (Xc)": xc
            })
            
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
        
        csv = df_results.to_csv(index=False)
        st.download_button("Download Results CSV", data=csv, file_name="crystallinity_results.csv", mime="text/csv")