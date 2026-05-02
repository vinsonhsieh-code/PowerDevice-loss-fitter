import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Evaluator - R&D Pro", layout="wide")

# ==========================================
# 1. 核心類別定義
# ==========================================
class SwitchingCurve:
    """切換損耗 (Method SW3: E = A + B*i + C*i^2)"""
    def __init__(self, name):
        self.name = name
        self.params = None 
        self.raw_pts = []
        self.real_pts = []

    def get_val(self, i):
        if self.params is None: return 0.0
        i_abs = abs(i)
        return self.params[0] + self.params[1]*i_abs + self.params[2]*(i_abs**2)

class ConductionDevice:
    """導通損耗 (Method Con1 & Eq 19)"""
    def __init__(self, name):
        self.name = name
        self.t_min_c = 25.0
        self.t_max_c = 150.0
        self.fit_tmin = None # [Vx, Rx]
        self.fit_tmax = None # [Vx, Rx]
        self.raw_pts_tmin, self.raw_pts_tmax = [], []
        self.real_pts_tmin, self.real_pts_tmax = [], []

    def get_eq19_params(self, tj_c):
        """解算當前溫度下之動態電阻 Rx 與閾值電壓 Vx (Eq 17, 18)"""
        if self.fit_tmin is None or self.fit_tmax is None: return None, None
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        T1, T2 = self.t_min_c + 273.15, self.t_max_c + 273.15
        Tj = tj_c + 273.15
        denom = T1 - T2
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * Tj
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * Tj
        return vx_tj, rx_tj

    def calc_pcon(self, tj_c, i_rms, i_avg):
        """計算導通功率損耗 (Eq 19)"""
        vx, rx = self.get_eq19_params(tj_c)
        if vx is None: return 0.0
        return rx * (i_rms**2) + vx * i_avg

# ==========================================
# 2. Session State 管理 (維持現狀)
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")
for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("🚀 Power Device 綜合損耗分析系統")

# ==========================================
# 3. Sidebar (維持現狀)
# ==========================================
with st.sidebar:
    st.header("📊 曲線管理")
    sw_n = st.text_input("新增切換曲線 (如 Eon)", "Eon_150C")
    if st.button("➕ 新增曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    st.divider()
    st.subheader("🌡️ 導通損溫度基準 (°C)")
    tc1 = st.number_input("T_min (°C)", value=25.0)
    tc2 = st.number_input("T_max (°C)", value=150.0)
    st.session_state.igbt_obj.t_min_c = st.session_state.fwd_obj.t_min_c = tc1
    st.session_state.igbt_obj.t_max_c = st.session_state.fwd_obj.t_max_c = tc2
    if st.button("🔄 重置標定"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 三大作業區 (切換損、IGBT導通、FWD導通)
# ==========================================
# 此處代碼維持之前標定與擬合功能不變... (省略中間重複的 render 部分以節省空間，請沿用上一版)
# 確保包含 render_zone 或對應的上傳與擬合邏輯

# ==========================================
# 5. 全域分析與結果排版優化 (核心修正區)
# ==========================================
st.divider()
st.header("🔍 全域損耗動態評估")

if st.session_state.igbt_obj.fit_tmax and st.session_state.fwd_obj.fit_tmax:
    # A. 輸入參數區
    with st.container():
        c1, c2, c3 = st.columns(3)
        tj_c = c1.number_input("工作接面溫度 Tj (°C)", value=100.0)
        i_pk = c2.number_input("峰值電流 I_peak (A)", value=100.0)
        fsw  = c3.number_input("切換頻率 f_sw (Hz)", value=15000.0)

    # B. 中間參數計算說明
    irms = i_pk / 2
    iavg = i_pk / np.pi
    with st.expander("💡 電流參數計算細節 (Eq 10, 16)", expanded=True):
        col_ia, col_ib = st.columns(2)
        col_ia.latex(r"I_{rms} = \frac{I_{peak}}{2} = " + f"{irms:.2f} A")
        col_ib.latex(r"I_{avg} = \frac{I_{peak}}{\pi} = " + f"{iavg:.2f} A")

    # C. 核心損耗計算與對齊排版
    pcon_igbt = st.session_state.igbt_obj.calc_pcon(tj_c, irms, iavg)
    pcon_fwd  = st.session_state.fwd_obj.calc_pcon(tj_c, irms, iavg)
    psw_total = 0.0
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None:
            e_avg = np.mean([o.get_val(i_pk * np.sin(t)) for t in np.linspace(0, np.pi, 100)])
            psw_total += e_avg * fsw * 1e-3

    st.markdown("---")
    
    # 使用 3 欄式卡片佈局
    card_col1, card_col2, card_col3 = st.columns(3)
    
    with card_col1:
        with st.container(border=True):
            st.subheader("IGBT 導通損耗")
            st.metric("P_con,IGBT", f"{pcon_igbt:.2f} W")
            st.caption("方程式說明 (Eq 19):")
            st.latex(r"R_{ce}(T_j) \cdot I_{rms}^2 + V_{ce}(T_j) \cdot I_{avg}")

    with card_col2:
        with st.container(border=True):
            st.subheader("FWD 導通損耗")
            st.metric("P_con,FWD", f"{pcon_fwd:.2f} W")
            st.caption("方程式說明 (Eq 19):")
            st.latex(r"R_{f}(T_j) \cdot I_{rms}^2 + V_{f}(T_j) \cdot I_{avg}")

    with card_col3:
        with st.container(border=True):
            st.subheader("總切換損耗")
            st.metric("P_sw", f"{psw_total:.2f} W")
            st.caption("方程式說明 (SW3 Integration):")
            st.latex(r"f_{sw} \cdot \frac{1}{\pi} \int_{0}^{\pi} E_{sw}(i(\theta)) d\theta")

    # D. 總結結果
    st.success(f"### 總預估損耗 (P_total): **{psw_total + pcon_igbt + pcon_fwd:.2f} W**")

else:
    st.info("💡 指示：請完成導通特性擬合後，此處將自動呈現結構化分析結果。")

# 擬合曲線可視化 (維持分開呈現)
st.divider()
st.subheader("📊 擬合曲線品質核對")
v_col1, v_col2, v_col3 = st.columns(3)
# 此處保留之前的繪圖代碼 (ax_sw, ax_igbt, ax_fwd)...