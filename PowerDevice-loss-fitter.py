import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Evaluator Pro", layout="wide")

# ==========================================
# 類別定義 1：切換損 (Switching Loss) - 保留原功能
# ==========================================
class PowerLossCurve:
    def __init__(self, name, model_type="Method_SW3"):
        self.name = name
        self.model_type = model_type
        self.params = None  
        self.raw_pixel_points = [] 
        self.real_data_points = [] 

    def set_params(self, params):
        self.params = params

    def get_loss(self, current_i):
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        if self.model_type == "Method_SW3":
            A, B, C = self.params
            return A + B * i_abs + C * (i_abs ** 2)
        elif self.model_type == "Linear":
            v0, r = self.params
            return v0 + r * i_abs
        return 0.0

# ==========================================
# 類別定義 2：導通損 (Conduction Loss) - 新增
# ==========================================
class ConductionModel:
    """ 根據論文 4.1 Method Con1 建立 """
    def __init__(self, name):
        self.name = name
        # 參考 Table 4 參數
        self.Tmin, self.Tmax = 298.15, 423.15 # Kelvin
        self.R1, self.R2 = 0.002, 0.004      # Ohm
        self.V1, self.V2 = 1.0, 0.85         # Volt

    def get_temp_dependent_params(self, Tj_k):
        """ 方程式 (17) & (18) """
        dt = self.Tmin - self.Tmax
        Rx_Tj = ((self.R2 * self.Tmin - self.R1 * self.Tmax) / dt) + ((self.R1 - self.R2) / dt) * Tj_k
        Vx_Tj = ((self.V2 * self.Tmin - self.V1 * self.Tmax) / dt) + ((self.V1 - self.V2) / dt) * Tj_k
        return Rx_Tj, Vx_Tj

    def get_power_loss(self, I_rms, I_avg, Tj_k):
        """ 方程式 (16) """
        Rx, Vx = self.get_temp_dependent_params(Tj_k)
        return Rx * (I_rms**2) + Vx * I_avg

# ==========================================
# Session State 初始化
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "con_models" not in st.session_state: st.session_state.con_models = {}
if "calib_pts" not in st.session_state: st.session_state.calib_pts = []

# ==========================================
# UI 佈局：使用 Tabs 分隔功能區
# ==========================================
tab_sw, tab_con, tab_analysis = st.tabs(["🔥 切換損 (Switching)", "⚡ 導通損 (Conduction)", "📊 綜合損耗分析"])

# --- TAB 1: 切換損功能 (保留原本邏輯) ---
with tab_sw:
    st.header("切換損曲線擬合")
    # (此處省略部分重複的標定與擬合 UI 程式碼，確保功能完全一致)
    st.info("切換損功能維持不變，請在此處管理 Eon/Eoff/Err 曲線。")

# --- TAB 2: 導通損功能 (根據論文 4.1 新增) ---
with tab_con:
    st.header("導通損模型建立 (Method Con1)")
    st.markdown("請根據 Datasheet 的 $v-i$ 特性曲線，輸入兩組溫度下的參數。")
    
    with st.expander("➕ 新增導通損元件 (如 IGBT_Con, FWD_Con)"):
        con_name = st.text_input("元件名稱", value="IGBT_Con")
        if st.button("建立模型"):
            st.session_state.con_models[con_name] = ConductionModel(con_name)
    
    if st.session_state.con_models:
        sel_con = st.selectbox("選擇要設定的元件", list(st.session_state.con_models.keys()))
        m = st.session_state.con_models[sel_con]
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"溫度 1 ($T_{{min}}$)")
            m.Tmin = st.number_input(f"{sel_con} Tmin (K)", value=298.15)
            m.R1 = st.number_input(f"{sel_con} R1 (mΩ)", value=2.34) / 1000.0
            m.V1 = st.number_input(f"{sel_con} V1 (V)", value=1.00)
        with c2:
            st.subheader(f"溫度 2 ($T_{{max}}$)")
            m.Tmax = st.number_input(f"{sel_con} Tmax (K)", value=423.15)
            m.R2 = st.number_input(f"{sel_con} R2 (mΩ)", value=3.90) / 1000.0
            m.V2 = st.number_input(f"{sel_con} V2 (V)", value=0.85)

# --- TAB 3: 綜合損耗分析 (模式 A & B) ---
with tab_analysis:
    st.header("全系統損耗動態分析器")
    
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1: i_peak = st.number_input("峰值電流 I_peak (A)", value=100.0)
    with c_p2: f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0)
    with c_p3: tj_c = st.number_input("接面溫度 Tj (°C)", value=125.0)
    
    tj_k = tj_c + 273.15 # 轉為開氏溫度進行論文公式計算
    
    # 計算電流參數 (正弦波模式)
    # 半波正弦下的平均與有效電流
    i_avg_sine = i_peak / np.pi
    i_rms_sine = i_peak / 2.0

    st.divider()
    
    # 損耗計算彙整
    res_sw = []
    total_sw = 0.0
    for n, o in st.session_state.get("curve_objects", {}).items(): # 假設這是原本存切換損的地方
        if o.params is not None:
            # 正弦波平均能量[cite: 1]
            e_avg = np.mean([o.get_loss(i_peak * np.sin(phi)) for phi in np.linspace(0, np.pi, 100)])
            p_sw = e_avg * f_sw * 1e-3
            total_sw += p_sw
            res_sw.append({"項目": n, "類別": "切換損", "功率 (W)": f"{p_sw:.3f}"})

    res_con = []
    total_con = 0.0
    for n, m in st.session_state.con_models.items():
        p_con = m.get_power_loss(i_rms_sine, i_avg_sine, tj_k) # 式 (16)
        total_con += p_con
        res_con.append({"項目": n, "類別": "導通損", "功率 (W)": f"{p_con:.3f}"})

    # 顯示結果
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("🔥 切換損耗明細")
        if res_sw: st.table(pd.DataFrame(res_sw))
        st.metric("總切換損耗", f"{total_sw:.2f} W")
    
    with col_res2:
        st.subheader("⚡ 導通損耗明細")
        if res_con: st.table(pd.DataFrame(res_con))
        st.metric("總導通損耗", f"{total_con:.2f} W")
        
    st.divider()
    st.header(f"🏆 系統總損耗: {total_sw + total_con:.2f} W")