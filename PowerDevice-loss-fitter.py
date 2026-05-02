import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - Full Suite", layout="wide")

# ==========================================
# 核心定義：曲線類別 (支援 Conduction 與 Switching)
# ==========================================
class PowerLossCurve:
    def __init__(self, name, model_type="Method_SW3", temp_ref=298.15):
        self.name = name
        self.model_type = model_type
        self.temp_ref = temp_ref # 該曲線對應的溫度 (K 或 C)
        self.params = None  
        self.raw_pixel_points = [] 
        self.real_data_points = [] 

    def set_params(self, params):
        self.params = params

    def get_value(self, current_i):
        """根據模型計算 Y 值 (Energy 或 Voltage)"""
        if self.params is None: return 0.0
        i = abs(current_i)
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2
            A, B, C = self.params
            return A + B * i + C * (i ** 2)
        elif self.model_type == "Method_Con1 (Linear V-I)":
            # v = V_threshold + R_on * i
            v_th, r_on = self.params
            return v_th + r_on * i
        return 0.0

# ==========================================
# Session State 初始化
# ==========================================
if "curve_objects" not in st.session_state:
    st.session_state.curve_objects = {}  
if "calib_pts" not in st.session_state:
    st.session_state.calib_pts = [] 

st.title("⚡ Power Electronics Loss Evaluator (Switching & Conduction)")

# ==========================================
# Sidebar：管理區
# ==========================================
st.sidebar.header("🛠️ 1. 曲線管理")
new_name = st.sidebar.text_input("曲線名稱 (如 IGBT_Vce_150C)", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("擬合模型", [
    "Method_SW3 (Energy: A+Bi+Ci^2)", 
    "Method_Con1 (Linear V-I: Vth + R*i)"
])
t_ref = st.sidebar.number_input("該數據對應的溫度 Tj", value=150.0)

if st.sidebar.button("➕ 新增曲線"):
    st.session_state.curve_objects[new_name] = PowerLossCurve(new_name, model_choice, t_ref)
    st.session_state.current_curve_name = new_name
    st.rerun()

all_names = list(st.session_state.curve_objects.keys())
if all_names:
    selected_name = st.sidebar.selectbox("編輯對象", all_names, index=0)
    current_obj = st.session_state.curve_objects[selected_name]
    if st.sidebar.button("🗑️ 刪除曲線"):
        del st.session_state.curve_objects[selected_name]
        st.rerun()
else:
    st.stop()

# ==========================================
# 主畫面：標定與擷取
# ==========================================
uploaded_file = st.file_uploader("2. 上傳 Datasheet V-I 或 Energy 曲線圖", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file); width, height = img.size
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"📍 擷取中: {current_obj.name}")
        value = streamlit_image_coordinates(img, key="img_click")
        if value:
            curr = (value["x"], value["y"])
            if len(st.session_state.calib_pts) < 2:
                if not st.session_state.calib_pts or curr != st.session_state.calib_pts[-1]:
                    st.session_state.calib_pts.append(curr); st.rerun()
            else:
                if not current_obj.raw_pixel_points or curr != current_obj.raw_pixel_points[-1]:
                    current_obj.raw_pixel_points.append(curr)
    with col2:
        st.subheader("📏 3. 座標校準")
        x_max = st.number_input("X 軸最大值 (A)", value=800.0)
        y_max = st.number_input("Y 軸最大值 (mJ 或 V)", value=5.0)
        if len(st.session_state.calib_pts) == 2:
            p0, pm = st.session_state.calib_pts
            sx, sy = x_max / (pm[0]-p0[0]), y_max / (pm[1]-p0[1])
            current_obj.real_data_points = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in current_obj.raw_pixel_points]
            st.dataframe(pd.DataFrame(current_obj.real_data_points, columns=["X","Y"]), height=150)

# ==========================================
# 擬合與損耗計算 (Mode B 整合導通損耗)
# ==========================================
if len(st.session_state.calib_pts) == 2 and current_obj.real_data_points:
    st.divider()
    if st.button("🚀 執行擬合", type="primary"):
        x, y = np.array([p[0] for p in current_obj.real_data_points]), np.array([p[1] for p in current_obj.real_data_points])
        if "SW3" in current_obj.model_type:
            popt, _ = curve_fit(lambda i, A, B, C: A + B*i + C*i**2, x, y)
        else:
            popt, _ = curve_fit(lambda i, Vth, Ron: Vth + Ron*i, x, y) # Eq 13
        current_obj.set_params(popt); st.success("擬合完成！")

    # --- 進階計算器 ---
    st.subheader("📋 4. 全域損耗分析 (含導通與切換)")
    objs = {n: o for n, o in st.session_state.curve_objects.items() if o.params is not None}
    if objs:
        f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0)
        tj_target = st.number_input("目標工作溫度 Tj (°C)", value=125.0)
        i_peak = st.number_input("正弦波峰值電流 I_peak (A)", value=400.0)
        
        # 正弦波參數
        i_avg = i_peak / np.pi # 半波平均電流
        i_rms = i_peak / 2     # 半波 RMS 電流
        
        p_sw_total = 0.0
        p_con_total = 0.0
        
        for name, obj in objs.items():
            # 這裡簡化處理：若有多個溫度曲線，程式會自動識別 (未來可加入 Eq 17/18 插值)
            if "SW3" in obj.model_type:
                e_avg = np.mean([obj.get_value(i_peak * np.sin(phi)) for phi in np.linspace(0, np.pi, 100)])
                p_sw_total += e_avg * f_sw * 1e-3 #[cite: 2]
            else:
                # 使用 Eq 16: Pcon = Ron*Irms^2 + Vth*Iave
                v_th, r_on = obj.params
                p_con = r_on * (i_rms**2) + v_th * i_avg
                p_con_total += p_con
        
        c1, c2 = st.columns(2)
        c1.metric("總切換損耗 P_sw", f"{p_sw_total:.2f} W")
        c2.metric("總導通損耗 P_con", f"{p_con_total:.2f} W")
        st.metric("🔥 總發熱量 P_total", f"{p_sw_total + p_con_total:.2f} W")

    # 波形圖
    fig, ax1 = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, np.pi, 200); i_w = i_peak * np.sin(t)
    ax1.plot(t, i_w, 'b--', alpha=0.3, label='Current i(t)')
    ax2 = ax1.twinx()
    for n, o in objs.items():
        ax2.plot(t, [o.get_value(v) for v in i_w], label=n)
    ax1.set_xlabel("Phase"); ax1.legend(); st.pyplot(fig)