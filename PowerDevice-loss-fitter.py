import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別
# ==========================================
class PowerLossCurve:
    def __init__(self, name, category="Switching", model_type="Method_SW3"):
        self.name = name
        self.category = category # "Switching" (能量 mJ) 或 "Conduction" (電壓 V)
        self.model_type = model_type
        self.params = None  
        self.raw_pixel_points = [] 
        self.real_data_points = [] 

    def set_params(self, params):
        self.params = params

    def get_value(self, current_i):
        """
        計算瞬時值：
        - 若為 Switching：回傳能量 E(i) [mJ]
        - 若為 Conduction：回傳順向壓降 v(i) [V]
        """
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2
            A, B, C = self.params
            return A + B * i_abs + C * (i_abs ** 2)
        elif self.model_type == "Linear":
            # v = VX + RX*i
            v0, r = self.params
            return v0 + r * i_abs
        elif self.model_type == "Power":
            a, b = self.params
            return a * (i_abs ** b)
        return 0.0

    def get_equation_string(self):
        if self.params is None: return "尚未擬合"
        if self.model_type == "Method_SW3":
            return f"{self.params[0]:.6e} + {self.params[1]:.6e}*i + {self.params[2]:.6e}*i^2"
        elif self.model_type == "Linear":
            return f"{self.params[0]:.4f} + {self.params[1]:.4f}*i"
        return f"Params: {self.params}"

# ==========================================
# Session State 初始化
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}  
if "con_curves" not in st.session_state: st.session_state.con_curves = {}  
if "sw_calib" not in st.session_state: st.session_state.sw_calib = [] 
if "con_calib" not in st.session_state: st.session_state.con_calib = [] 

st.title("⚡ Power Device 損耗評估系統 (切換 + 導通)")

# ==========================================
# 一、 切換損耗區域 (Switching Loss Section)
# ==========================================
st.header("📋 1. 切換損耗評估 (Switching Loss Evaluation)")
sw_col_side, sw_col_main = st.columns([1, 4])

with sw_col_side:
    st.subheader("管理切換曲線")
    new_sw = st.text_input("切換曲線名稱", value=f"Esw_{len(st.session_state.sw_curves)+1}")
    if st.button("➕ 新增切換曲線"):
        st.session_state.sw_curves[new_sw] = PowerLossCurve(new_sw, "Switching", "Method_SW3")
    
    sw_list = list(st.session_state.sw_curves.keys())
    selected_sw = st.selectbox("編輯切換對象", sw_list) if sw_list else None
    if selected_sw and st.button("🗑️ 刪除切換曲線"):
        del st.session_state.sw_curves[selected_sw]
        st.rerun()

if selected_sw:
    sw_obj = st.session_state.sw_curves[selected_sw]
    uploaded_sw = st.file_uploader("上傳切換損耗圖檔 (能量 vs 電流)", type=["png","jpg"], key="sw_up")
    
    if uploaded_sw:
        img_sw = Image.open(uploaded_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"正在擷取: {selected_sw}")
            val_sw = streamlit_image_coordinates(img_sw, key="sw_img")
            if val_sw:
                pt = (val_sw["x"], val_sw["y"])
                if len(st.session_state.sw_calib) < 2:
                    if not st.session_state.sw_calib or pt != st.session_state.sw_calib[-1]:
                        st.session_state.sw_calib.append(pt)
                        st.rerun()
                else:
                    if not sw_obj.raw_pixel_points or pt != sw_obj.raw_pixel_points[-1]:
                        sw_obj.raw_pixel_points.append(pt)
        with c2:
            sw_xmax = st.number_input("X軸最大 (A)", value=1000.0, key="sw_xm")
            sw_ymax = st.number_input("Y軸最大 (mJ)", value=125.0, key="sw_ym")
            if len(st.session_state.sw_calib) == 2:
                p0, pm = st.session_state.sw_calib
                sx, sy = sw_xmax/(pm[0]-p0[0]), sw_ymax/(pm[1]-p0[1])
                sw_obj.real_data_points = [((p[0]-p0[0])*sx, (p[1]-p0[1])*sy) for p in sw_obj.raw_pixel_points]
                if st.button("🚀 擬合切換損"):
                    x, y = np.array([p[0] for p in sw_obj.real_data_points]), np.array([p[1] for p in sw_obj.real_data_points])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, x, y)
                    sw_obj.set_params(popt)
                st.write(f"已擷取 {len(sw_obj.raw_pixel_points)} 點")

st.divider()

# ==========================================
# 二、 導通損耗區域 (Conduction Loss Section)
# ==========================================
st.header("📋 2. 導通損耗評估 (Conduction Loss Evaluation)")
st.info("依據論文 Method Con1：使用順向壓降 $v-i$ 特性建立線性模型 $v_X = V_X + R_X \cdot i$。")

con_col_side, con_col_main = st.columns([1, 4])

with con_col_side:
    st.subheader("管理導通曲線")
    new_con = st.text_input("導通曲線名稱", value=f"Vcon_{len(st.session_state.con_curves)+1}")
    if st.button("➕ 新增導通曲線"):
        st.session_state.con_curves[new_con] = PowerLossCurve(new_con, "Conduction", "Linear")
    
    con_list = list(st.session_state.con_curves.keys())
    selected_con = st.selectbox("編輯導通對象", con_list) if con_list else None
    if selected_con and st.button("🗑️ 刪除導通曲線"):
        del st.session_state.con_curves[selected_con]
        st.rerun()

if selected_con:
    con_obj = st.session_state.con_curves[selected_con]
    uploaded_con = st.file_uploader("上傳導通特性圖檔 (壓降 vs 電流)", type=["png","jpg"], key="con_up")
    
    if uploaded_con:
        img_con = Image.open(uploaded_con)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"正在擷取: {selected_con}")
            val_con = streamlit_image_coordinates(img_con, key="con_img")
            if val_con:
                pt = (val_con["x"], val_con["y"])
                if len(st.session_state.con_calib) < 2:
                    if not st.session_state.con_calib or pt != st.session_state.con_calib[-1]:
                        st.session_state.con_calib.append(pt)
                        st.rerun()
                else:
                    if not con_obj.raw_pixel_points or pt != con_obj.raw_pixel_points[-1]:
                        con_obj.raw_pixel_points.append(pt)
        with c2:
            con_xmax = st.number_input("X軸最大 (V)", value=5.0, key="con_xm")
            con_ymax = st.number_input("Y軸最大 (A)", value=800.0, key="con_ym")
            if len(st.session_state.con_calib) == 2:
                p0, pm = st.session_state.con_calib
                # 注意：導通圖通常 X 是電壓，Y 是電流
                sx, sy = con_xmax/(pm[0]-p0[0]), con_ymax/(pm[1]-p0[1])
                # 儲存時統一轉為 (Current, Voltage) 以符合擬合邏輯
                con_obj.real_data_points = [((p[1]-p0[1])*sy, (p[0]-p0[0])*sx) for p in con_obj.raw_pixel_points]
                if st.button("🚀 擬合導通損 (Linear)"):
                    x, y = np.array([p[0] for p in con_obj.real_data_points]), np.array([p[1] for p in con_obj.real_data_points])
                    popt, _ = curve_fit(lambda i,v0,r: v0+r*i, x, y)
                    con_obj.set_params(popt)
                st.write(f"已擷取 {len(con_obj.raw_pixel_points)} 點")

st.divider()

# ==========================================
# 三、 全域損耗分析與波形 (Global Calculator)
# ==========================================
st.header("📋 3. 全域損耗動態分析 (Switching + Conduction)")

all_fitted_sw = {n: o for n, o in st.session_state.sw_curves.items() if o.params is not None}
all_fitted_con = {n: o for n, o in st.session_state.con_curves.items() if o.params is not None}

if all_fitted_sw or all_fitted_con:
    c1, c2, c3 = st.columns(3)
    i_peak = c1.number_input("峰值電流 I_peak (A)", value=400.0)
    f_out = c2.number_input("基波頻率 f_out (Hz)", value=50.0)
    f_sw = c3.number_input("切換頻率 f_sw (Hz)", value=10000.0)

    # 時間與波形計算
    t = np.linspace(0, 1/f_out, 500)
    i_wave = i_peak * np.sin(2 * np.pi * f_out * t)
    i_abs = np.abs(i_wave)

    total_sw_p = 0.0
    total_con_p = 0.0
    
    # 1. 計算切換損耗 (能量 mJ -> 功率 W)[cite: 2]
    for n, o in all_fitted_sw.items():
        e_avg = np.mean([o.get_value(i_peak * np.sin(phi)) for phi in np.linspace(0, np.pi, 200)])
        total_sw_p += e_avg * f_sw * 1e-3

    # 2. 計算導通損耗 (電壓 V * 電流 A -> 功率 W)
    # Equation 14: p_con = v(i) * i
    for n, o in all_fitted_con.items():
        # 只在正半週計算導通損耗
        p_instant = np.array([o.get_value(curr) * curr if curr > 0 else 0 for curr in i_wave])
        total_con_p += np.mean(p_instant) # 平均功率

    st.columns(2)[0].metric("總切換損耗 P_sw", f"{total_sw_p:.2f} W")
    st.columns(2)[1].metric("總導通損耗 P_con", f"{total_con_p:.2f} W")
    st.success(f"🔥 總功率損耗: {total_sw_p + total_con_p:.2f} W")

    # 分別呈現擬合波形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 切換損波形 (Energy)
    for n, o in all_fitted_sw.items():
        ax1.plot(t*1000, [o.get_value(i) for i in i_wave], label=f"E_sw: {n}")
    ax1.set_title("Switching Energy Waveform (mJ)")
    ax1.set_ylabel("Energy (mJ)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 導通損波形 (Instantaneous Power)
    for n, o in all_fitted_con.items():
        ax2.plot(t*1000, [o.get_value(i)*i if i > 0 else 0 for i in i_wave], label=f"P_con: {n}")
    ax2.set_title("Instantaneous Conduction Power Waveform (W)")
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

# 係數表彙整
st.subheader("📋 系統係數彙整")
sum_data = []
for n, o in {**st.session_state.sw_curves, **st.session_state.con_curves}.items():
    if o.params is not None:
        sum_data.append({"名稱": n, "分類": o.category, "方程式": o.get_equation_string()})
if sum_data: st.table(pd.DataFrame(sum_data))