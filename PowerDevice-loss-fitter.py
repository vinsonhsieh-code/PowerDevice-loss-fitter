import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - Integrated Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別
# ==========================================
class PowerLossCurve:
    def __init__(self, name, category="Switching", model_type="Method_SW3"):
        self.name = name
        self.category = category # "Switching" 或 "Conduction"
        self.model_type = model_type
        self.params = None  
        self.raw_pixel_points = [] 
        self.real_data_points = [] 

    def set_params(self, params):
        self.params = params

    def get_value(self, current_i):
        """
        Switching: 回傳能量 (mJ)
        Conduction: 回傳壓降 v (V)
        """
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        if self.category == "Switching":
            if self.model_type == "Method_SW3":
                A, B, C = self.params
                return A + B * i_abs + C * (i_abs ** 2)
            elif self.model_type == "Power":
                a, b = self.params
                return a * (i_abs ** b)
        else: # Conduction: v = R*i + V
            R, V = self.params
            return R * i_abs + V
        return 0.0

    def get_conduction_loss(self, current_i):
        """計算導通瞬時損耗 p = v * i"""
        v = self.get_value(current_i)
        return v * abs(current_i)

    def get_equation_string(self):
        if self.params is None: return "尚未擬合"
        if self.category == "Switching":
            if self.model_type == "Method_SW3":
                return f"{self.params[0]:.4e} + {self.params[1]:.4e}*i + {self.params[2]:.4e}*i^2"
        else: # Conduction
            return f"v = {self.params[0]:.4f}*i + {self.params[1]:.4f}"
        return str(self.params)

# ==========================================
# Session State 初始化
# ==========================================
for key in ["sw_curves", "con_curves", "calib_sw", "calib_con"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "curves" in key else []

st.title("⚡ Power Device Loss Evaluator (Switching & Conduction)")

# ==========================================
# Sidebar：雙區域管理
# ==========================================
with st.sidebar:
    st.header("📋 1. 曲線管理中心")
    
    # --- 切換損管理 ---
    st.subheader("🔹 切換損 (Switching)")
    sw_name = st.text_input("切換損名稱", value=f"SW_{len(st.session_state.sw_curves)+1}")
    if st.button("➕ 新增切換損曲線"):
        st.session_state.sw_curves[sw_name] = PowerLossCurve(sw_name, "Switching", "Method_SW3")
        st.session_state.cur_sw = sw_name
    
    sw_list = list(st.session_state.sw_curves.keys())
    if sw_list:
        st.session_state.cur_sw = st.selectbox("編輯切換損", sw_list, index=sw_list.index(st.session_state.get("cur_sw", sw_list[0])))
        if st.button("🗑️ 刪除選中切換損"):
            del st.session_state.sw_curves[st.session_state.cur_sw]
            st.rerun()

    st.divider()
    
    # --- 導通損管理 ---
    st.subheader("🔸 導通損 (Conduction)")
    con_name = st.text_input("導通損名稱", value=f"CON_{len(st.session_state.con_curves)+1}")
    if st.button("➕ 新增導通損曲線"):
        st.session_state.con_curves[con_name] = PowerLossCurve(con_name, "Conduction", "Linear")
        st.session_state.cur_con = con_name
    
    con_list = list(st.session_state.con_curves.keys())
    if con_list:
        st.session_state.cur_con = st.selectbox("編輯導通損", con_list, index=con_list.index(st.session_state.get("cur_con", con_list[0])))
        if st.button("🗑️ 刪除選中導通損"):
            del st.session_state.con_curves[st.session_state.cur_con]
            st.rerun()

# ==========================================
# 主頁面：雙區並列顯示 (切換損 vs 導通損)
# ==========================================
col_sw, col_con = st.columns(2)

# --- 左側：切換損區域 ---
with col_sw:
    st.header("🔹 切換損建模 (E-i Chart)")
    up_sw = st.file_uploader("上傳切換損圖檔", type=["png", "jpg"], key="up_sw")
    if up_sw and sw_list:
        obj = st.session_state.sw_curves[st.session_state.cur_sw]
        img = Image.open(up_sw)
        st.write(f"📍 正在編輯: {obj.name}")
        val = streamlit_image_coordinates(img, key="click_sw")
        
        if val:
            pt = (val["x"], val["y"])
            if len(st.session_state.calib_sw) < 2:
                if not st.session_state.calib_sw or pt != st.session_state.calib_sw[-1]:
                    st.session_state.calib_sw.append(pt)
                    st.rerun()
            else:
                if not obj.raw_pixel_points or pt != obj.raw_pixel_points[-1]:
                    obj.raw_pixel_points.append(pt)
        
        # 標定與擬合
        mx_sw = st.number_input("SW X軸最大(A)", value=1000.0, key="mx_sw")
        my_sw = st.number_input("SW Y軸最大(mJ)", value=125.0, key="my_sw")
        if len(st.session_state.calib_sw) == 2:
            p0, pM = st.session_state.calib_sw
            sx, sy = mx_sw / (pM[0]-p0[0]), my_sw / (pM[1]-p0[1])
            obj.real_data_points = [((p[0]-p0[0])*sx, (p[1]-p0[1])*sy) for p in obj.raw_pixel_points]
            if st.button(f"🚀 擬合 {obj.name}"):
                x = np.array([p[0] for p in obj.real_data_points])
                y = np.array([p[1] for p in obj.real_data_points])
                popt, _ = curve_fit(lambda i, A, B, C: A + B*i + C*i**2, x, y)
                obj.set_params(popt)

# --- 右側：導通損區域 ---
with col_con:
    st.header("🔸 導通損建模 (v-i Chart)")
    up_con = st.file_uploader("上傳導通損圖檔", type=["png", "jpg"], key="up_con")
    if up_con and con_list:
        obj = st.session_state.con_curves[st.session_state.cur_con]
        img = Image.open(up_con)
        st.write(f"📍 正在編輯: {obj.name}")
        val = streamlit_image_coordinates(img, key="click_con")
        
        if val:
            pt = (val["x"], val["y"])
            if len(st.session_state.calib_con) < 2:
                if not st.session_state.calib_con or pt != st.session_state.calib_con[-1]:
                    st.session_state.calib_con.append(pt)
                    st.rerun()
            else:
                if not obj.raw_pixel_points or pt != obj.raw_pixel_points[-1]:
                    obj.raw_pixel_points.append(pt)
        
        # 標定與擬合
        mx_con = st.number_input("CON X軸最大(A)", value=800.0, key="mx_con")
        my_con = st.number_input("CON Y軸最大(V)", value=5.0, key="my_con")
        if len(st.session_state.calib_con) == 2:
            p0, pM = st.session_state.calib_con
            sx, sy = mx_con / (pM[0]-p0[0]), my_con / (pM[1]-p0[1])
            obj.real_data_points = [((p[0]-p0[0])*sx, (p[1]-p0[1])*sy) for p in obj.raw_pixel_points]
            if st.button(f"🚀 擬合 {obj.name}"):
                x = np.array([p[0] for p in obj.real_data_points])
                y = np.array([p[1] for p in obj.real_data_points])
                # Linear: v = R*i + V
                popt, _ = curve_fit(lambda i, R, V: R * i + V, x, y)
                obj.set_params(popt)

# ==========================================
# 全域分析與波形 (整合切換與導通)
# ==========================================
st.divider()
st.header("📊 全域損耗動態分析 (Switching + Conduction)")

fit_sw = {n: o for n, o in st.session_state.sw_curves.items() if o.params is not None}
fit_con = {n: o for n, o in st.session_state.con_curves.items() if o.params is not None}

if fit_sw or fit_con:
    c1, c2, c3 = st.columns(3)
    i_peak = c1.number_input("峰值電流 I_peak (A)", value=400.0)
    f_out = c2.number_input("基波頻率 f_out (Hz)", value=50.0)
    f_sw = c3.number_input("切換頻率 f_sw (Hz)", value=10000.0)

    t = np.linspace(0, 1/f_out, 500)
    i_t = i_peak * np.sin(2 * np.pi * f_out * t)
    
    # 計算損耗
    total_sw_w, total_con_w = 0.0, 0.0
    
    # 切換損功率 Psw = E_avg * f_sw[cite: 2]
    for obj in fit_sw.values():
        e_avg = np.mean([obj.get_value(i_peak * np.sin(phi)) for phi in np.linspace(0, np.pi, 200)])
        total_sw_w += e_avg * f_sw * 1e-3

    # 導通損功率 Pcon = 1/2pi * int(v*i) dt
    for obj in fit_con.values():
        p_instant = [obj.get_conduction_loss(i) for i in i_t if i > 0] # 僅正半週導通
        total_con_w += np.mean(p_instant) * 0.5 # 假設半週導通

    st.metric("總預估損耗 (Total Loss)", f"{total_sw_w + total_con_w:.2f} W", 
              delta=f"SW: {total_sw_w:.2f}W | CON: {total_con_w:.2f}W")

    # 繪圖
    fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = ax.twinx()
    ax.plot(t*1000, i_t, 'b--', alpha=0.3, label="Current i(t)")
    for obj in fit_sw.values():
        ax2.plot(t*1000, [obj.get_value(i) for i in i_t], label=f"SW: {obj.name} (mJ)")
    for obj in fit_con.values():
        ax2.plot(t*1000, [obj.get_conduction_loss(i) for i in i_t], label=f"CON: {obj.name} (W)")
    ax.set_xlabel("Time (ms)"); ax.legend(loc=2); ax2.legend(loc=1); st.pyplot(fig)

# 參數彙整表
st.divider()
st.subheader("📋 擬合係數彙整表")
all_data = []
for o in list(fit_sw.values()) + list(fit_con.values()):
    all_data.append({"名稱": o.name, "分類": o.category, "方程式": o.get_equation_string()})
if all_data: st.table(pd.DataFrame(all_data))