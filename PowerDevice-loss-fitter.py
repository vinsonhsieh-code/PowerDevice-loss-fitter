import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Analyzer Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別 (支援 SW 與 Conduction)
# ==========================================
class PowerLossCurve:
    def __init__(self, name, model_type="Method_SW3"):
        self.name = name
        self.model_type = model_type # "Method_SW3", "Power", "Conduction_VI"
        self.params = None  
        self.raw_pixel_points = [] 
        self.real_data_points = [] 

    def set_params(self, params):
        self.params = params

    def get_instant_power(self, current_i):
        """
        計算瞬時功率 (W)
        - 若為 SW 模型，回傳的是每次切換能量 (mJ)
        - 若為 Conduction 模型，回傳的是瞬時功率 (W)
        """
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2 (mJ)[cite: 1]
            A, B, C = self.params
            return A + B * i_abs + C * (i_abs ** 2)
        
        elif self.model_type == "Power":
            # E = a * i^b (mJ)
            a, b = self.params
            return a * (i_abs ** b)
        
        elif self.model_type == "Conduction_VI":
            # 依據論文 Eq(13)(14): v = V_X + R_X*i => P = V_X*i + R_X*i^2
            # params[0] 是 V_X (截距), params[1] 是 R_X (斜率)
            Vx, Rx = self.params
            return Vx * i_abs + Rx * (i_abs ** 2)
            
        return 0.0

    def get_equation_string(self):
        if self.params is None: return "尚未擬合"
        if self.model_type == "Method_SW3":
            return f"{self.params[0]:.4e} + {self.params[1]:.4e}*i + {self.params[2]:.4e}*i^2"
        elif self.model_type == "Conduction_VI":
            return f"v = {self.params[0]:.4f} + {self.params[1]:.4f}*i (V)"
        return f"Params: {self.params}"

# ==========================================
# Session State 初始化
# ==========================================
if "curve_objects" not in st.session_state:
    st.session_state.curve_objects = {}  
if "current_curve_name" not in st.session_state:
    st.session_state.current_curve_name = None
if "calib_pts" not in st.session_state:
    st.session_state.calib_pts = [] 

st.title("⚡ Power Device Loss Analyzer (SW & Conduction Integrated)")

# ==========================================
# Sidebar：多功能管理
# ==========================================
st.sidebar.header("🛠️ 1. 曲線管理中心")
new_name = st.sidebar.text_input("新曲線名稱 (如 Eon, Vce_150C)", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("擬合模型類型", [
    "Method_SW3 (切換損 E-i)", 
    "Conduction_VI (導通損 v-i)", 
    "Power (E = a*i^b)"
])

if st.sidebar.button("➕ 新增曲線檔案"):
    m_type = "Method_SW3" if "SW3" in model_choice else ("Conduction_VI" if "Conduction" in model_choice else "Power")
    if new_name not in st.session_state.curve_objects:
        st.session_state.curve_objects[new_name] = PowerLossCurve(name=new_name, model_type=m_type)
        st.session_state.current_curve_name = new_name
        st.rerun()

all_names = list(st.session_state.curve_objects.keys())
if all_names:
    if st.session_state.current_curve_name not in all_names:
        st.session_state.current_curve_name = all_names[0]
    selected_name = st.sidebar.selectbox("目前編輯對象", all_names, index=all_names.index(st.session_state.current_curve_name))
    st.session_state.current_curve_name = selected_name
    current_obj = st.session_state.curve_objects[selected_name]
    
    if st.sidebar.button("🗑️ 刪除選中曲線"):
        del st.session_state.curve_objects[selected_name]
        st.rerun()
else:
    st.info("👈 請先新增曲線。")
    st.stop()

if st.sidebar.button("🔄 重置標定"):
    st.session_state.calib_pts = []
    st.rerun()

# ==========================================
# 圖片標定區 (SW 與 Conduction 共用)
# ==========================================
uploaded_file = st.file_uploader("2. 上傳 Datasheet 特性圖 (E-i 或 v-i)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 擷取數據: {current_obj.name}")
        if len(st.session_state.calib_pts) < 2:
            st.warning("請先點擊原點 (0,0) 與 最大刻度點進行標定。")
        
        value = streamlit_image_coordinates(img, key="img_click")
        if value:
            curr_pt = (value["x"], value["y"])
            if len(st.session_state.calib_pts) < 2:
                if not st.session_state.calib_pts or curr_pt != st.session_state.calib_pts[-1]:
                    st.session_state.calib_pts.append(curr_pt)
                    st.rerun()
            else:
                if not current_obj.raw_pixel_points or curr_pt != current_obj.raw_pixel_points[-1]:
                    current_obj.raw_pixel_points.append(curr_pt)

    with col2:
        st.subheader("📏 座標轉換設定")
        real_x_max = st.number_input("X 軸最大值 (Current A)", value=1000.0)
        y_label = "Y 軸最大值 (mJ)" if "SW" in current_obj.model_type else "Y 軸最大值 (Voltage V)"
        real_y_max = st.number_input(y_label, value=125.0 if "SW" in current_obj.model_type else 5.0)
        
        if len(st.session_state.calib_pts) == 2:
            p0, p_max = st.session_state.calib_pts
            dx, dy = p_max[0] - p0[0], p_max[1] - p0[1]
            if dx != 0 and dy != 0:
                scale_x, scale_y = real_x_max / dx, real_y_max / dy
                current_obj.real_data_points = [((px - p0[0]) * scale_x, (py - p0[1]) * scale_y) for px, py in current_obj.raw_pixel_points]
                st.dataframe(pd.DataFrame(current_obj.real_data_points, columns=["X", "Y"]), height=150)

# ==========================================
# 擬合運算
# ==========================================
if len(st.session_state.calib_pts) == 2 and current_obj.real_data_points:
    st.divider()
    if st.button(f"🚀 執行擬合 ({current_obj.name})", type="primary"):
        x = np.array([p[0] for p in current_obj.real_data_points])
        y = np.array([p[1] for p in current_obj.real_data_points])
        try:
            if current_obj.model_type == "Method_SW3":
                popt, _ = curve_fit(lambda i, A, B, C: A + B*i + C*i**2, x, y)
            elif current_obj.model_type == "Conduction_VI":
                # v = Vx + Rx * i (Linear fit)
                popt, _ = curve_fit(lambda i, Vx, Rx: Vx + Rx*i, x, y)
            else:
                popt, _ = curve_fit(lambda i, a, b: a * (i**b), x, y)
            current_obj.set_params(popt)
            st.success("擬合成功！")
        except: st.error("擬合出錯。")

    # ==========================================
    # 5. 綜合損耗計算器 (SW + Conduction)
    # ==========================================
    fitted_objs = {n: o for n, o in st.session_state.curve_objects.items() if o.params is not None}
    if fitted_objs:
        st.subheader("📋 5. 全系統動態分析 (SW + Conduction)")
        c1, c2, c3 = st.columns(3)
        with c1: i_peak = st.number_input("峰值電流 I_peak (A)", value=float(real_x_max/2))
        with c2: f_out = st.number_input("基波頻率 f_out (Hz)", value=50.0)
        with c3: f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0)

        total_sw_w = 0.0
        total_con_w = 0.0
        time_vec = np.linspace(0, 1/f_out, 500)
        thetas = np.linspace(0, np.pi, 200)

        for name, obj in fitted_objs.items():
            # 計算平均值 (基於半個正弦波週期)
            inst_vals = [obj.get_instant_power(i_peak * np.sin(phi)) for phi in thetas]
            avg_val = np.mean(inst_vals)
            
            if obj.model_type == "Conduction_VI":
                total_con_w += avg_val # 直接就是功率 (W)
            else:
                total_sw_w += avg_val * f_sw * 1e-3 # 能量(mJ)轉功率(W)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("切換損耗 P_sw", f"{total_sw_w:.2f} W")
        col_m2.metric("導通損耗 P_con", f"{total_con_w:.2f} W")
        col_m3.metric("總損耗 P_total", f"{(total_sw_w + total_con_w):.2f} W", delta_color="inverse")

        # 波形繪圖
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (A)', color='tab:blue')
        i_wave = i_peak * np.sin(2 * np.pi * f_out * time_vec)
        ax1.plot(time_vec*1000, i_wave, 'b--', alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Instantaneous Loss (W / mJ)')
        for name, obj in fitted_objs.items():
            wave = [obj.get_instant_power(v) for v in i_wave]
            ax2.plot(time_vec*1000, wave, label=name)
        ax2.legend()
        st.pyplot(fig)

# 係數彙整表
st.divider()
st.subheader("📋 6. 擬合係數彙整表 (Device Library)")
summary = []
for n, o in st.session_state.curve_objects.items():
    if o.params is not None:
        summary.append({"名稱": n, "模型": o.model_type, "方程式": o.get_equation_string()})
if summary: st.table(pd.DataFrame(summary))