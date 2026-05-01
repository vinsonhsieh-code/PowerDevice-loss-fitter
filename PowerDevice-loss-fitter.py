import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - R&D Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別
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
        """計算瞬時損耗"""
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2
            A, B, C = self.params
            return A + B * i_abs + C * (i_abs ** 2)
        elif self.model_type == "Linear":
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

st.title("⚡ Power Device Loss Evaluator (Waveform Pro)")

# ==========================================
# Sidebar 曲線管理
# ==========================================
st.sidebar.header("🛠️ 1. 曲線與標定管理")
new_name = st.sidebar.text_input("新曲線名稱", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("擬合模型", ["Method_SW3", "Linear", "Power"])

if st.sidebar.button("➕ 新增曲線"):
    if new_name not in st.session_state.curve_objects:
        st.session_state.curve_objects[new_name] = PowerLossCurve(name=new_name, model_type=model_choice)
        st.session_state.current_curve_name = new_name
        st.rerun()

all_names = list(st.session_state.curve_objects.keys())
if all_names:
    selected_name = st.sidebar.selectbox("切換編輯對象", all_names, index=all_names.index(st.session_state.current_curve_name))
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
# 主畫面：標定與擬合
# ==========================================
uploaded_file = st.file_uploader("2. 上傳圖檔", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 編輯中: {current_obj.name}")
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
        st.subheader("📏 3. 座標設定")
        real_x_max = st.number_input("X 軸最大值 (A)", value=1000.0)
        real_y_max = st.number_input("Y 軸最大值 (mJ/V)", value=125.0)
        if len(st.session_state.calib_pts) == 2:
            p0, p_max = st.session_state.calib_pts
            dx, dy = p_max[0] - p0[0], p_max[1] - p0[1]
            if dx != 0 and dy != 0:
                scale_x, scale_y = real_x_max / dx, real_y_max / dy
                current_obj.real_data_points = [((px - p0[0]) * scale_x, (py - p0[1]) * scale_y) for px, py in current_obj.raw_pixel_points]
                st.dataframe(pd.DataFrame(current_obj.real_data_points, columns=["X", "Y"]), height=150)

# ==========================================
# 擬合與全域計算器
# ==========================================
if len(st.session_state.calib_pts) == 2 and current_obj.real_data_points:
    st.divider()
    if st.button(f"🚀 執行擬合 ({current_obj.name})", type="primary"):
        x, y = np.array([p[0] for p in current_obj.real_data_points]), np.array([p[1] for p in current_obj.real_data_points])
        try:
            if current_obj.model_type == "Method_SW3":
                popt, _ = curve_fit(lambda i, A, B, C: A + B*i + C*i**2, x, y)
            elif current_obj.model_type == "Linear":
                popt, _ = curve_fit(lambda i, v0, r: v0 + r*i, x, y)
            else:
                popt, _ = curve_fit(lambda i, a, b: a * (i**b), x, y)
            current_obj.set_params(popt)
            st.success("擬合成功！")
        except: st.error("擬合失敗。")

    # --- 5. 計算器與波形圖 ---
    fitted_objs = {n: o for n, o in st.session_state.curve_objects.items() if o.params is not None}
    if fitted_objs:
        st.subheader("📋 5. 全域損耗與波形動態分析")
        calc_mode = st.radio("模式", ["單點直流", "正弦波平均"], horizontal=True)
        f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0)
        calc_i = st.number_input("輸入電流 (A) - [單點值] 或 [正弦波峰值]", value=float(real_x_max/2))

        # 計算結果
        total_sw_power = 0.0
        wave_data = []
        thetas = np.linspace(0, np.pi, 200) # 用於繪圖的相位

        for name, obj in fitted_objs.items():
            if calc_mode == "單點直流":
                e_avg = obj.get_loss(calc_i)
            else:
                # 數值積分[cite: 1]
                e_avg = np.mean([obj.get_loss(calc_i * np.sin(t)) for t in thetas])
            
            p_loss = e_avg * f_sw * 1e-3 #
            total_sw_power += p_loss
            
        st.metric("預估總切換功率 (P_sw)", f"{total_sw_power:.4f} W")

        # --- 6. 波形繪圖區 ---
        if calc_mode == "正弦波平均":
            st.markdown("#### 📈 實時損耗波形對照 (Instantaneous Loss vs. Current)")
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            # 繪製電流波形 (左軸)
            i_wave = calc_i * np.sin(thetas)
            color_i = 'tab:blue'
            ax1.set_xlabel('Phase (rad)')
            ax1.set_ylabel('Current (A)', color=color_i)
            ax1.plot(thetas, i_wave, color=color_i, linestyle='--', label='Current i(t)', alpha=0.5)
            ax1.tick_params(axis='y', labelcolor=color_i)

            # 繪製損耗波形 (右軸)
            ax2 = ax1.twinx()
            color_e = 'tab:red'
            ax2.set_ylabel('Instantaneous Loss (mJ)', color=color_e)
            
            for name, obj in fitted_objs.items():
                e_wave = [obj.get_loss(val) for val in i_wave]
                ax2.plot(thetas, e_wave, label=f'Loss: {name}')
            
            ax2.tick_params(axis='y', labelcolor=color_e)
            ax2.legend(loc='upper right')
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("註：虛線為正弦電流波形，實線為各元件在該電流下的瞬時能量損耗 (mJ)。")

    # 基礎擬合曲線圖
    fig2, ax_base = plt.subplots(figsize=(10, 3))
    xi = np.linspace(0, real_x_max, 200)
    for name, obj in fitted_objs.items():
        ax_base.plot(xi, [obj.get_loss(v) for v in xi], label=name)
    ax_base.set_xlabel("Current (A)"); ax_base.set_ylabel("Value"); ax_base.legend(); st.pyplot(fig2)

# 係數彙整
st.divider()
st.subheader("📋 擬合係數彙整表")
summary = [{"曲線": n, "模型": o.model_type, "方程式": o.get_equation_string()} 
           for n, o in st.session_state.curve_objects.items() if o.params is not None]
if summary: st.table(pd.DataFrame(summary))