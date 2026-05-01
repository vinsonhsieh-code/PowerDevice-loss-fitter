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
        """根據擬合模型計算瞬時損耗"""
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2[cite: 1]
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

st.title("⚡ Power Device Loss Evaluator (動態基波分析版)")

# ==========================================
# Sidebar：曲線與標定管理
# ==========================================
st.sidebar.header("🛠️ 1. 曲線管理")
new_name = st.sidebar.text_input("新曲線名稱 (如 Eon_150C)", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("擬合模型", ["Method_SW3", "Linear", "Power"])

if st.sidebar.button("➕ 新增曲線"):
    if new_name not in st.session_state.curve_objects:
        st.session_state.curve_objects[new_name] = PowerLossCurve(name=new_name, model_type=model_choice)
        st.session_state.current_curve_name = new_name
        st.rerun()

all_names = list(st.session_state.curve_objects.keys())
if all_names:
    if st.session_state.current_curve_name not in all_names:
        st.session_state.current_curve_name = all_names[0]
    selected_name = st.sidebar.selectbox("切換編輯對象", all_names, index=all_names.index(st.session_state.current_curve_name))
    st.session_state.current_curve_name = selected_name
    current_obj = st.session_state.curve_objects[selected_name]
    
    if st.sidebar.button("🗑️ 刪除選中曲線"):
        del st.session_state.curve_objects[selected_name]
        remaining = list(st.session_state.curve_objects.keys())
        st.session_state.current_curve_name = remaining[0] if remaining else None
        st.rerun()
else:
    st.info("👈 請先新增曲線檔案。")
    st.stop()

if st.sidebar.button("🔄 重置標定"):
    st.session_state.calib_pts = []
    st.rerun()

# ==========================================
# 主畫面：標定、擷取與擬合
# ==========================================
uploaded_file = st.file_uploader("2. 上傳圖檔 (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 編輯中: {current_obj.name}")
        if len(st.session_state.calib_pts) == 0:
            st.warning("請點擊圖片中座標軸的【左下角原點 (0,0)】")
        elif len(st.session_state.calib_pts) == 1:
            st.warning("請點擊圖片中座標軸的【右上方最大刻度點】")
        else:
            st.success("標定完成！請點選曲線擷取數據點。")
        
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
        
        if st.button("🗑️ 清除目前點選點"):
            current_obj.raw_pixel_points = []
            st.rerun()

    with col2:
        st.subheader("📏 3. 座標設定")
        real_x_max = st.number_input("X 軸最大物理值 (A)", value=1000.0)
        real_y_max = st.number_input("Y 軸最大物理值 (mJ/V)", value=125.0)
        
        if len(st.session_state.calib_pts) == 2:
            p0, p_max = st.session_state.calib_pts
            dx, dy = p_max[0] - p0[0], p_max[1] - p0[1]
            if dx != 0 and dy != 0:
                scale_x, scale_y = real_x_max / dx, real_y_max / dy
                current_obj.real_data_points = [
                    ((px - p0[0]) * scale_x, (py - p0[1]) * scale_y) 
                    for px, py in current_obj.raw_pixel_points
                ]
                st.dataframe(pd.DataFrame(current_obj.real_data_points, columns=["X (A)", "Y"]), height=150)

# ==========================================
# 擬合與動態波形分析器
# ==========================================
if len(st.session_state.calib_pts) == 2 and current_obj.real_data_points:
    st.divider()
    if st.button(f"🚀 執行擬合 ({current_obj.name})", type="primary"):
        x = np.array([p[0] for p in current_obj.real_data_points])
        y = np.array([p[1] for p in current_obj.real_data_points])
        try:
            if current_obj.model_type == "Method_SW3":
                popt, _ = curve_fit(lambda i, A, B, C: A + B*i + C*i**2, x, y)
            elif current_obj.model_type == "Linear":
                popt, _ = curve_fit(lambda i, v0, r: v0 + r*i, x, y)
            else:
                popt, _ = curve_fit(lambda i, a, b: a * (i**b), x, y)
            current_obj.set_params(popt)
            st.success("擬合成功！")
        except: st.error("擬合出錯。")

    # --- 4. 動態損耗計算與可視化 ---
    fitted_objs = {n: o for n, o in st.session_state.curve_objects.items() if o.params is not None}
    
    if fitted_objs:
        st.subheader("📋 4. 全域動態分析器 (動態基波)")
        
        # 參數輸入列
        c1, c2, c3 = st.columns(3)
        with c1:
            i_peak = st.number_input("峰值電流 I_peak (A)", value=float(real_x_max/2), min_value=0.0)
        with c2:
            f_out = st.number_input("基波頻率 f_out (Hz)", value=50.0, min_value=0.1)
        with c3:
            f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0, min_value=1.0)
        
        # 計算時間軸 (繪製一個完整基波週期)
        t_period = 1 / f_out
        time_vec = np.linspace(0, t_period, 500)
        i_wave = i_peak * np.sin(2 * np.pi * f_out * time_vec)
        
        # 計算平均能量與總功率[cite: 1, 2]
        total_p_sw_w = 0.0
        summary_results = []
        
        for name, obj in fitted_objs.items():
            # 在半個週期內積分求平均[cite: 1]
            e_avg = np.mean([obj.get_loss(i_peak * np.sin(phi)) for phi in np.linspace(0, np.pi, 200)])
            p_loss = e_avg * f_sw * 1e-3 #
            total_p_sw_w += p_loss
            summary_results.append({"項目": name, "平均能量 (mJ)": f"{e_avg:.4f}", "功率損耗 (W)": f"{p_loss:.4f}"})
        
        st.metric("總預估切換功率損耗 (P_sw)", f"{total_p_sw_w:.4f} W")
        st.table(pd.DataFrame(summary_results))

        # --- 波形繪製 ---
        st.markdown(f"#### 📈 週期波形對照 ($f_{{out}} = {f_out}$ Hz, $I_{{peak}} = {i_peak}$ A)")
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        # 電流波形 (左軸)
        color_i = 'tab:blue'
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (A)', color=color_i)
        ax1.plot(time_vec * 1000, i_wave, color=color_i, linestyle='--', alpha=0.4, label='i(t)')
        ax1.tick_params(axis='y', labelcolor=color_i)
        ax1.axhline(0, color='black', lw=1, alpha=0.3)

        # 損耗波形 (右軸)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Instantaneous Loss (mJ)', color='tab:red')
        for name, obj in fitted_objs.items():
            e_wave = [obj.get_loss(val) for val in i_wave]
            ax2.plot(time_vec * 1000, e_wave, label=f'Loss: {name}')
        
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)

# 係數表
st.divider()
st.subheader("📋 5. 擬合係數彙整表")
summary = [{"曲線名稱": n, "模型": o.model_type, "方程式定義": o.get_equation_string()} 
           for n, o in st.session_state.curve_objects.items() if o.params is not None]
if summary: st.table(pd.DataFrame(summary))