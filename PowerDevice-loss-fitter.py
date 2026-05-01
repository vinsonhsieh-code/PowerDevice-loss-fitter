import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別 (封裝計算與擬合參數)
# ==========================================
class PowerLossCurve:
    def __init__(self, name, model_type="Method_SW3"):
        self.name = name
        self.model_type = model_type
        self.params = None  
        self.raw_pixel_points = [] # 儲存點擊的像素 (x, y)
        self.real_data_points = [] # 儲存轉換後的 (Current, Energy)

    def set_params(self, params):
        self.params = params

    def get_loss(self, current_i):
        """根據擬合模型計算損耗[cite: 1]"""
        if self.params is None: return 0.0
        i_abs = abs(current_i)
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2[cite: 1]
            A, B, C = self.params
            return A + B * i_abs + C * (i_abs ** 2)
        elif self.model_type == "Linear":
            # V = V0 + r*i
            v0, r = self.params
            return v0 + r * i_abs
        elif self.model_type == "Power":
            # E = a * i^b
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
    st.session_state.calib_pts = [] # 儲存 (0,0) 與 (Max, Max) 像素

st.title("⚡ Power Device Loss Evaluator (精準標定與波形版)")

# ==========================================
# Sidebar：曲線與標定管理
# ==========================================
st.sidebar.header("🛠️ 1. 曲線與標定管理")

# 新增曲線
new_name = st.sidebar.text_input("輸入新曲線名稱 (如 Eon)", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("擬合模型", ["Method_SW3", "Linear", "Power"])

if st.sidebar.button("➕ 新增曲線檔案"):
    if new_name not in st.session_state.curve_objects:
        st.session_state.curve_objects[new_name] = PowerLossCurve(name=new_name, model_type=model_choice)
        st.session_state.current_curve_name = new_name
        st.rerun()

# 曲線列表與刪除
all_names = list(st.session_state.curve_objects.keys())
if all_names:
    if st.session_state.current_curve_name not in all_names:
        st.session_state.current_curve_name = all_names[0]
    selected_name = st.sidebar.selectbox("切換編輯對象", all_names, index=all_names.index(st.session_state.current_curve_name))
    st.session_state.current_curve_name = selected_name
    current_obj = st.session_state.curve_objects[selected_name]
    
    if st.sidebar.button("🗑️ 刪除目前選中的曲線"):
        del st.session_state.curve_objects[selected_name]
        remaining = list(st.session_state.curve_objects.keys())
        st.session_state.current_curve_name = remaining[0] if remaining else None
        st.rerun()
else:
    st.info("👈 請先新增曲線檔案。")
    st.stop()

# 重置標定
if st.sidebar.button("🔄 重置標定 (換圖或校準錯誤時點擊)"):
    st.session_state.calib_pts = []
    for obj in st.session_state.curve_objects.values():
        obj.raw_pixel_points = []
        obj.real_data_points = []
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
        st.subheader(f"📍 編輯中: {current_obj.name} ({current_obj.model_type})")
        
        # 標定流程引導
        if len(st.session_state.calib_pts) == 0:
            st.warning("⚠️ 標定步驟 A：請點擊圖片中座標軸的【左下角原點 (0,0)】")
        elif len(st.session_state.calib_pts) == 1:
            st.warning("⚠️ 標定步驟 B：請點擊圖片中座標軸的【右上方最大刻度點】")
        else:
            st.success("✅ 標定已完成！現在請點選曲線路徑擷取數據點。")
        
        value = streamlit_image_coordinates(img, key="img_click")
        
        if value:
            curr_pt = (value["x"], value["y"])
            # 處理標定點
            if len(st.session_state.calib_pts) < 2:
                if not st.session_state.calib_pts or curr_pt != st.session_state.calib_pts[-1]:
                    st.session_state.calib_pts.append(curr_pt)
                    st.rerun()
            # 處理曲線數據點
            else:
                if not current_obj.raw_pixel_points or curr_pt != current_obj.raw_pixel_points[-1]:
                    current_obj.raw_pixel_points.append(curr_pt)
        
        if st.button("🗑️ 清除當前曲線點"):
            current_obj.raw_pixel_points = []
            st.rerun()

    with col2:
        st.subheader("📏 3. 座標校準設定")
        real_x_max = st.number_input("X 軸最大物理值 (A)", value=1000.0)
        real_y_max = st.number_input("Y 軸最大物理值 (mJ/V)", value=125.0)
        
        # 座標轉換邏輯
        if len(st.session_state.calib_pts) == 2:
            p0, p_max = st.session_state.calib_pts
            dx = p_max[0] - p0[0]
            dy = p_max[1] - p0[1] # Y 軸像素通常 p0 > p_max，故 dy 為負
            
            if dx != 0 and dy != 0:
                scale_x = real_x_max / dx
                scale_y = real_y_max / dy
                
                # 更新目前曲線的實體數值點
                current_obj.real_data_points = [
                    ((px - p0[0]) * scale_x, (py - p0[1]) * scale_y) 
                    for px, py in current_obj.raw_pixel_points
                ]
                
                df = pd.DataFrame(current_obj.real_data_points, columns=["X (A)", "Y (Value)"])
                st.write(f"已擷取 {len(df)} 個數據點：")
                st.dataframe(df, height=200)

# ==========================================
# 擬合與全域分析
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
        except: st.error("擬合出錯，請檢查數據點分佈。")

    # --- 5. 進階計算與波形對照 ---
    fitted_objs = {n: o for n, o in st.session_state.curve_objects.items() if o.params is not None}
    
    if fitted_objs:
        st.subheader("📋 4. 全域分析與波形對照")
        mode = st.radio("評估模式", ["單點直流 (Static)", "正弦波平均 (Sine Wave)"], horizontal=True)
        f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0)
        input_i = st.number_input("輸入電流 (A) [單點值 或 峰值 I_peak]", value=float(real_x_max/2))

        total_p_sw = 0.0
        thetas = np.linspace(0, np.pi, 200)
        
        for name, obj in fitted_objs.items():
            if mode == "單點直流 (Static)":
                e_avg = obj.get_loss(input_i)
            else:
                # 實作第 10 式積分求平均能量[cite: 1]
                e_avg = np.mean([obj.get_loss(input_i * np.sin(t)) for t in thetas])
            
            p_loss = e_avg * f_sw * 1e-3 #[cite: 2]
            total_p_sw += p_loss

        st.metric("總切換功率損耗 (P_sw)", f"{total_p_sw:.4f} W")

        # 繪圖區
        if mode == "正弦波平均 (Sine Wave)":
            fig, ax1 = plt.subplots(figsize=(10, 4))
            i_wave = input_i * np.sin(thetas)
            ax1.set_xlabel('Phase (rad)')
            ax1.set_ylabel('Current (A)', color='tab:blue')
            ax1.plot(thetas, i_wave, 'b--', label='Current i(t)', alpha=0.3)
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Instantaneous Loss (mJ)', color='tab:red')
            for name, obj in fitted_objs.items():
                e_wave = [obj.get_loss(v) for v in i_wave]
                ax2.plot(thetas, e_wave, label=f'Loss: {name}')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.legend(loc='upper right')
            st.pyplot(fig)

    # 擬合曲線預覽
    fig_base, ax_base = plt.subplots(figsize=(10, 3))
    xi = np.linspace(0, real_x_max, 200)
    for name, obj in fitted_objs.items():
        ax_base.plot(xi, [obj.get_loss(v) for v in xi], label=name)
    ax_base.set_xlabel("Current (A)"); ax_base.set_ylabel("Loss / Voltage"); ax_base.legend()
    st.pyplot(fig_base)

# 係數彙整表
st.divider()
st.subheader("📋 5. 擬合係數彙整表")
summary = [{"曲線名稱": n, "模型": o.model_type, "方程式定義": o.get_equation_string()} 
           for n, o in st.session_state.curve_objects.items() if o.params is not None]
if summary: st.table(pd.DataFrame(summary))