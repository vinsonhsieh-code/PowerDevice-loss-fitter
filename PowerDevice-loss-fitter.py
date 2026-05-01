import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別 (Curve Class Definition)
# ==========================================
class PowerLossCurve:
    """
    用於儲存與計算功率元件損耗特性的類別。
    方便後續程式開發與 Simulink 模型呼叫。
    """
    def __init__(self, name, model_type="Method_SW3"):
        self.name = name
        self.model_type = model_type
        self.params = None  # 將儲存擬合後的係數陣列
        self.raw_pixel_points = [] # 儲存使用者點擊的像素座標
        self.real_data_points = [] # 儲存轉換後的物理數值點

    def set_params(self, params):
        """儲存擬合係數"""
        self.params = params

    def get_loss(self, current_i):
        """輸入電流，根據擬合係數回傳計算出的損耗 (或電壓)"""
        if self.params is None:
            return None
            
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2
            A, B, C = self.params
            return A + B * current_i + C * (current_i ** 2)
        elif self.model_type == "Linear":
            # V = V0 + r*I
            v0, r = self.params
            return v0 + r * current_i
        elif self.model_type == "Power":
            # E = a * I^b
            a, b = self.params
            return a * (current_i ** b)
        return None

    def get_equation_string(self):
        """回傳方便閱讀的方程式字串"""
        if self.params is None: return "尚未擬合"
        if self.model_type == "Method_SW3":
            return f"{self.params[0]:.6f} + {self.params[1]:.6f}*i + {self.params[2]:.6f}*i^2"
        elif self.model_type == "Linear":
            return f"{self.params[0]:.4f} + {self.params[1]:.4f}*i"
        elif self.model_type == "Power":
            return f"{self.params[0]:.4f} * i^{self.params[1]:.4f}"

# ==========================================
# Session State 初始化
# ==========================================
if "curve_objects" not in st.session_state:
    st.session_state.curve_objects = {}  # 儲存 PowerLossCurve 物件的字典
if "current_curve_name" not in st.session_state:
    st.session_state.current_curve_name = None
if "calib_pts" not in st.session_state:
    st.session_state.calib_pts = [] # 儲存 (0,0) 和 (Max, Max)

st.title("⚡ Power Device Loss Evaluator (精準多曲線版)")

# ==========================================
# Sidebar 控制區：新增與切換曲線
# ==========================================
st.sidebar.header("🛠️ 1. 曲線管理")

new_name = st.sidebar.text_input("輸入新曲線名稱 (如 Eon_150C)", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("選擇擬合模型", [
    "Method_SW3", # E = A + B*i + C*i^2
    "Linear",     # V = V0 + r*I
    "Power"       # E = a*I^b
])

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
else:
    st.info("👈 請先從左側側邊欄「新增曲線」開始。")
    st.stop()

if st.sidebar.button("🔄 重置標定 (若換圖需點擊)"):
    st.session_state.calib_pts = []
    st.rerun()

# ==========================================
# 主畫面：上傳與標定
# ==========================================
uploaded_file = st.file_uploader("2. 上傳 Datasheet 圖檔", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 正在編輯: {current_obj.name} ({current_obj.model_type})")
        
        # 標定狀態提示
        if len(st.session_state.calib_pts) == 0:
            st.warning("⚠️ 步驟 A：請點擊圖表的【左下角原點 (0,0)】")
        elif len(st.session_state.calib_pts) == 1:
            st.warning("⚠️ 步驟 B：請點擊圖表的【右上方最大刻度交點】")
        else:
            st.success("✅ 標定完成！請沿著曲線點擊以擷取數據。")
        
        # 圖片點擊擷取
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
        
        if st.button("🗑️ 清除目前曲線點"):
            current_obj.raw_pixel_points = []
            current_obj.real_data_points = []
            st.rerun()

    with col2:
        st.subheader("📏 3. 座標轉換")
        real_x_max = st.number_input("X 軸最大刻度 (如 1000 A)", value=1000.0)
        real_y_max = st.number_input("Y 軸最大刻度 (如 125 mJ)", value=125.0)
        
        # 如果標定完成，進行像素到實體數值的轉換
        if len(st.session_state.calib_pts) == 2:
            p0 = st.session_state.calib_pts[0]
            p_max = st.session_state.calib_pts[1]
            
            # 避免除以零錯誤
            dx = p_max[0] - p0[0]
            dy = p_max[1] - p0[1]
            if dx != 0 and dy != 0:
                scale_x = real_x_max / dx
                scale_y = real_y_max / dy # 影像Y軸向下為正，若 p_max 在上方，dy 是負的，剛好抵銷
                
                # 即時更新當前曲線的實體數值點
                current_obj.real_data_points = []
                for px, py in current_obj.raw_pixel_points:
                    rx = (px - p0[0]) * scale_x
                    ry = (py - p0[1]) * scale_y
                    current_obj.real_data_points.append((rx, ry))
                
                df = pd.DataFrame(current_obj.real_data_points, columns=["X (A)", "Y (Value)"])
                st.write(f"已點選 {len(df)} 個點")
                st.dataframe(df)

# ==========================================
# 擬合與多曲線繪圖驗證
# ==========================================
if len(st.session_state.calib_pts) == 2 and current_obj.real_data_points:
    st.divider()
    if st.button(f"🚀 執行擬合 ({current_obj.name})", type="primary"):
        x_data = np.array([p[0] for p in current_obj.real_data_points])
        y_data = np.array([p[1] for p in current_obj.real_data_points])
        
        try:
            if current_obj.model_type == "Method_SW3":
                def func(i, A, B, C): return A + B * i + C * i**2
                popt, _ = curve_fit(func, x_data, y_data)
            elif current_obj.model_type == "Linear":
                def func(i, v0, r): return v0 + r * i
                popt, _ = curve_fit(func, x_data, y_data)
            else:
                def func(i, a, b): return a * (i**b)
                popt, _ = curve_fit(func, x_data, y_data)
                
            current_obj.set_params(popt)
            st.success(f"✅ {current_obj.name} 擬合成功！")
        except Exception as e:
            st.error(f"擬合失敗，請確認數據點分佈。錯誤: {e}")

    # 繪製所有擁有參數的曲線
    fig, ax = plt.subplots(figsize=(10, 5))
    has_plot = False
    
    xi = np.linspace(0, real_x_max, 200)
    
    for name, obj in st.session_state.curve_objects.items():
        if obj.params is not None:
            has_plot = True
            # 利用 Class 內建的 get_loss 方法計算 Y 值陣列
            yi = [obj.get_loss(x) for x in xi]
            ax.plot(xi, yi, label=f"{name} ({obj.model_type})")
            
            # 若為當前編輯對象，額外畫出紅點對照
            if name == current_obj.name and obj.real_data_points:
                pts = np.array(obj.real_data_points)
                ax.scatter(pts[:,0], pts[:,1], color='red', marker='x', label=f"{name} Data")

    if has_plot:
        ax.set_xlim(0, real_x_max)
        ax.set_ylim(0, real_y_max)
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Loss (mJ) / Voltage (V)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

# ==========================================
# 最終參數匯出表
# ==========================================
st.divider()
st.subheader("📋 系統曲線定義參數庫 (Parameters Library)")
summary = []
for name, obj in st.session_state.curve_objects.items():
    if obj.params is not None:
        summary.append({
            "Curve Name": name,
            "Model": obj.model_type,
            "Equation": obj.get_equation_string()
        })
if summary:
    st.table(pd.DataFrame(summary))
else:
    st.write("目前尚無已完成擬合的曲線。")