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
    def __init__(self, name, model_type="Method_SW3"):
        self.name = name
        self.model_type = model_type
        self.params = None  
        self.raw_pixel_points = [] 
        self.real_data_points = [] 

    def set_params(self, params):
        self.params = params

    def get_loss(self, current_i):
        """根據論文(8)式代入電流計算損耗"""
        if self.params is None: return 0.0
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2
            A, B, C = self.params
            return A + B * current_i + C * (current_i ** 2)
        elif self.model_type == "Linear":
            v0, r = self.params
            return v0 + r * current_i
        elif self.model_type == "Power":
            a, b = self.params
            return a * (current_i ** b)
        return 0.0

    def get_equation_string(self):
        if self.params is None: return "尚未擬合"
        if self.model_type == "Method_SW3":
            return f"{self.params[0]:.6e} + {self.params[1]:.6e}*i + {self.params[2]:.6e}*i^2"
        elif self.model_type == "Linear":
            return f"{self.params[0]:.4f} + {self.params[1]:.4f}*i"
        elif self.model_type == "Power":
            return f"{self.params[0]:.4f} * i^{self.params[1]:.4f}"

# ==========================================
# Session State 初始化
# ==========================================
if "curve_objects" not in st.session_state:
    st.session_state.curve_objects = {}  
if "current_curve_name" not in st.session_state:
    st.session_state.current_curve_name = None
if "calib_pts" not in st.session_state:
    st.session_state.calib_pts = [] 

st.title("⚡ Power Device Loss Evaluator (Pro Version)")

# ==========================================
# Sidebar 曲線管理
# ==========================================
st.sidebar.header("🛠️ 1. 曲線與標定管理")

new_name = st.sidebar.text_input("輸入新曲線名稱 (如 Eon, Eoff, Err)", value=f"Curve_{len(st.session_state.curve_objects) + 1}")
model_choice = st.sidebar.selectbox("選擇擬合模型", ["Method_SW3", "Linear", "Power"])

if st.sidebar.button("➕ 新增曲線檔案"):
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
    if st.sidebar.button("🗑️ 刪除目前選中的曲線", use_container_width=True):
        del st.session_state.curve_objects[selected_name]
        remaining = list(st.session_state.curve_objects.keys())
        st.session_state.current_curve_name = remaining[0] if remaining else None
        st.rerun()
else:
    st.info("👈 請先新增曲線。")
    st.stop()

if st.sidebar.button("🔄 重置座標標定 (換圖用)"):
    st.session_state.calib_pts = []
    st.rerun()

# ==========================================
# 主畫面：標定與擬合
# ==========================================
uploaded_file = st.file_uploader("2. 上傳 Datasheet 圖檔", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 編輯中: {current_obj.name}")
        if len(st.session_state.calib_pts) < 2:
            st.warning("請先完成兩點標定：(0,0) 與 (Max, Max)")
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
        except Exception as e: st.error(f"擬合失敗: {e}")

    # --- 5. 全域損耗計算器 ---
    st.subheader("📋 5. 全域損耗計算器 (Global Loss Calculator)")
    fitted_objs = {name: obj for name, obj in st.session_state.curve_objects.items() if obj.params is not None}
    
    if fitted_objs:
        calc_i = st.number_input("輸入目標電流 $i$ (A)，即時計算所有已擬合曲線數值：", value=float(real_x_max/2))
        
        calc_results = []
        total_sw_loss = 0.0
        
        for name, obj in fitted_objs.items():
            val = obj.get_loss(calc_i)
            calc_results.append({"曲線名稱": name, "計算結果 (Value)": f"{val:.4f}", "模型": obj.model_type})
            # 如果名稱包含 Eon, Eoff, Err 則自動累計總切換損耗
            if any(key in name.lower() for key in ["eon", "eoff", "err"]):
                total_sw_loss += val
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.table(pd.DataFrame(calc_results))
        with c2:
            st.metric(label="預估總切換損耗 (Eon+Eoff+Err)", value=f"{total_sw_loss:.4f} mJ")
            st.caption(f"基於電流 $i = {calc_i}$ A")
    else:
        st.info("請至少完成一條曲線的擬合以啟用全域計算器。")

    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 4))
    xi = np.linspace(0, real_x_max, 200)
    for name, obj in st.session_state.curve_objects.items():
        if obj.params is not None:
            ax.plot(xi, [obj.get_loss(val) for val in xi], label=f"{name}")
            if name == current_obj.name:
                pts = np.array(current_obj.real_data_points)
                ax.scatter(pts[:,0], pts[:,1], color='red', marker='x')
    ax.set_xlim(0, real_x_max); ax.set_ylim(0, real_y_max)
    ax.set_xlabel("Current (A)"); ax.set_ylabel("Value"); ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# 參數彙整表
st.divider()
st.subheader("📋 擬合係數彙整表 (A, B, C)")
summary = [{"曲線名稱": n, "模型": o.model_type, "方程式": o.get_equation_string()} 
           for n, o in st.session_state.curve_objects.items() if o.params is not None]
if summary: st.table(pd.DataFrame(summary))