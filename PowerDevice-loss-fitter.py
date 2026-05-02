import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - R&D Pro", layout="wide")

# ==========================================
# 核心定義：曲線類別 (切換損功能保持不變)
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
# 核心定義：導通損耗模型 (Method Con1)
# ==========================================
def calc_conduction_params(Tj, Tmin, Tmax, R1, R2, V1, V2):
    """依據 Eq 17 & 18 計算特定溫度下的 R 和 V"""
    # R_X(Tj) = (R2*Tmin - R1*Tmax)/(Tmin - Tmax) + (R1-R2)/(Tmin - Tmax) * Tj
    denom = Tmin - Tmax
    Rx = ((R2 * Tmin - R1 * Tmax) / denom) + ((R1 - R2) / denom) * Tj
    Vx = ((V2 * Tmin - V1 * Tmax) / denom) + ((V1 - V2) / denom) * Tj
    return Rx, Vx

# ==========================================
# Session State 初始化
# ==========================================
if "curve_objects" not in st.session_state:
    st.session_state.curve_objects = {}  
if "calib_pts" not in st.session_state:
    st.session_state.calib_pts = [] 

st.title("⚡ Power Device Total Loss Evaluator (R&D Pro)")

# ==========================================
# 區域 A：切換損耗 (Switching Loss) - 功能維持
# ==========================================
st.header("📊 1. 切換損耗擬合與分析 (Switching Loss)")

# (側邊欄管理切換損物件)
st.sidebar.header("🛠️ 切換損管理")
new_sw_name = st.sidebar.text_input("新增切換損曲線 (如 Eon)", value=f"SW_Curve_{len(st.session_state.curve_objects)+1}")
if st.sidebar.button("➕ 新增切換損曲線"):
    st.session_state.curve_objects[new_sw_name] = PowerLossCurve(name=new_sw_name)
    st.rerun()

all_sw_names = list(st.session_state.curve_objects.keys())
selected_sw = None
if all_sw_names:
    selected_sw = st.sidebar.selectbox("編輯切換損對象", all_sw_names)
    current_obj = st.session_state.curve_objects[selected_sw]
    if st.sidebar.button("🗑️ 刪除"):
        del st.session_state.curve_objects[selected_sw]
        st.rerun()

# 標定與擬合邏輯 (略，與前版相同以節省空間，但程式碼完整保留功能)
uploaded_file = st.file_uploader("上傳圖檔進行切換損標定", type=["png", "jpg", "jpeg"])
if uploaded_file and selected_sw:
    img = Image.open(uploaded_file)
    col_img, col_data = st.columns([2, 1])
    with col_img:
        st.write(f"正在擷取: {selected_sw}")
        val = streamlit_image_coordinates(img, key="sw_img")
        if val:
            curr = (val["x"], val["y"])
            if len(st.session_state.calib_pts) < 2:
                if not st.session_state.calib_pts or curr != st.session_state.calib_pts[-1]:
                    st.session_state.calib_pts.append(curr)
                    st.rerun()
            else:
                if not current_obj.raw_pixel_points or curr != current_obj.raw_pixel_points[-1]:
                    current_obj.raw_pixel_points.append(curr)
    with col_data:
        x_max = st.number_input("切換損 X 軸最大值 (A)", value=1000.0, key="sw_xmax")
        y_max = st.number_input("切換損 Y 軸最大值 (mJ)", value=125.0, key="sw_ymax")
        if len(st.session_state.calib_pts) == 2:
            p0, pm = st.session_state.calib_pts
            sc_x, sc_y = x_max/(pm[0]-p0[0]), y_max/(pm[1]-p0[1])
            current_obj.real_data_points = [((p[0]-p0[0])*sc_x, (p[1]-p0[1])*sc_y) for p in current_obj.raw_pixel_points]
            if st.button(f"🚀 執行擬合 {selected_sw}"):
                x = np.array([p[0] for p in current_obj.real_data_points])
                y = np.array([p[1] for p in current_obj.real_data_points])
                popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, x, y)
                current_obj.set_params(popt)
            st.dataframe(pd.DataFrame(current_obj.real_data_points, columns=["I(A)", "E(mJ)"]), height=150)

st.divider()

# ==========================================
# 區域 B：導通損耗 (Conduction Loss) - 追加區域
# ==========================================
st.header("🌡️ 2. 導通損耗評估 (Conduction Loss - Method Con1)")

# 建立參數輸入表 (依據論文 Table 4)
st.subheader("📋 參數輸入 (依據 Datasheet $v-i$ 特性)")
col_t, col_igbt, col_fwd = st.columns(3)

with col_t:
    st.write("**溫度基準 (Kelvin)**")
    Tmin = st.number_input("T_min (K)", value=298.15)
    Tmax = st.number_input("T_max (K)", value=423.15)
    Tj_target = st.number_input("工作接面溫度 Tj (K)", value=398.15)

with col_igbt:
    st.write("**IGBT 參數**")
    R1_I = st.number_input("R1 (mΩ) @ Tmin", value=2.34, key="R1I") / 1000.0 # 轉為 Ohm
    R2_I = st.number_input("R2 (mΩ) @ Tmax", value=3.90, key="R2I") / 1000.0
    V1_I = st.number_input("V1 (V) @ Tmin", value=1.00, key="V1I")
    V2_I = st.number_input("V2 (V) @ Tmax", value=0.85, key="V2I")

with col_fwd:
    st.write("**FWD 參數**")
    R1_F = st.number_input("R1 (mΩ) @ Tmin", value=2.40, key="R1F") / 1000.0
    R2_F = st.number_input("R2 (mΩ) @ Tmax", value=3.34, key="R2F") / 1000.0
    V1_F = st.number_input("V1 (V) @ Tmin", value=1.45, key="V1F")
    V2_F = st.number_input("V2 (V) @ Tmax", value=1.05, key="V2F")

# 計算當前 Tj 下的線性係數
Rx_I, Vx_I = calc_conduction_params(Tj_target, Tmin, Tmax, R1_I, R2_I, V1_I, V2_I)
Rx_F, Vx_F = calc_conduction_params(Tj_target, Tmin, Tmax, R1_F, R2_F, V1_F, V2_F)

st.info(f"💡 在 {Tj_target}K 時：\n"
        f"- IGBT: Rx = {Rx_I*1000:.4f} mΩ, Vx = {Vx_I:.4f} V\n"
        f"- FWD: Rx = {Rx_F*1000:.4f} mΩ, Vx = {Vx_F:.4f} V")

st.divider()

# ==========================================
# 區域 C：全域分析與整合計算器
# ==========================================
st.header("📊 3. 系統總損耗整合分析 (System Total Power)")

c1, c2, c3 = st.columns(3)
with c1:
    I_rms = st.number_input("有效電流 I_rms (A)", value=300.0)
with c2:
    I_avg = st.number_input("平均電流 I_avg (A)", value=200.0)
with c3:
    f_sw_total = st.number_input("切換頻率 f_sw (Hz)", value=10000.0, key="total_fsw")

# 1. 計算切換損 (Switching) - 依據之前擬合的物件
sw_power = 0.0
for obj in st.session_state.curve_objects.values():
    if obj.params is not None:
        # 這裡簡化為模式 B 的正弦平均概念
        phi = np.linspace(0, np.pi, 100)
        e_avg = np.mean([obj.get_loss(I_rms * np.sqrt(2) * np.sin(p)) for p in phi])
        sw_power += e_avg * f_sw_total * 1e-3

# 2. 計算導通損 (Conduction) - 依據 Eq 16 & 20[cite: 1, 2]
Pcon_IGBT = Rx_I * (I_rms**2) + Vx_I * I_avg
Pcon_FWD = Rx_F * (I_rms**2) + Vx_F * I_avg
Pcon_Total = Pcon_IGBT + Pcon_FWD

# 3. 顯示結果
res_sw, res_con, res_total = st.columns(3)
res_sw.metric("總切換損耗 (P_sw)", f"{sw_power:.2f} W")
res_con.metric("總導通損耗 (P_con)", f"{Pcon_Total:.2f} W")
res_total.metric("系統總功率損耗", f"{sw_power + Pcon_Total:.2f} W", delta=f"Conductions accounts for {Pcon_Total/(sw_power+Pcon_Total+1e-6)*100:.1f}%")

# 繪製導通損耗 V-I 曲線驗證
st.subheader("📈 導通特性 V-I 曲線 (線性化驗證)")
vi_i = np.linspace(0, I_rms*1.5, 100)
v_igbt = Rx_I * vi_i + Vx_I
v_fwd = Rx_F * vi_i + Vx_F

fig_vi, ax_vi = plt.subplots(figsize=(10, 4))
ax_vi.plot(vi_i, v_igbt, label=f"IGBT @ {Tj_target}K (Linear)", color='blue')
ax_vi.plot(vi_i, v_fwd, label=f"FWD @ {Tj_target}K (Linear)", color='red')
ax_vi.set_xlabel("Current i_x (A)")
ax_vi.set_ylabel("Forward Voltage v_x (V)")
ax_vi.legend()
ax_vi.grid(True, alpha=0.3)
st.pyplot(fig_vi)