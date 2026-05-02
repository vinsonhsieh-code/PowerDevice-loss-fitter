import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter Pro - SW & CON", layout="wide")

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
        Switching: 回傳單次能量 E (mJ)
        Conduction: 回傳瞬時電壓 V (V)
        """
        if self.params is None: return 0.0
        i = abs(current_i)
        if self.model_type == "Method_SW3":
            # E = A + B*i + C*i^2[cite: 1]
            A, B, C = self.params
            return A + B * i + C * (i ** 2)
        elif self.model_type == "Linear" or self.model_type == "Method_Con1":
            # V = Vx + Rx * i
            v_th, r = self.params
            return v_th + r * i
        elif self.model_type == "Power":
            a, b = self.params
            return a * (i ** b)
        return 0.0

    def get_equation_string(self):
        if self.params is None: return "尚未擬合"
        p = self.params
        if self.model_type == "Method_SW3":
            return f"{p[0]:.4e} + {p[1]:.4e}*i + {p[2]:.4e}*i^2"
        elif self.model_type == "Linear" or self.model_type == "Method_Con1":
            return f"V_th:{p[0]:.4f} + R:{p[1]:.4e}*i"
        elif self.model_type == "Power":
            return f"{p[0]:.4f} * i^{p[1]:.4f}"

# ==========================================
# Session State 初始化
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "con_curves" not in st.session_state: st.session_state.con_curves = {}
if "active_curve" not in st.session_state: st.session_state.active_curve = {"name": None, "cat": None}
if "calib_pts" not in st.session_state: st.session_state.calib_pts = []

st.title("⚡ Power Device Loss Evaluator (SW & CON Integrated)")

# ==========================================
# Sidebar：獨立管理區域
# ==========================================
st.sidebar.header("📊 1a. 切換損管理 (Switching)")
sw_name = st.sidebar.text_input("切換損曲線名稱", value=f"SW_{len(st.session_state.sw_curves)+1}")
sw_model = st.sidebar.selectbox("切換損模型", ["Method_SW3", "Power"], key="sw_m")
if st.sidebar.button("➕ 新增切換損曲線"):
    st.session_state.sw_curves[sw_name] = PowerLossCurve(sw_name, "Switching", sw_model)
    st.session_state.active_curve = {"name": sw_name, "cat": "Switching"}

st.sidebar.divider()
st.sidebar.header("🔌 1b. 導通損管理 (Conduction)")
con_name = st.sidebar.text_input("導通損曲線名稱", value=f"CON_{len(st.session_state.con_curves)+1}")
if st.sidebar.button("➕ 新增導通損曲線"):
    st.session_state.con_curves[con_name] = PowerLossCurve(con_name, "Conduction", "Method_Con1")
    st.session_state.active_curve = {"name": con_name, "cat": "Conduction"}

st.sidebar.divider()
all_sw = list(st.session_state.sw_curves.keys())
all_con = list(st.session_state.con_curves.keys())
selected_target = st.sidebar.selectbox("🎯 目前選中編輯對象", all_sw + all_con)
if selected_target:
    st.session_state.active_curve["name"] = selected_target
    st.session_state.active_curve["cat"] = "Switching" if selected_target in all_sw else "Conduction"

if st.sidebar.button("🗑️ 刪除選中曲線"):
    if selected_target in all_sw: del st.session_state.sw_curves[selected_target]
    else: del st.session_state.con_curves[selected_target]
    st.rerun()

if st.sidebar.button("🔄 重置標定"):
    st.session_state.calib_pts = []
    st.rerun()

# ==========================================
# 主區域 2：圖片擷取與校準
# ==========================================
uploaded_file = st.file_uploader("2. 上傳圖檔", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    col_img, col_data = st.columns([2, 1])

    with col_img:
        target_name = st.session_state.active_curve["name"]
        if target_name:
            st.subheader(f"📍 正在擷取: {target_name}")
            if len(st.session_state.calib_pts) < 2:
                steps = ["點擊原點 (0,0)", "點擊最大刻度點"]
                st.warning(f"請先完成標定：{steps[len(st.session_state.calib_pts)]}")
            
            value = streamlit_image_coordinates(img, key="img_main")
            if value:
                curr = (value["x"], value["y"])
                if len(st.session_state.calib_pts) < 2:
                    if not st.session_state.calib_pts or curr != st.session_state.calib_pts[-1]:
                        st.session_state.calib_pts.append(curr); st.rerun()
                else:
                    target_obj = st.session_state.sw_curves[target_name] if target_name in all_sw else st.session_state.con_curves[target_name]
                    if not target_obj.raw_pixel_points or curr != target_obj.raw_pixel_points[-1]:
                        target_obj.raw_pixel_points.append(curr)

    with col_data:
        st.subheader("📏 座標校正")
        rx_max = st.number_input("X 軸最大值 (A)", value=1000.0)
        ry_max = st.number_input("Y 軸最大值 (mJ 或 V)", value=125.0)
        if len(st.session_state.calib_pts) == 2:
            p0, pm = st.session_state.calib_pts
            sx, sy = rx_max/(pm[0]-p0[0]), ry_max/(pm[1]-p0[1])
            if target_name:
                obj = st.session_state.sw_curves[target_name] if target_name in all_sw else st.session_state.con_curves[target_name]
                obj.real_data_points = [((p[0]-p0[0])*sx, (p[1]-p0[1])*sy) for p in obj.raw_pixel_points]
                st.dataframe(pd.DataFrame(obj.real_data_points, columns=["X","Y"]), height=150)

# ==========================================
# 主區域 3：擬合與呈現
# ==========================================
if uploaded_file and selected_target:
    st.divider()
    if st.button(f"🚀 執行擬合 ({selected_target})", type="primary"):
        obj = st.session_state.sw_curves[selected_target] if selected_target in all_sw else st.session_state.con_curves[selected_target]
        x, y = np.array([p[0] for p in obj.real_data_points]), np.array([p[1] for p in obj.real_data_points])
        if len(x) > 2:
            if obj.model_type == "Method_SW3":
                popt, _ = curve_fit(lambda i,a,b,c: a+b*i+c*i**2, x, y)
            elif obj.model_type == "Method_Con1" or obj.model_type == "Linear":
                popt, _ = curve_fit(lambda i,v,r: v+r*i, x, y)
            else: popt, _ = curve_fit(lambda i,a,b: a*(i**b), x, y)
            obj.set_params(popt); st.success("擬合成功")

    # 分別呈現切換損與導通損圖表
    c_sw_plot, c_con_plot = st.columns(2)
    xi = np.linspace(0, rx_max, 200)
    
    with c_sw_plot:
        st.subheader("📈 切換損擬合曲線 (Energy vs I)")
        fig_sw, ax_sw = plt.subplots()
        for n, o in st.session_state.sw_curves.items():
            if o.params is not None: ax_sw.plot(xi, [o.get_value(v) for v in xi], label=n)
        ax_sw.set_xlabel("Current (A)"); ax_sw.set_ylabel("Energy (mJ)"); ax_sw.legend(); st.pyplot(fig_sw)

    with c_con_plot:
        st.subheader("📈 導通損擬合曲線 (Voltage vs I)")
        fig_con, ax_con = plt.subplots()
        for n, o in st.session_state.con_curves.items():
            if o.params is not None: ax_con.plot(xi, [o.get_value(v) for v in xi], label=n)
        ax_con.set_xlabel("Current (A)"); ax_con.set_ylabel("Voltage (V)"); ax_con.legend(); st.pyplot(fig_con)

# ==========================================
# 主區域 4：全域動態分析器 (加總 SW + CON)
# ==========================================
st.divider()
st.subheader("📋 4. 綜合損耗動態分析器")
c1, c2, c3 = st.columns(3)
with c1: i_peak = st.number_input("峰值電流 I_peak (A)", value=500.0)
with c2: f_out = st.number_input("基波頻率 f_out (Hz)", value=50.0)
with c3: f_sw = st.number_input("切換頻率 f_sw (Hz)", value=10000.0)

t = np.linspace(0, 1/f_out, 500); i_w = i_peak * np.sin(2*np.pi*f_out*t)
p_sw_total, p_con_total = 0.0, 0.0

# 計算功率
for o in st.session_state.sw_curves.values():
    if o.params is not None:
        e_avg = np.mean([o.get_value(i_peak * np.sin(phi)) for phi in np.linspace(0, np.pi, 200)])
        p_sw_total += e_avg * f_sw * 1e-3 #[cite: 1]

for o in st.session_state.con_curves.values():
    if o.params is not None:
        # P_con = mean(v(i)*i)
        p_instant = [o.get_value(iv) * abs(iv) for iv in i_w]
        p_con_total += np.mean(p_instant) #[cite: 2]

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("總切換損耗 (P_sw)", f"{p_sw_total:.2f} W")
col_m2.metric("總導通損耗 (P_con)", f"{p_con_total:.2f} W")
col_m3.metric("總功率損耗 (P_total)", f"{p_sw_total + p_con_total:.2f} W")

# 動態波形
fig_w, ax_w1 = plt.subplots(figsize=(10, 4))
ax_w1.plot(t*1000, i_w, 'b--', alpha=0.3, label="i(t)")
ax_w1.set_ylabel("Current (A)", color='b')
ax_w2 = ax_w1.twinx()
for n, o in st.session_state.sw_curves.items():
    if o.params is not None: ax_w2.plot(t*1000, [o.get_value(iv)*f_sw*1e-3 for iv in i_w], label=f"P_sw:{n}")
for n, o in st.session_state.con_curves.items():
    if o.params is not None: ax_w2.plot(t*1000, [o.get_value(iv)*abs(iv) for iv in i_w], label=f"P_con:{n}")
ax_w2.set_ylabel("Instantaneous Power (W)", color='r')
ax_w2.legend(loc='upper right', fontsize='small')
st.pyplot(fig_w)