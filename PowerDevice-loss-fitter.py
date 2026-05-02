import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Total Loss Fitter", layout="wide")

# ==========================================
# 核心定義：曲線類別 (Switching & Conduction)
# ==========================================
class SwitchingCurve:
    """切換損耗 (Method SW3: E = A + B*i + C*i^2)"""
    def __init__(self, name):
        self.name = name
        self.params = None # [A, B, C]
        self.raw_pts = []
        self.real_pts = []
    def get_val(self, i):
        if self.params is None: return 0.0
        return self.params[0] + self.params[1]*abs(i) + self.params[2]*(i**2)

class ConductionCurve:
    """導通損耗 (Method Con1: v = Vx + Rx*i)"""
    def __init__(self, name):
        self.name = name
        self.params = None # [Vx, Rx]
        self.raw_pts = []
        self.real_pts = []
    def get_voltage(self, i):
        """計算導通壓降 v_X"""
        if self.params is None: return 0.0
        return self.params[0] + self.params[1]*abs(i)
    def get_power(self, i_rms, i_avg):
        """計算導通功率損耗 (Eq 16)"""
        if self.params is None: return 0.0
        Vx, Rx = self.params
        return Rx * (i_rms**2) + Vx * i_avg

# ==========================================
# Session State 初始化
# ==========================================
# 分別管理三組標定數據與曲線物件
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "con_curves_igbt" not in st.session_state: st.session_state.con_curves_igbt = {}
if "con_curves_fwd" not in st.session_state: st.session_state.con_curves_fwd = {}

if "calib_sw" not in st.session_state: st.session_state.calib_sw = []
if "calib_igbt" not in st.session_state: st.session_state.calib_igbt = []
if "calib_fwd" not in st.session_state: st.session_state.calib_fwd = []

st.title("🚀 Power Device 綜合損耗建模系統 (三圖並行版)")

# ==========================================
# Sidebar：曲線管理區
# ==========================================
with st.sidebar:
    st.header("📊 曲線庫管理")
    
    st.subheader("🔹 切換損 (SW)")
    sw_name = st.text_input("SW 曲線名", "Eon_150C")
    if st.button("➕ 新增切換曲線"): st.session_state.sw_curves[sw_name] = SwitchingCurve(sw_name)
    
    st.subheader("🔸 IGBT 導通 (Con_IGBT)")
    igbt_name = st.text_input("IGBT 導通名", "IGBT_Vce_150C")
    if st.button("➕ 新增 IGBT 導通"): st.session_state.con_curves_igbt[igbt_name] = ConductionCurve(igbt_name)
    
    st.subheader("🔺 FWD 導通 (Con_FWD)")
    fwd_name = st.text_input("FWD 導通名", "FWD_Vf_150C")
    if st.button("➕ 新增 FWD 導通"): st.session_state.con_curves_fwd[fwd_name] = ConductionCurve(fwd_name)

    st.divider()
    if st.button("🔄 重置所有標定", type="primary"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# Helper Function: 建立作業區域
# ==========================================
def render_zone(title, curve_dict, calib_key, uploader_key, x_label, y_label, x_max_def, y_max_def, fit_type):
    st.header(f"📍 {title}")
    curve_list = list(curve_dict.keys())
    
    if not curve_list:
        st.info(f"💡 指示：請先在左側管理區新增一條『{title}』曲線名稱。")
        return

    active_name = st.selectbox(f"選擇編輯對象 ({title})", curve_list, key=f"sel_{uploader_key}")
    obj = curve_dict[active_name]
    
    up = st.file_uploader(f"上傳 {title} 圖檔", type=["png", "jpg"], key=uploader_key)
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # 引導指示
            if len(st.session_state[calib_key]) == 0:
                st.warning("👉 指示 Step 1：請點擊圖片座標軸的【左下角原點 (0,0)】")
            elif len(st.session_state[calib_key]) == 1:
                st.warning("👉 指示 Step 2：請點擊圖片座標軸的【右上方最大刻度點】")
            else:
                st.success("👉 指示 Step 3：請沿著曲線點擊擷取數據點。完成後點擊右側擬合。")
            
            val = streamlit_image_coordinates(img, key=f"click_{uploader_key}")
            if val:
                p = (val["x"], val["y"])
                if len(st.session_state[calib_key]) < 2:
                    if not st.session_state[calib_key] or p != st.session_state[calib_key][-1]:
                        st.session_state[calib_key].append(p)
                        st.rerun()
                else:
                    if not obj.raw_pts or p != obj.raw_pts[-1]:
                        obj.raw_pts.append(p)
        
        with c2:
            x_m = st.number_input(f"{x_label} 最大值", value=x_max_def, key=f"xm_{uploader_key}")
            y_m = st.number_input(f"{y_label} 最大值", value=y_max_def, key=f"ym_{uploader_key}")
            
            if len(st.session_state[calib_key]) == 2:
                p0, pm = st.session_state[calib_pts := calib_key][0], st.session_state[calib_pts][1]
                sx, sy = x_m/(pm[0]-p0[0]), y_m/(pm[1]-p0[1])
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                st.write(f"已擷取 {len(obj.real_pts)} 個點")
                
                if st.button(f"🚀 擬合 {active_name}", key=f"btn_{uploader_key}"):
                    xd, yd = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    if fit_type == "SW": # Method SW3[cite: 1]
                        popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, xd, yd)
                        obj.set_params(popt)
                    else: # Method Con1 (Linear)
                        z = np.polyfit(xd, yd, 1) # y = z[0]*x + z[1] -> V = Rx*I + Vx
                        obj.params = [z[1], z[0]] # [Vx, Rx]
                    st.success("擬合完成！")
            
            if st.button(f"🗑️ 清除點點", key=f"clr_{uploader_key}"):
                obj.raw_pts = []; obj.real_pts = []; st.rerun()

# ==========================================
# 4. 執行各作業區渲染 (同一頁面)
# ==========================================
# A. 切換損作業區
render_zone("切換損耗 (Switching Energy)", st.session_state.sw_curves, "calib_sw", "up_sw", "Ic (A)", "Energy (mJ)", 1000.0, 125.0, "SW")

st.divider()
# B. IGBT 導通作業區
render_zone("IGBT 導通特性 (Vce-Ic)", st.session_state.con_curves_igbt, "calib_igbt", "up_igbt", "Ic (A)", "Vce (V)", 800.0, 5.0, "CON")

st.divider()
# C. FWD 導通作業區
render_zone("FWD 導通特性 (Vf-If)", st.session_state.con_curves_fwd, "calib_fwd", "up_fwd", "If (A)", "Vf (V)", 800.0, 5.0, "CON")

# ==========================================
# 5. 擬合結果圖形 (分別呈現)
# ==========================================
st.divider()
st.header("📊 擬合結果對照圖")
res_c1, res_c2, res_c3 = st.columns(3)

with res_c1:
    st.subheader("切換損耗擬合結果")
    fig_s, ax_s = plt.subplots()
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None:
            xi = np.linspace(0, 1000, 100); ax_s.plot(xi, [o.get_val(x) for x in xi], label=n)
    ax_s.set_xlabel("Current (A)"); ax_s.set_ylabel("Energy (mJ)"); ax_s.legend(); st.pyplot(fig_s)

with res_c2:
    st.subheader("IGBT 導通特性 (Vce)")
    fig_i, ax_i = plt.subplots()
    for n, o in st.session_state.con_curves_igbt.items():
        if o.params is not None:
            xi = np.linspace(0, 800, 100); ax_i.plot(xi, [o.get_voltage(x) for x in xi], label=n)
    ax_i.set_xlabel("Current (A)"); ax_i.set_ylabel("Voltage (V)"); ax_i.legend(); st.pyplot(fig_i)

with res_c3:
    st.subheader("FWD 導通特性 (Vf)")
    fig_f, ax_f = plt.subplots()
    for n, o in st.session_state.con_curves_fwd.items():
        if o.params is not None:
            xi = np.linspace(0, 800, 100); ax_f.plot(xi, [o.get_voltage(x) for x in xi], label=n)
    ax_f.set_xlabel("Current (A)"); ax_f.set_ylabel("Voltage (V)"); ax_f.legend(); st.pyplot(fig_f)

# ==========================================
# 6. 係數匯總表
# ==========================================
st.divider()
st.subheader("📋 系統係數資料庫 (A, B, C / Vx, Rx)")
summary = []
for n, o in st.session_state.sw_curves.items():
    if o.params is not None: summary.append({"類型": "切換損", "名稱": n, "係數": f"A={o.params[0]:.2e}, B={o.params[1]:.2e}, C={o.params[2]:.2e}"})
for n, o in st.session_state.con_curves_igbt.items():
    if o.params is not None: summary.append({"類型": "IGBT導通", "名稱": n, "係數": f"Vx={o.params[0]:.4f}V, Rx={o.params[1]*1000:.4f}mΩ"})
for n, o in st.session_state.con_curves_fwd.items():
    if o.params is not None: summary.append({"類型": "FWD導通", "名稱": n, "係數": f"Vx={o.params[0]:.4f}V, Rx={o.params[1]*1000:.4f}mΩ"})
st.table(pd.DataFrame(summary))