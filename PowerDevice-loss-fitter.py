import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Evaluator - Eq 19", layout="wide")

# ==========================================
# 1. 核心定義：曲線管理物件
# ==========================================
class SwitchingCurve:
    def __init__(self, name):
        self.name = name
        self.params = None # [A, B, C]
        self.raw_pts = []
        self.real_pts = []
    def get_val(self, i):
        if self.params is None: return 0.0
        return self.params[0] + self.params[1]*abs(i) + self.params[2]*(i**2)

class ConductionDevice:
    """管理單一元件(IGBT或FWD)在不同溫度下的導通特性"""
    def __init__(self, name):
        self.name = name
        self.t_min = 298.15 # 預設 Tmin (K)
        self.t_max = 423.15 # 預設 Tmax (K)
        self.fit_tmin = None # [Vx, Rx] at Tmin
        self.fit_tmax = None # [Vx, Rx] at Tmax
        self.raw_pts_tmin = []
        self.raw_pts_tmax = []
        self.real_pts_tmin = []
        self.real_pts_tmax = []

    def calc_eq19(self, tj_k, i_rms, i_avg):
        """實作論文方程式 (19)"""
        if self.fit_tmin is None or self.fit_tmax is None: return 0.0
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        T1, T2 = self.t_min, self.t_max
        
        # 分母
        denom = T1 - T2
        # 計算 Rx(Tj) - Equation (17)
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * tj_k
        # 計算 Vx(Tj) - Equation (18)
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * tj_k
        
        # 回傳 P_con,X - Equation (19)
        return rx_tj * (i_rms**2) + vx_tj * i_avg

# ==========================================
# 2. Session State 初始化
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")

# 標定暫存
for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("🚀 Power Device 綜合損耗建模系統 (Equation 19)")

# ==========================================
# 3. 側邊欄：統一管理
# ==========================================
with st.sidebar:
    st.header("📊 系統參數設定")
    sw_n = st.text_input("新增切換曲線名", "Eon_150C")
    if st.button("➕ 新增切換曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    
    st.divider()
    st.subheader("🌡️ 導通損溫度設定 (K)")
    st.session_state.igbt_obj.t_min = st.number_input("T_min (例如 298.15)", value=298.15)
    st.session_state.igbt_obj.t_max = st.number_input("T_max (例如 423.15)", value=423.15)
    st.session_state.fwd_obj.t_min, st.session_state.fwd_obj.t_max = st.session_state.igbt_obj.t_min, st.session_state.igbt_obj.t_max

    if st.button("🔄 重置所有標定", type="primary"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 切換損作業區 (維持現狀)
# ==========================================
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list:
    st.info("💡 指示：請先在左側新增一條『切換損耗曲線』名稱。")
else:
    active_sw = st.selectbox("切換曲線編輯對象", sw_list)
    up_sw = st.file_uploader("上傳切換損圖檔", type=["png", "jpg"], key="up_sw")
    if up_sw:
        img = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_sw) < 2: st.warning("👉 指示：請先標定原點與最大刻度點。")
            v = streamlit_image_coordinates(img, key="click_sw")
            if v:
                p = (v["x"], v["y"])
                if len(st.session_state.calib_sw) < 2:
                    if not st.session_state.calib_sw or p != st.session_state.calib_sw[-1]:
                        st.session_state.calib_sw.append(p); st.rerun()
                else:
                    obj = st.session_state.sw_curves[active_sw]
                    if not obj.raw_pts or p != obj.raw_pts[-1]: obj.raw_pts.append(p)
        with c2:
            xm, ym = st.number_input("X軸最大(A)", 1000.0), st.number_input("Y軸最大(mJ)", 125.0)
            if len(st.session_state.calib_sw) == 2:
                p0, pm = st.session_state.calib_sw
                sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
                obj = st.session_state.sw_curves[active_sw]
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                if st.button(f"🚀 擬合 {active_sw}"):
                    xd, yd = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, xd, yd)
                    obj.params = popt; st.success("擬合成功")

# ==========================================
# 5. 導通損作業區 (IGBT & FWD 分別呈現)
# ==========================================
def render_con_zone(device_obj, calib_key, up_key, title):
    st.divider()
    st.header(f"2️⃣ {title} 導通特性建模 (Method Con1)")
    t_mode = st.radio(f"選擇標定溫度點 ({title})", ["T_min", "T_max"], horizontal=True, key=f"mode_{up_key}")
    up = st.file_uploader(f"上傳 {title} v-i 圖", type=["png", "jpg"], key=up_key)
    
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state[calib_key]) < 2: st.warning("👉 指示：請標定原點(0,0)與最大刻度點。")
            else: st.success(f"👉 指示：現在正在擷取 {t_mode} 的曲線數據。")
            v = streamlit_image_coordinates(img, key=f"click_{up_key}")
            if v:
                p = (v["x"], v["y"])
                if len(st.session_state[calib_key]) < 2:
                    if not st.session_state[calib_key] or p != st.session_state[calib_key][-1]:
                        st.session_state[calib_key].append(p); st.rerun()
                else:
                    target_list = device_obj.raw_pts_tmin if t_mode == "T_min" else device_obj.raw_pts_tmax
                    if not target_list or p != target_list[-1]: target_list.append(p)
        with c2:
            xm, ym = st.number_input(f"{title} X軸最大(V)", 5.0, key=f"xm_{up_key}"), st.number_input(f"{title} Y軸最大(A)", 800.0, key=f"ym_{up_key}")
            if len(st.session_state[calib_key]) == 2:
                p0, pm = st.session_state[calib_key]
                sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
                raw = device_obj.raw_pts_tmin if t_mode == "T_min" else device_obj.raw_pts_tmax
                real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
                if t_mode == "T_min": device_obj.real_pts_tmin = real
                else: device_obj.real_pts_tmax = real
                
                if st.button(f"🚀 擬合 {title} - {t_mode}", key=f"btn_{up_key}"):
                    curr_real = device_obj.real_pts_tmin if t_mode == "T_min" else device_obj.real_pts_tmax
                    v_d, i_d = np.array([p[0] for p in curr_real]), np.array([p[1] for p in curr_real])
                    z = np.polyfit(i_d, v_d, 1) # v = Rx*i + Vx
                    if t_mode == "T_min": device_obj.fit_tmin = [z[1], z[0]] # [Vx, Rx]
                    else: device_obj.fit_tmax = [z[1], z[0]]
                    st.success(f"{t_mode} 擬合完成：Vx={z[1]:.4f}V, Rx={z[0]*1000:.4f}mΩ")

render_con_zone(st.session_state.igbt_obj, "calib_igbt", "up_igbt", "IGBT")
render_con_zone(st.session_state.fwd_obj, "calib_fwd", "up_fwd", "FWD")

# ==========================================
# 6. 結果可視化與 Eq 19 計算器
# ==========================================
st.divider()
st.header("📊 損耗評估與計算 (Equation 19)")
res_c1, res_c2 = st.columns(2)

with res_c1:
    st.subheader("擬合曲線對比")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # IGBT Vce
    ii = np.linspace(0, 800, 100)
    for t, p in [("Tmin", st.session_state.igbt_obj.fit_tmin), ("Tmax", st.session_state.igbt_obj.fit_tmax)]:
        if p: ax[0].plot(p[0]+p[1]*ii, ii, label=t)
    ax[0].set_title("IGBT Vce"); ax[0].set_xlabel("Vce (V)"); ax[0].set_ylabel("Ic (A)"); ax[0].legend()
    # FWD Vf
    for t, p in [("Tmin", st.session_state.fwd_obj.fit_tmin), ("Tmax", st.session_state.fwd_obj.fit_tmax)]:
        if p: ax[1].plot(p[0]+p[1]*ii, ii, label=t)
    ax[1].set_title("FWD Vf"); ax[1].set_xlabel("Vf (V)"); ax[1].set_ylabel("If (A)"); ax[1].legend()
    st.pyplot(fig)

with res_c2:
    st.subheader("🔍 方程式 (19) 即時計算器")
    if st.session_state.igbt_obj.fit_tmax and st.session_state.fwd_obj.fit_tmax:
        tj_eval = st.number_input("工作接面溫度 Tj (K)", value=398.15)
        i_rms = st.number_input("RMS 電流 I_X (A)", value=400.0)
        i_avg = st.number_input("平均電流 i_X,ave (A)", value=250.0)
        
        p_con_igbt = st.session_state.igbt_obj.calc_eq19(tj_eval, i_rms, i_avg)
        p_con_fwd = st.session_state.fwd_obj.calc_eq19(tj_eval, i_rms, i_avg)
        
        st.metric("IGBT 導通損 (P_con,IGBT)", f"{p_con_igbt:.2f} W")
        st.metric("FWD 導通損 (P_con,FWD)", f"{p_con_fwd:.2f} W")
        st.metric("總導通損耗 (P_con)", f"{p_con_igbt + p_con_fwd:.2f} W")
    else:
        st.info("💡 指示：請完成 IGBT 與 FWD 的 Tmin/Tmax 擬合以啟用計算器。")