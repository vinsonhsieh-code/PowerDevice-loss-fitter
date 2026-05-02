import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Total Loss Evaluator", layout="wide")

# ==========================================
# 1. 核心類別定義
# ==========================================
class SwitchingCurve:
    """切換損耗 (Method SW3: E = A + B*i + C*i^2)"""
    def __init__(self, name):
        self.name = name
        self.params = None 
        self.raw_pts = []
        self.real_pts = []
    def get_val(self, i):
        if self.params is None: return 0.0
        i_abs = abs(i)
        return self.params[0] + self.params[1]*i_abs + self.params[2]*(i_abs**2)

class ConductionDevice:
    """導通損耗 (Method Con1 & Eq 19)"""
    def __init__(self, name):
        self.name = name
        self.t_min = 298.15 
        self.t_max = 423.15 
        self.fit_tmin = None # [Vx, Rx]
        self.fit_tmax = None # [Vx, Rx]
        self.raw_pts_tmin, self.raw_pts_tmax = [], []
        self.real_pts_tmin, self.real_pts_tmax = [], []

    def get_eq19_params(self, tj_k):
        """解算 Tj 下的 Rx 與 Vx (Eq 17, 18)"""
        if self.fit_tmin is None or self.fit_tmax is None: return None, None
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        T1, T2 = self.t_min, self.t_max
        denom = T1 - T2
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * tj_k
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * tj_k
        return vx_tj, rx_tj

    def calc_pcon(self, tj_k, i_rms, i_avg):
        """計算導通損耗 (Eq 16/19)[cite: 1]"""
        vx, rx = self.get_eq19_params(tj_k)
        if vx is None: return 0.0
        return rx * (i_rms**2) + vx * i_avg

# ==========================================
# 2. Session State 管理
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")
for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("🚀 Power Device 綜合損耗建模系統 (Equation 19)")

# ==========================================
# 3. Sidebar 與 標定重置
# ==========================================
with st.sidebar:
    st.header("📊 曲線庫管理")
    sw_n = st.text_input("新增切換曲線 (如 Eon)", "Eon_150C")
    if st.button("➕ 新增切換曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    st.divider()
    st.subheader("🌡️ 溫度基準 (K)")
    t1 = st.number_input("T_min", value=298.15)
    t2 = st.number_input("T_max", value=423.15)
    st.session_state.igbt_obj.t_min = st.session_state.fwd_obj.t_min = t1
    st.session_state.igbt_obj.t_max = st.session_state.fwd_obj.t_max = t2
    if st.button("🔄 重置所有標定", type="primary"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 三大作業作業區 (同一頁面)
# ==========================================

# --- A. 切換損區域 ---
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list: st.info("💡 指示：請在左側新增『切換曲線』名稱。")
else:
    active_sw = st.selectbox("切換曲線對象", sw_list)
    up_sw = st.file_uploader("上傳切換能量圖", type=["png", "jpg"], key="up_sw")
    if up_sw:
        img = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_sw) < 2: st.warning("👉 指示：請依序點擊原點(0,0)與最大刻度點進行標定。")
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
            xm, ym = st.number_input("SW X(A)最大", 1000.0), st.number_input("SW Y(mJ)最大", 125.0)
            if len(st.session_state.calib_sw) == 2:
                p0, pm = st.session_state.calib_sw
                sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
                obj = st.session_state.sw_curves[active_sw]
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                if st.button(f"🚀 擬合切換損 {active_sw}"):
                    xd, yd = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, xd, yd)
                    obj.params = popt; st.success("擬合完成")

# --- B. IGBT 導通區域 ---
st.divider()
st.header("2️⃣ IGBT 導通特性建模 (Method Con1)")
t_m_i = st.radio("IGBT 標定溫度點", ["T_min", "T_max"], horizontal=True, key="tm_igbt")
up_igbt = st.file_uploader("上傳 IGBT v-i 圖", type=["png", "jpg"], key="up_igbt")
if up_igbt:
    img = Image.open(up_igbt)
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(st.session_state.calib_igbt) < 2: st.warning("👉 指示：標定原點與最大點。")
        v = streamlit_image_coordinates(img, key="click_igbt")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_igbt) < 2:
                if not st.session_state.calib_igbt or p != st.session_state.calib_igbt[-1]:
                    st.session_state.calib_igbt.append(p); st.rerun()
            else:
                target = st.session_state.igbt_obj.raw_pts_tmin if t_m_i=="T_min" else st.session_state.igbt_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("IGBT V(V)最大", 5.0), st.number_input("IGBT I(A)最大", 800.0)
        if len(st.session_state.calib_igbt) == 2:
            p0, pm = st.session_state.calib_igbt
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            raw = st.session_state.igbt_obj.raw_pts_tmin if t_m_i=="T_min" else st.session_state.igbt_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if t_m_i=="T_min": st.session_state.igbt_obj.real_pts_tmin = real
            else: st.session_state.igbt_obj.real_pts_tmax = real
            if st.button(f"🚀 擬合 IGBT-{t_m_i}"):
                cur = st.session_state.igbt_obj.real_pts_tmin if t_m_i=="T_min" else st.session_state.igbt_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if t_m_i=="T_min": st.session_state.igbt_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.igbt_obj.fit_tmax = [z[1], z[0]]
                st.success(f"{t_m_i} 擬合成功")

# --- C. FWD 導通區域 ---
st.divider()
st.header("3️⃣ FWD 導通特性建模 (Method Con1)")
t_m_f = st.radio("FWD 標定溫度點", ["T_min", "T_max"], horizontal=True, key="tm_fwd")
up_fwd = st.file_uploader("上傳 FWD v-i 圖", type=["png", "jpg"], key="up_fwd")
if up_fwd:
    img = Image.open(up_fwd)
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(st.session_state.calib_fwd) < 2: st.warning("👉 指示：標定原點與最大點。")
        v = streamlit_image_coordinates(img, key="click_fwd")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_fwd) < 2:
                if not st.session_state.calib_fwd or p != st.session_state.calib_fwd[-1]:
                    st.session_state.calib_fwd.append(p); st.rerun()
            else:
                target = st.session_state.fwd_obj.raw_pts_tmin if t_m_f=="T_min" else st.session_state.fwd_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("FWD V(V)最大", 5.0, key="fx"), st.number_input("FWD I(A)最大", 800.0, key="fy")
        if len(st.session_state.calib_fwd) == 2:
            p0, pm = st.session_state.calib_fwd
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            raw = st.session_state.fwd_obj.raw_pts_tmin if t_m_f=="T_min" else st.session_state.fwd_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if t_m_f=="T_min": st.session_state.fwd_obj.real_pts_tmin = real
            else: st.session_state.fwd_obj.real_pts_tmax = real
            if st.button(f"🚀 擬合 FWD-{t_m_f}"):
                cur = st.session_state.fwd_obj.real_pts_tmin if t_m_f=="T_min" else st.session_state.fwd_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if t_m_f=="T_min": st.session_state.fwd_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.fwd_obj.fit_tmax = [z[1], z[0]]
                st.success(f"{t_m_f} 擬合成功")

# ==========================================
# 5. 全域分析與 Eq 19 計算器 (重要更新)
# ==========================================
st.divider()
st.header("📊 全域損耗分析器 (Switching + Conduction Eq 19)")
c1, c2, c3 = st.columns(3)
tj_eval = c1.number_input("工作接面溫度 Tj (K)", value=398.15)
i_peak = c2.number_input("峰值電流 I_peak (A)", value=400.0)
f_sw = c3.number_input("切換頻率 f_sw (Hz)", value=10000.0)

# 正弦波自動解算 (半波正弦波)
i_rms = i_peak / 2  # 正弦波半波 RMS (Ix)[cite: 1]
i_avg = i_peak / np.pi # 正弦波半波平均 (Ix,ave)[cite: 1]

st.info(f"💡 正弦波電流自動解算結果：RMS 電流 $I_X$ = {i_rms:.2f} A, 平均電流 $i_{{X,ave}}$ = {i_avg:.2f} A")

res_col1, res_col2 = st.columns(2)
with res_col1:
    st.subheader("擬合曲線可視化")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # SW
    xi = np.linspace(0, 1000, 100)
    for n, o in st.session_state.sw_curves.items():
        if o.params: ax[0].plot(xi, [o.get_val(x) for x in xi], label=n)
    ax[0].set_title("Switching Energy (mJ)"); ax[0].legend()
    # Con
    ii = np.linspace(0, 800, 100)
    for obj in [st.session_state.igbt_obj, st.session_state.fwd_obj]:
        vx, rx = obj.get_eq19_params(tj_eval)
        if vx: ax[1].plot(vx + rx*ii, ii, label=f"{obj.name}@{tj_eval}K")
    ax[1].set_title("Conduction V-I"); ax[1].legend()
    st.pyplot(fig)

with res_col2:
    st.subheader("計算結果彙整 (Eq 19 & SW3)")
    # 計算切換損 (模式 B 積分)[cite: 1]
    p_sw_total = 0.0
    for n, o in st.session_state.sw_curves.items():
        if o.params:
            e_avg = np.mean([o.get_val(i_peak * np.sin(t)) for t in np.linspace(0, np.pi, 100)])
            p_sw_total += e_avg * f_sw * 1e-3 #[cite: 1]
    
    p_con_igbt = st.session_state.igbt_obj.calc_pcon(tj_eval, i_rms, i_avg)
    p_con_fwd = st.session_state.fwd_obj.calc_pcon(tj_eval, i_rms, i_avg)
    
    st.metric("總切換功率損耗 (P_sw)", f"{p_sw_total:.2f} W")
    st.metric("IGBT 導通損 (P_con,IGBT)", f"{p_con_igbt:.2f} W")
    st.metric("FWD 導通損 (P_con,FWD)", f"{p_con_fwd:.2f} W")
    st.markdown(f"### **總預估損耗 (P_total): {p_sw_total + p_con_igbt + p_con_fwd:.2f} W**")