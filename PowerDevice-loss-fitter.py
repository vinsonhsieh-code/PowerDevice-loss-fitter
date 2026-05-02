import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - SOP Pro", layout="wide")

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
        self.t_min_c = 25.0
        self.t_max_c = 150.0
        self.fit_tmin = None # [Vx, Rx]
        self.fit_tmax = None # [Vx, Rx]
        self.raw_pts_tmin, self.raw_pts_tmax = [], []
        self.real_pts_tmin, self.real_pts_tmax = [], []

    def get_eq19_params(self, tj_c):
        """解算當前溫度下之動態電阻 Rx 與閾值電壓 Vx (Eq 17, 18)"""
        if self.fit_tmin is None or self.fit_tmax is None: return None, None
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        T1, T2 = self.t_min_c + 273.15, self.t_max_c + 273.15
        Tj = tj_c + 273.15
        denom = T1 - T2
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * Tj
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * Tj
        return vx_tj, rx_tj

    def calc_pcon(self, tj_c, i_rms, i_avg):
        """計算導通功率損耗 (Eq 19)"""
        vx, rx = self.get_eq19_params(tj_c)
        if vx is None: return 0.0
        return rx * (i_rms**2) + vx * i_avg

# ==========================================
# 2. Session State 狀態維護
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")
for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("⚡ Power Device 綜合損耗建模系統 (計算式說明版)")

# ==========================================
# 3. Sidebar
# ==========================================
with st.sidebar:
    st.header("📊 曲線庫管理")
    sw_n = st.text_input("新增切換曲線 (如 Eon)", "Eon_150C")
    if st.button("➕ 新增曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    st.divider()
    st.subheader("🌡️ 導通損基準設定 (°C)")
    tc1 = st.number_input("T_min (°C)", value=25.0)
    tc2 = st.number_input("T_max (°C)", value=150.0)
    st.session_state.igbt_obj.t_min_c = st.session_state.fwd_obj.t_min_c = tc1
    st.session_state.igbt_obj.t_max_c = st.session_state.fwd_obj.t_max_c = tc2
    if st.button("🔄 重置標定數據"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 三大作業區 (同一頁面呈現)
# ==========================================

# A. 切換損
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list: st.info("請在側邊欄新增曲線。")
else:
    act_sw = st.selectbox("切換曲線編輯", sw_list)
    up_sw = st.file_uploader("上傳切換損圖檔", type=["png", "jpg"], key="up_sw")
    if up_sw:
        img = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_sw) < 2: st.warning("👉 請先點選原點 (0,0) 與最大點進行座標軸標定。")
            v = streamlit_image_coordinates(img, key="cl_sw")
            if v:
                p = (v["x"], v["y"])
                if len(st.session_state.calib_sw) < 2:
                    if not st.session_state.calib_sw or p != st.session_state.calib_sw[-1]:
                        st.session_state.calib_sw.append(p); st.rerun()
                else:
                    obj = st.session_state.sw_curves[act_sw]
                    if not obj.raw_pts or p != obj.raw_pts[-1]: obj.raw_pts.append(p)
        with c2:
            xm, ym = st.number_input("I(A) Max", 1000.0), st.number_input("E(mJ) Max", 125.0)
            if len(st.session_state.calib_sw) == 2:
                p0, pm = st.session_state.calib_sw
                sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
                obj = st.session_state.sw_curves[act_sw]
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                if st.button(f"🚀 擬合 {act_sw}"):
                    xd, yd = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, xd, yd)
                    obj.params = popt; st.success("擬合成功")

# B. IGBT 導通
st.divider()
st.header("2️⃣ IGBT 導通特性建模 (Method Con1)")
tm_i = st.radio("IGBT 溫度點", [f"{tc1}°C", f"{tc2}°C"], horizontal=True, key="tm_i")
up_i = st.file_uploader("上傳 IGBT v-i 圖", type=["png", "jpg"], key="up_i")
if up_i:
    img = Image.open(up_i)
    c1, c2 = st.columns([2, 1])
    with c1:
        v = streamlit_image_coordinates(img, key="cl_i")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_igbt) < 2:
                if not st.session_state.calib_igbt or p != st.session_state.calib_igbt[-1]:
                    st.session_state.calib_igbt.append(p); st.rerun()
            else:
                target = st.session_state.igbt_obj.raw_pts_tmin if tm_i.startswith(str(tc1)) else st.session_state.igbt_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("IGBT V Max", 5.0), st.number_input("IGBT I Max", 800.0)
        if len(st.session_state.calib_igbt) == 2:
            p0, pm = st.session_state.calib_igbt
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            is_min = tm_i.startswith(str(tc1))
            raw = st.session_state.igbt_obj.raw_pts_tmin if is_min else st.session_state.igbt_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if is_min: st.session_state.igbt_obj.real_pts_tmin = real
            else: st.session_state.igbt_obj.real_pts_tmax = real
            if st.button("🚀 擬合 IGBT"):
                cur = st.session_state.igbt_obj.real_pts_tmin if is_min else st.session_state.igbt_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if is_min: st.session_state.igbt_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.igbt_obj.fit_tmax = [z[1], z[0]]
                st.success(f"IGBT {tm_i} 擬合成功")

# C. FWD 導通
st.divider()
st.header("3️⃣ FWD 導通特性建模 (Method Con1)")
tm_f = st.radio("FWD 溫度點", [f"{tc1}°C", f"{tc2}°C"], horizontal=True, key="tm_f")
up_f = st.file_uploader("上傳 FWD v-i 圖", type=["png", "jpg"], key="up_f")
if up_f:
    img = Image.open(up_f)
    c1, c2 = st.columns([2, 1])
    with c1:
        v = streamlit_image_coordinates(img, key="cl_f")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_fwd) < 2:
                if not st.session_state.calib_fwd or p != st.session_state.calib_fwd[-1]:
                    st.session_state.calib_fwd.append(p); st.rerun()
            else:
                target = st.session_state.fwd_obj.raw_pts_tmin if tm_f.startswith(str(tc1)) else st.session_state.fwd_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("FWD V Max", 5.0, key="fx"), st.number_input("FWD I Max", 800.0, key="fy")
        if len(st.session_state.calib_fwd) == 2:
            p0, pm = st.session_state.calib_fwd
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            is_min = tm_f.startswith(str(tc1))
            raw = st.session_state.fwd_obj.raw_pts_tmin if is_min else st.session_state.fwd_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if is_min: st.session_state.fwd_obj.real_pts_tmin = real
            else: st.session_state.fwd_obj.real_pts_tmax = real
            if st.button("🚀 擬合 FWD"):
                cur = st.session_state.fwd_obj.real_pts_tmin if is_min else st.session_state.fwd_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if is_min: st.session_state.fwd_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.fwd_obj.fit_tmax = [z[1], z[0]]
                st.success(f"FWD {tm_f} 擬合成功")

# ==========================================
# 5. 結果視覺化 (嚴格修復版)
# ==========================================
st.divider()
st.header("📊 擬合結果視覺化")
res_c1, res_c2, res_c3 = st.columns(3)

with res_c1:
    st.subheader("切換能量擬合 (mJ)")
    fig_sw, ax_sw = plt.subplots()
    xi = np.linspace(0, 1000, 100)
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None: ax_sw.plot(xi, [o.get_val(x) for x in xi], label=n)
    ax_sw.set_xlabel("Current (A)"); ax_sw.legend(); st.pyplot(fig_sw)

with res_c2:
    st.subheader("IGBT Vce 擬合")
    fig_i, ax_i = plt.subplots()
    ii = np.linspace(0, 800, 100)
    for t, p in [(f"{tc1}°C", st.session_state.igbt_obj.fit_tmin), (f"{tc2}°C", st.session_state.igbt_obj.fit_tmax)]:
        if p is not None: ax_i.plot(p[0]+p[1]*ii, ii, label=t)
    ax_i.set_xlabel("Voltage (V)"); ax_i.set_ylabel("Current (A)"); ax_i.legend(); st.pyplot(fig_i)

with res_c3:
    st.subheader("FWD Vf 擬合")
    fig_f, ax_f = plt.subplots()
    for t, p in [(f"{tc1}°C", st.session_state.fwd_obj.fit_tmin), (f"{tc2}°C", st.session_state.fwd_obj.fit_tmax)]:
        if p is not None: ax_f.plot(p[0]+p[1]*ii, ii, label=t)
    ax_f.set_xlabel("Voltage (V)"); ax_f.set_ylabel("Current (A)"); ax_f.legend(); st.pyplot(fig_f)

# ==========================================
# 6. 全域損耗分析器 (含方程式說明)
# ==========================================
st.divider()
st.header("🔍 全域損耗分析器 (Equation 19 & SW3)")
if st.session_state.igbt_obj.fit_tmax and st.session_state.fwd_obj.fit_tmax:
    c1, c2, c3, c4 = st.columns(4)
    tj_c = c1.number_input("工作接面溫度 Tj (°C)", value=100.0)
    i_pk = c2.number_input("峰值電流 I_peak (A)", value=400.0)
    fsw  = c3.number_input("切換頻率 f_sw (Hz)", value=10000.0)
    
    # 電流自動解算
    irms = i_pk / 2
    iavg = i_pk / np.pi
    
    st.info("💡 **電流參數解算說明**：")
    st.latex(r"I_{rms} = \frac{I_{peak}}{2} = " + f"{irms:.2f} A")
    st.latex(r"I_{avg} = \frac{I_{peak}}{\pi} = " + f"{iavg:.2f} A")
    
    # 損耗計算
    pcon_igbt = st.session_state.igbt_obj.calc_pcon(tj_c, irms, iavg)
    pcon_fwd  = st.session_state.fwd_obj.calc_pcon(tj_c, irms, iavg)
    
    # 切換損積分計算
    psw_total = 0.0
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None:
            e_avg = np.mean([o.get_val(i_pk * np.sin(t)) for t in np.linspace(0, np.pi, 100)])
            psw_total += e_avg * fsw * 1e-3
            
    # 結果呈現與方程式說明
    r1, r2 = st.columns(2)
    with r1:
        st.metric("IGBT 導通損耗", f"{pcon_igbt:.2f} W")
        st.markdown("**方程式說明 (Eq 19)**:")
        st.latex(r"P_{con,IGBT} = R_{ce}(T_j) \cdot I_{rms}^2 + V_{ce}(T_j) \cdot I_{avg}")
        
        st.metric("FWD 導通損耗", f"{pcon_fwd:.2f} W")
        st.markdown("**方程式說明 (Eq 19)**:")
        st.latex(r"P_{con,FWD} = R_{f}(T_j) \cdot I_{rms}^2 + V_{f}(T_j) \cdot I_{avg}")
        
    with r2:
        st.metric("總切換損耗 (P_sw)", f"{psw_total:.2f} W")
        st.markdown("**方程式說明 (SW3 Integration)**:")
        st.latex(r"P_{sw} = \left( \frac{1}{\pi} \int_{0}^{\pi} E_{sw}(i(\theta)) d\theta \right) \cdot f_{sw}")
        st.caption("其中 $E_{sw}(i) = A + B \cdot i + C \cdot i^2$")
        
    st.success(f"### **總預估損耗 (P_total): {psw_total + pcon_igbt + pcon_fwd:.2f} W**")
else:
    st.info("💡 指示：完成導通特性擬合後，計算器將自動啟用。")