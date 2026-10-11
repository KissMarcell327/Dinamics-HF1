import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json

# --- 1. Oldalbeállítások ---
st.set_page_config(
    page_title="BME Dinamika 1. HF Kinematikai Megoldó",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Egyedi CSS stílus
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #566573;
        margin-bottom: 1.5rem;
    }
    .hwchk-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2980b9;
        margin-bottom: 1rem;
    }
    .step-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Kinematikai Számítási Függvény ---
def calculate_kinematics(phi_deg, l1, l2, l3, R, v_Cx):
    """
    BME Dinamika 1. HF kinematikai állapotának meghatározása.
    """
    phi = np.radians(phi_deg)
    
    # 1. Geometria
    # Origó: E érintkezési pontban (0, 0)
    # A középpont: (0, l3)
    r_E = np.array([0.0, 0.0])
    r_A = np.array([0.0, l3])
    
    # B pont helyvektora E-ből: r_EB = [l3*sin(phi), l3*(1 + cos(phi))]
    r_EB = np.array([l3 * np.sin(phi), l3 * (1.0 + np.cos(phi))])
    x_EB, y_EB = float(r_EB[0]), float(r_EB[1])
    
    # Geometriai ellenőrzések
    if np.abs(x_EB) < 1e-9:
        raise ValueError("A henger phi szöge esetén x_EB = 0, így az egyenletrendszer szingulárissá válik (phi != 0, 180 fok)!")
    
    sin_beta = y_EB / l2
    if np.abs(sin_beta) > 1.0:
        raise ValueError(f"A mechanizmus geometriailag nem záródik! sin(beta) = y_EB / l2 = {sin_beta:.4f} > 1. A rúd hossza túl kicsi.")
    
    beta = float(np.arcsin(sin_beta))
    cos_beta = float(np.sqrt(1.0 - sin_beta**2))
    
    # C pont az x-tengelyen mozog (y_C = 0)
    x_C = float(x_EB + l2 * cos_beta)
    r_C = np.array([x_C, 0.0])
    r_B = r_EB.copy()
    
    # S2 súlypont: a BC szakasz felezőpontja
    r_S2 = np.array([(x_EB + x_C) / 2.0, y_EB / 2.0])
    
    # 2. Sebességállapot
    # v_B a (3) felől: [-omega_3z * y_EB, omega_3z * x_EB]
    # v_B a (2) felől: [v_Cx - omega_2z * l2 * sin(beta), -omega_2z * l2 * cos(beta)]
    # Megoldás:
    denom_omega2 = (y_EB / x_EB) * l2 * cos_beta + l2 * sin_beta
    if np.abs(denom_omega2) < 1e-9:
        raise ValueError("A sebességállapot meghatározása szinguláris (nevező ~ 0)!")
    
    omega_2z = float(v_Cx / denom_omega2)
    omega_3z = float(-omega_2z * (l2 * cos_beta) / x_EB)
    
    v_C = np.array([v_Cx, 0.0])
    v_B = np.array([-omega_3z * y_EB, omega_3z * x_EB])
    
    # v_S2:
    v_S2 = np.array([v_Cx - omega_2z * (l2 / 2.0) * sin_beta, -omega_2z * (l2 / 2.0) * cos_beta])
    v_S2_mag = float(np.linalg.norm(v_S2))
    
    # 3. P sebességpólus
    if np.abs(omega_2z) < 1e-9:
        y_CP = np.nan
        r_P = np.array([x_C, np.nan])
    else:
        y_CP = float(v_Cx / omega_2z)
        r_P = np.array([x_C, y_CP])
        
    # 4. Gyorsulásállapot
    # C pont állandó sebességű: a_C = [0, 0] -> G = C
    a_C = np.array([0.0, 0.0])
    
    # E pont gyorsulása R sugarú domború körpályán tiszta gördüléskor:
    a_E = float((R * l3 * (omega_3z**2)) / (R + l3))
    
    # 2x2 lineáris egyenletrendszer epsilon_2z-re és epsilon_3z-re:
    # [ y_EB,             -y_EB ] [eps_2z] = [ omega_3z^2 * x_EB + omega_2z^2 * l2 * cos(beta) ]
    # [ l2 * cos(beta),    x_EB ] [eps_3z]   [ (omega_3z^2 - omega_2z^2) * y_EB - a_E          ]
    M = np.array([
        [y_EB, -y_EB],
        [l2 * cos_beta, x_EB]
    ])
    rhs = np.array([
        (omega_3z**2) * x_EB + (omega_2z**2) * l2 * cos_beta,
        (omega_3z**2 - omega_2z**2) * y_EB - a_E
    ])
    
    sol = np.linalg.solve(M, rhs)
    eps_2z = float(sol[0])
    eps_3z = float(sol[1])
    
    # a_B vektor:
    a_B = np.array([
        -eps_3z * y_EB - (omega_3z**2) * x_EB,
        a_E + eps_3z * x_EB - (omega_3z**2) * y_EB
    ])
    
    # a_S2 = a_B / 2:
    a_S2 = a_B / 2.0
    a_S2_mag = float(np.linalg.norm(a_S2))
    
    # Gyorsulásszög: tan(alpha_2) = eps_2z / omega_2z^2
    tan_alpha2 = abs(eps_2z) / (omega_2z**2) if omega_2z != 0 else np.nan
    alpha_2_deg = float(np.degrees(np.arctan(tan_alpha2)))
    
    # 5. Tangenciális és normális gyorsulás, görbületi sugár:
    if v_S2_mag > 1e-12:
        e_t = v_S2 / v_S2_mag
        a_S2t_signed = float(np.dot(a_S2, e_t))
        a_S2t_abs = float(np.abs(a_S2t_signed))
        a_S2n = float(np.sqrt(max(0.0, a_S2_mag**2 - a_S2t_signed**2)))
        rho_S2 = float((v_S2_mag**2) / a_S2n) if a_S2n > 1e-12 else float('inf')
        vec_a_S2t = a_S2t_signed * e_t
        vec_a_S2n = a_S2 - vec_a_S2t
    else:
        e_t = np.array([0.0, 0.0])
        a_S2t_signed = 0.0
        a_S2t_abs = 0.0
        a_S2n = a_S2_mag
        rho_S2 = 0.0
        vec_a_S2t = np.array([0.0, 0.0])
        vec_a_S2n = a_S2.copy()
        
    return {
        'phi_deg': phi_deg, 'phi': phi, 'l1': l1, 'l2': l2, 'l3': l3, 'R': R, 'v_Cx': v_Cx,
        'r_E': r_E, 'r_A': r_A, 'r_EB': r_EB, 'r_B': r_B, 'r_C': r_C, 'r_S2': r_S2, 'r_P': r_P,
        'sin_beta': sin_beta, 'cos_beta': cos_beta, 'beta_deg': float(np.degrees(beta)), 'beta': beta,
        'omega_2z': omega_2z, 'omega_3z': omega_3z,
        'v_C': v_C, 'v_B': v_B, 'v_S2': v_S2, 'v_S2_mag': v_S2_mag,
        'a_C': a_C, 'a_E': a_E, 'eps_2z': eps_2z, 'eps_3z': eps_3z,
        'a_B': a_B, 'a_S2': a_S2, 'a_S2_mag': a_S2_mag,
        'tan_alpha2': tan_alpha2, 'alpha_2_deg': alpha_2_deg,
        'e_t': e_t, 'a_S2t_signed': a_S2t_signed, 'a_S2t_abs': a_S2t_abs,
        'a_S2n': a_S2n, 'rho_S2': rho_S2,
        'vec_a_S2t': vec_a_S2t, 'vec_a_S2n': vec_a_S2n,
        'M': M, 'rhs': rhs
    }

# --- 3. Műszaki Ábra Generáló Függvény ---
def generate_mechanism_figure(res, v_scale=0.06, a_scale=0.015, show_construction=True, show_labels=True, dark_mode=False):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(11, 8.2), dpi=150)
    
    bg_color = '#ffffff' if not dark_mode else '#1e1e1e'
    text_color = '#2c3e50' if not dark_mode else '#ecf0f1'
    grid_color = '#d5dbdb' if not dark_mode else '#444444'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    l1 = res['l1']
    l2 = res['l2']
    l3 = res['l3']
    R = res['R']
    r_A = res['r_A']
    r_B = res['r_B']
    r_C = res['r_C']
    r_S2 = res['r_S2']
    r_P = res['r_P']
    
    # 1. Domború pálya (középpont: (0, -R), sugár: R)
    theta_span = np.arcsin(min(0.85, 0.15 / R))
    thetas = np.linspace(np.pi/2 - theta_span, np.pi/2 + theta_span, 120)
    x_track = R * np.cos(thetas)
    y_track = -R + R * np.sin(thetas)
    ax.plot(x_track, y_track, color='#34495e', linewidth=3.2, zorder=2, label='Fix domború körpálya (R)')
    
    # Körpálya befalazás / rögzítés vonalkázása
    n_hatch = 18
    for th in np.linspace(np.pi/2 - theta_span*0.92, np.pi/2 + theta_span*0.92, n_hatch):
        xh = R * np.cos(th)
        yh = -R + R * np.sin(th)
        dxh = -0.016 * np.cos(th - 0.45)
        dyh = -0.016 * np.sin(th - 0.45)
        ax.plot([xh, xh + dxh], [yh, yh + dyh], color='#7f8c8d', lw=1.2, zorder=1)
        
    # 2. Vízszintes pálya az (1) henger alatt (y = -l1)
    x_ground_min = r_C[0] - 1.6 * l1
    x_ground_max = r_C[0] + 2.0 * l1
    ax.plot([x_ground_min, x_ground_max], [-l1, -l1], color='#34495e', lw=2.8, zorder=2, label='Vízszintes talaj / vezeték')
    for xg in np.arange(x_ground_min, x_ground_max, 0.018):
        ax.plot([xg, xg - 0.009], [-l1, -l1 - 0.009], color='#7f8c8d', lw=1.2, zorder=1)
        
    # Koordinátatengelyek / középvonalak
    ax.axhline(0, color=grid_color, linestyle='-.', lw=1.1, alpha=0.8)
    ax.axvline(0, color=grid_color, linestyle='-.', lw=1.1, alpha=0.8)
    
    # 3. (3) henger (középpont A=(0, l3), sugár l3)
    cylinder3 = patches.Circle((0, l3), l3, edgecolor='#2980b9', facecolor='#ebf5fb', lw=2.4, zorder=3, label='(3) henger')
    ax.add_patch(cylinder3)
    
    # 4. (1) henger (középpont C=(x_C, 0), sugár l1)
    cylinder1 = patches.Circle((r_C[0], 0), l1, edgecolor='#27ae60', facecolor='#eafaf1', lw=2.4, zorder=3, label='(1) henger')
    ax.add_patch(cylinder1)
    
    # 5. (2) rúd (B és C között)
    ax.plot([r_B[0], r_C[0]], [r_B[1], r_C[1]], color='#d35400', lw=5.5, solid_capstyle='round', zorder=4, label='(2) rúd')
    
    # 6. Szerkesztővonalak a P sebességpólushoz
    if show_construction and not np.isnan(r_P[1]):
        # E - B - P egyenes
        p_dir = r_P - np.array([0.0, 0.0])
        p_len = np.linalg.norm(p_dir)
        p_unit = p_dir / p_len if p_len > 0 else np.array([0, 1])
        ax.plot([0, r_P[0] + 0.035 * p_unit[0]], [0, r_P[1] + 0.035 * p_unit[1]], 
                color='#8e44ad', linestyle='--', lw=1.6, zorder=2, label='E-B-P szerkesztővonal')
        
        # C - P egyenes (függőleges C-ből P-be)
        ax.plot([r_C[0], r_P[0]], [r_C[1], r_P[1] + 0.02], 
                color='#8e44ad', linestyle=':', lw=2.0, zorder=2, label='C-P szerkesztővonal (merőleges v_C-re)')
        
    # 7. Pontok jelölése
    points = [
        ('E', np.array([0.0, 0.0]), '#c0392b', (-0.014, -0.016), 'o', 60),
        ('A', r_A, '#2980b9', (-0.022, 0.002), 'o', 50),
        ('B', r_B, '#e67e22', (0.006, 0.008), 'o', 65),
        ('C (G)', r_C, '#27ae60', (0.008, -0.018), 'o', 65),
        ('S_2', r_S2, '#d35400', (0.006, 0.010), 'D', 75),
    ]
    if not np.isnan(r_P[1]):
        points.append(('P', r_P, '#8e44ad', (0.008, 0.008), '*', 110))
        
    for name, pt, col, offset, marker, size in points:
        ax.scatter(pt[0], pt[1], color=col, s=size, marker=marker, zorder=6, edgecolor='black', linewidth=0.6)
        if show_labels:
            label_text = f"${name}$" if not '(' in name else name
            ax.annotate(label_text, 
                        xy=(pt[0], pt[1]), 
                        xytext=(pt[0] + offset[0], pt[1] + offset[1]),
                        fontsize=11.5, fontweight='bold', color=col, zorder=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    # 8. Vektorok rajzolása
    def draw_vector(start, vec, scale, color, label, lw=2.2):
        end = start + vec * scale
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15),
                    zorder=8)
        if show_labels and label:
            ax.text(end[0] + 0.003, end[1] + 0.003, label, color=color, fontsize=11, fontweight='bold', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

    # Sebességek: v_C, v_B, v_S2 (Kék árnyalatok)
    draw_vector(r_C, res['v_C'], v_scale, '#2980b9', r'$\mathbf{v}_C$')
    draw_vector(r_B, res['v_B'], v_scale, '#2980b9', r'$\mathbf{v}_B$')
    draw_vector(r_S2, res['v_S2'], v_scale, '#1b4f72', r'$\mathbf{v}_{S2}$')
    
    # Gyorsulások: a_B, a_S2 (Piros árnyalatok)
    draw_vector(r_B, res['a_B'], a_scale, '#c0392b', r'$\mathbf{a}_B$')
    draw_vector(r_S2, res['a_S2'], a_scale, '#e74c3c', r'$\mathbf{a}_{S2}$')
    
    # S2 gyorsuláskomponensek: a_S2t, a_S2n (Lila árnyalatok)
    draw_vector(r_S2, res['vec_a_S2t'], a_scale, '#9b59b6', r'$\mathbf{a}_{S2t}$', lw=1.8)
    draw_vector(r_S2, res['vec_a_S2n'], a_scale, '#8e44ad', r'$\mathbf{a}_{S2n}$', lw=1.8)

    ax.set_aspect('equal', 'datalim')
    ax.grid(True, linestyle='--', alpha=0.5, color=grid_color)
    ax.set_title(r'BME Dinamika 1. HF – Síkbeli mechanizmus kinematikai modellje', fontsize=13.5, pad=14, color=text_color)
    ax.set_xlabel('x tengely [m]', fontsize=10.5, color=text_color)
    ax.set_ylabel('y tengely [m]', fontsize=10.5, color=text_color)
    
    # Skáladoboz
    scale_text = f"Vektorskálák:\nSebesség: 1 m/s $\\approx$ {v_scale*1000:.0f} mm\nGyorsulás: 1 m/s² $\\approx$ {a_scale*1000:.0f} mm"
    ax.text(0.02, 0.98, scale_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9f9', edgecolor='#bdc3c7', alpha=0.9))
            
    # Jelmagyarázat
    custom_legend = [
        plt.Line2D([0], [0], color='#2980b9', lw=2.2, marker='>', label=r'Sebességvektorok ($\mathbf{v}$)'),
        plt.Line2D([0], [0], color='#c0392b', lw=2.2, marker='>', label=r'Gyorsulásvektorok ($\mathbf{a}$)'),
        plt.Line2D([0], [0], color='#8e44ad', lw=2.0, marker='>', label=r'Komponensek ($\mathbf{a}_{S2t}, \mathbf{a}_{S2n}$)'),
        plt.Line2D([0], [0], color='#8e44ad', lw=1.5, linestyle='--', label=r'Pólus szerkesztővonalak')
    ]
    ax.legend(handles=custom_legend, loc='lower right', framealpha=0.92, fontsize=9.5)
    
    ax.margins(0.12)
    fig.tight_layout()
    return fig

# --- 4. Streamlit Oldalsáv: Bemeneti Paraméterek ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/BME_logo.svg/320px-BME_logo.svg.png", width=140)
st.sidebar.title("Paraméterek (Bemenet)")

# Reset gomb
if st.sidebar.button("🔄 Alapértékek visszaállítása"):
    st.session_state['phi'] = 55.0
    st.session_state['l1'] = 0.06
    st.session_state['l2'] = 0.21
    st.session_state['l3'] = 0.05
    st.session_state['R'] = 0.15
    st.session_state['v_Cx'] = 0.96
    st.session_state['auto_R'] = True

if 'auto_R' not in st.session_state:
    st.session_state['auto_R'] = True

phi_deg = st.sidebar.slider(
    "phi [fok] – (3) henger szögelfordulása",
    min_value=10.0, max_value=85.0, value=float(st.session_state.get('phi', 55.0)), step=0.5
)

l3 = st.sidebar.number_input(
    "l3 [m] – a (3) henger sugara",
    min_value=0.01, max_value=0.50, value=float(st.session_state.get('l3', 0.05)), step=0.01, format="%.4f"
)

auto_R = st.sidebar.checkbox("R automatikusan 3 * l3", value=st.session_state.get('auto_R', True))
st.session_state['auto_R'] = auto_R

if auto_R:
    R = 3.0 * l3
    st.sidebar.text(f"R [m] = {R:.4f} (3 * l3)")
else:
    R = st.sidebar.number_input(
        "R [m] – fix domború pálya sugara",
        min_value=0.02, max_value=1.00, value=float(st.session_state.get('R', 0.15)), step=0.01, format="%.4f"
    )

l2 = st.sidebar.number_input(
    "l2 [m] – a (2) rúd hossza",
    min_value=0.05, max_value=1.00, value=float(st.session_state.get('l2', 0.21)), step=0.01, format="%.4f"
)

l1 = st.sidebar.number_input(
    "l1 [m] – az (1) henger sugara",
    min_value=0.01, max_value=0.50, value=float(st.session_state.get('l1', 0.06)), step=0.01, format="%.4f"
)

v_Cx = st.sidebar.number_input(
    "v_Cx [m/s] – C pont vízszintes sebessége",
    min_value=0.01, max_value=10.0, value=float(st.session_state.get('v_Cx', 0.96)), step=0.05, format="%.4f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Ábrázolási beállítások")
v_scale = st.sidebar.slider("Sebességvektor skála [m/(m/s)]", min_value=0.01, max_value=0.15, value=0.06, step=0.005)
a_scale = st.sidebar.slider("Gyorsulásvektor skála [m/(m/s²)]", min_value=0.005, max_value=0.05, value=0.015, step=0.002)
show_const = st.sidebar.checkbox("Pólus szerkesztővonalak megjelenítése", value=True)
show_labels = st.sidebar.checkbox("Vektor- és pontfeliratok", value=True)
precision = st.sidebar.selectbox("Tizedesjegyek száma az eredményeknél", [4, 6, 8], index=1)

# --- 5. Főoldal: Cím és Számítás Futtatása ---
st.markdown('<div class="main-header">BME Dinamika 1. Házi feladat – Kinematikai Megoldó</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Síkbeli összetett mechanizmus kinematikája (BME Műszaki Mechanikai Tanszék, BMEGEMMBXM3) '
    '– Automatikus megoldás és ellenőrzés tetszőleges paraméterekkel.</div>',
    unsafe_allow_html=True
)

calc_error = None
try:
    res = calculate_kinematics(phi_deg, l1, l2, l3, R, v_Cx)
except Exception as e:
    calc_error = str(e)

if calc_error:
    st.error(f"⚠️ Hiba a számítás során: {calc_error}")
    st.info("Kérjük, ellenőrizd az oldalsávban megadott geometriai paramétereket!")
    st.stop()

# --- Gyors Áttekintő Kártyák (KPI Metrics) ---
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("ω₂z [rad/s]", f"{res['omega_2z']:.{precision}f}")
c2.metric("ε₂z [rad/s²]", f"{res['eps_2z']:.{precision}f}")
c3.metric("ω₃z [rad/s]", f"{res['omega_3z']:.{precision}f}")
c4.metric("ε₃z [rad/s²]", f"{res['eps_3z']:.{precision}f}")
c5.metric("v_S2 [m/s]", f"{res['v_S2_mag']:.{precision}f}")
c6.metric("a_S2 [m/s²]", f"{res['a_S2_mag']:.{precision}f}")

# --- Fő Tartalmi Fülek (Tabs) ---
tab_hwchk, tab_plot, tab_steps, tab_sim, tab_latex = st.tabs([
    "📋 1. hwchk Eredménytáblázat",
    "🎨 2. Méretarányos Műszaki Ábra",
    "📝 3. Lépésről lépésre Kidolgozás",
    "🔄 4. Mozgástartomány & Elemzés",
    "📄 5. Letölthető LaTeX / Jegyzőkönyv"
])

# =========================================================================
# 1. FÜL: hwchk Eredménytáblázat
# =========================================================================
with tab_hwchk:
    st.subheader("BME MM HF Ellenőrző (hwchk) hivatalos mezőértékek")
    st.markdown("""
    Az alábbi táblázat pontosan a tanszéki **hwchk** automatikus ellenőrző rendszer mezőneveit tartalmazza.
    A mezőértékek az alatta lévő gombbal vagy szövegmezőből közvetlenül kimásolhatók.
    """)
    
    hwchk_data = [
        {"hwchk Mezőnév": "omega_2z", "Mennyiség leírása": "A (2) rúd szögsebessége", "Érték": f"{res['omega_2z']:.{precision}f}", "Mértékegység": "rad/s"},
        {"hwchk Mezőnév": "epsilon_2z", "Mennyiség leírása": "A (2) rúd szöggyorsulása", "Érték": f"{res['eps_2z']:.{precision}f}", "Mértékegység": "rad/s^2"},
        {"hwchk Mezőnév": "omega_3z", "Mennyiség leírása": "A (3) henger szögsebessége", "Érték": f"{res['omega_3z']:.{precision}f}", "Mértékegység": "rad/s"},
        {"hwchk Mezőnév": "epsilon_3z", "Mennyiség leírása": "A (3) henger szöggyorsulása", "Érték": f"{res['eps_3z']:.{precision}f}", "Mértékegység": "rad/s^2"},
        {"hwchk Mezőnév": "v_S2", "Mennyiség leírása": "A rúd S2 súlypontjának sebességnagysága", "Érték": f"{res['v_S2_mag']:.{precision}f}", "Mértékegység": "m/s"},
        {"hwchk Mezőnév": "a_S2", "Mennyiség leírása": "A rúd S2 súlypontjának gyorsulásnagysága", "Érték": f"{res['a_S2_mag']:.{precision}f}", "Mértékegység": "m/s^2"},
        {"hwchk Mezőnév": "|alpha_2|", "Mennyiség leírása": "A (2) rúd gyorsulásszöge", "Érték": f"{res['alpha_2_deg']:.{precision}f}", "Mértékegység": "fok"},
        {"hwchk Mezőnév": "|a_S2t|", "Mennyiség leírása": "S2 tangenciális (pályamenti) gyorsulása", "Érték": f"{res['a_S2t_abs']:.{precision}f}", "Mértékegység": "m/s^2"},
        {"hwchk Mezőnév": "|a_S2n|", "Mennyiség leírása": "S2 normális (centripetális) gyorsulása", "Érték": f"{res['a_S2n']:.{precision}f}", "Mértékegység": "m/s^2"},
        {"hwchk Mezőnév": "rho_S2", "Mennyiség leírása": "S2 pálya görbületi sugara", "Érték": f"{res['rho_S2']:.{precision}f}", "Mértékegység": "m"},
    ]
    df_hwchk = pd.DataFrame(hwchk_data)
    
    st.dataframe(
        df_hwchk, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "hwchk Mezőnév": st.column_config.TextColumn("hwchk Mezőnév", width="medium"),
            "Mennyiség leírása": st.column_config.TextColumn("Leírás", width="large"),
            "Érték": st.column_config.TextColumn("Számított érték", width="medium"),
            "Mértékegység": st.column_config.TextColumn("Mértékegység", width="small"),
        }
    )
    
    col_copy1, col_copy2 = st.columns(2)
    with col_copy1:
        st.markdown("##### 📌 Kimásolható mezőlista (Kulcs = Érték):")
        hwchk_raw_text = "\n".join([f"{row['hwchk Mezőnév']} = {row['Érték']}" for row in hwchk_data])
        st.text_area("hwchk formátum", hwchk_raw_text, height=220)
        
    with col_copy2:
        st.markdown("##### 📦 JSON Export / Beillesztés:")
        hwchk_dict = {row['hwchk Mezőnév']: float(row['Érték']) for row in hwchk_data}
        st.json(hwchk_dict)
        
    # Letöltési gombok
    csv_buffer = io.StringIO()
    df_hwchk.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Eredmények letöltése CSV formátumban",
        data=csv_buffer.getvalue(),
        file_name=f"BME_Dinamika_1HF_phi{phi_deg:.1f}.csv",
        mime="text/csv"
    )

# =========================================================================
# 2. FÜL: Méretarányos Műszaki Ábra
# =========================================================================
with tab_plot:
    st.subheader("Méretarányos műszaki ábra (Matplotlib)")
    st.markdown(r"""
    A szerkezet valós méretarányos geometriája a sebesség- és gyorsulásállapot vektoros megjelenítésével.
    - 🔵 **Kék nyilak:** $\mathbf{v}_C, \mathbf{v}_B, \mathbf{v}_{S2}$ sebességvektorok
    - 🔴 **Piros nyilak:** $\mathbf{a}_B, \mathbf{a}_{S2}$ gyorsulásvektorok
    - 🟣 **Lila nyilak:** $\mathbf{a}_{S2t}, \mathbf{a}_{S2n}$ érintő- és normális gyorsuláskomponensek
    - 🟣 **Szaggatott vonalak:** Sebességpólus ($P$) szerkesztése ($E-B-P$ és $C-P$ merőleges)
    """)
    
    fig = generate_mechanism_figure(res, v_scale, a_scale, show_const, show_labels)
    st.pyplot(fig, use_container_width=True)
    
    # PNG letöltés gomb
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    img_buf.seek(0)
    st.download_button(
        label="💾 Ábra letöltése nagy felbontású PNG képként (300 DPI)",
        data=img_buf,
        file_name=f"kinematika_abra_phi_{phi_deg:.1f}.png",
        mime="image/png"
    )

# =========================================================================
# 3. FÜL: Lépésről lépésre Kidolgozás (LaTeX)
# =========================================================================
with tab_steps:
    st.subheader("Részletes, lépésről lépésre levezetett feladatmegoldás")
    
    # 1. Lépés: Geometria
    with st.expander("1. Geometriai állapot meghatározása", expanded=True):
        st.markdown(rf"""
**Derékszögű koordináta-rendszer rögzítése:**
Origó $E$ az álló domború pálya és a (3) gördülő henger érintkezési pontjában: $\mathbf{{r}}_E = \begin{{bmatrix}} 0 \\ 0 \end{{bmatrix}}\,\text{{m}}$.
A (3) henger középpontja: $A = \begin{{bmatrix}} 0 \\ l_3 \end{{bmatrix}} = \begin{{bmatrix}} 0 \\ {res['l3']:.4f} \end{{bmatrix}}\,\text{{m}}$.

A (3) henger $B$ kerületi csuklópontjának helyvektora $E$-ből a megadott $\phi = {phi_deg:.1f}^\circ$ szögelfordulással:
$$
\mathbf{{r}}_{{EB}} = \begin{{bmatrix}} l_3 \sin\phi \\ l_3 (1 + \cos\phi) \end{{bmatrix}} = 
\begin{{bmatrix}} {res['l3']:.4f} \cdot \sin({phi_deg:.1f}^\circ) \\ {res['l3']:.4f} \cdot (1 + \cos({phi_deg:.1f}^\circ)) \end{{bmatrix}} =
\begin{{bmatrix}} {res['r_EB'][0]:.{precision}f} \\ {res['r_EB'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m}}
$$

A $C$ pont a vízszintes $x$-tengely mentén mozog, így $y_C = 0$.
A (2) rúd hossza $l_2 = {res['l2']:.4f}\,\text{{m}}$, vízszintessel bezárt szöge $\beta$:
$$
\sin\beta = \frac{{y_B - y_C}}{{l_2}} = \frac{{y_{{EB}}}}{{l_2}} = \frac{{{res['r_EB'][1]:.{precision}f}}}{{{res['l2']:.4f}}} = {res['sin_beta']:.{precision}f}
$$
Ebből a szög és annak koszinusza:
$$
\beta = \arcsin({res['sin_beta']:.{precision}f}) = {res['beta_deg']:.{precision}f}^\circ \quad ({res['beta']:.{precision}f}\,\text{{rad}})
$$
$$
\cos\beta = \sqrt{{1 - \sin^2\beta}} = {res['cos_beta']:.{precision}f}
$$

A $C$ pont koordinátája:
$$
x_C = x_B + l_2 \cos\beta = {res['r_EB'][0]:.{precision}f} + {res['l2']:.4f} \cdot {res['cos_beta']:.{precision}f} = {res['r_C'][0]:.{precision}f}\,\text{{m}}
$$
$$
\mathbf{{r}}_C = \begin{{bmatrix}} {res['r_C'][0]:.{precision}f} \\ 0 \end{{bmatrix}}\,\text{{m}}
$$

A (2) rúd $S_2$ súlypontja (felezőpontja):
$$
\mathbf{{r}}_{{S2}} = \frac{{\mathbf{{r}}_B + \mathbf{{r}}_C}}{{2}} = 
\begin{{bmatrix}} \frac{{{res['r_EB'][0]:.{precision}f} + {res['r_C'][0]:.{precision}f}}}{{2}} \\ \frac{{{res['r_EB'][1]:.{precision}f} + 0}}{{2}} \end{{bmatrix}} =
\begin{{bmatrix}} {res['r_S2'][0]:.{precision}f} \\ {res['r_S2'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m}}
$$
""")

    # 2. Lépés: Sebességállapot
    with st.expander("2. Sebességállapot és szögsebességek meghatározása", expanded=True):
        st.markdown(rf"""
A (3) henger tiszta gördülést végez a domború pályán az $E$ pontban, így a pillanatnyi sebessége zérus:
$$
\mathbf{{v}}_E = \mathbf{{0}}
$$

A merev test sebességeloszlási törvénye alapján a $B$ pont sebessége a (3) henger felől:
$$
\mathbf{{v}}_B = \mathbf{{v}}_E + \boldsymbol{{\omega}}_3 \times \mathbf{{r}}_{{EB}} =
\begin{{bmatrix}} 0 \\ 0 \\ \omega_{{3z}} \end{{bmatrix}} \times \begin{{bmatrix}} x_{{EB}} \\ y_{{EB}} \\ 0 \end{{bmatrix}} =
\begin{{bmatrix}} -\omega_{{3z}} y_{{EB}} \\ \omega_{{3z}} x_{{EB}} \end{{bmatrix}}
$$

Ugyanakkor a $B$ pont a (2) rúd része is, így a $C$ pont sebességéből kifejezve:
$$
\mathbf{{v}}_C = \begin{{bmatrix}} v_{{Cx}} \\ 0 \end{{bmatrix}} = \begin{{bmatrix}} {res['v_Cx']:.4f} \\ 0 \end{{bmatrix}}\,\text{{m/s}}
$$
$$
\mathbf{{r}}_{{CB}} = \mathbf{{r}}_B - \mathbf{{r}}_C = \begin{{bmatrix}} -l_2 \cos\beta \\ l_2 \sin\beta \end{{bmatrix}} = \begin{{bmatrix}} -l_2 \cos\beta \\ y_{{EB}} \end{{bmatrix}}
$$
$$
\mathbf{{v}}_B = \mathbf{{v}}_C + \boldsymbol{{\omega}}_2 \times \mathbf{{r}}_{{CB}} =
\begin{{bmatrix}} v_{{Cx}} \\ 0 \end{{bmatrix}} + \begin{{bmatrix}} 0 \\ 0 \\ \omega_{{2z}} \end{{bmatrix}} \times \begin{{bmatrix}} -l_2 \cos\beta \\ l_2 \sin\beta \\ 0 \end{{bmatrix}} =
\begin{{bmatrix}} v_{{Cx}} - \omega_{{2z}} l_2 \sin\beta \\ -\omega_{{2z}} l_2 \cos\beta \end{{bmatrix}}
$$

A $B$ pont sebességének kétféle felírása megegyezik ($\mathbf{{v}}_B^{{(3)}} = \mathbf{{v}}_B^{{(2)}}$):
$$
\begin{{cases}}
-\omega_{{3z}} y_{{EB}} = v_{{Cx}} - \omega_{{2z}} l_2 \sin\beta & (x\text{{-irány}}) \\
\omega_{{3z}} x_{{EB}} = -\omega_{{2z}} l_2 \cos\beta & (y\text{{-irány}})
\end{{cases}}
$$

A második egyenletből kifejezve $\omega_{{3z}}$-t:
$$
\omega_{{3z}} = -\omega_{{2z}} \frac{{l_2 \cos\beta}}{{x_{{EB}}}}
$$
Behelyettesítve az első egyenletbe:
$$
-\left(-\omega_{{2z}} \frac{{l_2 \cos\beta}}{{x_{{EB}}}}\right) y_{{EB}} + \omega_{{2z}} l_2 \sin\beta = v_{{Cx}}
$$
$$
\omega_{{2z}} \left[ \frac{{y_{{EB}}}}{{x_{{EB}}}} l_2 \cos\beta + l_2 \sin\beta \right] = v_{{Cx}}
$$
$$
\omega_{{2z}} = \frac{{v_{{Cx}}}}{{\frac{{y_{{EB}}}}{{x_{{EB}}}} l_2 \cos\beta + l_2 \sin\beta}} =
\frac{{{res['v_Cx']:.4f}}}{{\frac{{{res['r_EB'][1]:.{precision}f}}}{{{res['r_EB'][0]:.{precision}f}}} \cdot {res['l2']:.4f} \cdot {res['cos_beta']:.{precision}f} + {res['l2']:.4f} \cdot {res['sin_beta']:.{precision}f}}} =
\mathbf{{{res['omega_2z']:.{precision}f}}}\,\text{{rad/s}}
$$

Ebből a (3) henger szögsebessége:
$$
\omega_{{3z}} = -({res['omega_2z']:.{precision}f}) \cdot \frac{{{res['l2']:.4f} \cdot {res['cos_beta']:.{precision}f}}}{{{res['r_EB'][0]:.{precision}f}}} =
\mathbf{{{res['omega_3z']:.{precision}f}}}\,\text{{rad/s}}
$$

A $B$ és $S_2$ pontok sebességvektorai:
$$
\mathbf{{v}}_B = \begin{{bmatrix}} {res['v_B'][0]:.{precision}f} \\ {res['v_B'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m/s}}
$$
$$
\mathbf{{v}}_{{S2}} = \mathbf{{v}}_C + \boldsymbol{{\omega}}_2 \times \mathbf{{r}}_{{CS2}} =
\begin{{bmatrix}} v_{{Cx}} - \omega_{{2z}} \frac{{l_2}}{{2}} \sin\beta \\ -\omega_{{2z}} \frac{{l_2}}{{2}} \cos\beta \end{{bmatrix}} =
\begin{{bmatrix}} {res['v_S2'][0]:.{precision}f} \\ {res['v_S2'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m/s}}
$$
$$
v_{{S2}} = |\mathbf{{v}}_{{S2}}| = \sqrt{{v_{{S2x}}^2 + v_{{S2y}}^2}} = 
\sqrt{{({res['v_S2'][0]:.{precision}f})^2 + ({res['v_S2'][1]:.{precision}f})^2}} =
\mathbf{{{res['v_S2_mag']:.{precision}f}}}\,\text{{m/s}}
$$
""")

    # 3. Lépés: Sebességpólus
    with st.expander("3. Sebességpólus (P) szerkesztése és koordinátái", expanded=True):
        st.markdown(rf"""
A (2) rúd $P$ sebességpólusa az a pont, amelynek pillanatnyi sebessége zérus: $\mathbf{{v}}_P = \mathbf{{0}}$.

**Geometriai szerkesztés:**
- Mivel $\mathbf{{v}}_C$ vízszintes, a rá merőleges egyenes a $C$ ponton átmenő **függőleges egyenes**: $x = x_C$.
- Mivel a (3) henger az $E$ pont körül fordul el pillanatnyilag, a $B$ pont sebessége merőleges az $EB$ szakaszra ($\mathbf{{v}}_B \perp EB$).
  Ezért a $\mathbf{{v}}_B$-re merőleges egyenes éppen az **$E$ és $B$ pontokon átmenő egyenes**.
- A két merőleges egyenes metszéspontja adja a $P$ pólust.

**Analitikus számítás:**
$$
\mathbf{{v}}_P = \mathbf{{v}}_C + \boldsymbol{{\omega}}_2 \times \mathbf{{r}}_{{CP}} = 
\begin{{bmatrix}} v_{{Cx}} \\ 0 \end{{bmatrix}} + \begin{{bmatrix}} -\omega_{{2z}} y_{{CP}} \\ \omega_{{2z}} x_{{CP}} \end{{bmatrix}} = \mathbf{{0}}
$$
Ebből:
$$
x_{{CP}} = 0 \implies x_P = x_C = {res['r_C'][0]:.{precision}f}\,\text{{m}}
$$
$$
y_{{CP}} = \frac{{v_{{Cx}}}}{{\omega_{{2z}}}} = \frac{{{res['v_Cx']:.4f}}}{{{res['omega_2z']:.{precision}f}}} = {res['r_P'][1]:.{precision}f}\,\text{{m}}
$$
A pólus helyvektora:
$$
\mathbf{{r}}_P = \begin{{bmatrix}} {res['r_P'][0]:.{precision}f} \\ {res['r_P'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m}}
$$
""")

    # 4. Lépés: Gyorsulásállapot
    with st.expander("4. Gyorsulásállapot és szöggyorsulások meghatározása", expanded=True):
        st.markdown(rf"""
**Kényszerfeltételek a gyorsulásokra:**
- A $C$ pont sebessége állandó és egyenes vonalú: $\mathbf{{a}}_C = \mathbf{{0}}$. Ebből következően a rúd gyorsuláspólusa: $G \equiv C$.
- Az $E$ pont a (3) henger tiszta gördülési pontja egy fix, $R = {res['R']:.4f}\,\text{{m}}$ sugarú **domború** felületen.
  A gördülő pont gyorsulása normális irányú (merőleges a felületre, azaz függőlegesen felfelé mutat):
  $$
  a_E = \frac{{R \cdot l_3 \cdot \omega_{{3z}}^2}}{{R + l_3}} =
  \frac{{{res['R']:.4f} \cdot {res['l3']:.4f} \cdot ({res['omega_3z']:.{precision}f})^2}}{{{res['R']:.4f} + {res['l3']:.4f}}} =
  \mathbf{{{res['a_E']:.{precision}f}}}\,\text{{m/s}}^2
  $$
  $$
  \mathbf{{a}}_E = \begin{{bmatrix}} 0 \\ a_E \end{{bmatrix}} = \begin{{bmatrix}} 0 \\ {res['a_E']:.{precision}f} \end{{bmatrix}}\,\text{{m/s}}^2
  $$

**A $B$ pont gyorsulásának felírása:**
1. A (3) henger felől:
$$
\mathbf{{a}}_B = \mathbf{{a}}_E + \boldsymbol{{\varepsilon}}_3 \times \mathbf{{r}}_{{EB}} - \omega_{{3z}}^2 \mathbf{{r}}_{{EB}} =
\begin{{bmatrix}} 0 \\ a_E \end{{bmatrix}} + \begin{{bmatrix}} -\varepsilon_{{3z}} y_{{EB}} \\ \varepsilon_{{3z}} x_{{EB}} \end{{bmatrix}} - \omega_{{3z}}^2 \begin{{bmatrix}} x_{{EB}} \\ y_{{EB}} \end{{bmatrix}}
$$
$$
\mathbf{{a}}_B^{{(3)}} = \begin{{bmatrix}} -\varepsilon_{{3z}} y_{{EB}} - \omega_{{3z}}^2 x_{{EB}} \\ a_E + \varepsilon_{{3z}} x_{{EB}} - \omega_{{3z}}^2 y_{{EB}} \end{{bmatrix}}
$$

2. A (2) rúd felől ($C$-ből kiindulva, ahol $\mathbf{{a}}_C = \mathbf{{0}}$):
$$
\mathbf{{a}}_B = \mathbf{{a}}_C + \boldsymbol{{\varepsilon}}_2 \times \mathbf{{r}}_{{CB}} - \omega_{{2z}}^2 \mathbf{{r}}_{{CB}} =
\begin{{bmatrix}} -\varepsilon_{{2z}} l_2 \sin\beta + \omega_{{2z}}^2 l_2 \cos\beta \\ -\varepsilon_{{2z}} l_2 \cos\beta - \omega_{{2z}}^2 l_2 \sin\beta \end{{bmatrix}}
$$
Mivel $l_2 \sin\beta = y_{{EB}}$:
$$
\mathbf{{a}}_B^{{(2)}} = \begin{{bmatrix}} -\varepsilon_{{2z}} y_{{EB}} + \omega_{{2z}}^2 l_2 \cos\beta \\ -\varepsilon_{{2z}} l_2 \cos\beta - \omega_{{2z}}^2 y_{{EB}} \end{{bmatrix}}
$$

**Lineáris egyenletrendszer felállítása $\varepsilon_{{2z}}$ és $\varepsilon_{{3z}}$-re:**
Egyenlővé téve a komponenseket ($\mathbf{{a}}_B^{{(3)}} = \mathbf{{a}}_B^{{(2)}}$):
$$
\begin{{cases}}
y_{{EB}} (\varepsilon_{{2z}} - \varepsilon_{{3z}}) = \omega_{{3z}}^2 x_{{EB}} + \omega_{{2z}}^2 l_2 \cos\beta \\
l_2 \cos\beta \cdot \varepsilon_{{2z}} + x_{{EB}} \cdot \varepsilon_{{3z}} = (\omega_{{3z}}^2 - \omega_{{2z}}^2) y_{{EB}} - a_E
\end{{cases}}
$$

Mátrixos alakban:
$$
\begin{{bmatrix}}
y_{{EB}} & -y_{{EB}} \\
l_2 \cos\beta & x_{{EB}}
\end{{bmatrix}}
\begin{{bmatrix}}
\varepsilon_{{2z}} \\
\varepsilon_{{3z}}
\end{{bmatrix}}
=
\begin{{bmatrix}}
\omega_{{3z}}^2 x_{{EB}} + \omega_{{2z}}^2 l_2 \cos\beta \\
(\omega_{{3z}}^2 - \omega_{{2z}}^2) y_{{EB}} - a_E
\end{{bmatrix}}
$$
Behelyettesítve a numerikus együtthatókat:
$$
\begin{{bmatrix}}
{res['M'][0,0]:.{precision}f} & {res['M'][0,1]:.{precision}f} \\
{res['M'][1,0]:.{precision}f} & {res['M'][1,1]:.{precision}f}
\end{{bmatrix}}
\begin{{bmatrix}}
\varepsilon_{{2z}} \\
\varepsilon_{{3z}}
\end{{bmatrix}}
=
\begin{{bmatrix}}
{res['rhs'][0]:.{precision}f} \\
{res['rhs'][1]:.{precision}f}
\end{{bmatrix}}
$$

Az egyenletrendszer megoldása:
$$
\varepsilon_{{2z}} = \mathbf{{{res['eps_2z']:.{precision}f}}}\,\text{{rad/s}}^2
$$
$$
\varepsilon_{{3z}} = \mathbf{{{res['eps_3z']:.{precision}f}}}\,\text{{rad/s}}^2
$$

A $B$ és $S_2$ pontok gyorsulásvektorai:
$$
\mathbf{{a}}_B = \begin{{bmatrix}} {res['a_B'][0]:.{precision}f} \\ {res['a_B'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m/s}}^2
$$
Mivel $\mathbf{{a}}_C = \mathbf{{0}}$, a felezőpont gyorsulása a rúd végpontja gyorsulásának pontosan a fele:
$$
\mathbf{{a}}_{{S2}} = \frac{{\mathbf{{a}}_B}}{{2}} = 
\begin{{bmatrix}} {res['a_S2'][0]:.{precision}f} \\ {res['a_S2'][1]:.{precision}f} \end{{bmatrix}}\,\text{{m/s}}^2
$$
$$
a_{{S2}} = |\mathbf{{a}}_{{S2}}| = \sqrt{{a_{{S2x}}^2 + a_{{S2y}}^2}} =
\mathbf{{{res['a_S2_mag']:.{precision}f}}}\,\text{{m/s}}^2
$$

A (2) rúd gyorsulásszöge:
$$
\tan\alpha_2 = \frac{{|\varepsilon_{{2z}}|}}{{\omega_{{2z}}^2}} = 
\frac{{|{res['eps_2z']:.{precision}f}|}}{{({res['omega_2z']:.{precision}f})^2}} = {res['tan_alpha2']:.{precision}f}
$$
$$
|\alpha_2| = \arctan({res['tan_alpha2']:.{precision}f}) =
\mathbf{{{res['alpha_2_deg']:.{precision}f}^\circ}}
$$
""")

    # 5. Lépés: Természetes koordináta-rendszer
    with st.expander("5. Érintő- és normális gyorsulás, görbületi sugár", expanded=True):
        st.markdown(rf"""
A pályaérintő egységvektor a sebességvektor irányában mutat:
$$
\mathbf{{e}}_t = \frac{{\mathbf{{v}}_{{S2}}}}{{|\mathbf{{v}}_{{S2}}|}} = 
\begin{{bmatrix}} {res['e_t'][0]:.{precision}f} \\ {res['e_t'][1]:.{precision}f} \end{{bmatrix}}
$$

Az $S_2$ pont tangenciális (pályamenti) gyorsulása a gyorsulásvektor vetülete a sebességre:
$$
a_{{S2t}} = \mathbf{{a}}_{{S2}} \cdot \mathbf{{e}}_t = 
({res['a_S2'][0]:.{precision}f})({res['e_t'][0]:.{precision}f}) + 
({res['a_S2'][1]:.{precision}f})({res['e_t'][1]:.{precision}f}) = 
{res['a_S2t_signed']:.{precision}f}\,\text{{m/s}}^2
$$
$$
|a_{{S2t}}| = \mathbf{{{res['a_S2t_abs']:.{precision}f}}}\,\text{{m/s}}^2
$$

A normális (centripetális) gyorsulás nagysága Pitagorasz-tétellel:
$$
|a_{{S2n}}| = \sqrt{{|\mathbf{{a}}_{{S2}}|^2 - a_{{S2t}}^2}} = 
\sqrt{{({res['a_S2_mag']:.{precision}f})^2 - ({res['a_S2t_signed']:.{precision}f})^2}} = 
\mathbf{{{res['a_S2n']:.{precision}f}}}\,\text{{m/s}}^2
$$

Az $S_2$ pont pályájának pillanatnyi görbületi sugara:
$$
\rho_{{S2}} = \frac{{|\mathbf{{v}}_{{S2}}|^2}}{{|a_{{S2n}}|}} = 
\frac{{({res['v_S2_mag']:.{precision}f})^2}}{{{res['a_S2n']:.{precision}f}}} = 
\mathbf{{{res['rho_S2']:.{precision}f}}}\,\text{{m}}
$$
""")

# =========================================================================
# 4. FÜL: Mozgástartomány & Kinematikai Elemzés
# =========================================================================
with tab_sim:
    st.subheader(r"Mozgástartomány és kinematikai mennyiségek változása $\phi$ függvényében")
    st.markdown(r"""
    Ez a diagram bemutatja, hogyan változnak a rendszer szögsebességei, szöggyorsulásai és az $S_2$ súlypont sebessége 
    a (3) henger elfordulási szögének ($\phi$) teljes megengedett tartományában, változatlan geometriai és bemeneti sebesség mellett.
    """)
    
    # Érvényes phi tartomány kiszámítása
    phi_test_vals = np.linspace(15.0, 80.0, 150)
    phi_valid = []
    w2_vals = []
    w3_vals = []
    eps2_vals = []
    eps3_vals = []
    vS2_vals = []
    aS2_vals = []
    
    for p_val in phi_test_vals:
        try:
            r_sim = calculate_kinematics(p_val, l1, l2, l3, R, v_Cx)
            phi_valid.append(p_val)
            w2_vals.append(r_sim['omega_2z'])
            w3_vals.append(r_sim['omega_3z'])
            eps2_vals.append(r_sim['eps_2z'])
            eps3_vals.append(r_sim['eps_3z'])
            vS2_vals.append(r_sim['v_S2_mag'])
            aS2_vals.append(r_sim['a_S2_mag'])
        except Exception:
            continue
            
    if phi_valid:
        fig_sim, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=120)
        
        # 1. Szögsebességek
        ax1.plot(phi_valid, w2_vals, color='#2980b9', lw=2.2, label=r'$\omega_{2z}$ [rad/s]')
        ax1.plot(phi_valid, w3_vals, color='#e67e22', lw=2.2, label=r'$\omega_{3z}$ [rad/s]')
        ax1.axvline(phi_deg, color='#e74c3c', linestyle='--', lw=1.5, label=f'Jelenlegi: {phi_deg:.1f}°')
        ax1.set_ylabel('Szögsebesség [rad/s]', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')
        ax1.set_title(r'Kinematikai állapot a $\phi$ szög függvényében', fontsize=12)
        
        # 2. Szöggyorsulások
        ax2.plot(phi_valid, eps2_vals, color='#27ae60', lw=2.2, label=r'$\varepsilon_{2z}$ [rad/s²]')
        ax2.plot(phi_valid, eps3_vals, color='#8e44ad', lw=2.2, label=r'$\varepsilon_{3z}$ [rad/s²]')
        ax2.axvline(phi_deg, color='#e74c3c', linestyle='--', lw=1.5)
        ax2.set_ylabel('Szöggyorsulás [rad/s²]', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')
        
        # 3. S2 sebesség és gyorsulás
        ax3.plot(phi_valid, vS2_vals, color='#16a085', lw=2.2, label=r'$v_{S2}$ [m/s]')
        ax3.plot(phi_valid, aS2_vals, color='#c0392b', lw=2.2, label=r'$a_{S2}$ [m/s²]')
        ax3.axvline(phi_deg, color='#e74c3c', linestyle='--', lw=1.5)
        ax3.set_xlabel(r'$\phi$ [fok]', fontsize=11)
        ax3.set_ylabel('Nagyság [m/s, m/s²]', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='upper right')
        
        fig_sim.tight_layout()
        st.pyplot(fig_sim)
    else:
        st.warning("A megadott geometriai méretekkel nem találtunk érvényes összefüggő mozgástartományt.")

# =========================================================================
# 5. FÜL: Letölthető LaTeX / Jegyzőkönyv Forrás
# =========================================================================
with tab_latex:
    st.subheader("Letölthető BME jegyzőkönyv (LaTeX forráskód)")
    st.markdown("""
    Ez a forráskód a tanszéki mintának megfelelő formátumban tartalmazza a feladat teljes, behelyettesített levezetését.
    Közvetlenül kimásolható vagy letölthető `.tex` fájlként, és lefordítható pl. Overleaf vagy TeXMaker használatával.
    """)
    
    latex_template = f"""\\documentclass[12pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[magyar]{{babel}}
\\usepackage{{amsmath,amsfonts,amssymb,graphicx,booktabs}}

\\title{{\\textbf{{Dinamika (BMEGEMMBXM3)}}\\\\\\large 1. Házi feladat -- Síkbeli összetett mechanizmus kinematikája}}
\\author{{Név: \\dots\\dots\\dots\\dots\\dots\\quad Neptun: \\dots\\dots\\dots\\dots}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section*{{1. Bemeneti adatok}}
\\begin{{itemize}}
    \\item $\\phi = {phi_deg:.2f}^\\circ$ (a (3) henger szögelfordulása)
    \\item $l_1 = {res['l1']:.4f}\\,\\text{{m}}$ (az (1) henger sugara)
    \\item $l_2 = {res['l2']:.4f}\\,\\text{{m}}$ (a (2) rúd hossza)
    \\item $l_3 = {res['l3']:.4f}\\,\\text{{m}}$ (a (3) henger sugara)
    \\item $R = {res['R']:.4f}\\,\\text{{m}}$ (a fix domború körpálya sugara)
    \\item $v_{{Cx}} = {res['v_Cx']:.4f}\\,\\text{{m/s}}$ (a $C$ pont vízszintes sebessége)
\\end{{itemize}}

\\section*{{2. hwchk Eredménytáblázat}}
\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{llll}}
\\toprule
\\textbf{{hwchk mezőnév}} & \\textbf{{Megnevezés}} & \\textbf{{Számított érték}} & \\textbf{{Egység}} \\\\
\\midrule
\\texttt{{omega\\_2z}}    & (2) rúd szögsebessége           & ${res['omega_2z']:.{precision}f}$ & rad/s \\\\
\\texttt{{epsilon\\_2z}}  & (2) rúd szöggyorsulása         & ${res['eps_2z']:.{precision}f}$  & rad/s$^2$ \\\\
\\texttt{{omega\\_3z}}    & (3) henger szögsebessége        & ${res['omega_3z']:.{precision}f}$ & rad/s \\\\
\\texttt{{epsilon\\_3z}}  & (3) henger szöggyorsulása      & ${res['eps_3z']:.{precision}f}$  & rad/s$^2$ \\\\
\\texttt{{v\\_S2}}        & $S_2$ súlypont sebességnagysága & ${res['v_S2_mag']:.{precision}f}$ & m/s \\\\
\\texttt{{a\\_S2}}        & $S_2$ súlypont gyorsulásnagysága& ${res['a_S2_mag']:.{precision}f}$ & m/s$^2$ \\\\
\\texttt{{|alpha\\_2|}}   & (2) rúd gyorsulásszöge          & ${res['alpha_2_deg']:.{precision}f}^\\circ$ & fok \\\\
\\texttt{{|a\\_S2t|}}     & $S_2$ pályamenti gyorsulása     & ${res['a_S2t_abs']:.{precision}f}$ & m/s$^2$ \\\\
\\texttt{{|a\\_S2n|}}     & $S_2$ normális gyorsulása       & ${res['a_S2n']:.{precision}f}$ & m/s$^2$ \\\\
\\texttt{{rho\\_S2}}      & $S_2$ görbületi sugara          & ${res['rho_S2']:.{precision}f}$ & m \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{A számított eredmények összefoglalása}}
\\end{{table}}

\\section*{{3. Lépésről lépésre levezetés}}
\\subsection*{{3.1. Geometria}}
Origó az $E$ pontban: $\\mathbf{{r}}_E = \\begin{{bmatrix}} 0 \\\\ 0 \\end{{bmatrix}}\\,\\text{{m}}$.
$$
\\mathbf{{r}}_{{EB}} = \\begin{{bmatrix}} l_3 \\sin\\phi \\\\ l_3 (1 + \\cos\\phi) \\end{{bmatrix}} = \\begin{{bmatrix}} {res['r_EB'][0]:.{precision}f} \\\\ {res['r_EB'][1]:.{precision}f} \\end{{bmatrix}}\\,\\text{{m}}
$$
$$
\\sin\\beta = \\frac{{y_{{EB}}}}{{l_2}} = {res['sin_beta']:.{precision}f} \\implies \\beta = {res['beta_deg']:.{precision}f}^\\circ,\\quad \\cos\\beta = {res['cos_beta']:.{precision}f}
$$
$$
x_C = x_{{EB}} + l_2 \\cos\\beta = {res['r_C'][0]:.{precision}f}\\,\\text{{m}}, \\quad y_C = 0
$$

\\subsection*{{3.2. Sebességállapot és sebességpólus}}
A (3) henger tiszta gördülése miatt $\\mathbf{{v}}_E = \\mathbf{{0}}$.
$$
\\mathbf{{v}}_B = \\boldsymbol{{\\omega}}_3 \\times \\mathbf{{r}}_{{EB}} = \\begin{{bmatrix}} -\\omega_{{3z}} y_{{EB}} \\\\ \\omega_{{3z}} x_{{EB}} \\end{{bmatrix}} = 
\\mathbf{{v}}_C + \\boldsymbol{{\\omega}}_2 \\times \\mathbf{{r}}_{{CB}} = \\begin{{bmatrix}} v_{{Cx}} - \\omega_{{2z}} l_2 \\sin\\beta \\\\ -\\omega_{{2z}} l_2 \\cos\\beta \\end{{bmatrix}}
$$
Ebből:
$$
\\omega_{{2z}} = \\frac{{v_{{Cx}}}}{{\\frac{{y_{{EB}}}}{{x_{{EB}}}} l_2 \\cos\\beta + l_2 \\sin\\beta}} = {res['omega_2z']:.{precision}f}\\,\\text{{rad/s}}, \\quad
\\omega_{{3z}} = -\\omega_{{2z}} \\frac{{l_2 \\cos\\beta}}{{x_{{EB}}}} = {res['omega_3z']:.{precision}f}\\,\\text{{rad/s}}
$$
A sebességpólus $P$ koordinátái:
$$
x_P = x_C = {res['r_P'][0]:.{precision}f}\\,\\text{{m}}, \\quad y_P = \\frac{{v_{{Cx}}}}{{\\omega_{{2z}}}} = {res['r_P'][1]:.{precision}f}\\,\\text{{m}}
$$

\\subsection*{{3.3. Gyorsulásállapot}}
Mivel $\\mathbf{{v}}_C = \\text{{állandó}}$, ezért $\\mathbf{{a}}_C = \\mathbf{{0}}$, így a gyorsuláspólus $G \\equiv C$.
Az $E$ pont gyorsulása $R$ sugarú domború pályán tiszta gördüléskor:
$$
a_E = \\frac{{R l_3 \\omega_{{3z}}^2}}{{R + l_3}} = {res['a_E']:.{precision}f}\\,\\text{{m/s}}^2
$$
A $B$ pont gyorsulásának kétféle felírásából adódó egyenletrendszer megoldása:
$$
\\varepsilon_{{2z}} = {res['eps_2z']:.{precision}f}\\,\\text{{rad/s}}^2, \\quad \\varepsilon_{{3z}} = {res['eps_3z']:.{precision}f}\\,\\text{{rad/s}}^2
$$
A rúd gyorsulásszöge:
$$
\\tan\\alpha_2 = \\frac{{|\\varepsilon_{{2z}}|}}{{\\omega_{{2z}}^2}} \\implies |\\alpha_2| = {res['alpha_2_deg']:.{precision}f}^\\circ
$$

\\subsection*{{3.4. Pályamenti gyorsulás és görbületi sugár}}
$$
|a_{{S2t}}| = |\\mathbf{{a}}_{{S2}} \\cdot \\mathbf{{e}}_t| = {res['a_S2t_abs']:.{precision}f}\\,\\text{{m/s}}^2, \\quad
|a_{{S2n}}| = \\sqrt{{|\\mathbf{{a}}_{{S2}}|^2 - a_{{S2t}}^2}} = {res['a_S2n']:.{precision}f}\\,\\text{{m/s}}^2
$$
$$
\\rho_{{S2}} = \\frac{{|\\mathbf{{v}}_{{S2}}|^2}}{{|a_{{S2n}}|}} = {res['rho_S2']:.{precision}f}\\,\\text{{m}}
$$

\\end{{document}}
"""
    st.text_area("LaTeX forráskód", latex_template, height=350)
    
    st.download_button(
        label="📥 LaTeX forrás letöltése (.tex)",
        data=latex_template,
        file_name=f"BME_Dinamika_1HF_phi{phi_deg:.1f}.tex",
        mime="text/plain"
    )
