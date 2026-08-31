import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from metodos import (
    biseccion,
    falsa_posicion,
    punto_fijo,
    newton_raphson,
    secante,
    evaluar_seguro
)

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Métodos Numéricos",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ESTILO - CRIMSON / BLACK / IVORY
# =========================================================

st.markdown("""
<style>
:root{
    --bg:#0B0D12;
    --bg2:#11141B;
    --panel:#151923;
    --panel2:#1B2030;
    --line:rgba(255,255,255,.08);
    --text:#F7F4F7;
    --muted:#AAA2AC;
    --pink:#FF5FA2;
    --rose:#E8427A;
    --violet:#A66CFF;
    --cyan:#53D7FF;
    --green:#52D2A5;
    --glass:rgba(20,24,34,.72);
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Inter", sans-serif;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 12% 10%, rgba(255,95,162,.14), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(83,215,255,.11), transparent 30%),
        radial-gradient(circle at 70% 75%, rgba(166,108,255,.08), transparent 34%),
        linear-gradient(180deg, #090B10 0%, #0F1218 100%);
}

header[data-testid="stHeader"] {
    background: rgba(11,13,18,.76);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(255,255,255,.06);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1520px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(18,20,29,.98) 0%, rgba(22,20,31,.98) 100%);
    border-right: 1px solid rgba(255,95,162,.14);
}

section[data-testid="stSidebar"] * {
    color: #F6F0F5 !important;
}

section[data-testid="stSidebar"] label {
    color: #D8CBD4 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] input {
    background: #0F1219 !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    box-shadow: inset 0 0 0 1px rgba(255,95,162,.02);
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] input:focus {
    border-color: rgba(255,95,162,.55) !important;
    box-shadow: 0 0 0 2px rgba(255,95,162,.10) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.07) !important;
}

/* BUTTONS */
.stButton > button {
    width: 100%;
    border: 1px solid rgba(255,255,255,.10);
    color: white;
    background:
        linear-gradient(135deg, #FF5FA2 0%, #A66CFF 100%);
    border-radius: 12px;
    padding: .74rem 1rem;
    font-weight: 750;
    letter-spacing: .02em;
    box-shadow:
        0 10px 26px rgba(255,95,162,.18),
        0 0 22px rgba(166,108,255,.08);
    transition: all .16s ease;
}

.stButton > button:hover {
    color:white;
    transform: translateY(-1px);
    box-shadow:
        0 14px 34px rgba(255,95,162,.24),
        0 0 28px rgba(166,108,255,.12);
}

.stDownloadButton > button {
    border: 1px solid rgba(255,95,162,.28);
    color:#FFD7E8;
    background:#141821;
    border-radius:11px;
}

/* HERO */
.hero {
    border: 1px solid rgba(255,255,255,.08);
    background:
        linear-gradient(135deg, rgba(22,26,37,.82) 0%, rgba(27,31,45,.68) 100%);
    box-shadow:
        0 18px 48px rgba(0,0,0,.30),
        inset 0 1px 0 rgba(255,255,255,.04);
    border-radius: 22px;
    padding: 26px 30px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(14px);
}

.hero:before {
    content:"";
    position:absolute;
    left:-90px;
    bottom:-120px;
    width:260px;
    height:260px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(255,95,162,.20), transparent 66%);
}

.hero:after {
    content:"";
    position:absolute;
    right:-80px;
    top:-110px;
    width:280px;
    height:280px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(83,215,255,.14), transparent 68%);
}

.hero-kicker {
    color:#FF8DBB;
    text-transform:uppercase;
    letter-spacing:.18em;
    font-size:.72rem;
    font-weight:800;
}

.hero-title {
    color:#FFFFFF;
    font-size:2.15rem;
    font-weight:850;
    margin:.3rem 0 .2rem 0;
    text-shadow: 0 0 28px rgba(255,95,162,.08);
}

.hero-sub {
    color:#BBB2BC;
    font-size:1rem;
    margin:0;
}

.method-chip {
    display:inline-block;
    padding:6px 11px;
    border-radius:999px;
    background:linear-gradient(90deg, rgba(255,95,162,.15), rgba(166,108,255,.14));
    color:#FFD9E9;
    border:1px solid rgba(255,95,162,.22);
    font-size:.78rem;
    font-weight:700;
    margin-top:12px;
}

/* PANELS */
.panel {
    border:1px solid rgba(255,255,255,.07);
    background:var(--glass);
    border-radius:18px;
    padding:20px 22px;
    box-shadow:
        0 12px 30px rgba(0,0,0,.22),
        inset 0 1px 0 rgba(255,255,255,.025);
    margin-bottom:16px;
    backdrop-filter: blur(12px);
}

.accent-panel {
    border-left:3px solid #FF5FA2;
}

.section-title {
    color:#FFFFFF;
    font-weight:800;
    font-size:1.22rem;
    margin:0 0 .7rem 0;
}

.small-muted {
    color:#AAA2AC;
    font-size:.9rem;
}

/* METRIC CARDS */
.metric-card {
    height: 138px;
    border:1px solid rgba(255,255,255,.08);
    background:
        linear-gradient(145deg, rgba(22,26,37,.92), rgba(16,19,27,.92));
    border-radius:18px;
    padding:18px 20px;
    box-shadow:
        0 14px 32px rgba(0,0,0,.24),
        inset 0 1px 0 rgba(255,255,255,.03);
    position:relative;
    overflow:hidden;
}

.metric-card:before {
    content:"";
    position:absolute;
    left:0;
    top:0;
    bottom:0;
    width:4px;
    background:linear-gradient(180deg,#FF5FA2,#A66CFF);
}

.metric-card:after {
    content:"";
    position:absolute;
    right:-35px;
    top:-35px;
    width:95px;
    height:95px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(255,95,162,.11), transparent 68%);
}

.metric-label{
    color:#AFA4AE;
    font-size:.78rem;
    font-weight:800;
    letter-spacing:.05em;
    text-transform:uppercase;
}

.metric-number{
    color:#FFFFFF;
    font-size:1.84rem;
    font-weight:850;
    margin-top:10px;
}

.metric-success{
    color:#52D2A5;
}

/* TABLE */
[data-testid="stDataFrame"] {
    border:1px solid rgba(255,255,255,.07);
    border-radius:15px;
    overflow:hidden;
    background:#11151D;
}

/* ALERTS */
[data-testid="stAlert"] {
    border-radius:13px;
}

/* EXPANDER */
details {
    background:rgba(18,22,30,.86);
    border:1px solid rgba(255,255,255,.07) !important;
    border-radius:14px !important;
}

/* TÍTULOS NATIVOS */
h1,h2,h3 {
    color:#FFFFFF;
}

/* code */
code {
    color:#FFD3E5 !important;
}

/* plot container */
[data-testid="stPlotlyChart"] {
    border-radius:16px;
    overflow:hidden;
}

/* divider */
hr {
    border-color:rgba(255,255,255,.07);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# PARSEO
# =========================================================

def parsear_funcion(texto, variable="x"):
    simbolo = sp.symbols(variable)
    texto = texto.replace("^", "**")
    locales = {
        variable: simbolo,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "e": sp.E
    }
    try:
        expr = sp.sympify(texto, locals=locales)
        func_num = sp.lambdify(simbolo, expr, modules=["numpy"])
        return expr, func_num, simbolo
    except Exception as e:
        raise ValueError(f"Expresión no válida: {e}")

# =========================================================
# EJERCICIOS
# =========================================================

EJERCICIOS = {
    "Escribir mi propio ejercicio": None,

    "Sesión 2 · Bisección · Enlace de red": {
        "metodo": "Bisección",
        "f": "1/(x-8.5)-0.35*log(x-2)",
        "a": 9.0,
        "b": 10.0,
        "tol": 0.5,
        "descripcion": "Capacidad aproximada de un enlace de datos usando un intervalo con cambio de signo."
    },

    "Sesión 2 · Falsa Posición · Migración a la nube": {
        "metodo": "Falsa Posición",
        "f": "45+12*x-20*exp(0.4*x)",
        "a": 3.0,
        "b": 4.0,
        "tol": 0.5,
        "descripcion": "Punto de equilibrio entre el costo del sistema local y el costo acumulado en la nube."
    },

    "Sesión 3 · Punto Fijo · Temperatura": {
        "metodo": "Punto Fijo",
        "f": "18+8*exp(-0.15*x)-x",
        "g": "18+8*exp(-0.15*x)",
        "x0": 20.0,
        "tol": 0.01,
        "descripcion": "Temperatura de equilibrio de una sala de servidores mediante la iteración xₙ₊₁ = g(xₙ)."
    },

    "Sesión 3 · Newton-Raphson · Almacenamiento": {
        "metodo": "Newton-Raphson",
        "f": "x**3-7*x-5",
        "x0": 3.0,
        "tol": 0.01,
        "descripcion": "Solución positiva del modelo de tiempo de respuesta mediante Newton-Raphson."
    },

    "Sesión 3 · Secante · Servidor": {
        "metodo": "Secante",
        "f": "exp(-x)-x**2+0.2",
        "x0": 0.5,
        "x1": 1.0,
        "tol": 0.01,
        "descripcion": "Punto de operación normalizado de un servidor utilizando dos aproximaciones iniciales."
    }
}

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ◆ Panel de control")

ejercicio_elegido = st.sidebar.selectbox(
    "Ejercicio de ejemplo",
    list(EJERCICIOS.keys())
)
preset = EJERCICIOS[ejercicio_elegido]

metodos = [
    "Bisección",
    "Falsa Posición",
    "Punto Fijo",
    "Newton-Raphson",
    "Secante"
]

idx = metodos.index(preset["metodo"]) if preset else 0
metodo = st.sidebar.selectbox("Método numérico", metodos, index=idx)

st.sidebar.markdown("---")
st.sidebar.markdown("### Función")
variable = st.sidebar.text_input("Variable", value="x")

funcion_original_str = None
funcion_str = None
a = b = x0 = x1 = None

if metodo == "Punto Fijo":
    if preset and preset["metodo"] == "Punto Fijo":
        f_default, g_default = preset["f"], preset["g"]
        x0_default, tol_default = preset["x0"], preset["tol"]
    else:
        f_default, g_default = "cos(x)-x", "cos(x)"
        x0_default, tol_default = 0.5, 0.01

    funcion_original_str = st.sidebar.text_input("f(x) = 0", value=f_default)
    funcion_str = st.sidebar.text_input("g(x)", value=g_default)
    st.sidebar.markdown("### Valor inicial")
    x0 = st.sidebar.number_input("x₀", value=float(x0_default), format="%.6f")

elif metodo in ["Bisección", "Falsa Posición"]:
    if preset and preset["metodo"] == metodo:
        f_default = preset["f"]
        a_default, b_default, tol_default = preset["a"], preset["b"], preset["tol"]
    else:
        f_default, a_default, b_default, tol_default = "x**3-x-2", 1.0, 2.0, 0.5

    funcion_str = st.sidebar.text_input("f(x)", value=f_default)
    st.sidebar.markdown("### Intervalo")
    a = st.sidebar.number_input("Límite inferior a", value=float(a_default), format="%.6f")
    b = st.sidebar.number_input("Límite superior b", value=float(b_default), format="%.6f")

elif metodo == "Newton-Raphson":
    if preset and preset["metodo"] == metodo:
        f_default, x0_default, tol_default = preset["f"], preset["x0"], preset["tol"]
    else:
        f_default, x0_default, tol_default = "x**3-x-1", 1.5, 0.01

    funcion_str = st.sidebar.text_input("f(x)", value=f_default)
    st.sidebar.markdown("### Valor inicial")
    x0 = st.sidebar.number_input("x₀", value=float(x0_default), format="%.6f")

else:
    if preset and preset["metodo"] == metodo:
        f_default, x0_default, x1_default, tol_default = preset["f"], preset["x0"], preset["x1"], preset["tol"]
    else:
        f_default, x0_default, x1_default, tol_default = "x**3-x-1", 1.0, 2.0, 0.01

    funcion_str = st.sidebar.text_input("f(x)", value=f_default)
    st.sidebar.markdown("### Valores iniciales")
    x0 = st.sidebar.number_input("x₀", value=float(x0_default), format="%.6f")
    x1 = st.sidebar.number_input("x₁", value=float(x1_default), format="%.6f")

st.sidebar.markdown("---")
st.sidebar.markdown("### Criterios de parada")

tolerancia = st.sidebar.number_input(
    "Tolerancia (%)",
    min_value=0.000001,
    value=float(tol_default),
    format="%.6f"
)
max_iter = st.sidebar.number_input(
    "Máximo de iteraciones",
    min_value=1,
    max_value=500,
    value=50,
    step=1
)

resolver = st.sidebar.button("Resolver", use_container_width=True)

# =========================================================
# HERO
# =========================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">NUMERICAL SYSTEMS · ROOT ENGINE</div>
        <div class="hero-title">Numerical Methods Studio</div>
        <p class="hero-sub">
            Explora raíces, convergencia e iteraciones con una interfaz moderna y visual.
        </p>
        <span class="method-chip">{metodo}</span>
    </div>
    """,
    unsafe_allow_html=True
)

if preset and "descripcion" in preset:
    st.markdown(
        f"""
        <div class="panel accent-panel">
            <div class="section-title">Contexto del ejercicio</div>
            <div class="small-muted">{preset["descripcion"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# RESULTADOS
# =========================================================

def boton_csv(df, nombre):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar tabla CSV",
        csv,
        file_name=nombre,
        mime="text/csv"
    )

if resolver:
    try:
        if metodo == "Punto Fijo":
            expr_f, f_original_num, x_sym = parsear_funcion(funcion_original_str, variable)
            expr_g, g_num, _ = parsear_funcion(funcion_str, variable)
            filas, resumen = punto_fijo(
                g_num, x0,
                tol=tolerancia,
                max_iter=int(max_iter),
                f_num=f_original_num
            )
        else:
            expr, f_num, x_sym = parsear_funcion(funcion_str, variable)

            if metodo == "Bisección":
                filas, resumen = biseccion(f_num, a, b, tol=tolerancia, max_iter=int(max_iter))

            elif metodo == "Falsa Posición":
                filas, resumen = falsa_posicion(f_num, a, b, tol=tolerancia, max_iter=int(max_iter))

            elif metodo == "Newton-Raphson":
                derivada = sp.diff(expr, x_sym)
                derivada_num = sp.lambdify(x_sym, derivada, modules=["numpy"])
                filas, resumen = newton_raphson(
                    f_num, derivada_num, x0,
                    tol=tolerancia,
                    max_iter=int(max_iter)
                )

            else:
                filas, resumen = secante(
                    f_num, x0, x1,
                    tol=tolerancia,
                    max_iter=int(max_iter)
                )

        if "error_msg" in resumen:
            st.error(resumen["error_msg"])

        elif not filas:
            st.error("No se generaron iteraciones.")

        else:
            raiz = resumen["raiz"]
            err_final = resumen["error_final"]

            c1, c2, c3, c4 = st.columns(4)

            cards = [
                ("Raíz aproximada", f"{raiz:.8f}", ""),
                ("Iteraciones", str(resumen["iteraciones"]), ""),
                ("Error final (%)", f"{0 if err_final is None else err_final:.6f}", ""),
                ("Estado", "Convergió" if resumen["convergio"] else "Máx. iter.", "metric-success" if resumen["convergio"] else "")
            ]

            for col, (lab, val, cls) in zip([c1,c2,c3,c4], cards):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{lab}</div>
                            <div class="metric-number {cls}">{val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.write("")

            if metodo == "Newton-Raphson":
                st.markdown('<div class="panel accent-panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Derivada utilizada</div>', unsafe_allow_html=True)
                st.latex(f"f'(x)={sp.latex(derivada)}")
                st.markdown('</div>', unsafe_allow_html=True)

            if metodo == "Punto Fijo":
                derivada_g = sp.diff(expr_g, x_sym)
                derivada_g_num = sp.lambdify(x_sym, derivada_g, modules=["numpy"])
                valor = abs(evaluar_seguro(derivada_g_num, raiz))

                st.markdown('<div class="panel accent-panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Condición de convergencia</div>', unsafe_allow_html=True)
                st.latex(f"g'(x)={sp.latex(derivada_g)}")
                st.write(f"|g'(raíz)| ≈ **{valor:.8f}**")
                if valor < 1:
                    st.success("Se cumple localmente |g'(x)| < 1.")
                else:
                    st.warning("No se cumple localmente |g'(x)| < 1.")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Tabla de iteraciones</div>', unsafe_allow_html=True)
            df = pd.DataFrame(filas)
            st.dataframe(df, use_container_width=True, hide_index=True)
            boton_csv(df, f"tabla_{metodo.lower().replace(' ', '_')}.csv")

            st.write("")

            g1, g2 = st.columns(2)

            with g1:
                st.markdown('<div class="section-title">Función y raíz</div>', unsafe_allow_html=True)

                if metodo in ["Bisección", "Falsa Posición"]:
                    xmin, xmax = min(a,b), max(a,b)
                elif metodo == "Secante":
                    xmin, xmax = min(x0,x1,raiz)-1, max(x0,x1,raiz)+1
                else:
                    xmin, xmax = min(x0,raiz)-2, max(x0,raiz)+2

                xs = np.linspace(xmin, xmax, 500)

                if metodo == "Punto Fijo":
                    ys_g = [evaluar_seguro(g_num, xx) for xx in xs]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys_g, mode="lines", name="y=g(x)",
                        line=dict(color="#FF5FA2", width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=xs, y=xs, mode="lines", name="y=x",
                        line=dict(color="#A66CFF", width=2, dash="dash")
                    ))
                    fig.add_trace(go.Scatter(
                        x=[raiz], y=[raiz], mode="markers", name="Punto fijo",
                        marker=dict(size=12, color="#53D7FF")
                    ))
                else:
                    ys = [evaluar_seguro(f_num, xx) for xx in xs]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", name="f(x)",
                        line=dict(color="#FF5FA2", width=3)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="#A66CFF")
                    fig.add_trace(go.Scatter(
                        x=[raiz], y=[evaluar_seguro(f_num, raiz)],
                        mode="markers", name="Raíz",
                        marker=dict(size=12, color="#53D7FF")
                    ))

                fig.update_layout(
                    template="plotly_white",
                    height=420,
                    margin=dict(l=20,r=20,t=25,b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#10141C",
                    font=dict(color="#A66CFF")
                )
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                st.markdown('<div class="section-title">Convergencia</div>', unsafe_allow_html=True)
                errores = df[df["Ea (%)"].notna()].copy()

                if not errores.empty:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=list(range(1, len(errores)+1)),
                        y=errores["Ea (%)"],
                        mode="lines+markers",
                        name="Ea (%)",
                        line=dict(color="#A66CFF", width=3),
                        marker=dict(size=8, color="#53D7FF")
                    ))
                    fig2.add_hline(
                        y=tolerancia,
                        line_dash="dash",
                        line_color="#FF5FA2",
                        annotation_text=f"Tolerancia = {tolerancia}%"
                    )
                    fig2.update_layout(
                        template="plotly_white",
                        height=420,
                        margin=dict(l=20,r=20,t=25,b=20),
                        xaxis_title="Iteración",
                        yaxis_title="Error (%)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#10141C",
                        font=dict(color="#A66CFF")
                    )
                    if (errores["Ea (%)"] > 0).all():
                        fig2.update_yaxes(type="log")

                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                """
                <div class="panel accent-panel">
                    <div class="section-title">Verificación final</div>
                    <div class="small-muted">
                        Evaluación de la ecuación en la aproximación encontrada.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.code(f"f(raíz) = {resumen['f_raiz']:.12e}")

            if abs(resumen["f_raiz"]) < 1e-4:
                st.success("La solución queda verificada porque f(raíz) es cercana a cero.")
            else:
                st.info("Se alcanzó el criterio de error establecido.")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

else:
    st.markdown(
        """
        <div class="panel accent-panel">
            <div class="section-title">Listo para calcular</div>
            <div class="small-muted">
                Configura el método y los datos en el panel izquierdo, luego presiona <b>Resolver</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

with st.expander("Guía rápida para escribir funciones"):
    st.markdown("""
| Matemática | Escribir |
|---|---|
| x² | `x**2` |
| x³ | `x**3` |
| √x | `sqrt(x)` |
| sen(x) | `sin(x)` |
| cos(x) | `cos(x)` |
| eˣ | `exp(x)` |
| ln(x) | `log(x)` |

Ejemplos: `x**3-x-2`, `cos(x)-x`, `exp(-x)-x**2+0.2`
""")
