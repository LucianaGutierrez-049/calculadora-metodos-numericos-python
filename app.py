import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from metodos import (
    biseccion,
    falsa_posicion,
    punto_fijo,
    newton_raphson,
    secante,
    evaluar_seguro,
)
from polinomios import (
    analizar_estabilidad,
    cotas_lagrange,
    expresion_horner,
    formato_complejo,
    horner,
    muller,
    pasos_horner,
    regla_descartes,
    tabla_raices,
    todas_raices_muller,
    validar_coeficientes,
)


st.set_page_config(
    page_title="Calculadora de Métodos Numéricos",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #090b10;
    --panel: #141821;
    --panel-2: #1b2130;
    --panel-3: #0f131b;
    --line: rgba(255,255,255,.11);
    --line-strong: rgba(255,95,159,.30);
    --text: #f8f4f6;
    --muted: #b9b0b8;
    --rose: #ff5f9f;
    --rose-2: #e9487f;
    --violet: #9b7cff;
    --cyan: #59d5ff;
    --green: #59d49f;
    --amber: #ffd166;
}
.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 8% 4%, rgba(255,95,159,.14), transparent 30%),
        radial-gradient(circle at 86% 0%, rgba(89,213,255,.10), transparent 32%),
        linear-gradient(180deg, #080a0f 0%, #10131a 100%);
}
.block-container { padding-top: 1.1rem; padding-bottom: 2.8rem; max-width: 1480px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10131a 0%, #11141d 100%);
    border-right: 1px solid rgba(255,95,159,.18);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #f5eef4;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: #0d1118 !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 8px !important;
}
.hero, .panel, .metric-card, .soft-panel {
    border: 1px solid var(--line);
    background: rgba(20,24,33,.92);
    border-radius: 8px;
    box-shadow: 0 14px 34px rgba(0,0,0,.24);
}
.hero {
    margin-bottom: 16px;
    padding: 24px 26px;
    background:
        linear-gradient(135deg, rgba(255,95,159,.15), transparent 36%),
        linear-gradient(145deg, rgba(25,30,42,.96), rgba(17,21,31,.94));
}
.hero-grid {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: flex-end;
}
.hero-title { font-size: clamp(1.75rem, 2.4vw, 2.6rem); font-weight: 850; color: #fff; margin: 0; line-height: 1.05; }
.hero-sub { color: var(--muted); margin: .55rem 0 0 0; max-width: 760px; }
.hero-badges { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; min-width: 260px; }
.badge {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 999px;
    padding: 6px 10px;
    background: rgba(255,255,255,.045);
    color: #f6eef5;
    font-size: .78rem;
    font-weight: 750;
}
.panel { padding: 18px 20px; margin-bottom: 16px; }
.soft-panel { padding: 16px 18px; margin: 10px 0 16px; background: rgba(15,19,27,.82); }
.section-title { font-size: 1.15rem; font-weight: 850; color: #fff; margin: 0 0 .25rem 0; }
.section-sub { color: var(--muted); margin: 0 0 .85rem 0; font-size: .92rem; }
.small-muted { color: var(--muted); font-size: .92rem; }
.metric-card {
    min-height: 112px;
    padding: 17px 18px;
    background:
        linear-gradient(180deg, rgba(31,37,52,.92), rgba(18,22,31,.96));
    position: relative;
    overflow: hidden;
}
.metric-card:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--rose), var(--cyan));
}
.metric-label { color: var(--muted); font-size: .76rem; text-transform: uppercase; font-weight: 850; letter-spacing: .04em; }
.metric-number { color: #fff; font-size: 1.42rem; font-weight: 850; margin-top: .45rem; overflow-wrap: anywhere; line-height: 1.18; }
.ok { color: var(--green); }
.warn { color: var(--amber); }
.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,.12);
    background: linear-gradient(135deg, var(--rose), var(--violet));
    color: #fff;
    font-weight: 800;
    min-height: 2.7rem;
    box-shadow: 0 10px 24px rgba(255,95,159,.16);
}
.stButton > button:hover {
    border-color: rgba(255,255,255,.22);
    box-shadow: 0 14px 30px rgba(255,95,159,.23);
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #10141d; }
div[data-testid="stTabs"] button {
    border-radius: 8px 8px 0 0;
    font-weight: 750;
}
details { border-radius: 8px !important; border: 1px solid var(--line) !important; background: rgba(15,19,27,.82) !important; }
code { color: #ffd6e8 !important; }
@media (max-width: 860px) {
    .hero-grid { flex-direction: column; align-items: flex-start; }
    .hero-badges { justify-content: flex-start; min-width: 0; }
}
</style>
""",
    unsafe_allow_html=True,
)


EJERCICIOS = {
    "Escribir mi propio ejercicio": None,
    "Sesión 2 · Bisección · Enlace de red": {
        "metodo": "Biseccion",
        "f": "1/(x-8.5)-0.35*log(x-2)",
        "a": 9.0,
        "b": 10.0,
        "tol": 0.5,
        "descripcion": "Capacidad aproximada de un enlace de datos usando un intervalo con cambio de signo.",
    },
    "Sesión 2 · Falsa Posición · Migración a la nube": {
        "metodo": "Falsa Posicion",
        "f": "45+12*x-20*exp(0.4*x)",
        "a": 3.0,
        "b": 4.0,
        "tol": 0.5,
        "descripcion": "Punto de equilibrio entre el costo local y el costo acumulado en la nube.",
    },
    "Sesión 3 · Punto Fijo · Temperatura": {
        "metodo": "Punto Fijo",
        "f": "18+8*exp(-0.15*x)-x",
        "g": "18+8*exp(-0.15*x)",
        "x0": 20.0,
        "tol": 0.01,
        "descripcion": "Temperatura de equilibrio por iteración de punto fijo.",
    },
    "Sesión 3 · Newton-Raphson · Almacenamiento": {
        "metodo": "Newton-Raphson",
        "f": "x**3-7*x-5",
        "x0": 3.0,
        "tol": 0.01,
        "descripcion": "Solución positiva del modelo mediante Newton-Raphson.",
    },
    "Sesión 3 · Secante · Servidor": {
        "metodo": "Secante",
        "f": "exp(-x)-x**2+0.2",
        "x0": 0.5,
        "x1": 1.0,
        "tol": 0.01,
        "descripcion": "Punto de operación normalizado de un servidor.",
    },
    "Sesión 4 · Müller · Filtro digital IIR": {
        "metodo": "Muller",
        "forma": "Expresion",
        "variable": "z",
        "polinomio": "8*z**4 - 6*z**3 - 3*z**2 + 3*z - 1",
        "coeficientes": "8,-6,-3,3,-1",
        "x0": "0",
        "x1": "0.5",
        "x2": "1.0",
        "tol": 1e-5,
        "max_iter": 50,
        "estabilidad": True,
        "descripcion": "Análisis de estabilidad de un filtro digital IIR: todos los polos deben cumplir |z_i| < 1.",
    },
}

METODOS = ["Biseccion", "Falsa Posicion", "Punto Fijo", "Newton-Raphson", "Secante", "Muller"]
METODOS_VISIBLES = {
    "Biseccion": "Bisección",
    "Falsa Posicion": "Falsa Posición",
    "Punto Fijo": "Punto Fijo",
    "Newton-Raphson": "Newton-Raphson",
    "Secante": "Secante",
    "Muller": "Müller",
}


def nombre_metodo(metodo):
    return METODOS_VISIBLES.get(metodo, metodo)


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
        "e": sp.E,
    }
    try:
        expr = sp.sympify(texto, locals=locales)
        func_num = sp.lambdify(simbolo, expr, modules=["numpy"])
        return expr, func_num, simbolo
    except Exception as exc:
        raise ValueError(f"Expresión no válida: {exc}") from exc


def detectar_variable(expr, variable_preferida):
    simbolos = sorted(expr.free_symbols, key=lambda s: s.name)
    if not simbolos:
        return sp.symbols(variable_preferida)
    if len(simbolos) > 1:
        nombres = ", ".join(str(s) for s in simbolos)
        raise ValueError(f"El polinomio debe usar una sola variable. Se detectaron: {nombres}.")
    return simbolos[0]


def parsear_polinomio_expresion(texto, variable_preferida="x"):
    texto = texto.replace("^", "**")
    locales = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "e": sp.E,
    }
    expr = sp.sympify(texto, locals=locales)
    simbolo = detectar_variable(expr, variable_preferida)
    try:
        poly = sp.Poly(sp.expand(expr), simbolo)
    except Exception as exc:
        raise ValueError("La expresión ingresada no es polinomial.") from exc
    coeficientes = [float(c) if c.is_real else complex(c.evalf()) for c in poly.all_coeffs()]
    return validar_coeficientes(coeficientes, grado_minimo=2), simbolo, poly


def parsear_coeficientes(texto, variable="x"):
    partes = [p.strip() for p in re.split(r"[,;\n]+", texto) if p.strip()]
    if not partes:
        raise ValueError("Ingresa coeficientes separados por comas.")
    coeficientes = []
    for parte in partes:
        parte = parte.replace("i", "I")
        valor = sp.N(sp.sympify(parte))
        coeficientes.append(float(valor) if valor.is_real else complex(valor))
    coeficientes = validar_coeficientes(coeficientes, grado_minimo=2)
    simbolo = sp.symbols(variable or "x")
    poly = sp.Poly.from_list(coeficientes, gens=simbolo)
    return coeficientes, simbolo, poly


def parsear_numero(texto):
    valor = sp.N(sp.sympify(str(texto).replace("i", "I")))
    return float(valor) if valor.is_real else complex(valor)


def dataframe_limpio(filas):
    if not filas:
        return pd.DataFrame()

    columnas = sorted({clave for fila in filas for clave in fila.keys()})
    columnas_con_complejos = {
        columna
        for columna in columnas
        if any(isinstance(fila.get(columna), complex) for fila in filas)
    }

    def conv(valor, forzar_texto=False):
        if isinstance(valor, complex):
            return formato_complejo(valor)
        if isinstance(valor, list):
            return formato_lista_numeros(valor)
        if forzar_texto and valor is not None:
            return formato_complejo(valor)
        if isinstance(valor, float):
            return round(valor, 8)
        return valor

    return pd.DataFrame([
        {
            k: conv(v, k in columnas_con_complejos)
            for k, v in fila.items()
        }
        for fila in filas
    ])


def formato_lista_numeros(valores):
    return "[" + ", ".join(formato_complejo(v) for v in valores) + "]"


def formatear_decimal(valor, decimales=8):
    if valor is None:
        return ""
    try:
        return f"{float(valor):.{decimales}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(valor)


def formatear_posibilidades(valores):
    return " o ".join(str(v) for v in valores)


def polinomio_desde_coeficientes(coeficientes, variable="z"):
    coeficientes = list(coeficientes)
    grado = len(coeficientes) - 1
    partes = []
    for i, coef in enumerate(coeficientes):
        z = complex(coef)
        if abs(z) < 1e-12:
            continue
        potencia = grado - i
        valor = formato_complejo(abs(z.real) if abs(z.imag) < 1e-12 else z)
        if potencia == 0:
            termino = valor
        elif potencia == 1:
            termino = f"{valor}{variable}"
        else:
            termino = f"{valor}{variable}^{potencia}"
        signo = "-" if z.real < 0 and abs(z.imag) < 1e-12 else "+"
        partes.append((signo, termino))
    if not partes:
        return "0"
    primero_signo, primero = partes[0]
    texto = f"-{primero}" if primero_signo == "-" else primero
    for signo, termino in partes[1:]:
        texto += f" {signo} {termino}"
    return texto


def criterio_parada_muller(filas, tolerancia):
    if not filas:
        return "No hay iteraciones suficientes para determinar el criterio de parada."
    ultima = filas[-1]
    cumple_delta = ultima.get("|x3-x2|") is not None and ultima["|x3-x2|"] < tolerancia
    cumple_residuo = abs(complex(ultima.get("P(x3)", 0))) < tolerancia
    criterios = []
    if cumple_delta:
        criterios.append("|z3-z2| < ε")
    if cumple_residuo:
        criterios.append("|P(z3)| < ε")
    if not criterios:
        return "No se cumplió un criterio de parada antes del máximo de iteraciones."
    prefijo = "Criterio de parada cumplido" if len(criterios) == 1 else "Criterios de parada cumplidos"
    return f"{prefijo}: {' y '.join(criterios)}"


def tablas_muller(filas):
    principales = []
    detalles = []
    for fila in filas:
        principales.append({
            "Iteración": fila["Iteracion"],
            "z0": fila["x0"],
            "z1": fila["x1"],
            "z2": fila["x2"],
            "z3 nuevo": fila["x3"],
            "P(z3)": fila["P(x3)"],
            "Ea (%)": fila["Ea (%)"],
        })
        detalles.append({
            "Iteración": fila["Iteracion"],
            "h0": fila["h0"],
            "h1": fila["h1"],
            "delta0": fila["delta0"],
            "delta1": fila["delta1"],
            "a": fila["a"],
            "b": fila["b"],
            "c": fila["c"],
            "discriminante": fila["discriminante"],
            "sqrt(discriminante)": fila["sqrt(discriminante)"],
            "denominador elegido": fila["denominador"],
            "|z3-z2|": fila["|x3-x2|"],
        })
    return dataframe_limpio(principales), dataframe_limpio(detalles)


def boton_csv(df, nombre):
    st.download_button(
        "Descargar tabla CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=nombre,
        mime="text/csv",
        key=f"descarga_{nombre}",
    )


def tarjeta(col, etiqueta, valor, clase=""):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{etiqueta}</div>
                <div class="metric-number {clase}">{valor}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def encabezado_seccion(titulo, subtitulo=None):
    subtitulo_html = f'<p class="section-sub">{subtitulo}</p>' if subtitulo else ""
    st.markdown(
        f"""
        <div class="soft-panel">
            <p class="section-title">{titulo}</p>
            {subtitulo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def grafica_raices(raices, estabilidad=False):
    fig = go.Figure()
    xs = [complex(r).real for r in raices]
    ys = [complex(r).imag for r in raices]
    etiquetas = [f"z{i}" for i in range(1, len(raices) + 1)]
    customdata = [
        [etiquetas[i], formato_complejo(raices[i]), xs[i], ys[i], abs(complex(raices[i]))]
        for i in range(len(raices))
    ]
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=etiquetas,
        textposition="top center",
        name="Raíces encontradas",
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "valor: %{customdata[1]}<br>"
            "Re: %{customdata[2]:.8f}<br>"
            "Im: %{customdata[3]:.8f}<br>"
            "|z|: %{customdata[4]:.8f}<extra></extra>"
        ),
        marker=dict(size=12, color="#ff5f9f"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9b7cff")
    fig.add_vline(x=0, line_dash="dash", line_color="#9b7cff")
    if estabilidad:
        theta = np.linspace(0, 2 * np.pi, 300)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", name="|z| = 1", line=dict(color="#59d5ff", dash="dot")))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=430,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Eje real",
        yaxis_title="Eje imaginario",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div>
                <p class="hero-title">Calculadora de Métodos Numéricos</p>
                <p class="hero-sub">
                    Resuelve métodos clásicos, analiza polinomios y prepara capturas claras para informe o examen.
                </p>
            </div>
            <div class="hero-badges">
                <span class="badge">Bisección</span>
                <span class="badge">Newton</span>
                <span class="badge">Secante</span>
                <span class="badge">Müller</span>
                <span class="badge">Estabilidad</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Panel de control")
ejercicio_elegido = st.sidebar.selectbox("Ejercicio de ejemplo", list(EJERCICIOS.keys()))
preset = EJERCICIOS[ejercicio_elegido]
idx = METODOS.index(preset["metodo"]) if preset else 0
metodo = st.sidebar.selectbox("Método numérico", METODOS, index=idx, format_func=nombre_metodo)

if preset and "descripcion" in preset:
    st.markdown(
        f"""
        <div class="soft-panel">
            <p class="section-title">Contexto del ejercicio</p>
            <p class="section-sub">{preset["descripcion"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if metodo != "Muller":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Función")
    variable = st.sidebar.text_input("Variable", value="x")
    funcion_original_str = None
    funcion_str = None
    a = b = x0 = x1 = None

    if metodo == "Punto Fijo":
        f_default = preset["f"] if preset and preset["metodo"] == metodo else "cos(x)-x"
        g_default = preset["g"] if preset and preset["metodo"] == metodo else "cos(x)"
        x0_default = preset["x0"] if preset and preset["metodo"] == metodo else 0.5
        tol_default = preset["tol"] if preset and preset["metodo"] == metodo else 0.01
        funcion_original_str = st.sidebar.text_input("f(x) = 0", value=f_default)
        funcion_str = st.sidebar.text_input("g(x)", value=g_default)
        x0 = st.sidebar.number_input("x0", value=float(x0_default), format="%.8f")
    elif metodo in ["Biseccion", "Falsa Posicion"]:
        f_default = preset["f"] if preset and preset["metodo"] == metodo else "x**3-x-2"
        a_default = preset["a"] if preset and preset["metodo"] == metodo else 1.0
        b_default = preset["b"] if preset and preset["metodo"] == metodo else 2.0
        tol_default = preset["tol"] if preset and preset["metodo"] == metodo else 0.5
        funcion_str = st.sidebar.text_input("f(x)", value=f_default)
        a = st.sidebar.number_input("Límite inferior a", value=float(a_default), format="%.8f")
        b = st.sidebar.number_input("Límite superior b", value=float(b_default), format="%.8f")
    elif metodo == "Newton-Raphson":
        f_default = preset["f"] if preset and preset["metodo"] == metodo else "x**3-x-1"
        x0_default = preset["x0"] if preset and preset["metodo"] == metodo else 1.5
        tol_default = preset["tol"] if preset and preset["metodo"] == metodo else 0.01
        funcion_str = st.sidebar.text_input("f(x)", value=f_default)
        x0 = st.sidebar.number_input("x0", value=float(x0_default), format="%.8f")
    else:
        f_default = preset["f"] if preset and preset["metodo"] == metodo else "x**3-x-1"
        x0_default = preset["x0"] if preset and preset["metodo"] == metodo else 1.0
        x1_default = preset["x1"] if preset and preset["metodo"] == metodo else 2.0
        tol_default = preset["tol"] if preset and preset["metodo"] == metodo else 0.01
        funcion_str = st.sidebar.text_input("f(x)", value=f_default)
        x0 = st.sidebar.number_input("x0", value=float(x0_default), format="%.8f")
        x1 = st.sidebar.number_input("x1", value=float(x1_default), format="%.8f")

    st.sidebar.markdown("---")
    tolerancia = st.sidebar.number_input("Tolerancia (%)", min_value=0.000001, value=float(tol_default), format="%.8f")
    max_iter = st.sidebar.number_input("Máximo de iteraciones", min_value=1, max_value=500, value=50, step=1)
    resolver = st.sidebar.button("Resolver")

    encabezado_seccion(
        nombre_metodo(metodo),
        "Configura los datos en el panel izquierdo y revisa el resumen, la tabla, las gráficas y la verificación.",
    )
    if resolver:
        try:
            if metodo == "Punto Fijo":
                expr_f, f_original_num, x_sym = parsear_funcion(funcion_original_str, variable)
                expr_g, g_num, _ = parsear_funcion(funcion_str, variable)
                filas, resumen = punto_fijo(g_num, x0, tol=tolerancia, max_iter=int(max_iter), f_num=f_original_num)
            else:
                expr, f_num, x_sym = parsear_funcion(funcion_str, variable)
                if metodo == "Biseccion":
                    filas, resumen = biseccion(f_num, a, b, tol=tolerancia, max_iter=int(max_iter))
                elif metodo == "Falsa Posicion":
                    filas, resumen = falsa_posicion(f_num, a, b, tol=tolerancia, max_iter=int(max_iter))
                elif metodo == "Newton-Raphson":
                    derivada = sp.diff(expr, x_sym)
                    derivada_num = sp.lambdify(x_sym, derivada, modules=["numpy"])
                    filas, resumen = newton_raphson(f_num, derivada_num, x0, tol=tolerancia, max_iter=int(max_iter))
                else:
                    filas, resumen = secante(f_num, x0, x1, tol=tolerancia, max_iter=int(max_iter))

            if "error_msg" in resumen:
                st.error(resumen["error_msg"])
            elif not filas:
                st.error("No se generaron iteraciones.")
            else:
                raiz = resumen["raiz"]
                encabezado_seccion("Resumen del cálculo", "Resultado principal y criterio de parada alcanzado.")
                cols = st.columns(4)
                tarjeta(cols[0], "Raíz aproximada", f"{raiz:.10g}")
                tarjeta(cols[1], "Iteraciones", resumen["iteraciones"])
                tarjeta(cols[2], "Error final (%)", f"{0 if resumen['error_final'] is None else resumen['error_final']:.8g}")
                tarjeta(cols[3], "Estado", "Convergió" if resumen["convergio"] else "Máx. iter.", "ok" if resumen["convergio"] else "warn")

                if metodo == "Newton-Raphson":
                    st.latex(f"f'(x)={sp.latex(derivada)}")
                if metodo == "Punto Fijo":
                    derivada_g = sp.diff(expr_g, x_sym)
                    derivada_g_num = sp.lambdify(x_sym, derivada_g, modules=["numpy"])
                    valor = abs(evaluar_seguro(derivada_g_num, raiz))
                    st.write(f"Condición local: |g'(raíz)| ≈ **{valor:.8f}**")

                encabezado_seccion("Tabla de iteraciones", "Detalle numérico listo para revisar o exportar.")
                df = pd.DataFrame(filas)
                st.dataframe(df, width="stretch", hide_index=True)
                boton_csv(df, f"tabla_{metodo.lower().replace(' ', '_')}.csv")

                encabezado_seccion("Visualización", "Función, raíz aproximada y comportamiento del error.")
                g1, g2 = st.columns(2)
                with g1:
                    xs = np.linspace(min(raiz, x0 if x0 is not None else raiz, a if a is not None else raiz) - 2, max(raiz, x1 if x1 is not None else raiz, b if b is not None else raiz) + 2, 500)
                    fig = go.Figure()
                    if metodo == "Punto Fijo":
                        ys_g = [evaluar_seguro(g_num, xx) for xx in xs]
                        fig.add_trace(go.Scatter(x=xs, y=ys_g, mode="lines", name="g(x)", line=dict(color="#ff5f9f")))
                        fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y=x", line=dict(color="#9b7cff", dash="dash")))
                        marcador_y = raiz
                    else:
                        ys = [evaluar_seguro(f_num, xx) for xx in xs]
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#ff5f9f")))
                        fig.add_hline(y=0, line_dash="dash", line_color="#9b7cff")
                        marcador_y = 0
                    fig.add_trace(go.Scatter(x=[raiz], y=[marcador_y], mode="markers", name="Raíz", marker=dict(size=11, color="#59d5ff")))
                    fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, width="stretch", key=f"grafica_funcion_{metodo}")
                with g2:
                    errores = df[df["Ea (%)"].notna()].copy()
                    if not errores.empty:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=list(range(1, len(errores) + 1)), y=errores["Ea (%)"], mode="lines+markers", line=dict(color="#59d5ff")))
                        fig2.add_hline(y=tolerancia, line_dash="dash", line_color="#ff5f9f")
                        fig2.update_layout(height=420, template="plotly_dark", xaxis_title="Iteración", yaxis_title="Ea (%)", margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig2, width="stretch", key=f"grafica_error_{metodo}")
                encabezado_seccion("Verificación final", "Sustitución de la raíz aproximada en la ecuación original.")
                st.code(f"f(raiz) = {resumen['f_raiz']:.12e}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ocurrió un error: {exc}")
    else:
        st.info("Configura los datos en el panel izquierdo y presiona Resolver.")

else:
    p = preset if preset and preset["metodo"] == "Muller" else {}
    st.sidebar.markdown("---")
    modo = st.sidebar.radio("Modo", ["Académico / Informe", "Examen"], horizontal=False)
    forma = st.sidebar.radio("Forma de entrada", ["Expresión", "Coeficientes"], index=0 if p.get("forma", "Expresion") == "Expresion" else 1)
    variable_default = p.get("variable", "x")
    variable_poly = st.sidebar.text_input("Variable por defecto", value=variable_default)
    if forma == "Expresión":
        entrada = st.sidebar.text_area("Polinomio", value=p.get("polinomio", "x**4 - 3*x**2 + 2"), height=88)
    else:
        entrada = st.sidebar.text_area("Coeficientes", value=p.get("coeficientes", "1,0,-3,0,2"), height=88)
    st.sidebar.markdown("### Aproximaciones iniciales")
    x0_txt = st.sidebar.text_input("x0", value=p.get("x0", "0"))
    x1_txt = st.sidebar.text_input("x1", value=p.get("x1", "0.5"))
    x2_txt = st.sidebar.text_input("x2", value=p.get("x2", "1.0"))
    tolerancia = st.sidebar.number_input("Épsilon absoluto", min_value=1e-12, value=float(p.get("tol", 1e-5)), format="%.12f")
    max_iter = st.sidebar.number_input("Máximo de iteraciones", min_value=1, max_value=500, value=int(p.get("max_iter", 50)), step=1)
    analizar_circulo = st.sidebar.checkbox("Análisis de estabilidad |z| < 1", value=bool(p.get("estabilidad", False)))
    resolver = st.sidebar.button("Resolver análisis")

    encabezado_seccion(
        "Müller y análisis de polinomios",
        "Entrada flexible por expresión o coeficientes, pasos teóricos, deflación y estabilidad.",
    )
    if resolver:
        try:
            if forma == "Expresión":
                coeficientes, simbolo, poly = parsear_polinomio_expresion(entrada, variable_poly)
            else:
                coeficientes, simbolo, poly = parsear_coeficientes(entrada, variable_poly)
            grado = len(coeficientes) - 1
            x0_val, x1_val, x2_val = parsear_numero(x0_txt), parsear_numero(x1_txt), parsear_numero(x2_txt)

            filas_muller, resumen_muller = muller(coeficientes, x0_val, x1_val, x2_val, tol=tolerancia, max_iter=int(max_iter))
            raices, procesos, tabla = todas_raices_muller(coeficientes, x0_val, x1_val, x2_val, tol=tolerancia, max_iter=int(max_iter))
            tabla = tabla_raices(coeficientes, raices, estabilidad=analizar_circulo)
            estabilidad_tabla, estable = analizar_estabilidad(raices)
            desc = regla_descartes(coeficientes)
            lag = cotas_lagrange(coeficientes)
            pasos_h, valor_h = pasos_horner(coeficientes, x2_val)

            encabezado_seccion("Resumen del polinomio", "Datos detectados y primera raíz calculada por Müller.")
            cols = st.columns(4)
            tarjeta(cols[0], "Grado", grado)
            tarjeta(cols[1], "Raíces esperadas", grado)
            tarjeta(cols[2], "Primera raíz", formato_complejo(resumen_muller["raiz"]))
            tarjeta(cols[3], "Estado", "Convergió" if resumen_muller["convergio"] else "Máx. iter.", "ok" if resumen_muller["convergio"] else "warn")
            st.info(criterio_parada_muller(filas_muller, tolerancia))

            coef_visibles = [complex(c).real if abs(complex(c).imag) < 1e-12 else c for c in coeficientes]
            st.markdown(
                f"""
                <div class="panel">
                    <p class="section-title">Entrada interpretada</p>
                    <p class="section-sub">Variable detectada: <b>{simbolo}</b></p>
                    <p class="section-sub">Coeficientes: <code>{coef_visibles}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sp.latex(poly.as_expr()))

            if modo == "Examen":
                tabs = st.tabs(["Resultado", "Tabla de Müller", "Raíces", "Análisis completo"])
            else:
                tabs = st.tabs(["Resumen", "Descartes y Lagrange", "Horner", "Müller", "Deflación", "Estabilidad"])
            df_m_principal, df_m_detalle = tablas_muller(filas_muller)

            if modo == "Examen":
                with tabs[0]:
                    encabezado_seccion("Resultado", "Raíces, residuos y plano complejo para consulta rápida.")
                    st.dataframe(dataframe_limpio(tabla), width="stretch", hide_index=True)
                    st.plotly_chart(
                        grafica_raices(raices, analizar_circulo),
                        width="stretch",
                        key="grafica_raices_examen_resultado",
                    )
                with tabs[1]:
                    encabezado_seccion("Tabla de Müller", "Iteraciones principales y detalle del denominador estable.")
                    st.dataframe(df_m_principal, width="stretch", hide_index=True)
                    boton_csv(df_m_principal, "tabla_muller.csv")
                    with st.expander("Ver detalle matemático de la iteración"):
                        st.dataframe(df_m_detalle, width="stretch", hide_index=True)
                with tabs[2]:
                    encabezado_seccion("Raíces verificadas", "Valores encontrados con partes real e imaginaria, módulo y residuo.")
                    st.dataframe(dataframe_limpio(tabla), width="stretch", hide_index=True)
                with tabs[3]:
                    st.write("Abre el modo Académico / Informe para ver todos los pasos separados.")
            else:
                with tabs[0]:
                    encabezado_seccion("Raíces y plano complejo", "Resumen listo para captura del informe.")
                    st.dataframe(dataframe_limpio(tabla), width="stretch", hide_index=True)
                    if analizar_circulo:
                        if estable:
                            st.success("Todas las raíces o polos tienen módulo menor que 1 y se encuentran dentro del círculo unitario del plano z. Por lo tanto, el filtro digital IIR es estable.")
                        else:
                            st.warning("Al menos una raíz o polo se encuentra sobre o fuera del círculo unitario del plano z. Por lo tanto, el filtro digital IIR es inestable.")
                    else:
                        st.info(f"Se encontraron {len(raices)} raíces del polinomio y se verificaron sustituyéndolas en la ecuación original.")
                    st.plotly_chart(
                        grafica_raices(raices, analizar_circulo),
                        width="stretch",
                        key="grafica_raices_academico_resumen",
                    )
                with tabs[1]:
                    encabezado_seccion("Descartes y Lagrange", "Conteo de signos y cotas calculadas sin usar raíces numéricas.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### DESCARTES")
                        st.write(f"**P({simbolo})**")
                        st.dataframe(pd.DataFrame({"Coeficientes": desc["coeficientes_px"], "Signos": desc["signos_px"]}), width="stretch", hide_index=True)
                        st.write(f"V+ = **{desc['cambios_positivos']}**")
                        st.write(f"Raíces positivas posibles: **{formatear_posibilidades(desc['positivas_posibles'])}**")
                        st.write(f"**P(-{simbolo})**")
                        st.dataframe(pd.DataFrame({"Coeficientes": desc["coeficientes_p_menos_x"], "Signos": desc["signos_p_menos_x"]}), width="stretch", hide_index=True)
                        st.write(f"V- = **{desc['cambios_negativos']}**")
                        st.write(f"Raíces negativas posibles: **{formatear_posibilidades(desc['negativas_posibles'])}**")
                    with c2:
                        st.markdown("#### LAGRANGE")
                        st.write(f"Coeficiente principal: **{formatear_decimal(lag['positiva']['an'])}**")
                        st.write(f"K: **{formatear_decimal(lag['positiva']['K'])}**")
                        st.write(f"k: **{lag['positiva']['k']}**")
                        st.write("Fórmula utilizada:")
                        st.code(lag["positiva"]["formula"])
                        st.write(f"Cota positiva: **{lag['cota_positiva']:.10g}**")
                        st.write(f"Cota negativa: **{lag['cota_negativa']:.10g}**")
                        st.write(f"Cota global: **{lag['cota_global']:.10g}**")
                        st.code(f"Para P(-{simbolo}): " + lag["negativa"]["formula"])
                        st.dataframe(pd.DataFrame(lag["positiva"]["negativos"]), width="stretch", hide_index=True)
                with tabs[2]:
                    encabezado_seccion("Horner", "Representación anidada y evaluación eficiente del polinomio.")
                    st.markdown("**Polinomio original**")
                    st.latex(sp.latex(poly.as_expr()))
                    st.markdown("**Representación mediante Horner**")
                    st.code(f"P({simbolo}) = {expresion_horner(coeficientes, str(simbolo))}")
                    st.markdown("**Ejemplo de evaluación**")
                    st.write(f"Evaluado en {simbolo} = {x2_txt}")
                    st.write(f"Resultado: **{formato_complejo(valor_h)}**")
                    st.dataframe(dataframe_limpio(pasos_h), width="stretch", hide_index=True)
                    st.markdown("**Verificación de P(raíz)**")
                    st.write(f"Horner interno verifica P(raíz) = `{formato_complejo(horner(coeficientes, resumen_muller['raiz']))}`")
                with tabs[3]:
                    encabezado_seccion("Müller", "Tabla de convergencia con discriminante, denominador y error aproximado.")
                    st.info(criterio_parada_muller(filas_muller, tolerancia))
                    st.dataframe(df_m_principal, width="stretch", hide_index=True)
                    boton_csv(df_m_principal, "tabla_muller.csv")
                    with st.expander("Ver detalle matemático de la iteración"):
                        st.dataframe(df_m_detalle, width="stretch", hide_index=True)
                    errores = [fila["Ea (%)"] for fila in filas_muller if fila["Ea (%)"] is not None]
                    if errores:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(range(1, len(errores) + 1)), y=errores, mode="lines+markers", line=dict(color="#59d5ff")))
                        fig.add_hline(y=tolerancia, line_dash="dash", line_color="#ff5f9f")
                        fig.update_layout(height=360, template="plotly_dark", xaxis_title="Iteración", yaxis_title="Ea (%)")
                        st.plotly_chart(fig, width="stretch", key="grafica_muller_error")
                with tabs[4]:
                    encabezado_seccion("Deflación", "Reducción del polinomio después de cada raíz encontrada.")
                    resumen_def = []
                    coef_antes = list(coeficientes)
                    for proceso in procesos:
                        resumen_def.append({
                            "Grado antes": proceso["grado"],
                            "Raíz": formato_complejo(proceso["raiz"]),
                            "Residuo Müller": f"{proceso['residuo_muller']:.6e}",
                            "Residuo deflación": formato_complejo(proceso["residuo_deflacion"]),
                            "Coeficientes resultantes": formato_lista_numeros(proceso["coeficientes_resultantes"]),
                        })
                        coef_antes = proceso["coeficientes_resultantes"]
                    st.dataframe(pd.DataFrame(resumen_def), width="stretch", hide_index=True)
                    coef_antes = list(coeficientes)
                    for i, proceso in enumerate(procesos, start=1):
                        coef_despues = proceso["coeficientes_resultantes"]
                        with st.expander(f"Deflación {i}", expanded=i == 1):
                            st.markdown(f"**Grado anterior:** {proceso['grado']}")
                            st.markdown(f"**Raíz utilizada:** `{formato_complejo(proceso['raiz'])}`")
                            st.markdown("**Polinomio antes**")
                            st.code(f"P({simbolo}) = {polinomio_desde_coeficientes(coef_antes, str(simbolo))}")
                            st.markdown(f"**Coeficientes antes:** `{formato_lista_numeros(coef_antes)}`")
                            st.markdown(f"**Coeficientes resultantes:** `{formato_lista_numeros(coef_despues)}`")
                            st.markdown("**Polinomio reducido Q(z)**")
                            st.code(f"Q({simbolo}) = {polinomio_desde_coeficientes(coef_despues, str(simbolo))}")
                            st.markdown(f"**Residuo de la deflación:** `{formato_complejo(proceso['residuo_deflacion'])}`")
                            st.dataframe(dataframe_limpio(proceso["pasos_deflacion"]), width="stretch", hide_index=True)
                        coef_antes = coef_despues
                with tabs[5]:
                    encabezado_seccion("Estabilidad", "Módulos de las raíces frente al círculo unitario.")
                    st.dataframe(dataframe_limpio(estabilidad_tabla), width="stretch", hide_index=True)
                    st.plotly_chart(
                        grafica_raices(raices, True),
                        width="stretch",
                        key="grafica_raices_estabilidad",
                    )
                    if analizar_circulo and estable:
                        st.success("Todas las raíces o polos tienen módulo menor que 1 y se encuentran dentro del círculo unitario del plano z. Por lo tanto, el filtro digital IIR es estable.")
                    elif analizar_circulo:
                        st.warning("Al menos una raíz o polo se encuentra sobre o fuera del círculo unitario del plano z. Por lo tanto, el filtro digital IIR es inestable.")
                    else:
                        st.info("Activa el análisis de estabilidad para interpretar el círculo unitario.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ocurrió un error: {exc}")
    else:
        st.info("Elige una forma de entrada, ajusta los valores iniciales y presiona Resolver análisis.")

with st.expander("Guía rápida de entrada"):
    st.markdown(
        """
| Matemática | Escribir |
|---|---|
| x² | `x**2` |
| x³ | `x**3` |
| eˣ | `exp(x)` |
| √x | `sqrt(x)` |
| i | `I` o `i` en valores iniciales |

Ejemplos: `x**3-x-2`, `exp(-x)-x`, `8*z**4-6*z**3-3*z**2+3*z-1`.
"""
    )
