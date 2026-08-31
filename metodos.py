import numpy as np


# =========================================================
# UTILIDADES
# =========================================================

def evaluar_seguro(funcion, x):
    try:
        valor = funcion(x)

        if np.iscomplexobj(valor):
            return np.nan

        valor = float(valor)

        if not np.isfinite(valor):
            return np.nan

        return valor

    except Exception:
        return np.nan


def error_relativo_porcentual(nuevo, anterior):
    if anterior is None:
        return None

    if nuevo == 0:
        return None

    return abs((nuevo - anterior) / nuevo) * 100


# =========================================================
# BISECCIÓN
# =========================================================

def biseccion(f_num, a, b, tol=0.5, max_iter=50):
    filas = []

    fa = evaluar_seguro(f_num, a)
    fb = evaluar_seguro(f_num, b)

    if np.isnan(fa) or np.isnan(fb):
        return [], {
            "error_msg": "f(a) o f(b) está fuera del dominio."
        }

    if fa * fb > 0:
        return [], {
            "error_msg": (
                f"No existe cambio de signo en [{a}, {b}]. "
                f"f(a)={fa:.6f}, f(b)={fb:.6f}"
            )
        }

    xr_anterior = None
    convergio = False

    for i in range(1, max_iter + 1):
        fa = evaluar_seguro(f_num, a)
        fb = evaluar_seguro(f_num, b)

        xr = (a + b) / 2
        fxr = evaluar_seguro(f_num, xr)

        if np.isnan(fxr):
            return filas, {
                "error_msg": "f(xr) salió del dominio."
            }

        producto = fa * fxr
        err = error_relativo_porcentual(xr, xr_anterior)

        filas.append({
            "k": i,
            "xl": a,
            "xu": b,
            "xr": xr,
            "f(xl)": fa,
            "f(xu)": fb,
            "f(xr)": fxr,
            "f(xl)·f(xr)": producto,
            "Ea (%)": err
        })

        if fxr == 0:
            convergio = True
            break

        if err is not None and err <= tol:
            convergio = True
            break

        if producto < 0:
            b = xr
        else:
            a = xr

        xr_anterior = xr

    ultima = filas[-1]

    return filas, {
        "raiz": ultima["xr"],
        "f_raiz": ultima["f(xr)"],
        "iteraciones": len(filas),
        "convergio": convergio,
        "error_final": ultima["Ea (%)"]
    }


# =========================================================
# FALSA POSICIÓN
# =========================================================

def falsa_posicion(f_num, a, b, tol=0.5, max_iter=50):
    filas = []

    fa = evaluar_seguro(f_num, a)
    fb = evaluar_seguro(f_num, b)

    if np.isnan(fa) or np.isnan(fb):
        return [], {
            "error_msg": "f(a) o f(b) está fuera del dominio."
        }

    if fa * fb > 0:
        return [], {
            "error_msg": (
                f"No existe cambio de signo en [{a}, {b}]. "
                f"f(a)={fa:.6f}, f(b)={fb:.6f}"
            )
        }

    xr_anterior = None
    convergio = False

    for i in range(1, max_iter + 1):
        fa = evaluar_seguro(f_num, a)
        fb = evaluar_seguro(f_num, b)

        denominador = fa - fb

        if abs(denominador) < 1e-14:
            return filas, {
                "error_msg": "El denominador es aproximadamente cero."
            }

        xr = b - (fb * (a - b)) / denominador
        fxr = evaluar_seguro(f_num, xr)

        if np.isnan(fxr):
            return filas, {
                "error_msg": "f(xr) salió del dominio."
            }

        producto = fa * fxr
        err = error_relativo_porcentual(xr, xr_anterior)

        filas.append({
            "k": i,
            "xl": a,
            "xu": b,
            "xr": xr,
            "f(xl)": fa,
            "f(xu)": fb,
            "f(xr)": fxr,
            "f(xl)·f(xr)": producto,
            "Ea (%)": err
        })

        if fxr == 0:
            convergio = True
            break

        if err is not None and err <= tol:
            convergio = True
            break

        if producto < 0:
            b = xr
        else:
            a = xr

        xr_anterior = xr

    ultima = filas[-1]

    return filas, {
        "raiz": ultima["xr"],
        "f_raiz": ultima["f(xr)"],
        "iteraciones": len(filas),
        "convergio": convergio,
        "error_final": ultima["Ea (%)"]
    }


# =========================================================
# PUNTO FIJO
# =========================================================

def punto_fijo(g_num, x0, tol=0.01, max_iter=50, f_num=None):
    filas = []

    x_actual = x0
    convergio = False

    for i in range(1, max_iter + 1):
        x_siguiente = evaluar_seguro(g_num, x_actual)

        if np.isnan(x_siguiente) or abs(x_siguiente) > 1e12:
            return filas, {
                "error_msg": "La iteración diverge o salió del dominio de g(x).",
                "convergio": False
            }

        error_abs = abs(x_siguiente - x_actual)
        err = error_relativo_porcentual(x_siguiente, x_actual)

        if f_num is not None:
            fx = evaluar_seguro(f_num, x_siguiente)
        else:
            gx = evaluar_seguro(g_num, x_siguiente)
            fx = gx - x_siguiente

        filas.append({
            "n": i - 1,
            "x_n": x_actual,
            "g(x_n)=x_n+1": x_siguiente,
            "f(x_n+1)": fx,
            "|x_n+1-x_n|": error_abs,
            "Ea (%)": err
        })

        if err is not None and err <= tol:
            convergio = True
            x_actual = x_siguiente
            break

        x_actual = x_siguiente

    ultima = filas[-1]

    return filas, {
        "raiz": x_actual,
        "f_raiz": ultima["f(x_n+1)"],
        "iteraciones": len(filas),
        "convergio": convergio,
        "error_final": ultima["Ea (%)"]
    }


# =========================================================
# NEWTON-RAPHSON
# =========================================================

def newton_raphson(f_num, fprime_num, x0, tol=0.01, max_iter=50):
    filas = []

    x_actual = x0
    convergio = False

    for i in range(1, max_iter + 1):
        fx = evaluar_seguro(f_num, x_actual)
        fpx = evaluar_seguro(fprime_num, x_actual)

        if np.isnan(fx) or np.isnan(fpx):
            return filas, {
                "error_msg": "f(x) o f'(x) salió del dominio.",
                "convergio": False
            }

        if abs(fpx) < 1e-14:
            return filas, {
                "error_msg": "f'(x_n) es aproximadamente cero. No se puede continuar.",
                "convergio": False
            }

        x_siguiente = x_actual - fx / fpx
        err = error_relativo_porcentual(x_siguiente, x_actual)

        filas.append({
            "n": i - 1,
            "x_n": x_actual,
            "f(x_n)": fx,
            "f'(x_n)": fpx,
            "x_n+1": x_siguiente,
            "Ea (%)": err
        })

        if err is not None and err <= tol:
            convergio = True
            x_actual = x_siguiente
            break

        x_actual = x_siguiente

    return filas, {
        "raiz": x_actual,
        "f_raiz": evaluar_seguro(f_num, x_actual),
        "iteraciones": len(filas),
        "convergio": convergio,
        "error_final": filas[-1]["Ea (%)"]
    }


# =========================================================
# SECANTE
# =========================================================

def secante(f_num, x0, x1, tol=0.01, max_iter=50):
    filas = []

    x_prev = x0
    x_curr = x1
    convergio = False

    for i in range(1, max_iter + 1):
        f_prev = evaluar_seguro(f_num, x_prev)
        f_curr = evaluar_seguro(f_num, x_curr)

        if np.isnan(f_prev) or np.isnan(f_curr):
            return filas, {
                "error_msg": "La función salió del dominio.",
                "convergio": False
            }

        denominador = f_curr - f_prev

        if abs(denominador) < 1e-14:
            return filas, {
                "error_msg": "El denominador es aproximadamente cero.",
                "convergio": False
            }

        x_next = x_curr - f_curr * (x_curr - x_prev) / denominador
        err = error_relativo_porcentual(x_next, x_curr)

        filas.append({
            "n": i - 1,
            "x_n-1": x_prev,
            "x_n": x_curr,
            "f(x_n-1)": f_prev,
            "f(x_n)": f_curr,
            "x_n+1": x_next,
            "Ea (%)": err
        })

        if err is not None and err <= tol:
            convergio = True
            x_curr = x_next
            break

        x_prev = x_curr
        x_curr = x_next

    return filas, {
        "raiz": x_curr,
        "f_raiz": evaluar_seguro(f_num, x_curr),
        "iteraciones": len(filas),
        "convergio": convergio,
        "error_final": filas[-1]["Ea (%)"]
    }
