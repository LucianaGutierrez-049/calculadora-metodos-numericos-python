import cmath
import math


EPS = 1e-12


def limpiar_numero(valor, tol=1e-10):
    z = complex(valor)
    real = 0.0 if abs(z.real) < tol else z.real
    imag = 0.0 if abs(z.imag) < tol else z.imag
    if imag == 0.0:
        return float(real)
    return complex(real, imag)


def formato_complejo(valor, decimales=5):
    z = complex(valor)
    z = complex(limpiar_numero(z))
    if abs(z.imag) < 10 ** (-(decimales - 2)):
        return f"{z.real:.{decimales}f}"
    signo = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{decimales}f} {signo} {abs(z.imag):.{decimales}f}i"


def validar_coeficientes(coeficientes, grado_minimo=1):
    if not coeficientes:
        raise ValueError("Debes ingresar al menos un coeficiente.")

    coef = [complex(c) for c in coeficientes]

    while len(coef) > 1 and abs(coef[0]) < EPS:
        coef.pop(0)

    if abs(coef[0]) < EPS:
        raise ValueError("El coeficiente principal no puede ser cero.")

    if len(coef) - 1 < grado_minimo:
        raise ValueError(f"El polinomio debe tener grado mínimo {grado_minimo}.")

    return [limpiar_numero(c) for c in coef]


def signos_coeficientes(coeficientes):
    signos = []
    for coef in coeficientes:
        c = complex(coef)
        if abs(c) < EPS:
            signos.append("0")
        elif c.real > 0:
            signos.append("+")
        else:
            signos.append("-")
    return signos


def cambios_signo(coeficientes):
    valores = []
    for coef in coeficientes:
        c = complex(coef)
        if abs(c) > EPS:
            valores.append(1 if c.real > 0 else -1)

    return sum(1 for i in range(len(valores) - 1) if valores[i] != valores[i + 1])


def posibilidades_descartes(cambios):
    return list(range(cambios, -1, -2))


def coeficientes_p_menos_x(coeficientes):
    grado = len(coeficientes) - 1
    resultado = []
    for i, coef in enumerate(coeficientes):
        potencia = grado - i
        resultado.append(coef if potencia % 2 == 0 else -complex(coef))
    return [limpiar_numero(c) for c in resultado]


def regla_descartes(coeficientes):
    coef = validar_coeficientes(coeficientes)
    coef_neg = coeficientes_p_menos_x(coef)
    v_pos = cambios_signo(coef)
    v_neg = cambios_signo(coef_neg)

    return {
        "coeficientes_px": coef,
        "signos_px": signos_coeficientes(coef),
        "cambios_positivos": v_pos,
        "positivas_posibles": posibilidades_descartes(v_pos),
        "coeficientes_p_menos_x": coef_neg,
        "signos_p_menos_x": signos_coeficientes(coef_neg),
        "cambios_negativos": v_neg,
        "negativas_posibles": posibilidades_descartes(v_neg),
    }


def detalle_lagrange_positiva(coeficientes):
    coef = [float(complex(c).real) for c in validar_coeficientes(coeficientes)]
    if coef[0] < 0:
        coef = [-c for c in coef]

    an = coef[0]
    negativos = []
    for posicion, c in enumerate(coef[1:], start=1):
        if c < 0:
            negativos.append({"posicion": posicion, "coeficiente": c, "abs": abs(c)})

    if not negativos:
        return {
            "coeficientes": coef,
            "an": an,
            "negativos": [],
            "K": 0.0,
            "k": None,
            "cota": 1.0,
            "formula": "No hay coeficientes negativos; se usa B = 1.",
        }

    k = negativos[0]["posicion"]
    K = max(item["abs"] for item in negativos)
    cota = 1 + (K / an) ** (1 / k)

    return {
        "coeficientes": coef,
        "an": an,
        "negativos": negativos,
        "K": K,
        "k": k,
        "cota": cota,
        "formula": f"B = 1 + ({K:g}/{an:g})^(1/{k}) = {cota:g}",
    }


def cotas_lagrange(coeficientes):
    coef = validar_coeficientes(coeficientes)
    positiva = detalle_lagrange_positiva(coef)
    coef_neg = coeficientes_p_menos_x(coef)
    negativa = detalle_lagrange_positiva(coef_neg)

    return {
        "positiva": positiva,
        "negativa": negativa,
        "cota_positiva": positiva["cota"],
        "cota_negativa": -negativa["cota"],
        "cota_global": max(positiva["cota"], negativa["cota"]),
        "coeficientes_p_menos_x": coef_neg,
    }


def horner(coeficientes, z):
    coef = validar_coeficientes(coeficientes)
    resultado = complex(coef[0])
    for c in coef[1:]:
        resultado = resultado * z + complex(c)
    return limpiar_numero(resultado)


def pasos_horner(coeficientes, z):
    coef = validar_coeficientes(coeficientes)
    acumulado = complex(coef[0])
    pasos = [{"coeficiente": limpiar_numero(coef[0]), "acumulado": limpiar_numero(acumulado)}]
    for c in coef[1:]:
        anterior = acumulado
        acumulado = acumulado * z + complex(c)
        pasos.append({
            "anterior": limpiar_numero(anterior),
            "z": limpiar_numero(z),
            "coeficiente": limpiar_numero(c),
            "acumulado": limpiar_numero(acumulado),
        })
    return pasos, limpiar_numero(acumulado)


def expresion_horner(coeficientes, variable="x"):
    coef = validar_coeficientes(coeficientes)
    expr = f"{coef[0]:g}"
    for c in coef[1:]:
        signo = "+" if complex(c).real >= 0 else "-"
        expr = f"({expr}{variable} {signo} {abs(complex(c).real):g})"
    return expr


def error_relativo_complejo(nuevo, anterior):
    if anterior is None or abs(nuevo) < EPS:
        return None
    return abs((nuevo - anterior) / nuevo) * 100


def muller(coeficientes, x0, x1, x2, tol=1e-5, max_iter=50):
    coef = validar_coeficientes(coeficientes)
    x0, x1, x2 = complex(x0), complex(x1), complex(x2)

    if min(abs(x1 - x0), abs(x2 - x1), abs(x2 - x0)) < EPS:
        raise ValueError("x0, x1 y x2 deben ser diferentes.")

    filas = []
    x3 = x2
    fx3 = horner(coef, x3)
    ea = None

    for iteracion in range(1, max_iter + 1):
        f0, f1, f2 = complex(horner(coef, x0)), complex(horner(coef, x1)), complex(horner(coef, x2))
        h0, h1 = x1 - x0, x2 - x1

        if abs(h0) < EPS or abs(h1) < EPS:
            raise ValueError("Dos aproximaciones consecutivas son demasiado cercanas.")

        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1

        if abs(h0 + h1) < EPS:
            raise ValueError("No se puede calcular a porque h0 + h1 es casi cero.")

        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2
        discriminante = b * b - 4 * a * c
        raiz_disc = cmath.sqrt(discriminante)
        den1, den2 = b + raiz_disc, b - raiz_disc
        denominador = den1 if abs(den1) >= abs(den2) else den2

        if abs(denominador) < EPS:
            raise ValueError("El denominador de Müller es aproximadamente cero.")

        x3 = x2 - (2 * c / denominador)
        if not (math.isfinite(x3.real) and math.isfinite(x3.imag)):
            raise ValueError("Müller produjo un valor no finito.")

        fx3 = complex(horner(coef, x3))
        error_abs = abs(x3 - x2)
        ea = error_relativo_complejo(x3, x2)

        filas.append({
            "Iteracion": iteracion,
            "x0": limpiar_numero(x0),
            "x1": limpiar_numero(x1),
            "x2": limpiar_numero(x2),
            "x3": limpiar_numero(x3),
            "P(x3)": limpiar_numero(fx3),
            "Ea (%)": ea,
            "h0": limpiar_numero(h0),
            "h1": limpiar_numero(h1),
            "delta0": limpiar_numero(delta0),
            "delta1": limpiar_numero(delta1),
            "a": limpiar_numero(a),
            "b": limpiar_numero(b),
            "c": limpiar_numero(c),
            "discriminante": limpiar_numero(discriminante),
            "sqrt(discriminante)": limpiar_numero(raiz_disc),
            "denominador": limpiar_numero(denominador),
            "|x3-x2|": error_abs,
        })

        if error_abs < tol or abs(fx3) < tol:
            return filas, {
                "raiz": limpiar_numero(x3),
                "iteraciones": iteracion,
                "error_final": ea,
                "residuo": abs(fx3),
                "convergio": True,
            }

        x0, x1, x2 = x1, x2, x3

    return filas, {
        "raiz": limpiar_numero(x3),
        "iteraciones": max_iter,
        "error_final": ea,
        "residuo": abs(fx3),
        "convergio": False,
    }


def deflacion_sintetica(coeficientes, raiz):
    coef = validar_coeficientes(coeficientes)
    r = complex(raiz)
    nuevos = [complex(coef[0])]
    pasos = [{"coeficiente": coef[0], "b": limpiar_numero(nuevos[0])}]

    for c in coef[1:-1]:
        b = complex(c) + r * nuevos[-1]
        nuevos.append(b)
        pasos.append({"coeficiente": c, "b": limpiar_numero(b)})

    residuo = complex(coef[-1]) + r * nuevos[-1]
    return [limpiar_numero(c) for c in nuevos], limpiar_numero(residuo), pasos


def resolver_cuadratico(coeficientes):
    a, b, c = [complex(v) for v in validar_coeficientes(coeficientes)]
    disc = b * b - 4 * a * c
    raiz = cmath.sqrt(disc)
    return [limpiar_numero((-b + raiz) / (2 * a)), limpiar_numero((-b - raiz) / (2 * a))]


def semillas_muller(base0, base1, base2, intento):
    if intento == 0:
        return complex(base0), complex(base1), complex(base2)
    ang = 2 * math.pi * intento / 9
    radio = 0.45 + 0.2 * intento
    centro = complex(base2) + complex(math.cos(ang), math.sin(ang)) * radio
    return centro - (0.4 + 0.15j), centro + (0.2 - 0.25j), centro + (0.55 + 0.35j)


def todas_raices_muller(coeficientes, x0=0, x1=0.5, x2=1.0, tol=1e-5, max_iter=50):
    coef_actual = [complex(c) for c in validar_coeficientes(coeficientes, grado_minimo=2)]
    originales = list(coef_actual)
    raices = []
    procesos = []

    while len(coef_actual) > 3:
        ultimo_error = None
        elegido = None
        for intento in range(10):
            s0, s1, s2 = semillas_muller(x0, x1, x2, intento)
            try:
                filas, resumen = muller(coef_actual, s0, s1, s2, tol=tol, max_iter=max_iter)
                if resumen["convergio"] or resumen["residuo"] < max(1e-4, tol * 100):
                    elegido = (filas, resumen, intento)
                    break
                ultimo_error = f"No convergió con intento {intento}."
            except ValueError as exc:
                ultimo_error = str(exc)

        if elegido is None:
            raise ValueError(f"No fue posible encontrar una raiz para deflactar. {ultimo_error}")

        filas, resumen, intento = elegido
        raiz = complex(resumen["raiz"])
        nuevos, residuo_deflacion, pasos = deflacion_sintetica(coef_actual, raiz)
        raices.append(limpiar_numero(raiz))
        procesos.append({
            "grado": len(coef_actual) - 1,
            "raiz": limpiar_numero(raiz),
            "intento": intento,
            "residuo_muller": resumen["residuo"],
            "residuo_deflacion": residuo_deflacion,
            "iteraciones": filas,
            "pasos_deflacion": pasos,
            "coeficientes_resultantes": nuevos,
        })

        coef_actual = [complex(c) for c in nuevos]
        x0, x1, x2 = raiz - 0.37 - 0.21j, raiz + 0.19 + 0.31j, raiz + 0.63 - 0.17j

    if len(coef_actual) == 3:
        raices.extend(resolver_cuadratico(coef_actual))
    elif len(coef_actual) == 2:
        a, b = coef_actual
        raices.append(limpiar_numero(-b / a))

    tabla = tabla_raices(originales, raices)
    return raices, procesos, tabla


def tabla_raices(coeficientes, raices, estabilidad=False):
    filas = []
    for i, raiz in enumerate(raices, start=1):
        z = complex(raiz)
        residuo = abs(complex(horner(coeficientes, z)))
        tipo = "Real" if abs(z.imag) < 1e-8 else "Compleja"
        fila = {
            "Raíz": f"z{i}",
            "Valor": formato_complejo(z),
            "Re": z.real,
            "Im": z.imag,
            "|z|": abs(z),
            "|P(z)|": residuo,
            "Tipo": tipo,
        }
        if estabilidad:
            fila["Dentro de |z| < 1"] = "Sí" if abs(z) < 1 else "No"
        filas.append(fila)
    return filas


def analizar_estabilidad(raices):
    filas = []
    estable = True
    for i, raiz in enumerate(raices, start=1):
        z = complex(raiz)
        dentro = abs(z) < 1
        estable = estable and dentro
        filas.append({
            "Raíz": f"z{i}",
            "Valor": formato_complejo(z),
            "|z|": abs(z),
            "Dentro del círculo unitario": "Sí" if dentro else "No",
        })
    return filas, estable
