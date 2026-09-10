# Calculadora de Métodos Numéricos

Aplicación en Streamlit para resolver ejercicios de métodos numéricos y analizar raíces de polinomios. Incluye presets del curso, pero también permite escribir ejercicios propios para práctica, informes y exámenes.

## Métodos

- Bisección
- Falsa Posición
- Punto Fijo
- Newton-Raphson
- Secante
- Müller

## Análisis de polinomios

El módulo de polinomios incluye:

- Regla de Descartes
- Cotas de Lagrange
- Evaluación por Horner
- Método de Müller
- Raíces complejas
- Deflación sintética
- Verificación de residuos
- Módulos de raíces
- Plano complejo
- Análisis de estabilidad con el círculo unitario

## Modo examen

La calculadora no está limitada a los ejercicios precargados. Puedes introducir un polinomio propio en dos formas:

- Expresión: `x**4 - 3*x**2 + 2`
- Coeficientes: `1,0,-3,0,2`

La aplicación detecta la variable, el grado, los coeficientes y las potencias faltantes. Los valores iniciales, tolerancia y número máximo de iteraciones son editables.

## Cómo ejecutar

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Ejemplos

```text
x**3-x-2
exp(-x)-x
8*z**4-6*z**3-3*z**2+3*z-1
x**4 + 1
(x-1)**3*(x+2)
```

## Preset de Sesión 4

Selecciona:

```text
Sesión 4 · Müller · Filtro digital IIR
```

El preset carga automáticamente:

```text
D(z) = 8*z**4 - 6*z**3 - 3*z**2 + 3*z - 1
z0 = 0
z1 = 0.5
z2 = 1.0
epsilon = 1e-5
max_iter = 50
```

El análisis de estabilidad verifica si todas las raíces cumplen `|z_i| < 1`.
