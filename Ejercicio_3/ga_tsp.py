"""Algoritmo genético para el TSP (funciones del Ejercicio 1, sin las pruebas del cuaderno)."""



import time
from itertools import permutations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
plt.rcParams["animation.embed_limit"] = 40

AZUL, ROJO, VERDE, GRIS = "#4C72B0", "#C44E52", "#55A868", "#999999"


def matriz_de_distancias(coordenadas, redondear=False):
    """Distancia euclidiana entre todas las parejas de ciudades."""
    coordenadas = np.asarray(coordenadas, dtype=float)
    diferencias = coordenadas[:, None, :] - coordenadas[None, :, :]
    matriz = np.sqrt((diferencias ** 2).sum(axis=2))
    return np.floor(matriz + 0.5) if redondear else matriz

def leer_archivo(ruta):
    """Lee las filas numéricas de un archivo de texto, CSV o TSPLIB."""
    filas = []
    for linea in Path(ruta).read_text().splitlines():
        partes = linea.replace(",", " ").replace(";", " ").split()
        try:
            numeros = [float(parte) for parte in partes]
        except ValueError:
            continue  # encabezados y líneas de texto
        if numeros:
            filas.append(numeros)
    return filas

def cargar_problema(entrada, tipo=None, redondear=False):
    """Devuelve (matriz de distancias, coordenadas).

    `entrada` puede ser la ruta de un archivo o un listado. Las coordenadas
    son None cuando la entrada es una matriz de distancias.
    """
    if isinstance(entrada, (str, Path)):
        entrada = leer_archivo(entrada)
    try:
        datos = np.asarray(entrada, dtype=float)
    except ValueError:
        raise ValueError("Todas las filas deben tener la misma cantidad de números.")
    if datos.ndim != 2:
        raise ValueError("La entrada debe ser una tabla de coordenadas o una matriz de distancias.")

    filas, columnas = datos.shape
    parece_matriz = (
        filas == columnas
        and np.allclose(datos, datos.T)
        and np.allclose(np.diag(datos), 0)
    )
    if tipo is None:
        tipo = "matriz" if parece_matriz else "coordenadas"

    if tipo == "matriz":
        if not parece_matriz:
            raise ValueError("La matriz debe ser cuadrada, simétrica y con ceros en la diagonal.")
        return datos, None
    if tipo == "coordenadas":
        if columnas == 3:
            datos = datos[:, 1:]  # la primera columna es el número de ciudad
        if datos.shape[1] != 2:
            raise ValueError("Las coordenadas deben tener dos columnas (x, y).")
        return matriz_de_distancias(datos, redondear), datos
    raise ValueError("tipo debe ser 'coordenadas' o 'matriz'.")

def posiciones_aproximadas(matriz):
    """Ubica las ciudades en el plano a partir de la matriz de distancias (MDS clásico)."""
    n = len(matriz)
    centrado = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * centrado @ (matriz ** 2) @ centrado
    valores, vectores = np.linalg.eigh(B)
    mayores = np.argsort(valores)[::-1][:2]
    return vectores[:, mayores] * np.sqrt(np.maximum(valores[mayores], 0))


def longitudes(poblacion, matriz):
    """Distancia total de cada recorrido, contando el regreso a la ciudad inicial."""
    poblacion = np.atleast_2d(poblacion)
    siguiente = np.roll(poblacion, -1, axis=1)
    return matriz[poblacion, siguiente].sum(axis=1)

def poblacion_inicial(N, n, rng):
    """N recorridos aleatorios sobre n ciudades."""
    return rng.permuted(np.tile(np.arange(n), (N, 1)), axis=1)


def pesos_por_ranking(N, f=None):
    """Peso de cada posición del ordenamiento (posición 0 = mejor individuo)."""
    posicion = np.arange(N)
    if callable(f):
        return np.asarray(f(posicion), dtype=float)
    if f in (None, "lineal"):
        return (N - posicion).astype(float)  # el mejor pesa N y el peor pesa 1
    if f == "exponencial":
        return 0.5 ** (10 * posicion / N)  # el peso baja a la mitad cada 10 % del ordenamiento
    raise ValueError("f debe ser 'lineal', 'exponencial' o una función.")

def pesos_por_fitness(distancias, f=None):
    """Peso de cada individuo según su distancia."""
    if callable(f):
        return np.asarray(f(distancias), dtype=float)
    if f in (None, "inversa"):
        return 1.0 / (distancias + 1e-12)  # fitness de clase, f = 1/D
    if f == "ventana":
        return distancias.max() - distancias + 1e-12  # fitness medido respecto al peor
    raise ValueError("f debe ser 'inversa', 'ventana' o una función.")

def seleccionar(distancias, cuantos, rng, seleccion="torneo", k=3, f=None, n_elite=2):
    """Devuelve las posiciones de `cuantos` individuos elegidos.

    Supone que `distancias` está ordenado de menor a mayor.
    """
    N = len(distancias)
    if seleccion == "torneo":
        participantes = rng.integers(0, N, size=(cuantos, k))
        return participantes.min(axis=1)  # la posición menor es la de menor distancia
    if seleccion == "elitismo":
        return rng.integers(0, max(n_elite, 2), size=cuantos)
    if seleccion == "ranking":
        pesos = pesos_por_ranking(N, f)
    elif seleccion == "fitness":
        pesos = pesos_por_fitness(distancias, f)
    else:
        raise ValueError("seleccion debe ser 'torneo', 'ranking', 'fitness' o 'elitismo'.")
    return rng.choice(N, size=cuantos, p=pesos / pesos.sum())


def cruce_ox(p1, p2, a, b):
    """Order Crossover: tramo [a, b) de p1 y el resto en el orden de p2."""
    n = len(p1)
    hijo = np.empty(n, dtype=p1.dtype)
    hijo[a:b] = p1[a:b]

    usado = np.zeros(n, dtype=bool)
    usado[p1[a:b]] = True
    orden_p2 = np.concatenate((p2[b:], p2[:b]))  # p2 leído desde el segundo corte
    faltantes = orden_p2[~usado[orden_p2]]

    hijo[b:] = faltantes[: n - b]
    hijo[:a] = faltantes[n - b:]
    return hijo

def cruce_pmx(p1, p2, a, b):
    """Partially Mapped Crossover: tramo [a, b) de p1 y el resto de p2."""
    n = len(p1)
    hijo = p2.copy()
    hijo[a:b] = p1[a:b]

    en_tramo = np.zeros(n, dtype=bool)
    en_tramo[p1[a:b]] = True
    mapa = np.empty(n, dtype=p1.dtype)
    mapa[p1[a:b]] = p2[a:b]  # ciudad del tramo de p1 -> ciudad de p2 en esa posición

    posiciones = np.arange(n)
    fuera_del_tramo = (posiciones < a) | (posiciones >= b)
    for i in posiciones[fuera_del_tramo & en_tramo[p2]]:  # posiciones con ciudad repetida
        ciudad = p2[i]
        while en_tramo[ciudad]:
            ciudad = mapa[ciudad]
        hijo[i] = ciudad
    return hijo

CRUCES = {"ox": cruce_ox, "pmx": cruce_pmx}


def mutar(recorrido, rng, tipo="inversion"):
    """Devuelve una copia mutada del recorrido."""
    nuevo = recorrido.copy()
    n = len(nuevo)
    i = rng.integers(0, n)
    j = (i + rng.integers(1, n)) % n  # siempre distinto de i
    i, j = min(i, j), max(i, j)

    if tipo == "inversion":
        nuevo[i:j + 1] = recorrido[i:j + 1][::-1]
    elif tipo == "intercambio":
        nuevo[i], nuevo[j] = recorrido[j], recorrido[i]
    elif tipo == "insercion":
        nuevo[i] = recorrido[j]
        nuevo[i + 1:j + 1] = recorrido[i:j]
    elif tipo == "mezcla":
        rng.shuffle(nuevo[i:j + 1])
    else:
        raise ValueError("mutacion debe ser 'inversion', 'intercambio', 'insercion' o 'mezcla'.")
    return nuevo


def forma_canonica(poblacion):
    """Escribe cada recorrido empezando en la ciudad 0 y en un sentido fijo."""
    N, n = poblacion.shape
    inicio = np.argmax(poblacion == 0, axis=1)
    columnas = (inicio[:, None] + np.arange(n)) % n
    canonica = np.take_along_axis(poblacion, columnas, axis=1)
    invertir = canonica[:, 1] > canonica[:, -1]
    canonica[invertir, 1:] = canonica[invertir, :0:-1]
    return canonica

def indices_de_clones(poblacion):
    """Posiciones de los recorridos que repiten a otro que aparece antes."""
    _, primeros = np.unique(forma_canonica(poblacion), axis=0, return_index=True)
    es_clon = np.ones(len(poblacion), dtype=bool)
    es_clon[primeros] = False
    return np.flatnonzero(es_clon)

def diversidad(poblacion, mejor):
    """Porcentaje promedio de aristas de cada individuo que no están en el mejor recorrido."""
    n = poblacion.shape[1]

    def aristas(recorridos):
        siguiente = np.roll(recorridos, -1, axis=-1)
        return np.minimum(recorridos, siguiente) * n + np.maximum(recorridos, siguiente)

    en_el_mejor = np.zeros(n * n, dtype=bool)
    en_el_mejor[aristas(mejor)] = True
    return 100 * (1 - en_el_mejor[aristas(poblacion)].mean())


def a_cantidad(valor, N):
    """Convierte un porcentaje entre 0 y 1, o un tamaño entero, en número de individuos."""
    return int(round(valor * N)) if 0 < valor < 1 else int(valor)

def algoritmo_genetico_tsp(entrada, N=200, maxIter=1000, seleccion="torneo", k=3, s=0.05,
                           f=None, c=0.70, m=0.25, cruce="ox", mutacion="inversion",
                           evitar_clones=True, tipo=None, redondear=False, semilla=None):
    """Algoritmo genético para el TSP.

    entrada        archivo o listado de coordenadas, o archivo o matriz de distancias
    N              tamaño de la población
    maxIter        número máximo de iteraciones (generaciones)
    seleccion      "torneo", "ranking", "fitness" o "elitismo"
    k              tamaño del torneo
    s              tamaño o porcentaje de sobrevivientes por elitismo
    f              distribución de probabilidad para "ranking" o "fitness"
    c              tamaño o porcentaje de la población creada por cruce
    m              tamaño o porcentaje de la población creada por mutación
    cruce          "ox" o "pmx"
    mutacion       "inversion", "intercambio", "insercion" o "mezcla"
    evitar_clones  si es True, los recorridos repetidos se mutan
    tipo           "coordenadas" o "matriz"; si es None se detecta
    redondear      redondea las distancias al entero más cercano (convención de TSPLIB)
    semilla        semilla del generador aleatorio

    Devuelve (best, D, historial).
    """
    matriz, coordenadas = cargar_problema(entrada, tipo, redondear)
    n = len(matriz)
    rng = np.random.default_rng(semilla)
    operador_cruce = CRUCES[cruce]

    n_elite, n_cruce, n_mutacion = (a_cantidad(valor, N) for valor in (s, c, m))
    n_inmigrantes = N - n_elite - n_cruce - n_mutacion
    if n_inmigrantes < 0:
        raise ValueError("s + c + m no puede superar el tamaño de la población.")

    # Población inicial, ordenada de menor a mayor distancia
    poblacion = poblacion_inicial(N, n, rng)
    distancias = longitudes(poblacion, matriz)
    orden = np.argsort(distancias)
    poblacion, distancias = poblacion[orden], distancias[orden]

    best, D = poblacion[0].copy(), distancias[0]
    mejor_por_generacion = [D]
    promedio_por_generacion = [distancias.mean()]
    diversidad_por_generacion = [diversidad(poblacion, best)]
    mejoras = [(0, best.copy(), float(D))]

    for generacion in range(1, maxIter + 1):
        nueva = np.empty_like(poblacion)

        # 1. Sobrevivientes por elitismo
        nueva[:n_elite] = poblacion[:n_elite]

        # 2. Selección de los padres para el cruce y de los individuos que se mutan
        elegidos = seleccionar(distancias, 2 * n_cruce + n_mutacion, rng, seleccion, k, f, n_elite)
        padres_1 = elegidos[:n_cruce]
        padres_2 = elegidos[n_cruce:2 * n_cruce]
        repetidos = padres_1 == padres_2
        padres_2[repetidos] = (padres_2[repetidos] + 1) % N  # dos padres distintos

        # 3. Hijos creados por cruce
        cortes = np.sort(rng.integers(0, n, size=(n_cruce, 2)), axis=1)
        for h in range(n_cruce):
            a, b = cortes[h]
            nueva[n_elite + h] = operador_cruce(poblacion[padres_1[h]], poblacion[padres_2[h]], a, b + 1)

        # 4. Individuos creados por mutación
        inicio = n_elite + n_cruce
        for h, indice in enumerate(elegidos[2 * n_cruce:]):
            nueva[inicio + h] = mutar(poblacion[indice], rng, mutacion)

        # 5. Inmigrantes aleatorios, si s + c + m < 1
        if n_inmigrantes > 0:
            nueva[N - n_inmigrantes:] = poblacion_inicial(n_inmigrantes, n, rng)

        # 6. Control de diversidad: los recorridos repetidos se mutan
        if evitar_clones:
            for indice in indices_de_clones(nueva):
                nueva[indice] = mutar(nueva[indice], rng, mutacion)

        # 7. Evaluación y ordenamiento de la nueva generación
        distancias = longitudes(nueva, matriz)
        orden = np.argsort(distancias)
        poblacion, distancias = nueva[orden], distancias[orden]

        if distancias[0] < D:
            best, D = poblacion[0].copy(), distancias[0]
            mejoras.append((generacion, best.copy(), float(D)))
        mejor_por_generacion.append(D)
        promedio_por_generacion.append(distancias.mean())
        diversidad_por_generacion.append(diversidad(poblacion, best))

    historial = {
        "mejor": np.array(mejor_por_generacion),
        "promedio": np.array(promedio_por_generacion),
        "diversidad": np.array(diversidad_por_generacion),
        "mejoras": mejoras,
        "posiciones": coordenadas if coordenadas is not None else posiciones_aproximadas(matriz),
        "posiciones_exactas": coordenadas is not None,
    }
    return best, float(D), historial


def nota_posiciones(historial):
    return "" if historial["posiciones_exactas"] else " (posiciones aproximadas)"

def dibujar_recorrido(ax, posiciones, recorrido, titulo=""):
    ciclo = np.append(recorrido, recorrido[0])
    ax.plot(posiciones[ciclo, 0], posiciones[ciclo, 1], color=AZUL, linewidth=1.5, zorder=1)
    ax.scatter(posiciones[:, 0], posiciones[:, 1], color=ROJO, s=25, zorder=2)
    ax.set_title(titulo)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

def dibujar_distancia(ax, historial, optimo=None):
    generaciones = np.arange(len(historial["mejor"]))
    ax.plot(generaciones, historial["promedio"], color=GRIS, linewidth=1, label="Promedio de la población")
    ax.plot(generaciones, historial["mejor"], color=AZUL, linewidth=2, label="Mejor recorrido")
    if optimo is not None:
        ax.axhline(optimo, color=VERDE, linestyle="--", label=f"Óptimo conocido ({optimo:g})")
    ax.set_title("Distancia por generación")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Distancia total")
    ax.legend()
    ax.grid(alpha=0.25)

def graficar_resultado(best, D, historial, optimo=None):
    fig, (ax_ruta, ax_dist) = plt.subplots(1, 2, figsize=(14, 5.5))
    titulo = f"Mejor recorrido encontrado, D = {D:.2f}" + nota_posiciones(historial)
    dibujar_recorrido(ax_ruta, historial["posiciones"], best, titulo)
    dibujar_distancia(ax_dist, historial, optimo)
    plt.tight_layout()
    plt.show()

def graficar_evolucion(historial, paneles=6):
    mejoras = historial["mejoras"]
    elegidas = np.unique(np.linspace(0, len(mejoras) - 1, paneles).astype(int))
    fig, ejes = plt.subplots(2, 3, figsize=(14, 9))
    for ax in ejes.ravel():
        ax.axis("off")
    for ax, indice in zip(ejes.ravel(), elegidas):
        generacion, recorrido, distancia = mejoras[indice]
        ax.axis("on")
        dibujar_recorrido(ax, historial["posiciones"], recorrido,
                          f"Generación {generacion}, D = {distancia:.2f}")
    fig.suptitle("Evolución del mejor recorrido" + nota_posiciones(historial), fontsize=14)
    plt.tight_layout()
    plt.show()

def animar_evolucion(historial, max_cuadros=40, intervalo=200, archivo=None, optimo=None):
    """Animación del mejor recorrido y de su distancia. Si se indica `archivo`, guarda un GIF."""
    mejoras = historial["mejoras"]
    posiciones = historial["posiciones"]
    mejor = historial["mejor"]
    generaciones = np.arange(len(mejor))

    elegidas = np.unique(np.linspace(0, len(mejoras) - 1, max_cuadros).astype(int))
    cuadros = [mejoras[indice] for indice in elegidas]
    final = (generaciones[-1], mejoras[-1][1], mejoras[-1][2])
    cuadros += [final] * 8  # pausa sobre el resultado final

    fig, (ax_ruta, ax_dist) = plt.subplots(1, 2, figsize=(12, 5), dpi=72)
    ax_ruta.scatter(posiciones[:, 0], posiciones[:, 1], color=ROJO, s=25, zorder=2)
    linea_ruta, = ax_ruta.plot([], [], color=AZUL, linewidth=1.5, zorder=1)
    ax_ruta.set_aspect("equal")
    ax_ruta.grid(alpha=0.25)

    linea_dist, = ax_dist.plot([], [], color=AZUL, linewidth=2, label="Mejor recorrido")
    punto, = ax_dist.plot([], [], "o", color=ROJO)
    if optimo is not None:
        ax_dist.axhline(optimo, color=VERDE, linestyle="--", label=f"Óptimo conocido ({optimo:g})")
    piso = mejor.min() if optimo is None else min(mejor.min(), optimo)
    ax_dist.set_xlim(0, generaciones[-1])
    ax_dist.set_ylim(0.95 * piso, 1.03 * mejor.max())
    ax_dist.set_title("Distancia del mejor recorrido")
    ax_dist.set_xlabel("Generación")
    ax_dist.set_ylabel("Distancia total")
    ax_dist.legend()
    ax_dist.grid(alpha=0.25)
    fig.tight_layout()

    def actualizar(cuadro):
        generacion, recorrido, distancia = cuadro
        ciclo = np.append(recorrido, recorrido[0])
        linea_ruta.set_data(posiciones[ciclo, 0], posiciones[ciclo, 1])
        ax_ruta.set_title(f"Generación {generacion}, D = {distancia:.2f}" + nota_posiciones(historial))
        linea_dist.set_data(generaciones[:generacion + 1], mejor[:generacion + 1])
        punto.set_data([generacion], [distancia])
        return linea_ruta, linea_dist, punto

    animacion = FuncAnimation(fig, actualizar, frames=cuadros, interval=intervalo)
    if archivo is not None:
        animacion.save(archivo, writer=PillowWriter(fps=1000 // intervalo))
    plt.close(fig)
    return animacion
