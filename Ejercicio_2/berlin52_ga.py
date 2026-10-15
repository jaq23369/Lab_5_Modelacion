"""Algoritmo genético reproducible para TSP con distancias TSPLIB EUC_2D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ResultadoGA:
    """Resultado y trazabilidad de una ejecución del algoritmo genético."""

    recorrido: np.ndarray
    distancia: int
    historial_mejor: np.ndarray
    historial_promedio: np.ndarray
    mejoras: list[tuple[int, int]]
    semilla: int | None
    parametros: dict[str, Any]


def leer_tsplib(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Lee una instancia TSPLIB con sección ``NODE_COORD_SECTION``."""

    meta: dict[str, Any] = {}
    puntos: list[tuple[int, float, float]] = []
    en_coordenadas = False

    with Path(path).open(encoding="utf-8") as archivo:
        for numero_linea, linea_original in enumerate(archivo, start=1):
            linea = linea_original.strip()
            if not linea or linea.startswith("COMMENT"):
                continue
            if linea.upper() == "NODE_COORD_SECTION":
                en_coordenadas = True
                continue
            if linea.upper() == "EOF":
                break
            if en_coordenadas:
                partes = linea.split()
                if len(partes) < 3:
                    raise ValueError(f"Coordenada inválida en la línea {numero_linea}.")
                try:
                    puntos.append((int(partes[0]), float(partes[1]), float(partes[2])))
                except ValueError as error:
                    raise ValueError(
                        f"Coordenada inválida en la línea {numero_linea}."
                    ) from error
                continue

            if ":" in linea:
                clave, valor = (parte.strip() for parte in linea.split(":", 1))
            else:
                partes = linea.split(maxsplit=1)
                if len(partes) != 2:
                    continue
                clave, valor = partes
            meta[clave.upper()] = valor

    if not puntos:
        raise ValueError("El archivo no contiene NODE_COORD_SECTION con coordenadas.")
    if meta.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper() != "EUC_2D":
        raise ValueError("Solo se admite EDGE_WEIGHT_TYPE EUC_2D.")

    puntos.sort(key=lambda punto: punto[0])
    ids = [punto[0] for punto in puntos]
    if len(ids) != len(set(ids)):
        raise ValueError("Los identificadores de ciudad deben ser únicos.")

    if "DIMENSION" in meta:
        try:
            meta["DIMENSION"] = int(meta["DIMENSION"])
        except ValueError as error:
            raise ValueError("DIMENSION debe ser un entero.") from error
        if meta["DIMENSION"] != len(puntos):
            raise ValueError("DIMENSION no coincide con el número de coordenadas.")
    else:
        meta["DIMENSION"] = len(puntos)

    coords = np.asarray([(x, y) for _, x, y in puntos], dtype=float)
    return coords, meta


def matriz_euc_2d(coords: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Construye la matriz EUC_2D: ``floor(distancia + 0.5)``."""

    arreglo = np.asarray(coords, dtype=float)
    if arreglo.ndim != 2 or arreglo.shape[1] != 2 or len(arreglo) < 2:
        raise ValueError("coords debe tener forma (n, 2), con n >= 2.")
    if not np.all(np.isfinite(arreglo)):
        raise ValueError("coords solo puede contener valores finitos.")
    diferencias = arreglo[:, None, :] - arreglo[None, :, :]
    return np.floor(np.sqrt(np.sum(diferencias * diferencias, axis=2)) + 0.5).astype(
        np.int64
    )


def es_recorrido_valido(route: Sequence[int] | np.ndarray, n: int) -> bool:
    """Indica si ``route`` es una permutación de ``range(n)``."""

    arreglo = np.asarray(route)
    if n < 0 or arreglo.ndim != 1 or len(arreglo) != n:
        return False
    if not np.issubdtype(arreglo.dtype, np.integer):
        return False
    return bool(np.array_equal(np.sort(arreglo), np.arange(n)))


def distancia_recorrido(
    route: Sequence[int] | np.ndarray, matrix: Sequence[Sequence[int]] | np.ndarray
) -> int:
    """Suma un ciclo cerrado, incluida la arista de regreso al origen."""

    matriz = np.asarray(matrix)
    recorrido = np.asarray(route)
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("matrix debe ser cuadrada.")
    if not es_recorrido_valido(recorrido, len(matriz)):
        raise ValueError("route no es una permutación válida.")
    return int(matriz[recorrido, np.roll(recorrido, -1)].sum())


def _longitudes(poblacion: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    return matriz[poblacion, np.roll(poblacion, -1, axis=1)].sum(axis=1)


def _seleccionar_torneo(
    distancias: np.ndarray, cuantos: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    participantes = rng.integers(0, len(distancias), size=(cuantos, k))
    valores = distancias[participantes]
    return participantes[np.arange(cuantos), np.argmin(valores, axis=1)]


def _cruce_ox(
    padre_1: np.ndarray, padre_2: np.ndarray, a: int, b: int
) -> np.ndarray:
    n = len(padre_1)
    hijo = np.full(n, -1, dtype=np.int64)
    hijo[a:b] = padre_1[a:b]
    usados = np.zeros(n, dtype=bool)
    usados[padre_1[a:b]] = True
    orden = np.concatenate((padre_2[b:], padre_2[:b]))
    faltantes = orden[~usados[orden]]
    posiciones = np.concatenate((np.arange(b, n), np.arange(0, a)))
    hijo[posiciones] = faltantes
    return hijo


def _mutar_inversion(recorrido: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nuevo = recorrido.copy()
    a, b = np.sort(rng.choice(len(nuevo), size=2, replace=False))
    nuevo[a : b + 1] = nuevo[a : b + 1][::-1]
    return nuevo


def _dos_opt_acotado(
    recorrido: np.ndarray,
    matriz: np.ndarray,
    max_pasadas: int = 80,
    max_evaluaciones: int = 100_000,
) -> tuple[np.ndarray, int]:
    """Aplica first-improvement 2-opt con límites explícitos de trabajo."""

    mejor = recorrido.copy()
    distancia = distancia_recorrido(mejor, matriz)
    n = len(mejor)
    evaluaciones = 0

    for _ in range(max_pasadas):
        mejor_cambio = 0
        mejores_cortes: tuple[int, int] | None = None
        for i in range(n - 1):
            a = mejor[i - 1]
            b = mejor[i]
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                c = mejor[j]
                d = mejor[(j + 1) % n]
                evaluaciones += 1
                cambio = int(matriz[a, c] + matriz[b, d] - matriz[a, b] - matriz[c, d])
                if cambio < mejor_cambio:
                    mejor_cambio = cambio
                    mejores_cortes = (i, j)
                if evaluaciones >= max_evaluaciones:
                    break
            if evaluaciones >= max_evaluaciones:
                break
        if mejores_cortes is None:
            break
        i, j = mejores_cortes
        mejor[i : j + 1] = mejor[i : j + 1][::-1]
        distancia += mejor_cambio
        if evaluaciones >= max_evaluaciones:
            break
    return mejor, distancia


def _cantidad(valor: float | int, N: int, nombre: str) -> int:
    if isinstance(valor, (bool, np.bool_)):
        raise TypeError(f"{nombre} debe ser una proporción o un conteo entero.")
    if isinstance(valor, (int, np.integer)):
        cantidad = int(valor)
        if not 0 <= cantidad <= N:
            raise ValueError(f"{nombre} como conteo debe estar entre 0 y N.")
        return cantidad
    if isinstance(valor, (float, np.floating)):
        proporcion = float(valor)
        if not np.isfinite(proporcion) or not 0 <= proporcion <= 1:
            raise ValueError(
                f"{nombre} como proporción debe ser finita y estar entre 0 y 1."
            )
        return int(round(proporcion * N))
    raise TypeError(f"{nombre} debe ser una proporción o un conteo entero.")


def algoritmo_genetico_tsp(
    coords: Sequence[Sequence[float]] | np.ndarray,
    N: int = 200,
    max_iter: int = 500,
    seleccion: str = "torneo",
    k: int = 3,
    s: float | int = 0.05,
    c: float | int = 0.70,
    m: float | int = 0.20,
    semilla: int | None = None,
) -> ResultadoGA:
    """Resuelve un TSP por GA: torneo, OX, inversión, élite e inmigrantes."""

    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N debe ser un entero mayor o igual a 2.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter debe ser un entero no negativo.")
    if seleccion != "torneo":
        raise ValueError("seleccion debe ser 'torneo'.")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k debe ser un entero positivo.")

    matriz = matriz_euc_2d(coords)
    n = len(matriz)
    n_elite = _cantidad(s, N, "s")
    n_cruce = _cantidad(c, N, "c")
    n_mutacion = _cantidad(m, N, "m")
    n_inmigrantes = N - n_elite - n_cruce - n_mutacion
    if n_inmigrantes < 0:
        raise ValueError("s + c + m no puede superar N.")

    rng = np.random.default_rng(semilla)
    poblacion = np.asarray([rng.permutation(n) for _ in range(N)], dtype=np.int64)
    distancias = _longitudes(poblacion, matriz)
    orden = np.argsort(distancias, kind="stable")
    poblacion, distancias = poblacion[orden], distancias[orden]

    mejor, mejor_distancia = _dos_opt_acotado(poblacion[0], matriz)
    mejor_distancia = int(mejor_distancia)
    historial_mejor = [mejor_distancia]
    historial_promedio = [float(distancias.mean())]
    mejoras: list[tuple[int, int]] = [(0, mejor_distancia)]

    for generacion in range(1, max_iter + 1):
        nueva = np.empty_like(poblacion)
        if n_elite:
            nueva[:n_elite] = poblacion[:n_elite]

        cuantos_padres = 2 * n_cruce + n_mutacion
        elegidos = _seleccionar_torneo(distancias, cuantos_padres, int(k), rng)
        for indice in range(n_cruce):
            p1 = elegidos[indice]
            p2 = elegidos[n_cruce + indice]
            if p1 == p2:
                p2 = (p2 + 1) % N
            a, b = np.sort(rng.choice(n, size=2, replace=False))
            nueva[n_elite + indice] = _cruce_ox(
                poblacion[p1], poblacion[p2], int(a), int(b) + 1
            )

        inicio_mutacion = n_elite + n_cruce
        for desplazamiento, indice in enumerate(elegidos[2 * n_cruce :]):
            nueva[inicio_mutacion + desplazamiento] = _mutar_inversion(
                poblacion[indice], rng
            )

        inicio_inmigrantes = n_elite + n_cruce + n_mutacion
        for indice in range(inicio_inmigrantes, N):
            nueva[indice] = rng.permutation(n)

        # Control de clones: cada copia se perturba; si persiste, se reemplaza.
        vistos: set[tuple[int, ...]] = set()
        for indice in range(N):
            clave = tuple(int(x) for x in nueva[indice])
            intentos = 0
            while clave in vistos and intentos < 3:
                nueva[indice] = _mutar_inversion(nueva[indice], rng)
                clave = tuple(int(x) for x in nueva[indice])
                intentos += 1
            if clave in vistos:
                nueva[indice] = rng.permutation(n)
                clave = tuple(int(x) for x in nueva[indice])
            vistos.add(clave)

        distancias = _longitudes(nueva, matriz)
        orden = np.argsort(distancias, kind="stable")
        poblacion, distancias = nueva[orden], distancias[orden]

        # Refinar el mejor de cada generación permite explorar distintas cuencas
        # 2-opt, incluso cuando su distancia sin refinar aún no supera el récord.
        candidato, distancia_candidato = _dos_opt_acotado(poblacion[0], matriz)
        if distancia_candidato < mejor_distancia:
            mejor = candidato
            mejor_distancia = int(distancia_candidato)
            mejoras.append((generacion, mejor_distancia))

        historial_mejor.append(mejor_distancia)
        historial_promedio.append(float(distancias.mean()))

    parametros = {
        "N": int(N),
        "max_iter": int(max_iter),
        "seleccion": seleccion,
        "k": int(k),
        "s": s,
        "c": c,
        "m": m,
    }
    return ResultadoGA(
        recorrido=mejor.copy(),
        distancia=mejor_distancia,
        historial_mejor=np.asarray(historial_mejor, dtype=np.int64),
        historial_promedio=np.asarray(historial_promedio, dtype=float),
        mejoras=mejoras,
        semilla=semilla,
        parametros=parametros,
    )


def ejecutar_experimentos(
    coords: Sequence[Sequence[float]] | np.ndarray,
    semillas: Iterable[int],
    **params: Any,
) -> tuple[ResultadoGA, list[ResultadoGA]]:
    """Ejecuta varias semillas y devuelve el mejor resultado y la lista completa."""

    resultados = [
        algoritmo_genetico_tsp(coords, semilla=int(semilla), **params)
        for semilla in semillas
    ]
    if not resultados:
        raise ValueError("semillas debe contener al menos un valor.")
    return min(resultados, key=lambda resultado: resultado.distancia), resultados
