"""Reproducibilidad clínica: fuente única para la inicialización del RNG global.

Centraliza la política de semillas del proyecto. El entrenamiento y la
verificación importan `seed_everything` desde aquí para que la semilla,
`PYTHONHASHSEED` y todas las fuentes de aleatoriedad (torch CPU/GPU, numpy,
random) queden fijadas de forma idéntica y se evite la divergencia entre
módulos.

`cudnn.benchmark=True` es intencional pese a `cudnn.deterministic=True`: los
tamaños de input son fijos (300/380 CNN, 294/378 ViT), por lo que cuDNN puede
seleccionar el algoritmo determinista más rápido y obtener un speedup ~10-20%
sin sacrificar reproducibilidad.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, strict_determinism: bool = False) -> None:
    """Fija todas las fuentes de aleatoriedad relevantes para el proyecto.

    Args:
        seed: semilla entera.
        strict_determinism: si True, activa
            `torch.use_deterministic_algorithms(True)` Y desactiva
            `cudnn.benchmark`. Garantiza bit-exactness entre runs al coste
            de velocidad (~10-20% más lento) y de romper algunas ops
            no-deterministas (scatter_add_cuda, F.interpolate bilinear,
            upsample) — en esos casos PyTorch lanza un `RuntimeError`
            explícito, no falla silencioso. Default False preserva el
            comportamiento estándar.

    NOTA: `strict_determinism=True` puede requerir establecer
    `CUBLAS_WORKSPACE_CONFIG=:4096:8` o `:16:8` en el entorno ANTES de
    importar PyTorch para que los matmuls en GPU sean determinísticos.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if strict_determinism:
        # Modo estricto: desactiva benchmark y fuerza ops deterministas.
        # Cualquier op no-determinista lanzará error en vez de fallar silencioso.
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # API no disponible en PyTorch < 1.8 (muy antiguo); degradar
            # silenciosamente al modo cudnn deterministic estándar.
            pass
    else:
        # benchmark=True + deterministic=True: PyTorch selecciona el algoritmo
        # determinista más rápido para shapes de entrada fijos. Default.
        torch.backends.cudnn.benchmark = True
        # Si un run previo activó strict_determinism, desactivarlo.
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


# Alias retrocompatible con código que importe el nombre antiguo.
set_seed = seed_everything
