"""I/O seguro para checkpoints clínicos.

Centraliza la escritura de state_dicts en un único helper para que todos los
puntos que persisten pesos (SWA, EMA, Model Soup y el best_checkpoint del
`EarlyStopping`) compartan la misma protección frente a una desconexión a mitad
de guardado (escenario habitual en Colab).

Contrato:

1. Validar `torch.isfinite` en TODOS los tensors del state_dict ANTES de
   persistir → `RuntimeError` con el nombre del param corrupto si alguno
   contiene NaN/Inf (no se escribe nada a disco).
2. Escribir a archivo temporal (`path.tmp`) + `Path.replace()` atómico sobre
   el destino → si el proceso se interrumpe a mitad, el checkpoint previo válido
   (si existe) se preserva intacto (rename es atómico en POSIX).
3. Si falla la escritura temp, limpia el `.tmp` y re-raise.

No afecta rendimiento: 1 barrido `.isfinite()` sobre todos los tensors
~5ms en un EffNetB4, negligible vs el propio `torch.save()` (~500ms).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def save_checkpoint_safe(
    state_dict: Mapping[str, torch.Tensor],
    save_path: str | Path,
    checkpoint_label: str,
) -> None:
    """Guarda `state_dict` con validación `isfinite` + rename atómico.

    Args:
        state_dict: resultado de `model.state_dict()` (raw_model — ya
            desenvuelto de DataParallel/DDP/torch.compile por el caller).
        save_path: ruta final del checkpoint (`.pth`). La escritura
            intermedia ocurre en `save_path + ".tmp"` y se promueve por
            `Path.replace()` tras éxito.
        checkpoint_label: etiqueta humana ("SWA", "EMA", "SOUP",
            "best binary standard") usada en el mensaje del `RuntimeError`
            para que el operador identifique qué pass produjo pesos
            corruptos sin leer el stack trace.

    Raises:
        RuntimeError: si algún tensor contiene NaN/Inf. El archivo destino
            NO se toca — el checkpoint previo válido queda intacto.
    """
    # 1. Validación previa: rechazar tensors no-finitos antes de tocar el disco
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue  # buffers non-tensor (poco frecuente) — saltar
        if not torch.isfinite(tensor).all():
            n_nan = (
                torch.isnan(tensor).sum().item()
                if tensor.is_floating_point()
                else 0
            )
            n_inf = (
                torch.isinf(tensor).sum().item()
                if tensor.is_floating_point()
                else 0
            )
            raise RuntimeError(
                f"Checkpoint '{checkpoint_label}' contiene pesos no-finitos en "
                f"param '{name}' (shape={tuple(tensor.shape)}, NaN={n_nan}, Inf={n_inf}). "
                f"NO se guarda para evitar corrupción del checkpoint previo válido en disco. "
                f"Causas probables: BN degenerado, AMP overflow en últimas épocas, Model Soup "
                f"con candidato corrupto. Revisar logs class_health y métricas val."
            )
    # 2. Save atómico: temp + rename
    save_path = Path(save_path)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    try:
        torch.save(state_dict, tmp_path)
        tmp_path.replace(save_path)  # atomic rename en POSIX
    except Exception:
        # Limpiar temp file si la escritura falló
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
