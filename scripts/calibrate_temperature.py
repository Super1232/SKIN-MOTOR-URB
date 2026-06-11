"""
Calibra TemperatureScaler sobre el val set.

Genera `temperature_{binary,multiclass}.pt` que `predict.py` consume para calibrar
las probabilidades antes del threshold clínico (0.20) y antes de los
calibradores downstream (per-site threshold, conformal y risk control).

Filosofía Guo et al. 2017 (Temperature Scaling): aprende un escalar T sobre el
val set tal que `sigmoid(logits/T)` (binario) o `softmax(logits/T)` (multiclase)
quede calibrado. Dos matices:
  - Vector Scaling multiclase: T es un vector [num_classes] (Guo 2017 Tabla 3
    reduce el ECE 15-30% en clases raras).
  - Clamp clínico [0.3, 5.0] en `calibrate()` y error explícito en `fit()` tras la
    optimización (una T fuera de rango indica un modelo patológicamente mal
    calibrado, no desplegable).

Uso:
    python scripts/calibrate_temperature.py --mode binary \\
        --binary-checkpoint models/checkpoints/best_binary_model_SOUP.pth

    python scripts/calibrate_temperature.py --mode multiclass \\
        --multiclass-checkpoint models/checkpoints/best_multiclass_model_SOUP.pth

Salida: `models/checkpoints/temperature_{binary,multiclass}.pt`

Reusa `get_dataloaders` del pipeline oficial (paridad exacta con val split de
training: mismas augmentations deterministas, mismo STAIN_METHOD). Sin TTA, sin
MC Dropout — temperature scaling calibra el predictor determinista. Single
forward pass.
"""
import argparse
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Raíz del repo en sys.path (permite `python scripts/...` directo).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import (  # noqa: E402
    DEFAULT_BINARY_BACKBONE,
    DEFAULT_MULTICLASS_BACKBONE,
    MULTICLASS_LABEL_MAP,
)
from src.data.dataset import get_dataloaders  # noqa: E402
from src.inference.calibration import TemperatureScaler  # noqa: E402
from src.inference.predict import _detect_backbone_config, _pool_kwargs, _fitnet_kwargs  # noqa: E402
from src.models.architectures import SkinCancerModel  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def _build_and_load(mode, checkpoint_path, device):
    """Construye SkinCancerModel con la config detectada del checkpoint y carga pesos.

    Paridad estructural con `_build_and_load` de `calibrate_conformal.py` y
    `_build_and_load_binary` de `calibrate_per_site_threshold.py`. Detecta:
      - backbone (CNN vs ViT) via `_detect_backbone_config`
      - projection_head — para multiclase entrenado con SupCon
      - prefijos `module.` (DataParallel) y `_orig_mod.` (torch.compile)
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    stripped = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod."):]
        stripped[name] = v
    state_dict = stripped

    backbone_name, fusion_mode = _detect_backbone_config(state_dict)
    has_proj = any(k.startswith("projection_head.") for k in state_dict.keys())
    has_cbam = any(
        k.startswith("cbam.") or k.startswith("backbone.cbam.")
        or ".cbam." in k
        for k in state_dict.keys()
    )

    kwargs = dict(
        num_classes=1 if mode == "binary" else len(MULTICLASS_LABEL_MAP),
        pretrained=False,
        use_metadata=True,
        use_attention=has_cbam,
        fusion_mode=fusion_mode,
        mode=mode,
        use_projection_head=has_proj if mode == "multiclass" else False,
    )
    if backbone_name is None:
        kwargs["model_name"] = (
            DEFAULT_BINARY_BACKBONE if mode == "binary" else DEFAULT_MULTICLASS_BACKBONE
        )
    else:
        kwargs["backbone_name"] = backbone_name

    # Detectar la variante de pooling (hybrid/multiscale/attention) del
    # checkpoint. Sin esto, un checkpoint entrenado con --use_hybrid_pool rompe
    # load_state_dict(strict=True) — mismo vector que la detección de
    # backbone/projection_head/CBAM. El state_dict ya viene sin los prefijos
    # module./_orig_mod. de arriba.
    kwargs.update(_pool_kwargs(state_dict))
    # Detectar FitNets (use_fitnet + fitnet_teacher_dim). Sin esto, un checkpoint
    # entrenado con --use_fitnet rompe load_state_dict(strict=True) con
    # "Unexpected keys: fitnet_projection.{weight,bias}".
    kwargs.update(_fitnet_kwargs(state_dict))

    model = SkinCancerModel(**kwargs)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def collect_val_logits(model, loader, device, mode):
    """Forward sobre val set acumulando logits + labels (sin TTA, sin MC Dropout).

    Returns:
        logits: torch.Tensor [N, C] (multiclase) o [N, 1] (binario, sin squeeze)
        labels: torch.Tensor [N] (int64 multiclase, float32 binario)
    """
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            logits = model(images, meta)
            # Binario: [B, 1] → preservar shape para fit() de TemperatureScaler.
            # Multiclase: [B, C] sin cambios.
            all_logits.append(logits.detach().cpu().float())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", required=True, choices=["binary", "multiclass"],
        help="Modo de calibracion (binario o multiclase)"
    )
    parser.add_argument(
        "--binary-checkpoint", default=None,
        help="Ruta al best_binary_model.pth (recomendado: SOUP). Requerido si --mode=binary."
    )
    parser.add_argument(
        "--multiclass-checkpoint", default=None,
        help="Ruta al best_multiclass_model.pth (recomendado: SOUP). Requerido si --mode=multiclass."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size para inferencia en val set"
    )
    parser.add_argument("--output-dir", default="models/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-iter", type=int, default=50,
        help="Iteraciones LBFGS (Guo et al. 2017 default 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="LR de LBFGS (default 0.01)"
    )
    args = parser.parse_args()

    # Validaciones de los argumentos
    if args.mode == "binary":
        if not args.binary_checkpoint:
            parser.error("--mode binary requiere --binary-checkpoint")
        ckpt_path = args.binary_checkpoint
    else:
        if not args.multiclass_checkpoint:
            parser.error("--mode multiclass requiere --multiclass-checkpoint")
        ckpt_path = args.multiclass_checkpoint

    if not Path(ckpt_path).exists():
        parser.error(f"Checkpoint no existe: {ckpt_path}")

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, mode={args.mode}, checkpoint={ckpt_path}")

    # Build model
    print(f"\n=== Cargando modelo {args.mode} ===")
    model = _build_and_load(args.mode, ckpt_path, device)

    # Collect val logits + labels
    print(f"\n=== Forward sobre val set ({args.mode}) ===")
    _, val_loader, _ = get_dataloaders(mode=args.mode, batch_size=args.batch_size, seed=args.seed)
    logits, labels = collect_val_logits(model, val_loader, device, args.mode)
    print(f"Val samples: {len(labels)}, logits shape: {tuple(logits.shape)}")

    # Fit Temperature Scaler
    if args.mode == "binary":
        print(f"\n=== Calibrando Temperature (binario, num_classes=1) ===")
        scaler = TemperatureScaler(num_classes=1)
        # fit() lanza RuntimeError si T cae fuera de [0.3, 5.0] tras la
        # optimización. NO se captura — no debe poder desplegarse una T patológica.
        T = scaler.fit(logits, labels, mode="binary", max_iter=args.max_iter, lr=args.lr)
        print(f"T optimizada: {T:.4f}")
    else:
        num_classes = len(MULTICLASS_LABEL_MAP)
        print(f"\n=== Calibrando Vector Scaling (multiclase, num_classes={num_classes}) ===")
        # Vector Scaling: una T por clase para mejor calibración en clases raras.
        scaler = TemperatureScaler(num_classes=num_classes)
        T = scaler.fit(logits, labels, mode="multiclass", max_iter=args.max_iter, lr=args.lr)
        # T es una lista: fit retorna lista si num_classes>1.
        T_str = ", ".join(f"{t:.4f}" for t in T) if isinstance(T, list) else f"{T:.4f}"
        print(f"T optimizada (vector): [{T_str}]")

    # Save
    output_path = out_dir / f"temperature_{args.mode}.pt"
    scaler.save(str(output_path))
    print(f"\nDone. Temperature guardada en {output_path}")
    # Mini-audit: imprimir estadísticas del val set para trazabilidad
    if args.mode == "binary":
        prevalence = float((labels == 1).sum()) / len(labels)
        print(f"Val prevalencia maligno: {prevalence:.4f}")
    else:
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        for c, cnt in zip(unique, counts):
            print(f"  Clase {int(c)}: n={int(cnt)}")


if __name__ == "__main__":
    main()
