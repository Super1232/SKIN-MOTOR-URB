"""
Calibra PerSiteThresholdCalibrator sobre el val set.

Pre-requisito: temperature scaling ya aplicado y guardado (recomendado, no
obligatorio). El threshold per-site se calibra sobre PROBABILIDADES post-sigmoid
(opcionalmente post-temperature). Sin temperature, la calibración sigue
funcionando pero las probabilidades pueden estar sobreconfiadas y los
thresholds derivados sesgados — usar temperature mejora la fidelidad.

Uso:
    python scripts/calibrate_per_site_threshold.py \\
        --binary-checkpoint models/checkpoints/best_binary_model_SOUP.pth \\
        --temperature-binary models/checkpoints/temperature_binary.pt \\
        --target-recall 0.95 \\
        --min-subgroup-n 30 \\
        --min-positive 10

Salida: models/checkpoints/per_site_threshold_binary.pt

Reusa `get_dataloaders` del pipeline oficial (paridad exacta con val split de
training: mismas augmentations deterministas, mismo STAIN_METHOD). Sin TTA,
sin MC Dropout — el calibrador opera sobre probabilidades del modelo
determinista (single forward).

Reporta a stdout una tabla por sitio con:
    site_code | n | n_pos | threshold | recall@thresh | spec@thresh | source
para que el operador clínico audite el resultado antes de desplegar.

Filosofía:
- target_recall=0.95 alineado con el sesgo a recall (NUNCA perder maligno)
- min_positive=10 alineado con un Wilson CI estable
- Threshold clamp a [0.10, 0.50] aplicado en PerSiteThresholdCalibrator
"""
import argparse
import hashlib
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import (  # noqa: E402
    DEFAULT_BINARY_BACKBONE,
    PER_SITE_THRESHOLD_MIN,
    PER_SITE_THRESHOLD_MAX,
    SITE_MAP,
)
from src.data.dataset import get_dataloaders  # noqa: E402
from src.inference.calibration import PerSiteThresholdCalibrator, TemperatureScaler  # noqa: E402
from src.inference.predict import _detect_backbone_config, _pool_kwargs, _fitnet_kwargs  # noqa: E402
from src.models.architectures import SkinCancerModel  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def _build_and_load_binary(checkpoint_path, device):
    """Construye SkinCancerModel binario detectando backbone/fusion del checkpoint.

    Paridad con `ClinicalInferenceEngine.__init__` de predict.py: misma
    detección automática de backbone y CBAM, desenvolviendo los prefijos
    `module.` (DataParallel) y `_orig_mod.` (torch.compile).
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
    has_cbam = any(
        k.startswith("cbam.") or k.startswith("backbone.cbam.")
        or ".cbam." in k
        for k in state_dict.keys()
    )

    kwargs = dict(
        num_classes=1,  # binario
        pretrained=False,
        use_metadata=True,
        use_attention=has_cbam,
        fusion_mode=fusion_mode,
        mode="binary",
    )
    if backbone_name is None:
        kwargs["model_name"] = DEFAULT_BINARY_BACKBONE
    else:
        kwargs["backbone_name"] = backbone_name

    # Variante de pooling (hybrid/multiscale/attention) detectada del checkpoint.
    # El state_dict ya viene sin los prefijos module./_orig_mod. de arriba.
    kwargs.update(_pool_kwargs(state_dict))
    # FitNets (use_fitnet + fitnet_teacher_dim), igual que en predict.py y los
    # otros scripts. Si el checkpoint no tiene fitnet_projection, el helper
    # devuelve {} y la construcción no cambia.
    kwargs.update(_fitnet_kwargs(state_dict))

    model = SkinCancerModel(**kwargs)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _compute_val_csv_hash() -> str:
    """SHA256 del 04_val.csv usado por get_dataloaders.

    Persiste dentro del .pt de per-site threshold y predict.py lo verifica
    al cargar. Un mismatch indica que el val set cambió desde la calibración.
    """
    p = REPO_ROOT / "data" / "processed" / "04_val.csv"
    if not p.exists():
        warnings.warn(
            f"val CSV {p} no encontrado — hash=''. "
            "predict.py no podra verificar integridad del val set.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ""
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_temp_safe(path):
    """Carga TemperatureScaler con guards de finitud y aviso de fallback."""
    if not path or not Path(path).exists():
        return None
    try:
        # Peek num_classes para reconstruir scaler con el shape correcto.
        peek = torch.load(path, map_location="cpu", weights_only=True)
        nc = int(peek.get("num_classes", 1)) if isinstance(peek, dict) else 1
        if nc < 1:
            raise ValueError(f"num_classes inválido: {nc}")
        scaler = TemperatureScaler(num_classes=nc)
        scaler.load(path)
        t_vec = scaler.temperature.detach().cpu().tolist()
        if isinstance(t_vec, (int, float)):
            t_vec = [t_vec]
        if not all(np.isfinite(x) and x > 0 for x in t_vec):
            raise ValueError(f"Temperatura invalida: T={t_vec}")
        if len(t_vec) == 1:
            warnings.warn(
                f"Temperatura cargada desde {path}: T={t_vec[0]:.3f}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            _fmt = ", ".join(f"{x:.3f}" for x in t_vec)
            warnings.warn(
                f"Temperatura cargada desde {path}: T=[{_fmt}] (vector, N={len(t_vec)})",
                RuntimeWarning,
                stacklevel=2,
            )
        return scaler
    except Exception as e:
        warnings.warn(
            f"Temperatura en {path} no disponible ({e}). Calibrando sin temperatura.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def collect_val_probs_with_sites(model, loader, device, temp_scaler=None):
    """
    Forward sobre val set acumulando probs binarias + labels + site_codes.

    site_code se extrae del tensor de metadata `meta` ([B, 3] = [age_norm, sex_code, site_code]).
    Es el índice entero [0..NUM_SITE_CODES-1] del mapping `SITE_MAP` aplicado por dataset.py.
    """
    all_probs, all_labels, all_sites = [], [], []
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            logits = model(images, meta).squeeze(1)
            if temp_scaler is not None:
                logits = temp_scaler.calibrate(logits)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            # site_code es la 3a columna de meta (paridad con encode_clinical_metadata_scalar)
            sites = meta[:, 2].cpu().numpy().astype(np.int64)
            all_sites.append(sites)
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_sites),
    )


def _site_name_for(site_code: int) -> str:
    """Resuelve el nombre canónico del site_code para auditoría. Como varios
    aliases mapean al mismo code, se devuelve el primero encontrado."""
    for k, v in SITE_MAP.items():
        if int(v) == int(site_code):
            return k
    return f"unknown(code={site_code})"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--binary-checkpoint", required=True,
        help="Ruta a best_binary_model.pth (recomendado: SOUP)"
    )
    parser.add_argument(
        "--temperature-binary", default=None,
        help="Ruta a temperature_binary.pt (opcional, recomendado)"
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.95,
        help="Recall minimo a preservar por sitio. Default 0.95"
    )
    parser.add_argument(
        "--min-subgroup-n", type=int, default=30,
        help="Minimo de muestras por sitio para calibrar. Default 30"
    )
    parser.add_argument(
        "--min-positive", type=int, default=10,
        help="Minimo de malignos por sitio (Wilson CI estable). Default 10"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size para inferencia en val set"
    )
    parser.add_argument("--output-dir", default="models/checkpoints")
    parser.add_argument(
        "--output-name", default="per_site_threshold_binary.pt",
        help="Nombre del archivo de salida"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validaciones de rango de los argumentos
    if not (0.0 < args.target_recall < 1.0):
        parser.error(f"--target-recall debe estar en (0, 1), recibido {args.target_recall}")
    if args.min_subgroup_n < 1:
        parser.error(f"--min-subgroup-n debe ser >= 1, recibido {args.min_subgroup_n}")
    if args.min_positive < 1:
        parser.error(f"--min-positive debe ser >= 1, recibido {args.min_positive}")

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Device: {device}, target_recall={args.target_recall}, "
        f"min_subgroup_n={args.min_subgroup_n}, min_positive={args.min_positive}, "
        f"output_dir={out_dir}"
    )

    # Build model
    print("\n=== Cargando modelo binario ===")
    model = _build_and_load_binary(args.binary_checkpoint, device)

    # Load temperature (opcional)
    temp = _load_temp_safe(args.temperature_binary)
    if temp is not None:
        temp.to(device)

    # Collect val probs + sites
    print("\n=== Forward sobre val set (binario) ===")
    _, val_loader, _ = get_dataloaders(mode="binary", batch_size=args.batch_size, seed=args.seed)
    probs, labels, sites = collect_val_probs_with_sites(model, val_loader, device, temp)
    print(f"Total val samples: {len(probs)}")
    print(f"Class distribution: malignant={int((labels==1).sum())}, "
          f"benign={int((labels==0).sum())}")
    print(f"Site distribution:")
    for s in np.unique(sites):
        mask_s = sites == s
        print(f"  site_code={s} ({_site_name_for(int(s))}): "
              f"n={int(mask_s.sum())}, n_pos={int(((mask_s) & (labels==1)).sum())}")

    # Fit calibrator
    print(f"\n=== Calibrando per-site threshold ===")
    cal = PerSiteThresholdCalibrator(
        target_recall=args.target_recall,
        min_subgroup_n=args.min_subgroup_n,
        min_positive=args.min_positive,
    )
    cal.fit(probs, labels, sites)

    # Audit table
    print(f"\n=== Per-site threshold table ===")
    print(f"{'site_code':<10}{'name':<22}{'n':<8}{'n_pos':<8}"
          f"{'threshold':<12}{'recall':<10}{'spec':<10}{'source':<18}{'skip_reason'}")
    print("-" * 130)
    for site_code in sorted(cal.per_site_metrics.keys()):
        m = cal.per_site_metrics[site_code]
        thr = m["threshold"]
        thr_str = f"{thr:.3f}" if thr is not None else "FALLBACK"
        rec_str = f"{m['recall_at_threshold']:.3f}" if m['recall_at_threshold'] is not None else "—"
        spec_str = f"{m['specificity_at_threshold']:.3f}" if m['specificity_at_threshold'] is not None else "—"
        source = "per_site" if thr is not None else "global_fallback"
        reason = m.get("skip_reason") or ""
        print(f"{site_code:<10}{_site_name_for(site_code):<22}{m['n']:<8}{m['n_pos']:<8}"
              f"{thr_str:<12}{rec_str:<10}{spec_str:<10}{source:<18}{reason}")

    # Save
    output_path = out_dir / args.output_name
    val_csv_hash = _compute_val_csv_hash()
    cal.save(str(output_path), val_csv_hash=val_csv_hash)
    print(f"\nDone. Per-site threshold guardado en {output_path}")
    print(f"val_csv_hash: {val_csv_hash[:16]}...")
    print(f"\nClamp clinico aplicado: thresholds en [{PER_SITE_THRESHOLD_MIN}, {PER_SITE_THRESHOLD_MAX}].")
    n_per_site = sum(1 for v in cal.thresholds.values() if v is not None)
    n_fallback = sum(1 for v in cal.thresholds.values() if v is None)
    print(f"Sitios calibrados: {n_per_site} / {n_per_site + n_fallback} "
          f"(restantes usaran fallback global={cal.fallback_threshold})")


if __name__ == "__main__":
    main()
