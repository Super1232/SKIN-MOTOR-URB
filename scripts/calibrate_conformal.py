"""
Calibra BinaryConformal y APSConformal sobre el val set.

Pre-requisito: temperature scaling ya aplicado y guardado (opcional pero
recomendado). Conformal opera sobre probabilidades calibradas — fit sobre
logits crudos desperdicia la suposición de intercambio (Vovk asume scores
bien comportados, con un modelo mal calibrado el threshold queda sesgado).

Uso:
    python scripts/calibrate_conformal.py \\
        --binary-checkpoint models/checkpoints/best_binary_model.pth \\
        --multiclass-checkpoint models/checkpoints/best_multiclass_model.pth \\
        --temperature-binary models/checkpoints/temperature_binary.pt \\
        --temperature-multiclass models/checkpoints/temperature_multiclass.pt \\
        --alpha 0.05

Salida: models/checkpoints/conformal_{binary,multiclass}.pt

Reutiliza `get_dataloaders` del pipeline oficial (paridad exacta con val split
de training: mismas augmentations deterministas, mismo STAIN_METHOD). Sin TTA
ni MC Dropout — conformal calibra un predictor determinista, la estocasticidad
rompería la hipótesis de intercambio.
"""
import argparse
import hashlib
import math
import warnings
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Raíz del repo en sys.path (permite `python scripts/...` directo).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import (  # noqa: E402
    DEFAULT_BINARY_BACKBONE,
    DEFAULT_MULTICLASS_BACKBONE,
    MULTICLASS_INDEX_MAP,
    MULTICLASS_LABEL_MAP,
)
from src.data.dataset import get_dataloaders  # noqa: E402
from src.inference.calibration import TemperatureScaler  # noqa: E402
from src.inference.conformal import (  # noqa: E402
    APSConformal,
    BinaryConformal,
    ConformalRiskControl,
)
from src.inference.predict import _detect_backbone_config, _pool_kwargs, _fitnet_kwargs  # noqa: E402
from src.models.architectures import SkinCancerModel  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402

# `_detect_backbone_config` se importa directamente desde `src.inference.predict`
# como fuente de verdad única. Mantener una copia local divergiría en silencio
# si se añadiera soporte para un backbone nuevo (p.ej. otro ViT) en predict.py
# sin actualizar este script de calibración.


def _build_and_load(mode, checkpoint_path, device):
    """Construye SkinCancerModel con la config detectada del checkpoint y carga pesos.
    Paridad con `ClinicalInferenceEngine.__init__` de predict.py: misma
    detección automática de backbone y projection_head.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Desenvolver el wrapping de DataParallel (`module.`) y torch.compile
    # (`_orig_mod.`). Un checkpoint con esos prefijos (p.ej. de un fork externo
    # que compiló sin desenvolver) rompería load_state_dict(strict=True) con un
    # "Missing/Unexpected keys" críptico.
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
    # Detectar CBAM para construir el modelo con el flag correcto. Confiar en el
    # default `use_attention=True` de `SkinCancerModel.__init__` solo funciona si
    # el checkpoint lo incluye; un checkpoint entrenado con `use_attention=False`
    # no tendría keys `cbam.*` y `load_state_dict(strict=True)` fallaría con
    # "Missing keys: cbam.channel_attention.*". Mismo criterio que los otros dos
    # scripts de calibración.
    has_cbam = any(
        k.startswith("cbam.") or k.startswith("backbone.cbam.")
        or ".cbam." in k
        for k in state_dict.keys()
    )

    kwargs = dict(
        # num_classes derivado de constants.py, no hardcodeado.
        num_classes=1 if mode == "binary" else len(MULTICLASS_LABEL_MAP),
        pretrained=False,
        use_metadata=True,
        use_attention=has_cbam,
        fusion_mode=fusion_mode,
        mode=mode,
        use_projection_head=has_proj if mode == "multiclass" else False,
    )
    if backbone_name is None:
        # Backbone por defecto desde constantes centralizadas (paridad con predict.py).
        kwargs["model_name"] = (
            DEFAULT_BINARY_BACKBONE if mode == "binary" else DEFAULT_MULTICLASS_BACKBONE
        )
    else:
        kwargs["backbone_name"] = backbone_name

    # Variante de pooling (hybrid/multiscale/attention) detectada del checkpoint,
    # igual que en los otros scripts y en predict.py. El state_dict ya viene sin
    # los prefijos module./_orig_mod. de arriba.
    kwargs.update(_pool_kwargs(state_dict))
    # FitNets (use_fitnet + fitnet_teacher_dim). Sin esto, calibrar conformal
    # sobre un checkpoint entrenado con --use_fitnet fallaría con
    # "Unexpected keys: fitnet_projection.*" → no se generaría el .pt → el engine
    # caería a modo sin-conformal en silencio.
    kwargs.update(_fitnet_kwargs(state_dict))

    model = SkinCancerModel(**kwargs)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _compute_val_csv_hash(mode: str) -> str:
    """SHA256 del val CSV usado por `get_dataloaders` para la calibración.

    Se persiste dentro del `.pt` conformal y `predict.py` lo verifica al cargar.
    Un mismatch indica que el val set cambió desde la calibración → la garantía
    de cobertura del Split Conformal queda rota porque la exchangeability se
    asumió contra un conjunto que ya no existe.

    `mode` decide qué CSV leer: `binary` → `04_val.csv`; `multiclass` → `07_val_multiclass.csv`.
    Mismo fallback que `get_dataloaders` para resolver el path relativo al
    raíz del proyecto.
    """
    candidates = {
        "binary": [
            REPO_ROOT / "data" / "processed" / "04_val.csv",
        ],
        "multiclass": [
            REPO_ROOT / "data" / "processed" / "07_val_multiclass.csv",
        ],
    }
    for p in candidates.get(mode, []):
        if p.exists():
            sha = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(chunk)
            return sha.hexdigest()
    warnings.warn(
        f"val CSV para mode='{mode}' no encontrado — hash=''. "
        f"predict.py no podrá verificar integridad del val set.",
        RuntimeWarning,
        stacklevel=2,
    )
    return ""


def _load_temp_safe(path):
    """Paridad con `ClinicalInferenceEngine._load_temperature_safe`: fallback
    a None con aviso si el checkpoint está corrupto o T no es finita.

    Vector-scaling safe: con Vector Scaling multiclase `scaler.temperature` es un
    tensor con `num_classes` elementos, así que se extrae vía `.tolist()` y se
    valida la finitud por componente (un `.item()` directo lanzaría
    `RuntimeError: a Tensor with N elements cannot be converted to Scalar`).

    El aviso de fallback usa `warnings.warn` en vez de `print` para que sea
    capturable por los handlers de logging en producción clínica.
    """
    if not path or not Path(path).exists():
        return None
    try:
        scaler = TemperatureScaler()
        scaler.load(path)
        t_vec = scaler.temperature.detach().cpu().tolist()
        if not isinstance(t_vec, list):
            t_vec = [t_vec]
        if not all(math.isfinite(x) and x > 0 for x in t_vec):
            raise ValueError(f"Temperatura invalida: T={t_vec}")
        # warnings.warn (no print) para captura estructurada en el log,
        # coherente con la rama except.
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
            f"Temperatura en {path} no disponible ({e}). Usando T=1.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def collect_val_probs_binary(model, loader, device, temp_scaler=None):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            logits = model(images, meta).squeeze(1)
            if temp_scaler is not None:
                # TemperatureScaler.calibrate es una división pura, soporta cualquier shape.
                logits = temp_scaler.calibrate(logits)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def collect_val_probs_multiclass(model, loader, device, temp_scaler=None):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            logits = model(images, meta)
            if temp_scaler is not None:
                logits = temp_scaler.calibrate(logits)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary-checkpoint", required=True, help="Ruta a best_binary_model.pth")
    parser.add_argument("--multiclass-checkpoint", default=None, help="Ruta a best_multiclass_model.pth (opcional)")
    parser.add_argument("--temperature-binary", default=None, help="Ruta a temperature_binary.pt (opcional)")
    parser.add_argument("--temperature-multiclass", default=None, help="Ruta a temperature_multiclass.pt (opcional)")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha global (deprecated: usar --alpha-binary/--alpha-multiclass). Si se pasa, sobreescribe ambos.")
    parser.add_argument("--alpha-binary", type=float, default=0.03, help="Alpha binario (default 0.03 = 97%% cobertura, safety-critical)")
    parser.add_argument("--alpha-multiclass", type=float, default=0.10, help="Alpha multiclase (default 0.10 = 90%% cobertura, informativo)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size para inferencia en val set")
    parser.add_argument("--output-dir", default="models/checkpoints")
    # Se propaga el seed explícito a get_dataloaders (igual que los otros dos
    # scripts de calibración) para garantizar un orden reproducible del val set
    # sin depender del default interno del sampler.
    parser.add_argument("--seed", type=int, default=42, help="Seed reproducible (default 42)")
    # ConformalRiskControl (Angelopoulos 2022). `--risk-control-fnr` activa la
    # calibración del control directo de FNR sobre las probabilidades binarias
    # calibradas. Si se pasa, genera `conformal_risk_binary.pt` además del
    # conformal estándar, que `predict.py` carga vía `risk_control_binary_path`.
    # El FNR (fracción de melanomas perdidos) es la métrica que regula la
    # autoridad sanitaria — más alineada con el despliegue clínico que la
    # cobertura marginal `1-α`.
    parser.add_argument(
        "--risk-control-fnr",
        type=float,
        default=None,
        help="Si se especifica, calibra ConformalRiskControl con este target FNR "
        "(ej. 0.1 = 10%% de falsos negativos tolerados). Salida: "
        "conformal_risk_binary.pt. Opcional (default None = no calibrar).",
    )
    args = parser.parse_args()
    # Validación temprana del target FNR (el constructor también valida, pero
    # detenerse aquí da un mensaje más útil para el usuario del script).
    if args.risk_control_fnr is not None and not (0.0 < args.risk_control_fnr < 1.0):
        parser.error(
            f"--risk-control-fnr debe estar en (0, 1), recibido {args.risk_control_fnr}"
        )

    # Alpha asimétrico: binario safety-critical (97% cov), multiclase informativo (90% cov).
    # --alpha override preserva comportamiento legacy (mismo alpha para ambos).
    if args.alpha is not None:
        alpha_binary = args.alpha
        alpha_multiclass = args.alpha
    else:
        alpha_binary = args.alpha_binary
        alpha_multiclass = args.alpha_multiclass

    # seed_everything al inicio garantiza un orden reproducible del val set entre
    # runs. Sin esto, bajo num_workers>0 el shuffle por worker podría introducir
    # variabilidad sutil en el threshold conformal y romper la asunción de
    # exchangeability del Split Conformal.
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Device: {device}, alpha_binary={alpha_binary}, "
        f"alpha_multiclass={alpha_multiclass}, output_dir={out_dir}"
    )

    # Estrategia en dos pases: construir ambos objetos en memoria ANTES de tocar
    # el disco. Si la multiclase falla a mitad (OOM, NaN en collect_val_probs,
    # hash mismatch), el binario NO queda huérfano en disco.
    #
    # Pase 1: construir (puede fallar — todavía no se ha tocado el disco).
    print("\n=== Pase 1: construir objetos conformal en memoria ===")

    # --- Binario ---
    print("\n=== BinaryConformal (fit) ===")
    _, val_loader_bin, _ = get_dataloaders(mode="binary", batch_size=args.batch_size, seed=args.seed)
    model_bin = _build_and_load("binary", args.binary_checkpoint, device)
    temp_bin = _load_temp_safe(args.temperature_binary)
    if temp_bin is not None:
        temp_bin.to(device)  # TemperatureScaler.load() deja el parámetro en CPU.
    probs_bin, labels_bin = collect_val_probs_binary(
        model_bin, val_loader_bin, device, temp_bin
    )
    conf_bin = BinaryConformal(alpha=alpha_binary)
    conf_bin.fit(probs_bin, labels_bin)
    # Persistir el hash del val CSV usado para la calibración.
    conf_bin.val_csv_hash = _compute_val_csv_hash("binary")

    # ConformalRiskControl (opcional). Reusa las probs/labels del binario ya
    # computadas — no hace falta un segundo forward sobre el val set.
    conf_risk = None
    if args.risk_control_fnr is not None:
        print(f"\n=== ConformalRiskControl (fit, target_fnr={args.risk_control_fnr}) ===")
        conf_risk = ConformalRiskControl(target_fnr=args.risk_control_fnr)
        conf_risk.fit(probs_bin, labels_bin)
        # Hash val CSV simétrico a BinaryConformal (mismo val set).
        conf_risk.val_csv_hash = _compute_val_csv_hash("binary")

    del model_bin
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Multiclase (opcional) ---
    conf_multi = None
    if args.multiclass_checkpoint:
        print("\n=== APSConformal (fit) ===")
        _, val_loader_multi, _ = get_dataloaders(
            mode="multiclass", batch_size=args.batch_size, seed=args.seed
        )
        model_multi = _build_and_load("multiclass", args.multiclass_checkpoint, device)
        temp_multi = _load_temp_safe(args.temperature_multiclass)
        if temp_multi is not None:
            temp_multi.to(device)
        probs_multi, labels_multi = collect_val_probs_multiclass(
            model_multi, val_loader_multi, device, temp_multi
        )
        class_names = [MULTICLASS_INDEX_MAP[i] for i in range(probs_multi.shape[1])]
        conf_multi = APSConformal(alpha=alpha_multiclass, class_names=class_names)
        conf_multi.fit(probs_multi, labels_multi)
        # Persistir el hash del val CSV usado para la calibración.
        conf_multi.val_csv_hash = _compute_val_csv_hash("multiclass")
        del model_multi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Pase 2: persistir. Si llegamos aquí, ambos objetos están validados en
    # memoria. Cualquier fallo previo abortó el script sin tocar el disco.
    print("\n=== Pase 2: persistir a disco ===")
    out_bin = out_dir / "conformal_binary.pt"
    conf_bin.save(out_bin)
    print(
        f"[OK] BinaryConformal alpha={alpha_binary} threshold={conf_bin.threshold:.4f} "
        f"n_calib={conf_bin.n_calib} -> {out_bin}"
    )

    if conf_multi is not None:
        out_multi = out_dir / "conformal_multiclass.pt"
        conf_multi.save(out_multi)
        print(
            f"[OK] APSConformal alpha={alpha_multiclass} threshold={conf_multi.threshold:.4f} "
            f"n_calib={conf_multi.n_calib} -> {out_multi}"
        )

    # Persistir ConformalRiskControl en el pase 2, simétrico a los demás.
    if conf_risk is not None:
        out_risk = out_dir / "conformal_risk_binary.pt"
        conf_risk.save(out_risk)
        print(
            f"[OK] ConformalRiskControl target_fnr={conf_risk.target_fnr} "
            f"threshold={conf_risk.threshold:.4f} "
            f"n_calib_pos={conf_risk.n_calib_pos} n_calib_neg={conf_risk.n_calib_neg} "
            f"-> {out_risk}"
        )


if __name__ == "__main__":
    main()
