"""Verificación de entorno clínico.

Comprueba que las versiones instaladas de las dependencias runtime caen dentro
del rango declarado en `requirements.txt`. Un `__import__(module)` que solo
verifica disponibilidad no basta: bajo `pip install -U` un paquete crítico
puede quedar fuera de rango y el import seguir funcionando (p.ej. albumentations
>=2.0 elimina `always_apply`, rompiendo el pipeline de augmentations sin avisar).
Aquí se parsea `requirements.txt` y se compara `module.__version__` contra el
rango declarado, fallando de forma explícita ante cualquier mismatch.

Solo se chequean los paquetes que training e inferencia importan directamente;
las dependencias dev-only (DVC, nbformat, mlflow) quedan fuera para no generar
falsos negativos.
"""
import re
import sys
import warnings
from pathlib import Path

# Mapeo de nombre-pip → módulo importable. pip usa guiones, Python usa
# underscores o nombres alternativos en algunos casos.
_PIP_TO_MODULE = {
    "opencv-python": "cv2",
    "python-multipart": "multipart",
    "scikit-learn": "sklearn",
}

# Subset de paquetes RUNTIME relevantes para training/inferencia. DVC y
# nbformat son dev-only; mlflow es opcional. Los que faltan aquí no generan
# fail — solo se validan los explícitamente listados.
_RUNTIME_PACKAGES = {
    "torch", "torchvision", "timm", "torchmetrics",
    "fastapi", "uvicorn",
    "pandas", "numpy", "opencv-python", "albumentations",
    "scikit-learn",
}


def _parse_requirements(path: Path) -> dict:
    """Parsea `requirements.txt` → {pip_name: spec_str}.

    Soporta el formato `name>=X,<Y` usado en este proyecto. Ignora comentarios
    y líneas vacías. No soporta markers PEP 508 complejos (no los usamos).
    """
    result = {}
    pattern = re.compile(r"^([A-Za-z0-9_\-]+)\s*([^#]*)$")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        name, spec = m.group(1).lower(), m.group(2).strip()
        result[name] = spec
    return result


def _parse_version(v: str) -> tuple:
    """(major, minor, patch) como ints. Ignora sufijos tipo `+cu118`, `rc1`."""
    v = re.split(r"[+\-]", v)[0]
    parts = []
    for p in v.split(".")[:3]:
        try:
            parts.append(int(re.match(r"\d+", p).group()))
        except (AttributeError, ValueError):
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _check_spec(installed: str, spec: str) -> tuple[bool, str]:
    """Verifica `installed` contra `spec` (ej. `>=2.1.0,<3.0`).

    Retorna (ok, razón_si_falla).
    """
    if not spec:
        return True, ""
    inst = _parse_version(installed)
    for constraint in spec.split(","):
        constraint = constraint.strip()
        m = re.match(r"([<>=!]+)\s*([0-9.]+)", constraint)
        if not m:
            continue
        op, ver = m.group(1), m.group(2)
        target = _parse_version(ver)
        if op == ">=" and not (inst >= target):
            return False, f"{installed} < {ver}"
        if op == ">" and not (inst > target):
            return False, f"{installed} <= {ver}"
        if op == "<=" and not (inst <= target):
            return False, f"{installed} > {ver}"
        if op == "<" and not (inst < target):
            return False, f"{installed} >= {ver}"
        if op == "==" and not (inst == target):
            return False, f"{installed} != {ver}"
        if op == "!=" and not (inst != target):
            return False, f"{installed} == {ver}"
    return True, ""


def verify_environment() -> bool:
    print(f"Versión de Python: {sys.version.split()[0]}")

    repo_root = Path(__file__).resolve().parents[2]
    req_file = repo_root / "requirements.txt"
    if not req_file.exists():
        print(f"[ERROR] requirements.txt no encontrado en {req_file}")
        return False

    requirements = _parse_requirements(req_file)
    all_ok = True
    print("\nVerificando librerías runtime...")

    for pip_name in sorted(_RUNTIME_PACKAGES):
        spec = requirements.get(pip_name, "")
        module_name = _PIP_TO_MODULE.get(pip_name, pip_name.replace("-", "_"))
        try:
            mod = __import__(module_name)
        except ImportError as e:
            print(f"[ERROR] {pip_name} ({module_name}): ImportError — {e}")
            all_ok = False
            continue
        version = getattr(mod, "__version__", "unknown")
        if version == "unknown":
            # Se avisa con warnings.warn (capturable por el handler del proceso)
            # en vez de print, para que sea un evento de validación auditable.
            warnings.warn(
                f"{pip_name} ({module_name}): __version__ no disponible (spec={spec!r})",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        ok, reason = _check_spec(version, spec)
        if ok:
            print(f"[OK] {pip_name} {version} (spec {spec or 'any'})")
        else:
            print(f"[ERROR] {pip_name} {version} VIOLA spec {spec!r}: {reason}")
            all_ok = False

    if all_ok:
        print("\n[OK] Entorno correcto — todas las versiones dentro del rango de requirements.txt.")
    else:
        print(
            "\n[AVISO] Mismatches detectados. Riesgo de comportamiento divergente "
            "(p.ej. albumentations >=2.0 rompió `always_apply`). "
            "Ejecuta `pip install -r requirements.txt` para corregir."
        )
    return all_ok


if __name__ == "__main__":
    ok = verify_environment()
    sys.exit(0 if ok else 1)
