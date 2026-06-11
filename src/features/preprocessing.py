import warnings

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from src.data.constants import STAIN_METHOD


def build_stain_transform():
    """Construye el transform de normalización de color del pipeline.

    Esta función es la fuente única de verdad compartida por el entrenamiento
    (dataset.py) y la inferencia (predict.py): garantiza que ambos apliquen
    exactamente la misma normalización de color. Construir esta transformación
    por separado en cada lado introduciría una divergencia silenciosa entre
    train e inferencia que degradaría la precisión sin error visible.

    El único método soportado es ShadesOfGray (constancia de color Minkowski
    p=6), validado empíricamente como suficiente sobre el dataset completo.
    Cualquier otro valor de STAIN_METHOD aborta el arranque con ValueError en
    lugar de degradar en silencio.
    """
    if STAIN_METHOD == "shades_of_gray":
        return ShadesOfGray(p=1.0)
    raise ValueError(
        f"STAIN_METHOD={STAIN_METHOD!r} no soportado. "
        "Único valor válido: 'shades_of_gray'."
    )


class HairRemoval(ImageOnlyTransform):
    """
    Transformación para la eliminación digital del vello en imágenes dermatoscópicas.

    Fundamento Teórico:
    El vello ocluye la lesión y crea texturas falsas que confunden a las redes neuronales.
    Esta implementación sigue la lógica del algoritmo clásico 'DullRazor':
    1. Detección: Uso de 'BlackHat Transform' morfológico para resaltar estructuras oscuras y finas.
    2. Enmascaramiento: Binarización (Thresholding) para aislar los pelos.
    3. Restauración: Inpainting (Telea Algorithm) para interpolar la piel bajo el pelo.
    """

    def __init__(self, kernel_size=17, p=1.0):
        # Albumentations >=2.0 eliminó `always_apply` del constructor de
        # ImageOnlyTransform; pasamos solo `p`. Con p=1.0 la transformación se
        # aplica siempre, que es el comportamiento deseado para el preprocesado.
        super(HairRemoval, self).__init__(p=p)
        self.kernel_size = kernel_size

    def apply(self, image, **params):
        # 1. Convertir a escala de grises para análisis estructural
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 2. Operación BlackHat (TopHat Inverso)
        # BlackHat = (Cierre Morfológico) - (Imagen Original)
        # El Cierre elimina elementos oscuros pequeños (pelos). Al restar, nos quedamos SOLO con los pelos.
        # Kernel adaptativo: escala el tamaño del kernel según la resolución de la imagen.
        # A 300px, kernel_size=17 fijo puede ser demasiado grande para pelos finos.
        # Referencia: 512px → 17 (tamaño DullRazor estándar).
        max_dim = max(image.shape[:2])
        adaptive_kernel = max(5, int(self.kernel_size * max_dim / 512))
        if adaptive_kernel % 2 == 0:
            adaptive_kernel += 1  # Kernels morfológicos deben ser impares
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (adaptive_kernel, adaptive_kernel))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # 3. Thresholding (Umbralización)
        # Creamos una máscara binaria donde Blanco = Pelo, Negro = Piel.
        # Umbral bajo (10) para capturar incluso pelos finos.
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # 4. Inpainting
        # Rellena los píxeles marcados en la máscara usando información de los vecinos.
        # cv2.INPAINT_TELEA usa el método Fast Marching.
        im_pain = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

        return im_pain


class ShadesOfGray(ImageOnlyTransform):
    """
    Algoritmo de Constancia de Color (Color Constancy) 'Shades of Gray'.

    Fundamento Teórico:
    Las imágenes dermatoscópicas varían en color debido a diferentes fuentes de luz y sensores.
    Este algoritmo normaliza la iluminación asumiendo que el promedio espacial de la escena
    debería ser gris neutro (Grey World Assumption generalizado).

    Fórmula: Minkowski Norm (p-norm)
    - p=1 equivale al algoritmo "Grey World" (promedio simple).
    - p=infinity equivale a "White Patch" (máximo reflejo).
    - p=6 es un estándar empírico robusto para dermatoscopia (Finlayson et al.).
    """

    def __init__(self, power=6, gamma=None, p=1.0):
        # Albumentations >=2.0: pasar solo `p` (ver HairRemoval).
        super(ShadesOfGray, self).__init__(p=p)
        self.power = power
        self.gamma = gamma
        # Auditoría de imágenes degeneradas. Cuando el estimador del iluminante
        # colapsa (parches completamente negros, viñeteado total, vello eliminado
        # de forma demasiado agresiva) la imagen se devuelve sin corregir. Llevamos
        # la cuenta y emitimos un único warning si los fallbacks superan el 1% sobre
        # al menos 100 muestras: señal de que el pipeline previo está roto.
        self._degenerate_count = 0
        self._total_count = 0
        self._warned = False

    def apply(self, image, **params):
        img_dtype = image.dtype
        self._total_count += 1

        # Crítico: trabajar en float32 antes de elevar a la potencia. Elevar un
        # uint8 (máx 255) a la sexta potencia desbordaría el rango del tipo de
        # forma cíclica, produciendo valores basura que destruyen el color.
        image_float = image.astype("float32")

        # Corrección Gamma previa opcional
        if self.gamma is not None:
            image_float = image_float / 255.0
            image_float = np.power(image_float, self.gamma)

        # 1. Estimación del Iluminante (Luz de la escena)
        R = image_float[:, :, 0]
        G = image_float[:, :, 1]
        B = image_float[:, :, 2]

        # Calculamos la norma Minkowski P promediada
        r_norm = np.power(np.mean(np.power(R, self.power)), 1 / self.power)
        g_norm = np.power(np.mean(np.power(G, self.power)), 1 / self.power)
        b_norm = np.power(np.mean(np.power(B, self.power)), 1 / self.power)

        # Vector de iluminación estimado
        som_norm = np.sqrt(r_norm**2 + g_norm**2 + b_norm**2)

        # Prevención de división por cero. Si la imagen es un parche
        # completamente negro (común en bordes de máscaras), som_norm vale 0:
        # dividir 0/0 produce NaN que envenenaría todo el lote.
        eps = 1e-6
        if som_norm < eps or np.isnan(som_norm):
            # Cuenta el fallback y avisa una sola vez si supera el 1%. Parches
            # negros aislados son normales (bordes tras crop o eliminación de
            # vello); superar el 1% del lote sugiere un pipeline previo roto.
            self._degenerate_count += 1
            if (
                not self._warned
                and self._total_count >= 100
                and self._degenerate_count / self._total_count > 0.01
            ):
                warnings.warn(
                    f"ShadesOfGray: {self._degenerate_count}/{self._total_count} "
                    f"({100 * self._degenerate_count / self._total_count:.1f}%) "
                    f"imágenes con iluminante degenerado (parches negros, "
                    f"viñeteado, HairRemoval excesivo). Revisar pipeline upstream.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return np.clip(image_float, 0, 255).astype(img_dtype)

        r_color = r_norm / som_norm
        g_color = g_norm / som_norm
        b_color = b_norm / som_norm

        # 2. Corrección Cromática (Von Kries Model)
        # Dividimos cada canal por su estimación de luz para "blanquear" la iluminación.

        # Evitar división por cero, NaNs sutiles, y overflow en corrección cromática.
        # Sin clip superior, una imagen muy brillante puede producir r_color >> 1 → artefactos.
        r_color = np.clip(r_color, eps, 100.0) if not np.isnan(r_color) else 1.0
        g_color = np.clip(g_color, eps, 100.0) if not np.isnan(g_color) else 1.0
        b_color = np.clip(b_color, eps, 100.0) if not np.isnan(b_color) else 1.0

        new_image = image_float.copy() # Usamos la versión casteada
        new_image[:, :, 0] /= r_color * np.sqrt(3)
        new_image[:, :, 1] /= g_color * np.sqrt(3)
        new_image[:, :, 2] /= b_color * np.sqrt(3)

        # Restaurar Gamma y rango 0-255
        if self.gamma is not None:
            new_image = np.power(new_image, 1 / self.gamma)
            new_image = new_image * 255.0

        new_image = np.clip(new_image, 0, 255).astype(img_dtype)

        return new_image
