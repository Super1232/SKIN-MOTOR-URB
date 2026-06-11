import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss para problemas de extremo desbalance de clases (Ej. Melanoma 2% vs Benigno 98%).

    Teoría Clínica:
    La Cross-Entropy estándar se conforma rápido con aprender a predecir la clase mayoritaria
    ("Todo es Benigno"). Focal Loss soluciona esto mediante dos enfoques:

    1. alpha: Pesa más a la clase minoritaria (Melanoma).
    2. gamma: Reduce dinámicamente el peso de los ejemplos "fáciles" (lunares obvios) y
       obliga al modelo a centrarse en los "difíciles" (melanomas sutiles).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction="mean", poly_epsilon=0.0):
        super(FocalLoss, self).__init__()
        # alpha debe ser explícito. El default habitual de la Focal Loss (0.25)
        # da MENOS peso a la clase minoritaria, lo contrario de lo que necesita
        # la detección de cáncer. Exigir alpha explícito evita usarla con la
        # semántica invertida por descuido.
        if alpha is None:
            raise ValueError(
                "alpha debe ser explícito para FocalLoss. "
                "Usar alpha=0.7 para detección de melanoma (up-weight clase minoritaria)."
            )
        if poly_epsilon < 0.0:
            raise ValueError(
                f"poly_epsilon debe ser >= 0. Recibido: {poly_epsilon}. "
                "Default 0.0 preserva el comportamiento del Focal estándar. "
                "El paper recomienda poly_epsilon=1.0 para Poly-1-Focal."
            )
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # Polyloss (Leng et al. ICCV 2022, arXiv:2204.12511):
        # `L_poly-1 = L_focal + epsilon * (1 - pt)`. El término lineal `(1-pt)`
        # complementa el polinomio `(1-pt)^gamma` del focal: cerca de pt≈1 el
        # término focal plateau-ea (gradiente → 0), pero `(1-pt)` mantiene
        # gradiente lineal → el modelo sigue afinando ejemplos casi correctos.
        # Útil en clases raras donde pt~0.9 no es "suficiente" y se necesita
        # empujar hasta pt~0.95. Usa pt ORIGINAL (sin smoothing), igual que el
        # factor focal.
        self.poly_epsilon = poly_epsilon
        # Este wrapper NO expone `pos_weight`. El desacoplamiento (pt con targets
        # originales + alpha_t por clase + (1-pt)^gamma) ya penaliza la clase
        # positiva; combinarlo con `pos_weight` produciría una triple penalización
        # sobre los melanomas difíciles (pos_weight × alpha × (1-pt)^gamma). Se
        # documenta para que no se añada creyendo que es la forma canónica de
        # balancear el binario con PyTorch, sin notar que el wrapper ya aplica
        # `alpha`.

    def forward(self, inputs, targets):
        """
        Inputs: Logits crudos de la red [Batch] (salida del modelo tras squeeze)
        Targets: Etiquetas binarias [Batch] floats (0.0 o 1.0)
        """
        # Cálculo en float32 explícito. Bajo fp16, `(1-pt)^gamma` cerca de pt≈0
        # pierde precisión en los gradientes y sigmoid/BCE de logits extremos
        # saturan. En bf16 (H100) es inocuo, pero mantiene consistencia con el
        # resto de losses del archivo.
        with torch.amp.autocast('cuda', enabled=False):
            inputs_f32 = inputs.float()
            targets_f32 = targets.float()

            # Medical Label Smoothing (0.1)
            smoothing = self.label_smoothing
            smoothed_targets = targets_f32 * (1.0 - smoothing) + 0.5 * smoothing

            bce_loss = F.binary_cross_entropy_with_logits(
                inputs_f32, smoothed_targets, reduction="none"
            )

            # Desacoplamiento: pt con targets originales (sin suavizar) para
            # preservar la semántica del factor focal.
            p = torch.sigmoid(inputs_f32)
            pt = p * targets_f32 + (1 - p) * (1 - targets_f32)

            # Clamp simétrico con MulticlassFocalLoss. `pt` puede alcanzar
            # exactamente 1.0 (cuando la predicción coincide con el target con
            # confianza saturada), haciendo `(1-pt) = 0`. Con un gamma no-entero
            # (gamma=2.0 es entero hoy, pero si se cambia a 2.5 el contrato se
            # rompe), `0**2.5` y su gradiente quedan mal definidos. El clamp a
            # `[0, 1-1e-7]` garantiza `(1-pt) > 0` estricto. Epsilon 1e-7 por
            # simetría con el multiclase.
            pt = pt.clamp(min=0.0, max=1.0 - 1e-7)

            alpha_t = targets_f32 * self.alpha + (1 - targets_f32) * (1 - self.alpha)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

            # Término Poly-1 aditivo. Solo si poly_epsilon > 0 (default 0.0
            # preserva el comportamiento exacto del FocalLoss original). pt es la
            # probabilidad de la clase verdadera con targets originales,
            # consistente con el factor focal. `(1-pt)` añade gradiente lineal
            # cerca de pt≈1 donde `(1-pt)^gamma` plateau-ea.
            if self.poly_epsilon > 0.0:
                focal_loss = focal_loss + self.poly_epsilon * (1.0 - pt)

            if self.reduction == "mean":
                return torch.mean(focal_loss)
            elif self.reduction == "sum":
                return torch.sum(focal_loss)
            else:
                return focal_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Optimiza el espacio de features para que embeddings de la misma clase se acerquen
    y de clases distintas se alejen. Complementario a Focal Loss: mientras Focal opera
    sobre logits finales, SupCon opera sobre features intermedias, creando un espacio
    de representación más discriminativo.

    Especialmente útil para clases visualmente similares (AK vs BKL, MEL vs NV).
    Solo se aplica en batches sin MixUp (labels mezclados no tienen pares claros).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Tensor [B, D] — embeddings L2-normalizados
            labels: Tensor [B] — etiquetas de clase (enteros)
        Returns:
            loss: Tensor escalar
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            # Aviso en vez de retorno silencioso. SupCon necesita al menos 2
            # muestras para formar pares positivos (la similitud consigo mismo se
            # excluye con fill_diagonal_(0)). Un batch<2 suele indicar
            # `drop_last=False` en el DataLoader combinado con
            # `len(dataset) % batch_size == 1` en la última época — silenciarlo
            # podría enmascarar un problema del data pipeline.
            warnings.warn(
                f"SupConLoss: batch_size={batch_size} < 2, sin pares positivos. "
                "Retornando 0.0 (no-op). Causa típica: drop_last=False en el "
                "DataLoader con |dataset| % batch_size == 1 en la última época. "
                "Ver src/data/dataset.py::get_dataloaders (drop_last=True es el "
                "default en el pipeline oficial).",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cálculo en float32 explícito para toda la computación de SupCon.
        # `similarity = matmul(features, features.T) / 0.07` produce valores
        # hasta ~14.3; `exp(14.3) = 1.6M` desborda el max de fp16 (65504), y
        # `torch.matmul` promociona a fp16 bajo autocast → overflow silencioso.
        # Mismo motivo que en GeM, SeesawLoss y BalancedSoftmaxLoss.
        with torch.amp.autocast('cuda', enabled=False):
            features = features.float()
            labels = labels.long()

            # L2 normalize features
            features = F.normalize(features, dim=1)

            # Similaridad coseno pairwise: [B, B]
            similarity = torch.matmul(features, features.T) / self.temperature

            # Máscara de positivos: misma clase, excluyendo la diagonal
            labels = labels.view(-1, 1)
            mask_positives = (labels == labels.T).float()
            mask_positives.fill_diagonal_(0)  # Excluir self-similarity

            # Máscara de todos excepto diagonal
            logits_mask = torch.ones_like(similarity)
            logits_mask.fill_diagonal_(0)

            # Estabilidad numérica: restar max
            logits_max, _ = similarity.detach().max(dim=1, keepdim=True)
            similarity = similarity - logits_max

            # Log-sum-exp sobre negativos + positivos (denominador). Si todos los
            # exponenciales hacen underflow a ~0 (similitudes muy negativas),
            # `log(suma_diminuta + eps) ≈ -18.4` crea log_probs extremadamente
            # negativos → picos del loss. `.clamp(min=1e-8)` en el sum es más
            # robusto que sumar eps antes del log.
            exp_logits = torch.exp(similarity) * logits_mask
            log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))

            # Promedio sobre positivos
            n_positives = mask_positives.sum(dim=1)
            # Guard: si una muestra no tiene positivos, skip (no contribuye al loss)
            has_positives = n_positives > 0

            if not has_positives.any():
                return torch.tensor(0.0, device=device, requires_grad=True)

            mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1)
            mean_log_prob_pos = mean_log_prob_pos[has_positives] / n_positives[has_positives]

            loss = -mean_log_prob_pos.mean()
        return loss


class SeesawLoss(nn.Module):
    """
    Seesaw Loss (Wang et al., CVPR 2021) para clasificación multiclase con
    long-tail extrema — adaptado al pipeline dermatoscópico.

    Motivación clínica
    ------------------
    La Focal Loss trata a todas las clases por igual en el factor `(1-pt)^γ`.
    En nuestro dataset el ratio Nevus:Dermatofibroma es ~100:1 — el modelo
    puede considerar "difíciles" a muestras de Nevus que ni siquiera
    corresponden a Dermatofibroma, desperdiciando capacidad.

    Seesaw introduce dos factores que modulan los logits de clases
    no-target ANTES del softmax:

    - Mitigation factor M_ij: si la clase target i tiene MUCHAS más muestras
      que la clase j (head), penaliza menos al modelo por dar probabilidad
      alta a j. Expresado como `(n_j / n_i)^p` si `n_i > n_j`, si no 1.0.
      Evita que las clases tail "arrastren" al modelo contra las head.

    - Compensation factor C_ij: si la probabilidad predicha para la clase
      no-target j supera a la de la target i, incrementa la penalización
      por `(p_j / p_i)^q`. Castiga errores confiados.

    El loss final es cross-entropy con logits `z'_j = z_j + log(M_ij * C_ij)`
    para j ≠ i, preservando `z_i` inalterado.

    Balance de clases
    -----------------
    NO usa `class_weights`. La información de balance entra vía los ratios
    `n_j/n_i`, no vía un reweighting global del loss. El
    `WeightedRandomSampler` (β=0.999) sigue activo; Seesaw opera ortogonal.

    Args:
        class_counts: list/tensor [num_classes] con conteos por clase en train.
        p: exponente de mitigation (default 0.8 del paper).
        q: exponente de compensation (default 2.0 del paper).
        label_smoothing: suavizado uniforme sobre target (default 0.1).
        eps: estabilidad numérica para ratios de probabilidad.
    """

    def __init__(
        self,
        class_counts,
        p: float = 0.8,
        q: float = 2.0,
        label_smoothing: float = 0.1,
        eps: float = 1e-2,
        reduction: str = "mean",
    ):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or (counts <= 0).any():
            raise ValueError(
                "class_counts debe ser 1D con todos los valores > 0 "
                f"(recibido shape={tuple(counts.shape)}, min={counts.min().item()})"
            )
        # Buffer para que se mueva con .to(device) y se guarde en state_dict
        # (persistente para auditoría post-entrenamiento).
        self.register_buffer("class_counts", counts)
        self.p = p
        self.q = q
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits crudos [B, C].
            targets: índices de clase [B].
        """
        num_classes = inputs.size(1)
        if self.class_counts.numel() != num_classes:
            raise ValueError(
                f"class_counts tiene {self.class_counts.numel()} entradas pero "
                f"inputs tiene {num_classes} clases."
            )

        # Cálculo en float32 explícito para todo el forward (compensation factor,
        # adjusted logits, log_softmax, NLL, label smoothing y reduction). Bajo
        # AMP, `ratio_p.pow(q)` puede desbordar el max de fp16 (65504) cuando p_i
        # es pequeña (p.ej. p_i=0.001, ratio=1000, 1000^2=1M). Además `eps=1e-2`
        # es representable en fp16, pero cualquier eps < 6.1e-5 haría underflow a
        # 0 → clamp inútil. El guard cubre también el label smoothing y la
        # reduction para no depender de que las ops no-cast preserven el dtype en
        # versiones futuras de PyTorch (consistencia con MulticlassFocalLoss).
        with torch.amp.autocast('cuda', enabled=False):
            inputs_f32 = inputs.float()
            targets = targets.long()
            counts = self.class_counts

            # ---- Mitigation factor M_ij = (n_j / n_i)^p si n_i > n_j, si no 1 ----
            # Construimos una fila por muestra i usando n_i = counts[target_i].
            n_i = counts[targets].unsqueeze(1)  # [B, 1]
            ratio = counts.unsqueeze(0) / n_i.clamp(min=1.0)  # [B, C], elemento j = n_j/n_i
            mitigation = torch.where(
                ratio < 1.0,
                ratio.pow(self.p),
                torch.ones_like(ratio),
            )  # [B, C] — para j donde n_i > n_j, aplica (n_j/n_i)^p; si no, 1

            # ---- Compensation factor C_ij = (p_j/p_i)^q si p_j > p_i, si no 1 ----
            # Probabilidades sin gradiente para evitar bucle de retroalimentación.
            with torch.no_grad():
                probs = F.softmax(inputs_f32.detach(), dim=1).clamp(min=self.eps)
            p_i = probs.gather(1, targets.unsqueeze(1))  # [B, 1]
            ratio_p = probs / p_i  # [B, C]
            compensation = torch.where(
                ratio_p > 1.0,
                ratio_p.pow(self.q),
                torch.ones_like(ratio_p),
            )

            # ---- Combinar y aplicar a logits de clases no-target ----
            seesaw_weight = mitigation.float() * compensation  # [B, C]
            # Para la clase target la corrección debe ser neutra (log(1)=0).
            target_mask = F.one_hot(targets, num_classes=num_classes).float()
            seesaw_weight = seesaw_weight * (1.0 - target_mask) + target_mask

            # Cross-entropy con logits modificados z'_j = z_j + log(weight_j).
            log_weight = torch.log(seesaw_weight.clamp(min=1e-8))
            adjusted_logits = inputs_f32 + log_weight
            log_probs = F.log_softmax(adjusted_logits, dim=1)

            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            if self.label_smoothing > 0:
                smooth_loss = -log_probs.mean(dim=1)
                loss = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss
            else:
                loss = nll

            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax (Ren et al., NeurIPS 2020).

    Inyecta el prior de cada clase directamente en el softmax:
        p(y=k | x) = softmax(z_k + log(n_k))
    Equivale a entrenar como si los batches fueran balanceados, mientras los
    logits siguen reflejando la distribución real. Matemáticamente equivalente
    a hacer logit-adjustment post-hoc pero optimizado durante el training.

    Combinable con Seesaw como pérdida auxiliar (peso 0.2) para estabilizar la
    convergencia de clases raras sin doble corrección con el sampler: Balanced
    Softmax actúa sobre la función de decisión, no sobre el muestreo.

    No usa `class_weights`.
    """

    def __init__(self, class_counts, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or (counts <= 0).any():
            raise ValueError(
                "class_counts debe ser 1D con todos los valores > 0 "
                f"(recibido shape={tuple(counts.shape)}, min={counts.min().item()})"
            )
        # log(prior) se almacena precomputado; se suma directamente a logits.
        log_prior = torch.log(counts / counts.sum())
        self.register_buffer("log_prior", log_prior)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.log_prior.numel() != inputs.size(1):
            raise ValueError(
                f"log_prior tiene {self.log_prior.numel()} entradas pero "
                f"inputs tiene {inputs.size(1)} clases."
            )
        # Cálculo en float32 explícito para todo el forward. Bajo AMP, el
        # log_prior de clases raras (p.ej. Dermatofibroma: log(84/30000) ~ -5.87)
        # reduce la precisión de fp16 en log_softmax y degrada los gradientes de
        # las clases minoritarias. El guard cubre NLL + label smoothing +
        # reduction por consistencia con SeesawLoss y MulticlassFocalLoss.
        with torch.amp.autocast('cuda', enabled=False):
            targets = targets.long()
            adjusted_logits = inputs.float() + self.log_prior.unsqueeze(0)
            log_probs = F.log_softmax(adjusted_logits, dim=1)
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            if self.label_smoothing > 0:
                smooth_loss = -log_probs.mean(dim=1)
                loss = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss
            else:
                loss = nll
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss


class MulticlassFocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase (8 tipos de lesión cutánea).

    CrossEntropyLoss trata todos los errores por igual: fallar un Nevus obvio pesa lo mismo
    que confundir Actinic Keratosis con Benign Keratosis. Focal Loss corrige esto:

    - Ejemplos fáciles (pt alto): peso reducido → el modelo no desperdicia capacidad.
    - Ejemplos difíciles (pt bajo): peso alto → el modelo se concentra en las fronteras
      de decisión confusas entre clases visualmente similares.

    Compatible con label smoothing y MixUp (targets son índices enteros).
    NO usa class_weights (el WeightedRandomSampler ya maneja el equilibrio de clases).
    """

    def __init__(self, gamma=2.0, label_smoothing=0.1, reduction="mean", poly_epsilon=0.0):
        super().__init__()
        if poly_epsilon < 0.0:
            raise ValueError(
                f"poly_epsilon debe ser >= 0. Recibido: {poly_epsilon}. "
                "Default 0.0 preserva el comportamiento del Focal estándar. "
                "El paper recomienda poly_epsilon=1.0 para Poly-1-Focal."
            )
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # Polyloss (Leng et al. ICCV 2022). Misma semántica que la FocalLoss
        # binaria: el término aditivo `epsilon * (1-pt)` mantiene gradiente
        # lineal cerca de pt≈1 donde `(1-pt)^gamma` plateau-ea. Especialmente
        # beneficioso para Dermatofibroma (84 imgs), donde empujar pt de 0.85 a
        # 0.95 importa más que optimizar pt=0.99→1.0.
        self.poly_epsilon = poly_epsilon

    def forward(self, inputs, targets):
        """
        Inputs: Logits crudos [Batch, num_classes]
        Targets: Índices de clase [Batch] (enteros)
        """
        # Cálculo en float32 explícito por consistencia con las demás losses
        # (Seesaw, BalancedSoftmax, SupCon, GeM). En bf16 (H100) el impacto
        # numérico es nulo, pero en fp16 (Colab T4) `(1-pt)^gamma` cerca de pt≈0
        # puede perder precisión y `F.log_softmax` de 8 clases con logits
        # extremos satura.
        with torch.amp.autocast('cuda', enabled=False):
            inputs_f32 = inputs.float()
            # Cast explícito a long. `gather` requiere índices int64; un caller
            # que pase targets float32 (p.ej. por un MixUp mal gestionado)
            # obtendría un error críptico "gather_cuda_" sin contexto.
            targets = targets.long()
            # Log-softmax para estabilidad numérica (evita log(0))
            log_probs = F.log_softmax(inputs_f32, dim=1)
            probs = torch.exp(log_probs)

            # pt = probabilidad asignada a la clase correcta (sin suavizar)
            # Desacoplamiento: pt usa targets originales para que (1-pt)^gamma refleje
            # la dificultad REAL del ejemplo, no la versión suavizada.
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Guard numérico: torch.exp(log_softmax) puede producir pt ligeramente
            # > 1.0 por precisión de punto flotante. Con gamma no-entero,
            # (1-pt)^gamma con base negativa produce NaN. Clamp conservador a
            # [0, 1-eps].
            pt = pt.clamp(min=0.0, max=1.0 - 1e-7)

            # NLL de la clase correcta
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Label Smoothing: suaviza la distribución target para absorber ruido de etiquetado
            # (1-s)*nll_hard + s*nll_uniform — mismo principio que CrossEntropyLoss(label_smoothing)
            if self.label_smoothing > 0:
                smooth_loss = -log_probs.mean(dim=1)
                nll = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss

            # Factor focal: (1-pt)^gamma down-weights ejemplos fáciles
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * nll

            # Término Poly-1 aditivo. Solo si poly_epsilon > 0 (default 0.0
            # preserva el comportamiento exacto del FocalLoss original). pt usa
            # targets originales (clamp ya aplicado) → consistente con el
            # decoupling del factor focal.
            if self.poly_epsilon > 0.0:
                loss = loss + self.poly_epsilon * (1.0 - pt)

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss
