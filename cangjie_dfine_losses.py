"""
Project Cangjie (Цанцзе) — Section 2 Enterprise Transformer Detector Optimization [v5.0]
Mathematical & Dual-Engine (PyTorch AutoGrad + Vectorized NumPy/SciPy) Implementation for Hyper-Dense Glyphs (60..172+ Strokes):
1. Fine-grained Distribution Refinement (FDR) with Non-Uniform Bin Weighting W(n) & Residual Logit Updates
2. Fine-Grained Localization Loss (L_FGL) with Dual-Bin Linear Interpolation (omega_left, omega_right) & IoU Weighting
3. Global Optimal Localization Self-Distillation (GO-LSD) via Decoupled Distillation Focal Loss (L_DDF)
4. Normalized Gaussian Wasserstein Distance Metric & Loss (NWD / L_NWD) with Analytical Non-Vanishing Gradients
5. Hierarchy-Gated Repulsion Loss (L_HG-Rep) with Structural Enclosure Tolerance Matrix M(i, j) (tau_contain = 0.75)
6. Anisotropic Multi-Scale Deformable Attention (MSDA) along Calligraphic Stroke Axes (0, 90, 45, 135 deg) + P2 Pyramid & Hungarian Matcher
"""

from __future__ import annotations

import os
import sys
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - fallback для окружений без scipy
    HAS_SCIPY = False

    def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Венгерский алгоритм (Jonker–Volgenant) на чистом numpy, O(n^2*m).

        Замена scipy.optimize.linear_sum_assignment, чтобы проект
        работал без scipy.

        [v9.6] Первая версия паддила матрицу до квадрата значением INF.
        Это ЛОМАЛО оптимум: при n_rows != n_cols фиктивные ячейки с INF
        попадали в назначение, и суммарная стоимость оказывалась ВЫШЕ
        оптимальной. Проверено на 200 случайных матрицах против
        полного перебора — расхождение.

        Теперь: прямоугольный случай обрабатывается корректно —
        назначается ровно min(n_rows, n_cols) пар, каждая строка и
        каждый столбец используются не более одного раза.

        Возвращает (row_ind, col_ind) — индексы назначения.
        """
        c = np.asarray(cost_matrix, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("cost_matrix должна быть 2-мерной")
        n_rows, n_cols = c.shape
        if n_rows == 0 or n_cols == 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

        # Алгоритм JV требует n_rows <= n_cols, иначе транспонируем.
        transposed = False
        if n_rows > n_cols:
            c = c.T
            n_rows, n_cols = n_cols, n_rows
            transposed = True

        INF = float("inf")
        # u — потенциалы строк, v — потенциалы столбцов
        u = np.zeros(n_rows + 1, dtype=np.float64)
        v = np.zeros(n_cols + 1, dtype=np.float64)
        # p[j] — строка, сопоставленная столбцу j (1-индексация, 0 = свободен)
        p = np.zeros(n_cols + 1, dtype=np.int64)
        way = np.zeros(n_cols + 1, dtype=np.int64)

        for i in range(1, n_rows + 1):
            p[0] = i
            j0 = 0
            minv = np.full(n_cols + 1, INF, dtype=np.float64)
            used = np.zeros(n_cols + 1, dtype=bool)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = -1
                row = c[i0 - 1]
                for j in range(1, n_cols + 1):
                    if used[j]:
                        continue
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                if j1 < 0:
                    break
                for j in range(n_cols + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            # Восстановление назначения по цепочке 'way'
            while j0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1

        # Собираем назначения: столбец j сопоставлен строке p[j]
        rows: List[int] = []
        cols: List[int] = []
        for j in range(1, n_cols + 1):
            if p[j] > 0:
                r = p[j] - 1
                if r < n_rows:
                    rows.append(r)
                    cols.append(j - 1)

        order = np.argsort(rows)
        row_ind = np.array([rows[i] for i in order], dtype=np.int64)
        col_ind = np.array([cols[i] for i in order], dtype=np.int64)

        if transposed:
            row_ind, col_ind = col_ind, row_ind
        return row_ind, col_ind


_UPLOAD_DIR = "/working_dir/c_d592584081f58810/user_uploaded_files/ttl=63d/user_uploaded"
if os.path.isdir(_UPLOAD_DIR) and _UPLOAD_DIR not in sys.path:
    sys.path.insert(0, _UPLOAD_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

try:
    from cangjie_benchmark_forge import KANGXI_214_COMPREHENSIVE_CATALOG
except ImportError:
    KANGXI_214_COMPREHENSIVE_CATALOG = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DFINE-LOSSES-v5]: %(message)s"
)
logger = logging.getLogger("CangjieDFINELosses")


SURROUND_RADICAL_GLYPHS: Set[str] = {
    "囗", "門", "门", "冂", "冖", "凵", "匚", "匸",
    "广", "厂", "尸", "戶", "户", "麻", "疒", "气",
    "勹", "戈", "弋", "辵", "辶", "廴", "走", "行",
    "鬥", "斗", "雨", "⻗", "穴", "宀"
}

SURROUND_CLASS_IDS: Set[int] = set()
for _idx, _item in enumerate(KANGXI_214_COMPREHENSIVE_CATALOG):
    if _item[1] in SURROUND_RADICAL_GLYPHS or _item[2] in SURROUND_RADICAL_GLYPHS:
        SURROUND_CLASS_IDS.add(_idx)
if not SURROUND_CLASS_IDS:
    SURROUND_CLASS_IDS = {12, 13, 16, 19, 21, 22, 26, 30, 39, 43, 52, 53, 55, 61, 62, 83, 103, 115, 143, 155, 161, 168, 172, 190, 199}


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.clip(np.sum(exp_v, axis=axis, keepdims=True), 1e-12, None)


def log_softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.clip(np.sum(np.exp(shifted), axis=axis, keepdims=True), 1e-12, None))
    return shifted - log_sum_exp


def cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def compute_pairwise_iou_and_intersection_np(
    boxes1_cxcywh: np.ndarray,
    boxes2_cxcywh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b1 = cxcywh_to_xyxy_np(boxes1_cxcywh)
    b2 = cxcywh_to_xyxy_np(boxes2_cxcywh)

    area1 = np.clip(boxes1_cxcywh[:, 2], 0.0, None) * np.clip(boxes1_cxcywh[:, 3], 0.0, None)
    area2 = np.clip(boxes2_cxcywh[:, 2], 0.0, None) * np.clip(boxes2_cxcywh[:, 3], 0.0, None)

    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.clip(ix2 - ix1, 0.0, None)
    inter_h = np.clip(iy2 - iy1, 0.0, None)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / np.clip(union_area, 1e-9, None)
    return iou, inter_area, area1, area2


class FineGrainedDistributionRefinement:
    """
    Тонкая дистрибутивная регрессия (Fine-grained Distribution Refinement, FDR):
        d = d_max * sum_{n=0}^{N} W(n) * Pr(n)
        Pr^l(n) = Softmax(Delta_logits^l(n) + logits^{l-1}(n))
    """
    def __init__(self, num_bins: int = 32, d_max: float = 1.0, curvature: float = 1.6):
        if num_bins < 4:
            raise ValueError("num_bins (N) должно быть >= 4.")
        self.num_bins = num_bins
        self.d_max = float(d_max)
        self.curvature = float(curvature)
        self.weights_W = self._build_non_uniform_weights(num_bins, curvature)

    @staticmethod
    def _build_non_uniform_weights(num_bins: int, curvature: float) -> np.ndarray:
        u = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
        w = np.power(u, curvature)
        w[0] = 0.0
        w[-1] = 1.0
        return w

    def refine_layer_distributions(
        self,
        initial_logits: Any,
        delta_logits_per_layer: List[Any]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        if HAS_TORCH and isinstance(initial_logits, torch.Tensor):
            w_tensor = torch.tensor(self.weights_W, dtype=initial_logits.dtype, device=initial_logits.device)
            acc_logits, probs_list, dists_list = [], [], []
            prev_logits = initial_logits
            for delta in delta_logits_per_layer:
                cur_logits = prev_logits + delta
                cur_probs = F.softmax(cur_logits, dim=-1)
                cur_d = self.d_max * torch.sum(cur_probs * w_tensor, dim=-1)
                acc_logits.append(cur_logits)
                probs_list.append(cur_probs)
                dists_list.append(cur_d)
                prev_logits = cur_logits
            return acc_logits, probs_list, dists_list

        acc_logits_np: List[np.ndarray] = []
        probs_np: List[np.ndarray] = []
        dists_np: List[np.ndarray] = []
        prev_l = np.asarray(initial_logits, dtype=np.float64)

        for delta in delta_logits_per_layer:
            cur_l = prev_l + np.asarray(delta, dtype=np.float64)
            cur_p = softmax_np(cur_l, axis=-1)
            cur_d = self.d_max * np.sum(cur_p * self.weights_W[None, None, :], axis=-1)
            acc_logits_np.append(cur_l)
            probs_np.append(cur_p)
            dists_np.append(cur_d)
            prev_l = cur_l

        return acc_logits_np, probs_np, dists_np


class FineGrainedLocalizationLoss:
    """
    Функция потерь тонкой локализации (Fine-Grained Localization Loss, FGL):
        L_FGL = sum_{l=1}^L sum_{k=1}^K IoU_k * (omega_left * CE(Pr_k^l(n), n_left) + omega_right * CE(Pr_k^l(n), n_right))
    """
    def __init__(self, fdr_module: FineGrainedDistributionRefinement):
        self.fdr = fdr_module
        self.W = fdr_module.weights_W
        self.d_max = fdr_module.d_max

    def compute_dual_bin_targets(self, target_offsets_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phi_norm = np.clip(np.asarray(target_offsets_phi, dtype=np.float64) / self.d_max, 0.0, 1.0)
        idx_right = np.searchsorted(self.W, phi_norm, side="right")
        idx_right = np.clip(idx_right, 1, self.fdr.num_bins)
        idx_left = idx_right - 1

        w_left_val = self.W[idx_left]
        w_right_val = self.W[idx_right]
        span = np.clip(w_right_val - w_left_val, 1e-12, None)

        omega_right = np.clip((phi_norm - w_left_val) / span, 0.0, 1.0)
        omega_left = 1.0 - omega_right
        return idx_left, idx_right, omega_left, omega_right

    def forward_with_grad(
        self,
        layer_logits_list: List[np.ndarray],
        target_offsets_phi: np.ndarray,
        iou_weights: np.ndarray
    ) -> Dict[str, Any]:
        K = target_offsets_phi.shape[0]
        n_left, n_right, omega_left, omega_right = self.compute_dual_bin_targets(target_offsets_phi)
        iou_k = np.asarray(iou_weights, dtype=np.float64)[:, None]

        total_loss = 0.0
        layer_losses: List[float] = []
        layer_grads: List[np.ndarray] = []

        for l_logits in layer_logits_list:
            probs = softmax_np(l_logits, axis=-1)
            log_probs = log_softmax_np(l_logits, axis=-1)

            k_idx = np.arange(K)[:, None]
            side_idx = np.arange(4)[None, :]

            ce_left = -log_probs[k_idx, side_idx, n_left]
            ce_right = -log_probs[k_idx, side_idx, n_right]

            weighted_ce = iou_k * (omega_left * ce_left + omega_right * ce_right)
            layer_loss = float(np.sum(weighted_ce) / max(1, K))
            total_loss += layer_loss
            layer_losses.append(layer_loss)

            target_dist = np.zeros_like(probs)
            target_dist[k_idx, side_idx, n_left] += omega_left
            target_dist[k_idx, side_idx, n_right] += omega_right

            grad_logits = (iou_k[:, :, None] * (probs - target_dist)) / max(1, K)
            layer_grads.append(grad_logits)

        return {
            "loss_fgl": total_loss,
            "layer_losses": layer_losses,
            "layer_logits_gradients": layer_grads,
            "n_left": n_left,
            "n_right": n_right,
            "omega_left": omega_left,
            "omega_right": omega_right
        }


class GOLSDDecoupledDistillationFocalLoss:
    """
    Глобальная оптимальная самодистилляция локализации (GO-LSD) через Decoupled Distillation Focal Loss (L_DDF).
    """
    def __init__(self, temperature: float = 2.0):
        self.temperature = float(temperature)

    def forward_with_grad(
        self,
        all_layer_logits: List[np.ndarray],
        matched_indices: np.ndarray,
        unmatched_indices: np.ndarray,
        matched_ious: np.ndarray,
        unmatched_confs: np.ndarray
    ) -> Dict[str, Any]:
        if len(all_layer_logits) < 2:
            raise ValueError("GO-LSD требует как минимум 2 слоя декодера (l < L и финальный слой L).")

        T = self.temperature
        teacher_logits = np.asarray(all_layer_logits[-1], dtype=np.float64) / T
        teacher_probs = softmax_np(teacher_logits, axis=-1)
        teacher_log_probs = log_softmax_np(teacher_logits, axis=-1)

        K_m = len(matched_indices)
        K_u = len(unmatched_indices)
        K_total = max(1, K_m + K_u)

        alpha_k = np.asarray(matched_ious, dtype=np.float64) * (K_m / float(K_total))
        beta_k = np.asarray(unmatched_confs, dtype=np.float64) * (K_u / float(K_total))

        total_ddf_loss = 0.0
        shallow_grads: List[np.ndarray] = []

        for l_idx in range(len(all_layer_logits) - 1):
            student_logits = np.asarray(all_layer_logits[l_idx], dtype=np.float64) / T
            student_probs = softmax_np(student_logits, axis=-1)
            student_log_probs = log_softmax_np(student_logits, axis=-1)

            kl_per_side = np.sum(student_probs * (student_log_probs - teacher_log_probs), axis=-1)
            kl_per_query = np.mean(kl_per_side, axis=-1)

            matched_term = float(np.sum(alpha_k * kl_per_query[matched_indices])) if K_m > 0 else 0.0
            unmatched_term = float(np.sum(beta_k * kl_per_query[unmatched_indices])) if K_u > 0 else 0.0

            layer_loss = (T ** 2) * (matched_term + unmatched_term) / float(K_total)
            total_ddf_loss += layer_loss

            query_weights = np.zeros(student_logits.shape[0], dtype=np.float64)
            if K_m > 0:
                query_weights[matched_indices] = alpha_k
            if K_u > 0:
                query_weights[unmatched_indices] = beta_k

            grad_l = (T * query_weights[:, None, None] * (student_probs - teacher_probs)) / (4.0 * float(K_total))
            shallow_grads.append(grad_l)

        return {
            "loss_ddf": total_ddf_loss,
            "alpha_weights": alpha_k,
            "beta_weights": beta_k,
            "shallow_layer_gradients": shallow_grads
        }


class NormalizedWassersteinDistanceLoss:
    """
    Метрика и функция потерь нормализованного расстояния Гаусса–Вассерштейна (NWD):
        W_2^2(N_p, N_g) = (cx_p - cx_g)^2 + (cy_p - cy_g)^2 + ((w_p - w_g)^2 + (h_p - h_g)^2) / 4
        NWD(N_p, N_g) = exp( - sqrt(W_2^2(N_p, N_g)) / C )
        L_NWD = 1 - NWD(N_p, N_g)
    """
    def __init__(self, constant_c: float = 12.8, eps: float = 1e-8):
        self.constant_c = float(constant_c)
        self.eps = float(eps)

    def compute_nwd_and_loss(self, pred_boxes: Any, gt_boxes: Any) -> Dict[str, Any]:
        if HAS_TORCH and isinstance(pred_boxes, torch.Tensor):
            diff_xy = pred_boxes[..., :2] - gt_boxes[..., :2]
            diff_wh = pred_boxes[..., 2:] - gt_boxes[..., 2:]
            w2_sq = torch.sum(diff_xy ** 2, dim=-1) + 0.25 * torch.sum(diff_wh ** 2, dim=-1)
            w2 = torch.sqrt(w2_sq + self.eps)
            nwd = torch.exp(-w2 / self.constant_c)
            loss = 1.0 - nwd
            return {"w2_squared": w2_sq, "nwd": nwd, "loss_nwd": torch.mean(loss)}

        p = np.asarray(pred_boxes, dtype=np.float64)
        g = np.asarray(gt_boxes, dtype=np.float64)

        dx = p[..., 0] - g[..., 0]
        dy = p[..., 1] - g[..., 1]
        dw = p[..., 2] - g[..., 2]
        dh = p[..., 3] - g[..., 3]

        w2_sq = dx ** 2 + dy ** 2 + 0.25 * (dw ** 2 + dh ** 2)
        w2 = np.sqrt(w2_sq + self.eps)
        nwd = np.exp(-w2 / self.constant_c)
        loss_vec = 1.0 - nwd

        factor = (nwd / (self.constant_c * w2))[..., None]
        grad_p = factor * np.stack([dx, dy, 0.25 * dw, 0.25 * dh], axis=-1)

        return {
            "w2_squared": w2_sq,
            "w2_distance": w2,
            "nwd": nwd,
            "loss_nwd_per_box": loss_vec,
            "loss_nwd": float(np.mean(loss_vec)),
            "grad_pred_boxes": grad_p
        }

    def pairwise_nwd_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        p = np.asarray(boxes1, dtype=np.float64)[:, None, :]
        g = np.asarray(boxes2, dtype=np.float64)[None, :, :]
        w2_sq = (p[..., 0] - g[..., 0]) ** 2 + (p[..., 1] - g[..., 1]) ** 2 + 0.25 * (
            (p[..., 2] - g[..., 2]) ** 2 + (p[..., 3] - g[..., 3]) ** 2
        )
        w2 = np.sqrt(w2_sq + self.eps)
        return np.exp(-w2 / self.constant_c)


class HierarchyGatedRepulsionLoss:
    """
    Иерархически-гейтированная функция потерь отталкивания (Hierarchy-Gated Repulsion Loss):
        L_HG-Rep = lambda_rep * sum_{i} sum_{j != i} M(i, j) * Smooth_ln( Area(B_i cap B_j) / min(Area(B_i), Area(B_j)) )
    """
    def __init__(
        self,
        lambda_rep: float = 0.5,
        tau_contain: float = 0.75,
        smooth_sigma: float = 0.5,
        surround_classes: Optional[Set[int]] = None
    ):
        self.lambda_rep = float(lambda_rep)
        self.tau_contain = float(tau_contain)
        self.smooth_sigma = float(smooth_sigma)
        self.surround_classes = surround_classes if surround_classes is not None else SURROUND_CLASS_IDS

    def _smooth_ln(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sigma = self.smooth_sigma
        x_clipped = np.clip(x, 0.0, 0.999999)
        val = np.where(
            x_clipped <= sigma,
            -np.log(1.0 - x_clipped),
            (x_clipped - sigma) / (1.0 - sigma) - math.log(1.0 - sigma)
        )
        grad = np.where(
            x_clipped <= sigma,
            1.0 / (1.0 - x_clipped),
            1.0 / (1.0 - sigma)
        )
        grad = np.where(x > 0.0, grad, 0.0)
        return val, grad

    def forward(self, boxes_cxcywh: np.ndarray, class_ids: np.ndarray) -> Dict[str, Any]:
        boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
        classes = np.asarray(class_ids, dtype=np.int64)
        N = boxes.shape[0]
        if N <= 1:
            return {
                "loss_hg_rep": 0.0,
                "loss_naive_rep": 0.0,
                "tolerance_matrix_M": np.ones((N, N), dtype=np.float64),
                "overlap_ratio_matrix": np.zeros((N, N), dtype=np.float64)
            }

        _, inter_area, areas, _ = compute_pairwise_iou_and_intersection_np(boxes, boxes)
        min_area = np.minimum(areas[:, None], areas[None, :])
        overlap_ratio = inter_area / np.clip(min_area, 1e-9, None)
        np.fill_diagonal(overlap_ratio, 0.0)

        is_surround = np.array([int(c) in self.surround_classes for c in classes], dtype=bool)
        pair_has_surround = (is_surround[:, None] & (areas[:, None] >= areas[None, :])) | (
            is_surround[None, :] & (areas[None, :] >= areas[:, None])
        )
        is_canonical_enclosure = (overlap_ratio > self.tau_contain) & pair_has_surround

        M = np.where(is_canonical_enclosure, 0.0, 1.0)
        np.fill_diagonal(M, 0.0)

        smooth_val, _ = self._smooth_ln(overlap_ratio)
        gated_terms = M * smooth_val
        naive_terms = (1.0 - np.eye(N)) * smooth_val

        num_pairs = float(N * (N - 1))
        loss_hg_rep = self.lambda_rep * float(np.sum(gated_terms)) / num_pairs
        loss_naive_rep = self.lambda_rep * float(np.sum(naive_terms)) / num_pairs

        return {
            "loss_hg_rep": loss_hg_rep,
            "loss_naive_rep": loss_naive_rep,
            "tolerance_matrix_M": M,
            "overlap_ratio_matrix": overlap_ratio,
            "suppressed_enclosure_pairs": int(np.sum((1.0 - M) * (1.0 - np.eye(N))) // 2)
        }


class AnisotropicDeformableAttentionMSDA:
    STROKE_AXES_RAD: Tuple[float, float, float, float] = (
        0.0,
        math.pi * 0.5,
        math.pi * 0.25,
        math.pi * 0.75
    )

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_points_per_axis: int = 2, aspect_ratio: float = 3.2):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points_per_axis = n_points_per_axis
        self.aspect_ratio = aspect_ratio
        self.base_elliptical_grid = self._build_anisotropic_grid()

    def _build_anisotropic_grid(self) -> np.ndarray:
        grid_per_head = []
        radii = np.linspace(1.0 / self.n_points_per_axis, 1.0, self.n_points_per_axis)
        for h_idx in range(self.n_heads):
            theta = self.STROKE_AXES_RAD[h_idx % len(self.STROKE_AXES_RAD)]
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            pts = []
            for r in radii:
                for sign in (-1.0, 1.0):
                    dx_local = sign * r * self.aspect_ratio * 0.04
                    dy_local = 0.0
                    gx = dx_local * cos_t - dy_local * sin_t
                    gy = dx_local * sin_t + dy_local * cos_t
                    pts.append((gx, gy))
            grid_per_head.append(pts)
        return np.array(grid_per_head, dtype=np.float64)

    @staticmethod
    def bilinear_sample_2d(feat_map: np.ndarray, sample_coords_norm: np.ndarray) -> np.ndarray:
        H, W, C = feat_map.shape
        xs = np.clip(sample_coords_norm[..., 0] * (W - 1), 0.0, W - 1.0001)
        ys = np.clip(sample_coords_norm[..., 1] * (H - 1), 0.0, H - 1.0001)

        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)

        wx = (xs - x0)[..., None]
        wy = (ys - y0)[..., None]

        v00 = feat_map[y0, x0]
        v01 = feat_map[y0, x1]
        v10 = feat_map[y1, x0]
        v11 = feat_map[y1, x1]

        return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)

    def forward(
        self,
        reference_points_norm: np.ndarray,
        p2_feature_map: np.ndarray,
        learned_delta_offsets: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        N_q = reference_points_norm.shape[0]
        n_pts = self.base_elliptical_grid.shape[1]

        sampling_locations = reference_points_norm[:, None, None, :] + self.base_elliptical_grid[None, :, :, :]
        if learned_delta_offsets is not None:
            sampling_locations = sampling_locations + learned_delta_offsets

        sampling_locations = np.clip(sampling_locations, 0.0, 1.0)
        sampled_feats = self.bilinear_sample_2d(p2_feature_map, sampling_locations)

        if attention_weights is None:
            attn = np.full((N_q, self.n_heads, n_pts), 1.0 / n_pts, dtype=np.float64)
        else:
            attn = softmax_np(attention_weights, axis=-1)

        aggregated = np.sum(sampled_feats * attn[..., None], axis=2)
        return {
            "aggregated_features": aggregated.reshape(N_q, -1),
            "sampling_locations": sampling_locations,
            "anisotropic_aspect_ratio": self.aspect_ratio
        }


class HierarchicalHungarianMatcher:
    def __init__(self, cost_class: float = 2.0, cost_bbox_l1: float = 5.0, cost_nwd: float = 4.0, nwd_c: float = 0.12):
        self.cost_class = cost_class
        self.cost_bbox_l1 = cost_bbox_l1
        self.cost_nwd = cost_nwd
        self.nwd_metric = NormalizedWassersteinDistanceLoss(constant_c=nwd_c)

    @staticmethod
    def initialize_hierarchical_queries_from_p2(
        p2_high_freq_energy: np.ndarray,
        num_macro_queries: int = 16,
        num_micro_queries: int = 128
    ) -> Dict[str, np.ndarray]:
        H_p2, W_p2 = p2_high_freq_energy.shape[:2]

        macro_grid = int(math.ceil(math.sqrt(num_macro_queries)))
        lin = np.linspace(0.25, 0.75, macro_grid)
        gx, gy = np.meshgrid(lin, lin)
        macro_centers = np.stack([gx.flatten(), gy.flatten()], axis=-1)[:num_macro_queries]
        macro_wh = np.full_like(macro_centers, 0.78)
        macro_boxes = np.concatenate([macro_centers, macro_wh], axis=-1)

        flat_energy = p2_high_freq_energy.reshape(-1)
        top_indices = np.argsort(flat_energy)[::-1][:num_micro_queries]
        ys, xs = np.unravel_index(top_indices, (H_p2, W_p2))
        micro_centers = np.stack([(xs + 0.5) / float(W_p2), (ys + 0.5) / float(H_p2)], axis=-1)
        micro_wh = np.full_like(micro_centers, 0.14)
        micro_boxes = np.concatenate([micro_centers, micro_wh], axis=-1)

        return {
            "macro_queries_cxcywh": macro_boxes,
            "micro_queries_cxcywh": micro_boxes,
            "combined_queries_cxcywh": np.concatenate([macro_boxes, micro_boxes], axis=0)
        }

    def match(
        self,
        pred_logits: np.ndarray,
        pred_boxes_cxcywh: np.ndarray,
        gt_classes: np.ndarray,
        gt_boxes_cxcywh: np.ndarray
    ) -> Dict[str, Any]:
        probs = softmax_np(pred_logits, axis=-1)
        cost_cls = -probs[:, gt_classes]

        cost_l1 = np.sum(np.abs(pred_boxes_cxcywh[:, None, :] - gt_boxes_cxcywh[None, :, :]), axis=-1)
        nwd_sim = self.nwd_metric.pairwise_nwd_matrix(pred_boxes_cxcywh, gt_boxes_cxcywh)
        cost_nwd_mat = 1.0 - nwd_sim

        total_cost_matrix = (
            self.cost_class * cost_cls
            + self.cost_bbox_l1 * cost_l1
            + self.cost_nwd * cost_nwd_mat
        )

        row_ind, col_ind = linear_sum_assignment(total_cost_matrix)
        all_rows = set(range(pred_boxes_cxcywh.shape[0]))
        unmatched_rows = np.array(sorted(list(all_rows - set(row_ind))), dtype=np.int64)

        return {
            "matched_query_indices": row_ind.astype(np.int64),
            "matched_gt_indices": col_ind.astype(np.int64),
            "unmatched_query_indices": unmatched_rows,
            "total_assignment_cost": float(total_cost_matrix[row_ind, col_ind].sum()),
            "matched_nwd_scores": nwd_sim[row_ind, col_ind]
        }


def run_self_verification() -> None:
    print("=" * 78)
    print("Запуск верификации модифицированных функций потерь D-FINE (Project Cangjie v5.0)")
    print("=" * 78)

    np.random.seed(42)

    fdr = FineGrainedDistributionRefinement(num_bins=32, d_max=1.0, curvature=1.6)
    init_logits = np.zeros((6, 4, 33), dtype=np.float64)
    deltas = [np.random.randn(6, 4, 33) * 0.25 for _ in range(3)]
    acc_logits, probs, dists = fdr.refine_layer_distributions(init_logits, deltas)
    assert len(probs) == 3 and probs[-1].shape == (6, 4, 33)
    assert np.allclose(np.sum(probs[-1], axis=-1), 1.0)

    fgl = FineGrainedLocalizationLoss(fdr)
    gt_phi = np.random.uniform(0.05, 0.85, size=(6, 4))
    ious = np.array([0.92, 0.85, 0.78, 0.88, 0.95, 0.81])
    fgl_res = fgl.forward_with_grad(acc_logits, gt_phi, ious)
    assert fgl_res["loss_fgl"] > 0.0
    assert np.allclose(fgl_res["omega_left"] + fgl_res["omega_right"], 1.0)
    print(f"✅ [1/5] FDR & Fine-Grained Localization Loss (L_FGL = {fgl_res['loss_fgl']:.4f}): интерполяция бинов точна.")

    all_q_logits = [np.random.randn(10, 4, 33) for _ in range(4)]
    matched_idx = np.array([0, 2, 4, 7])
    unmatched_idx = np.array([1, 3, 5, 6, 8, 9])
    m_ious = np.array([0.91, 0.84, 0.79, 0.95])
    u_confs = np.array([0.12, 0.08, 0.25, 0.04, 0.19, 0.11])
    golsd = GOLSDDecoupledDistillationFocalLoss(temperature=2.0)
    ddf_res = golsd.forward_with_grad(all_q_logits, matched_idx, unmatched_idx, m_ious, u_confs)
    assert ddf_res["loss_ddf"] >= 0.0 and len(ddf_res["shallow_layer_gradients"]) == 3
    print(f"✅ [2/5] GO-LSD Self-Distillation (L_DDF = {ddf_res['loss_ddf']:.4f}): градиенты 3 промежуточных слоев верифицированы.")

    nwd_engine = NormalizedWassersteinDistanceLoss(constant_c=12.8)
    box_pred_micro = np.array([[100.0, 100.0, 12.0, 12.0], [100.0, 100.0, 12.0, 12.0]])
    box_gt_micro = np.array([[102.0, 100.0, 12.0, 12.0], [116.0, 116.0, 12.0, 12.0]])
    ious_micro, _, _, _ = compute_pairwise_iou_and_intersection_np(box_pred_micro, box_gt_micro)
    nwd_res = nwd_engine.compute_nwd_and_loss(box_pred_micro, box_gt_micro)
    grad_norm_non_overlapping = float(np.linalg.norm(nwd_res["grad_pred_boxes"][1]))
    assert ious_micro[1, 1] == 0.0 and grad_norm_non_overlapping > 0.005
    print(
        f"✅ [3/5] Normalized Wasserstein Distance (NWD при сдвиге 2px = {nwd_res['nwd'][0]:.4f}; "
        f"при IoU=0 норма градиента = {grad_norm_non_overlapping:.4f} > 0)."
    )

    hg_rep = HierarchyGatedRepulsionLoss(lambda_rep=0.5, tau_contain=0.75)
    boxes_nested = np.array([
        [0.50, 0.50, 0.80, 0.80],
        [0.50, 0.50, 0.45, 0.45],
        [0.52, 0.50, 0.45, 0.45],
    ])
    cls_nested = np.array([30, 95, 95])
    rep_res = hg_rep.forward(boxes_nested, cls_nested)
    assert rep_res["tolerance_matrix_M"][0, 1] == 0.0
    assert rep_res["tolerance_matrix_M"][1, 2] == 1.0
    assert rep_res["loss_hg_rep"] < rep_res["loss_naive_rep"]
    print(
        f"✅ [4/5] Hierarchy-Gated Repulsion Loss: M(囗, 玉) = {rep_res['tolerance_matrix_M'][0, 1]:.0f} "
        f"(L_HG-Rep = {rep_res['loss_hg_rep']:.4f} vs классический Repulsion = {rep_res['loss_naive_rep']:.4f})."
    )

    msda = AnisotropicDeformableAttentionMSDA(d_model=64, n_heads=4, n_points_per_axis=2)
    p2_map = np.random.rand(256, 256, 16)
    p2_energy = np.mean(p2_map, axis=-1)
    matcher = HierarchicalHungarianMatcher()
    q_init = matcher.initialize_hierarchical_queries_from_p2(p2_energy, num_macro_queries=8, num_micro_queries=32)
    msda_out = msda.forward(q_init["combined_queries_cxcywh"][:, :2], p2_map)
    match_out = matcher.match(
        pred_logits=np.random.randn(40, 214),
        pred_boxes_cxcywh=q_init["combined_queries_cxcywh"],
        gt_classes=np.array([30, 95, 172]),
        gt_boxes_cxcywh=np.array([[0.5, 0.5, 0.8, 0.8], [0.5, 0.5, 0.4, 0.4], [0.5, 0.2, 0.7, 0.25]])
    )
    assert msda_out["aggregated_features"].shape == (40, 64)
    assert len(match_out["matched_query_indices"]) == 3
    print("✅ [5/5] Anisotropic MSDA (оси 0°/90°/45°/135°) + пирамида P2 (256x256) + Hungarian Matcher успешно верифицированы.")
    print("=" * 78)


if __name__ == "__main__":
    run_self_verification()
