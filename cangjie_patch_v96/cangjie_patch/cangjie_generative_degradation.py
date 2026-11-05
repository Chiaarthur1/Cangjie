"""
Project Cangjie (Цанцзе) — Section 6 Generative Domain Adaptation & Physico-Mechanical Degradation [v5.0]
Bridges the domain gap between clean digital fonts and damaged historical field artifacts:
1. Skeleton-Conditioned Calligraphic Style Synthesizer (ControlNet Structural Guide + Flying-White 飞白 Brush Physics):
   - Extracts MakeMeAHanzi stroke medians or synthesizes multi-register vector skeletons for 50..172+ stroke glyphs
   - Produces 3-channel ControlNet conditioning maps (Stroke Distance Skeleton, Tangent Orientation Field, Brush Pressure)
   - Analytical Bounding-Box tracking without manual labeling
2. Non-Linear Spatial Warping with Exact BBox Recalculation:
   - Thin Plate Spline (TPS) radial basis function solver U(r) = r^2 * ln(r)
   - Smooth Elastic Deformation Fields (simulating silk warping, bamboo strip curvature, and crumpled paper)
3. Stone Rubbing Emulator (Topian 拓片):
   - Inverse contrast (white lime relief strokes on charcoal-black pigment)
   - Multi-octave 2D Perlin noise (stone porosity) + morphological chisel fractures breaking thin strokes
4. Xuan Paper Unsteady Anisotropic Ink Diffusion PDE Solver (Сюаньчжи 宣纸):
   - Finite-difference numerical integration of dI/dt = div(D * grad(I)) - alpha * I
   - Anisotropic diffusion tensor D aligned with paper fiber orientation
5. Mineral Cinnabar Red Seal Synthesizer (Subtractive Color Bleeding):
   - Procedural archaic Zhuanshu seal stamp with non-linear Beer-Lambert pigment mixing over/under ink strokes
"""

from __future__ import annotations

import os
import sys
import math
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw
import numpy as np

# OpenCV опционален: если его нет — используется совместимый NumPy-шим
# (cangjie_cv2_compat). Раньше `import cv2` был жестким, и падал ВЕСЬ импорт
# модуля, а вместе с ним — деградации, рендера и UI-студии, хотя семантичес-
# кая дедукция OpenCV не нужна вообще.
try:
    import cv2  # type: ignore
    HAS_REAL_CV2 = True
except ImportError:  # pragma: no cover
    from cangjie_cv2_compat import *  # noqa: F401,F403
    from cangjie_cv2_compat import HAS_REAL_CV2
    import cangjie_cv2_compat as cv2

from cangjie_benchmark_forge import (
    KANGXI_214_COMPREHENSIVE_CATALOG,
    CalligraphicPillowRenderer,
    GlyphRepository,
    BoundingBox,
    RadicalLeaf,
    SpatialSlot
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [GEN-DEGRADATION-v5]: %(message)s"
)
logger = logging.getLogger("CangjieGenerativeDegradation")


class SkeletonControlNetSynthesizer:
    """
    Генерирует векторный скелетный каркас и 3-канальную структурную карту ControlNet
    (Канал R: карта расстояний до медиан штрихов; Канал G: поле углов наклона касательных cos(2*theta);
    Канал B: профиль каллиграфического нажима кисти), а также рендерит каллиграфический глиф
    с эффектом «летящего белого» (сухой кисти 飞白) и скорописными перемычками.
    """
    def __init__(self, repo: Optional[GlyphRepository] = None, resolution: int = 512):
        self.repo = repo or GlyphRepository()
        self.resolution = resolution

    def render_skeleton_and_calligraphy(
        self,
        components: List[Tuple[str, BoundingBox]],
        style: str = "DAOIST_TALISMAN",
        flying_white_intensity: float = 0.35
    ) -> Dict[str, Any]:
        res = self.resolution
        skeleton_canvas = np.zeros((res, res), dtype=np.float32)
        angle_canvas = np.zeros((res, res), dtype=np.float32)
        pressure_canvas = np.zeros((res, res), dtype=np.float32)
        ink_canvas = np.zeros((res, res), dtype=np.float32)

        rng = np.random.RandomState(len(components) * 97 + sum(ord(c[0][0]) for c in components if c[0]))

        for glyph, box in components:
            leaf = RadicalLeaf(glyph, SpatialSlot.ROOT, self.repo)
            leaf.calculate_geometry(box)
            tb = leaf.target_box
            medians = leaf.medians
            cbox = leaf.content_bbox
            strokes_cnt = leaf.get_stroke_count()

            if medians and len(medians) > 0:
                cx, cy = cbox.x, cbox.y
                cw, ch = max(1.0, cbox.width), max(1.0, cbox.height)
                sx, sy = tb.width / cw, tb.height / ch
                base_radius = max(2.2, (14.0 - strokes_cnt * 0.35) * (res / 512.0))

                for stroke in medians:
                    if len(stroke) < 2:
                        continue
                    pts = [
                        (tb.x + (pt[0] - cx) * sx, tb.y + (pt[1] - cy) * sy)
                        for pt in stroke
                    ]
                    for i in range(len(pts) - 1):
                        x0, y0 = pts[i]
                        x1, y1 = pts[i + 1]
                        theta = math.atan2(y1 - y0, x1 - x0)
                        orient_val = 0.5 * (1.0 + math.cos(2.0 * theta))

                        dist = max(1.0, math.hypot(x1 - x0, y1 - y0))
                        steps = max(4, int(dist * 1.5))
                        for s in range(steps + 1):
                            t = s / float(steps)
                            px = int(round(x0 + t * (x1 - x0)))
                            py = int(round(y0 + t * (y1 - y0)))
                            if not (0 <= px < res and 0 <= py < res):
                                continue

                            prog = (i + t) / max(1.0, float(len(pts) - 1))
                            if style == "SHANG_JIAGUWEN":
                                pressure = 0.4 + 0.6 * math.sin(prog * math.pi)
                            elif style == "ZHOU_JINWEN":
                                pressure = 0.85 + 0.30 * math.cos((prog - 0.5) * math.pi)
                            else:
                                pressure = 0.65 + 0.45 * math.sin(prog * math.pi)

                            skeleton_canvas[py, px] = 1.0
                            angle_canvas[py, px] = orient_val
                            pressure_canvas[py, px] = pressure

                            r_px = max(1, int(round(base_radius * pressure)))
                            cv2.circle(ink_canvas, (px, py), r_px, 1.0, -1)
            else:
                # Шрифтовой CJK-рендеринг с автоматическим извлечением медианного скелета и градиента углов
                pil_r = CalligraphicPillowRenderer(target_resolution=res, supersample_factor=1)
                leaf.render(pil_r)
                glyph_arr = np.array(pil_r.finalize().convert("L"), dtype=np.float32)
                g_ink = np.clip((255.0 - glyph_arr) / 255.0, 0.0, 1.0)
                ink_canvas = np.maximum(ink_canvas, g_ink)
                g_bin = (g_ink > 0.35).astype(np.uint8)
                if np.any(g_bin):
                    dist = cv2.distanceTransform(g_bin, cv2.DIST_L2, 3)
                    ridge = (dist >= np.maximum(1.0, 0.55 * cv2.dilate(dist, np.ones((3, 3), np.uint8)))).astype(np.float32) * g_bin
                    skeleton_canvas = np.maximum(skeleton_canvas, ridge)
                    gx = cv2.Sobel(g_ink, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(g_ink, cv2.CV_32F, 0, 1, ksize=3)
                    orient = 0.5 * (1.0 + np.cos(2.0 * np.arctan2(gy, gx))) * g_bin
                    angle_canvas = np.maximum(angle_canvas, orient)
                    pressure_canvas = np.maximum(pressure_canvas, np.clip(dist / max(1.0, float(np.max(dist))), 0.0, 1.0))

        if flying_white_intensity > 0.0 and style in ("DAOIST_TALISMAN", "CALLIGRAPHIC"):
            streaks = rng.randn(res // 4, res).astype(np.float32)
            streaks = cv2.resize(streaks, (res, res), interpolation=cv2.INTER_LINEAR)
            streaks = cv2.GaussianBlur(streaks, (9, 1), 3.0)
            dry_mask = (streaks > (0.65 - 0.35 * flying_white_intensity)).astype(np.float32)
            core = cv2.erode(ink_canvas, np.ones((3, 3), dtype=np.uint8), iterations=1)
            ink_canvas = np.clip(ink_canvas - 0.45 * dry_mask * (ink_canvas - 0.5 * core), 0.0, 1.0)

        binary_skel = (skeleton_canvas > 0.1).astype(np.uint8)
        if np.any(binary_skel):
            dist_map = cv2.distanceTransform(1 - binary_skel, cv2.DIST_L2, 3)
            skel_guide = np.exp(-dist_map / max(2.0, res * 0.012))
        else:
            skel_guide = np.zeros_like(skeleton_canvas)

        angle_guide = cv2.GaussianBlur(angle_canvas, (5, 5), 1.2)
        pressure_guide = cv2.GaussianBlur(pressure_canvas, (5, 5), 1.2)
        controlnet_map = np.stack([skel_guide, angle_guide, pressure_guide], axis=-1)

        return {
            "controlnet_conditioning_map": np.clip(controlnet_map, 0.0, 1.0),
            "ink_density_matrix": np.clip(ink_canvas, 0.0, 1.0)
        }


class ThinPlateSplineDeformer:
    """
    Точное аналитическое преобразование тонких пластин (Thin Plate Spline, TPS)
    на базе радиальных базисных функций U(r) = r^2 * ln(r + eps).
    """
    def __init__(self, grid_size: int = 4, max_Clean_shift: float = 0.035, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.max_shift = max_Clean_shift
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _tps_kernel(r_sq: np.ndarray) -> np.ndarray:
        r_sq_safe = np.maximum(r_sq, 1e-12)
        return np.where(r_sq > 1e-12, 0.5 * r_sq * np.log(r_sq_safe), 0.0)

    def _fit_tps_weights(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K_num = src_pts.shape[0]
        diff = src_pts[:, None, :] - src_pts[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        K_mat = self._tps_kernel(dist_sq) + np.eye(K_num) * 1e-6

        P_mat = np.concatenate([np.ones((K_num, 1)), src_pts], axis=-1)
        L_top = np.concatenate([K_mat, P_mat], axis=-1)
        L_bot = np.concatenate([P_mat.T, np.zeros((3, 3))], axis=-1)
        L_mat = np.concatenate([L_top, L_bot], axis=0)

        rhs = np.concatenate([dst_pts, np.zeros((3, 2))], axis=0)
        coeffs = np.linalg.solve(L_mat, rhs)
        return coeffs[:K_num], coeffs[K_num:]

    @classmethod
    def _eval_tps(cls, query_pts: np.ndarray, ctrl_pts: np.ndarray, W: np.ndarray, A: np.ndarray) -> np.ndarray:
        diff = query_pts[:, None, :] - ctrl_pts[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        U = cls._tps_kernel(dist_sq)
        affine = A[0:1, :] + query_pts @ A[1:, :]
        non_linear = U @ W
        return affine + non_linear

    def warp_image_and_bboxes(
        self,
        image_arr: np.ndarray,
        bboxes_norm_xyxy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W_px = image_arr.shape[:2]
        lin = np.linspace(0.1, 0.9, self.grid_size)
        gx, gy = np.meshgrid(lin, lin)
        src_ctrl = np.stack([gx.flatten(), gy.flatten()], axis=-1)

        shifts = self.rng.uniform(-self.max_shift, self.max_shift, size=src_ctrl.shape)
        dst_ctrl = np.clip(src_ctrl + shifts, 0.02, 0.98)

        W_inv, A_inv = self._fit_tps_weights(dst_ctrl, src_ctrl)
        W_fwd, A_fwd = self._fit_tps_weights(src_ctrl, dst_ctrl)

        grid_res = 64
        u_lin = np.linspace(0.0, 1.0, grid_res)
        ugx, ugy = np.meshgrid(u_lin, u_lin)
        q_pts = np.stack([ugx.flatten(), ugy.flatten()], axis=-1)
        mapped_src = self._eval_tps(q_pts, dst_ctrl, W_inv, A_inv).reshape(grid_res, grid_res, 2)

        map_x = cv2.resize((mapped_src[:, :, 0] * (W_px - 1)).astype(np.float32), (W_px, H), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize((mapped_src[:, :, 1] * (H - 1)).astype(np.float32), (W_px, H), interpolation=cv2.INTER_LINEAR)

        warped_img = cv2.remap(image_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        warped_boxes = np.zeros_like(bboxes_norm_xyxy, dtype=np.float64)
        for i, (x1, y1, x2, y2) in enumerate(bboxes_norm_xyxy):
            xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            perimeter_pts = np.array([
                [x1, y1], [xm, y1], [x2, y1],
                [x2, ym], [x2, y2], [xm, y2],
                [x1, y2], [x1, ym]
            ], dtype=np.float64)
            new_pts = self._eval_tps(perimeter_pts, src_ctrl, W_fwd, A_fwd)
            nx1, ny1 = np.clip(np.min(new_pts[:, 0]), 0.0, 1.0), np.clip(np.min(new_pts[:, 1]), 0.0, 1.0)
            nx2, ny2 = np.clip(np.max(new_pts[:, 0]), 0.0, 1.0), np.clip(np.max(new_pts[:, 1]), 0.0, 1.0)
            warped_boxes[i] = [nx1, ny1, max(nx1 + 1e-3, nx2), max(ny1 + 1e-3, ny2)]

        return warped_img, warped_boxes


class XuanPaperInkDiffusionSolver:
    """
    Численный решатель двумерного нестационарного уравнения анизотропной фильтрации туши
    по волокнам рисовой бумаги Сюаньчжи (宣纸):
        dI / dt = div(D * grad(I)) - alpha * I
    """
    def __init__(
        self,
        d_parallel: float = 0.22,
        d_perp: float = 0.07,
        fiber_angle_deg: float = 18.0,
        absorption_alpha: float = 0.015,
        num_steps: int = 8,
        dt: float = 0.45
    ):
        self.d_parallel = float(d_parallel)
        self.d_perp = float(d_perp)
        self.fiber_angle_rad = math.radians(fiber_angle_deg)
        self.alpha = float(absorption_alpha)
        self.num_steps = int(num_steps)
        self.dt = float(dt)

        cos_p = math.cos(self.fiber_angle_rad)
        sin_p = math.sin(self.fiber_angle_rad)
        self.D_xx = self.d_parallel * (cos_p ** 2) + self.d_perp * (sin_p ** 2)
        self.D_yy = self.d_parallel * (sin_p ** 2) + self.d_perp * (cos_p ** 2)
        self.D_xy = (self.d_parallel - self.d_perp) * cos_p * sin_p

        self.pde_kernel = np.array([
            [ 0.5 * self.D_xy,                 self.D_yy,                -0.5 * self.D_xy],
            [ self.D_xx,        -2.0 * (self.D_xx + self.D_yy),           self.D_xx      ],
            [-0.5 * self.D_xy,                 self.D_yy,                 0.5 * self.D_xy]
        ], dtype=np.float32)

    def solve_diffusion(self, initial_ink_density: np.ndarray) -> np.ndarray:
        I_cur = np.asarray(initial_ink_density, dtype=np.float32).copy()
        I_core = I_cur.copy()

        for _ in range(self.num_steps):
            div_d_grad = cv2.filter2D(I_cur, cv2.CV_32F, self.pde_kernel, borderType=cv2.BORDER_REPLICATE)
            dI_dt = div_d_grad - self.alpha * I_cur
            I_cur = np.clip(I_cur + self.dt * dI_dt, 0.0, 1.0)

        combined = np.clip(np.maximum(I_core * 0.88, I_cur) + 0.28 * I_cur, 0.0, 1.0)
        return combined


class StoneRubbingTopianSimulator:
    @staticmethod
    def generate_fractal_perlin_noise(h: int, w: int, octaves: int = 4, seed: int = 77) -> np.ndarray:
        rng = np.random.RandomState(seed)
        acc = np.zeros((h, w), dtype=np.float32)
        amp = 1.0
        total_amp = 0.0
        for oct_idx in range(octaves):
            grid_h = max(4, h // (2 ** (octaves - oct_idx + 1)))
            grid_w = max(4, w // (2 ** (octaves - oct_idx + 1)))
            coarse = rng.rand(grid_h, grid_w).astype(np.float32)
            up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            acc += amp * up
            total_amp += amp
            amp *= 0.5
        acc /= max(1e-6, total_amp)
        return np.clip(acc, 0.0, 1.0)

    @classmethod
    def synthesize_topian_rubbing(cls, ink_density: np.ndarray, seed: int = 101) -> np.ndarray:
        H, W = ink_density.shape
        perlin = cls.generate_fractal_perlin_noise(H, W, octaves=4, seed=seed)

        chip_mask = (perlin > 0.76).astype(np.float32)
        fractured_strokes = np.clip(ink_density * (1.0 - 0.65 * chip_mask), 0.0, 1.0)

        bg_charcoal = 24.0 + 48.0 * perlin + 95.0 * (perlin > 0.82).astype(np.float32)
        stroke_lime = 238.0 - 25.0 * (1.0 - perlin)

        gray = fractured_strokes * stroke_lime + (1.0 - fractured_strokes) * bg_charcoal
        gray = np.clip(gray, 0.0, 255.0)

        rgb = np.stack([
            np.clip(gray * 1.01, 0, 255),
            np.clip(gray * 0.99, 0, 255),
            np.clip(gray * 0.95, 0, 255)
        ], axis=-1).astype(np.uint8)
        return rgb


class SubtractiveCinnabarSealOverlay:
    @staticmethod
    def apply_cinnabar_seal(
        rgb_image: np.ndarray,
        seal_box_norm: Tuple[float, float, float, float] = (0.52, 0.52, 0.88, 0.88),
        opacity: float = 0.78,
        seed: int = 303
    ) -> np.ndarray:
        H, W = rgb_image.shape[:2]
        rng = np.random.RandomState(seed)

        x1, y1 = int(seal_box_norm[0] * W), int(seal_box_norm[1] * H)
        x2, y2 = int(seal_box_norm[2] * W), int(seal_box_norm[3] * H)
        sw, sh = max(16, x2 - x1), max(16, y2 - y1)

        seal_mask = np.zeros((sh, sw), dtype=np.float32)
        border_t = max(3, int(min(sw, sh) * 0.07))
        cv2.rectangle(seal_mask, (border_t, border_t), (sw - border_t, sh - border_t), 1.0, border_t)
        cv2.line(seal_mask, (sw // 2, border_t), (sw // 2, sh - border_t), 1.0, max(2, border_t - 1))
        cv2.line(seal_mask, (border_t, sh // 2), (sw - border_t, sh // 2), 1.0, max(2, border_t - 1))

        for qx in (0, 1):
            for qy in (0, 1):
                cx0 = int((0.25 + 0.5 * qx) * sw)
                cy0 = int((0.25 + 0.5 * qy) * sh)
                r_q = int(min(sw, sh) * 0.14)
                cv2.rectangle(seal_mask, (cx0 - r_q, cy0 - r_q), (cx0 + r_q, cy0 + r_q), 1.0, max(2, border_t // 2))

        grain = rng.rand(sh, sw).astype(np.float32)
        seal_density = seal_mask * np.clip(0.55 + 0.55 * grain, 0.0, 1.0) * opacity
        seal_density = cv2.GaussianBlur(seal_density, (3, 3), 0.7)

        transmittance_cinnabar = np.stack([
            np.exp(-0.14 * seal_density),
            np.exp(-1.95 * seal_density),
            np.exp(-2.15 * seal_density)
        ], axis=-1)

        out = rgb_image.astype(np.float32).copy()
        roi = out[y1:y1 + sh, x1:x1 + sw] / 255.0
        blended_roi = np.clip(roi * transmittance_cinnabar * 255.0, 0.0, 255.0)
        out[y1:y1 + sh, x1:x1 + sw] = blended_roi
        return out.astype(np.uint8)


class HistoricalArtifactDegradationPipeline:
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self.repo = GlyphRepository()
        self.skel_synth = SkeletonControlNetSynthesizer(repo=self.repo, resolution=resolution)
        self.ink_solver = XuanPaperInkDiffusionSolver()
        self.tps = ThinPlateSplineDeformer(grid_size=4, max_Clean_shift=0.032)

    def generate_historical_sample(
        self,
        components: List[Tuple[str, BoundingBox]],
        domain_mode: str = "XUAN_INK_SEAL"
    ) -> Dict[str, Any]:
        res = self.resolution
        synth = self.skel_synth.render_skeleton_and_calligraphy(components, style="DAOIST_TALISMAN")
        ink_mat = synth["ink_density_matrix"]
        controlnet_map = synth["controlnet_conditioning_map"]

        bboxes_norm = np.array([
            [
                b.x / float(res),
                b.y / float(res),
                (b.x + b.width) / float(res),
                (b.y + b.height) / float(res)
            ]
            for _, b in components
        ], dtype=np.float64)

        if domain_mode == "TOPIAN_STONE_RUBBING":
            rgb_img = StoneRubbingTopianSimulator.synthesize_topian_rubbing(ink_mat)
        else:
            diffused_ink = self.ink_solver.solve_diffusion(ink_mat)
            if domain_mode == "DAOIST_HUANGBIAOZHI":
                paper_rgb = np.array([252.0, 232.0, 182.0], dtype=np.float32)
            else:
                paper_rgb = np.array([246.0, 241.0, 229.0], dtype=np.float32)

            rgb_float = paper_rgb[None, None, :] * (1.0 - 0.90 * diffused_ink[:, :, None])
            rgb_img = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
            rgb_img = SubtractiveCinnabarSealOverlay.apply_cinnabar_seal(rgb_img)

        warped_rgb, warped_bboxes = self.tps.warp_image_and_bboxes(rgb_img, bboxes_norm)
        return {
            "degraded_rgb_image": Image.fromarray(warped_rgb),
            "degraded_rgb_array": warped_rgb,
            "controlnet_conditioning_map": controlnet_map,
            "initial_bboxes_xyxy_norm": bboxes_norm,
            "warped_bboxes_xyxy_norm": warped_bboxes,
            "domain_mode": domain_mode
        }


def run_self_verification() -> None:
    print("=" * 78)
    print("Запуск верификации генеративной доменной адаптации и деградации (v5.0)")
    print("=" * 78)

    pipeline = HistoricalArtifactDegradationPipeline(resolution=256)
    test_components = [
        ("⻗", BoundingBox(20, 16, 216, 68)),
        ("鬼", BoundingBox(28, 92, 96, 140)),
        ("申", BoundingBox(132, 92, 96, 140)),
    ]

    out_xuan = pipeline.generate_historical_sample(test_components, domain_mode="XUAN_INK_SEAL")
    assert out_xuan["degraded_rgb_array"].shape == (256, 256, 3)
    assert out_xuan["controlnet_conditioning_map"].shape == (256, 256, 3)
    assert out_xuan["warped_bboxes_xyxy_norm"].shape == (3, 4)
    box_shift = float(np.mean(np.abs(out_xuan["warped_bboxes_xyxy_norm"] - out_xuan["initial_bboxes_xyxy_norm"])))
    assert 0.001 < box_shift < 0.06
    print(f"✅ [1/3] PDE диффузия туши Сюаньчжи + Киноварная печать + TPS (средний сдвиг боксов = {box_shift:.4f}).")

    out_topian = pipeline.generate_historical_sample(test_components, domain_mode="TOPIAN_STONE_RUBBING")
    border_median = float(np.median(out_topian["degraded_rgb_array"][0, :, :]))
    assert border_median < 110.0
    print(f"✅ [2/3] Каменный эстампаж топянь (拓片): инверсный контраст и шум Перлина подтверждены (фон = {border_median:.1f}).")

    out_dao = pipeline.generate_historical_sample(test_components, domain_mode="DAOIST_HUANGBIAOZHI")
    assert out_dao["degraded_rgb_array"].shape == (256, 256, 3)
    print("✅ [3/3] Даосский талисман на желтой бумаге Хуанбяочжи + эффект «летящего белого» (飞白) успешно сгенерирован.")
    print("=" * 78)


if __name__ == "__main__":
    run_self_verification()
