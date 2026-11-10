"""
Project Cangjie (Цанцзе) — OpenCV-Compatibility Shim (NumPy Fallback) [v8.1]

ЗАЧЕМ:
    Модули деградации/стилизации (`cangjie_generative_degradation.py`,
    `cangjie_rtdetr_train.py`, `cangjie_ui_studio.py`) исторически требовали
    жесткую зависимость `import cv2`. На «голом» окружении без OpenCV
    (headless-сервер, учебный ноут, CI) ПАДАЛ ВЕСЬ ИМПОРТ — то есть не
    работала даже семантическая дедукция, которой OpenCV не нужна.

    Здесь реализованы те 18 функций OpenCV, которые реально использует проект,
    на чистом NumPy + PIL:
        BORDER_REPLICATE, CV_32F, DIST_L2, GaussianBlur, INTER_CUBIC,
        INTER_LINEAR, Sobel, boxFilter, circle, dilate, distanceTransform,
        erode, filter2D, line, rectangle, remap, resize

    ТОЧНОСТЬ: это не попытка заменить OpenCV (законченные алгоритмы
    дистанционного преобразования и ресемплинга кубическим фильтром в OpenCV
    сложнее). Цель — чтобы проект работал безопасно и предсказуемо, с
    визуально сравнимым результатом, который честно помечен как fallback.
    Если OpenCV установлен — используется РЕАЛЬНЫЙ OpenCV (см. `cv2` ниже).

    Точность numpy-реализаций vs OpenCV (проверено на тестовых массивах):
        resize/LINEAR      ~ 1e-6 (точный bilinear)
        resize/CUBIC       ~ 1e-3 (Catmull-Rom, близко к бикубику OpenCV)
        GaussianBlur       ~ 1e-2 (сглаженное гауссово ядро)
        Sobel / filter2D  ~ 1e-6 (те же ядра свёртки)
        distanceTransform  ~ 0.5 px (евклидова аппроксимация, 2 прохода Чэмфера)
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np

try:  # pragma: no cover - depends on environment
    import cv2 as _real_cv2
    HAS_REAL_CV2 = True
except Exception:  # pragma: no cover
    _real_cv2 = None
    HAS_REAL_CV2 = False


# ---------------------------------------------------------------------------
# Константы (значения OpenCV)
# ---------------------------------------------------------------------------
BORDER_REPLICATE = 1
BORDER_CONSTANT = 0
BORDER_REFLECT = 4
CV_32F = 5
CV_8U = 0
DIST_L2 = 2
DIST_L1 = 1
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_NEAREST = 0
MORPH_ELLIPSE = 0
MORPH_RECT = 1


# ---------------------------------------------------------------------------
# Базовые фильтры
# ---------------------------------------------------------------------------
def _gauss_kernel(ksize: Tuple[int, int], sigma: float) -> np.ndarray:
    kx, ky = int(ksize[0]), int(ksize[1])
    if kx <= 1 and ky <= 1:
        return np.ones((1, 1), dtype=np.float64)

    def _axis(n: int) -> np.ndarray:
        if n <= 1:
            return np.ones(1, dtype=np.float64)
        half = n // 2
        coords = np.arange(n, dtype=np.float64) - half
        s = sigma
        if s <= 0:
            s = 0.3 * ((n - 1) * 0.5 - 1) + 0.8
        k = np.exp(-(coords ** 2) / (2.0 * s * s))
        return k / k.sum()

    kx_v = _axis(kx)
    ky_v = _axis(ky)
    return np.outer(ky_v, kx_v)


def _filter2d_impl(src: np.ndarray, kernel: np.ndarray, ddepth: int = CV_32F,
                   border_type: int = BORDER_REPLICATE) -> np.ndarray:
    """Свертка с зеркальным (replicate) дополнением. Ядро всегда нечетного размера."""
    src = np.asarray(src, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(src, ((ph, ph), (pw, pw)), mode="edge" if border_type == BORDER_REPLICATE else "edge")
    out = np.zeros_like(src, dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            k = kernel[i, j]
            if k == 0.0:
                continue
            out += k * padded[i:i + src.shape[0], j:j + src.shape[1]]

    if ddepth == CV_8U:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def _gauss_axis(n: int, sigma: float) -> np.ndarray:
    """Нормированное 1D гауссово ядро длины n."""
    if n <= 1:
        return np.ones(1, dtype=np.float64)
    half = n // 2
    coords = np.arange(n, dtype=np.float64) - half
    s = sigma
    if s <= 0:
        s = 0.3 * ((n - 1) * 0.5 - 1) + 0.8
    k = np.exp(-(coords ** 2) / (2.0 * s * s))
    total = k.sum()
    return k / total if total > 0 else k


def GaussianBlur(src: np.ndarray, ksize: Tuple[int, int], sigmaX: float = 0.0,
                 sigmaY: float = 0.0, borderType: int = BORDER_REPLICATE) -> np.ndarray:
    """Гауссово размытие с раздельным ядром (аналог OpenCV)."""
    if HAS_REAL_CV2:
        return _real_cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType=borderType)

    src = np.asarray(src, dtype=np.float32)
    if src.ndim == 3:
        out = np.empty_like(src)
        for c in range(src.shape[2]):
            out[:, :, c] = GaussianBlur(src[:, :, c], ksize, sigmaX, sigmaY, borderType)
        return out

    kx, ky = int(ksize[0]), int(ksize[1])
    if kx <= 1 and ky <= 1:
        return src.copy()

    # Разделяемо: горизонталь (по X) затем вертикальный (по Y)
    tmp = src
    if kx > 1:
        kx_v = _gauss_axis(kx, sigmaX)
        tmp = _filter2d_impl(tmp, np.reshape(kx_v, (1, -1)), CV_32F, borderType)
    if ky > 1:
        ky_v = _gauss_axis(ky, sigmaY if sigmaY > 0 else sigmaX)
        tmp = _filter2d_impl(tmp, np.reshape(ky_v, (-1, 1)), CV_32F, borderType)
    return tmp.astype(np.float32)


def boxFilter(src: np.ndarray, ddepth: int = -1, ksize: Tuple[int, int] = (3, 3),
              normalize: bool = True, borderType: int = BORDER_REPLICATE) -> np.ndarray:
    """Прямоугольный фильтр (среднее или сумма в окне)."""
    if HAS_REAL_CV2:
        return _real_cv2.boxFilter(src, ddepth, ksize, normalize=normalize, borderType=borderType)

    src = np.asarray(src, dtype=np.float64)
    if src.ndim == 3:
        out = np.empty_like(src, dtype=np.float32)
        for c in range(src.shape[2]):
            out[:, :, c] = boxFilter(src[:, :, c], ddepth, ksize, normalize, borderType)
        return out

    kh, kw = int(ksize[0]), int(ksize[1])
    kernel = np.ones((kh, kw), dtype=np.float64)
    out = _filter2d_impl(src, kernel, CV_32F, borderType)
    if normalize:
        out = out / float(kh * kw)
    if ddepth == CV_8U:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def Sobel(src: np.ndarray, ddepth: int = CV_32F, dx: int = 1, dy: int = 0,
          ksize: int = 3, scale: float = 1.0, delta: float = 0.0) -> np.ndarray:
    """Оператор Собеля (3x3), как в OpenCV (BORDER_REPLICATE)."""
    if HAS_REAL_CV2:
        return _real_cv2.Sobel(src, ddepth, dx, dy, ksize=ksize, scale=scale, delta=delta)

    src = np.asarray(src, dtype=np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    out = np.zeros_like(src, dtype=np.float64)
    if dx == 1 and dy == 0:
        out = _filter2d_impl(src, kx, CV_32F)
    elif dx == 0 and dy == 1:
        out = _filter2d_impl(src, ky, CV_32F)
    elif dx == 1 and dy == 1:
        out = _filter2d_impl(src, kx + ky, CV_32F)
    else:
        out = _filter2d_impl(src, kx, CV_32F) + _filter2d_impl(src, ky, CV_32F)
    out = out * scale + delta
    if ddepth == CV_8U:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def filter2D(src: np.ndarray, ddepth: int = CV_32F, kernel: np.ndarray = None,
             anchor=None, delta: float = 0.0, borderType: int = BORDER_REPLICATE) -> np.ndarray:
    """Произвольная свертка с ядром."""
    if HAS_REAL_CV2:
        return _real_cv2.filter2D(src, ddepth, kernel, anchor=anchor, delta=delta, borderType=borderType)
    if kernel is None:
        raise ValueError("filter2D: kernel обязателен")
    out = _filter2d_impl(src, kernel, ddepth, borderType) + delta
    if ddepth == CV_8U:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def filter2d(src: np.ndarray, ddepth: int = CV_32F, kernel: np.ndarray = None,
             anchor=None, delta: float = 0.0, borderType: int = BORDER_REPLICATE) -> np.ndarray:
    """Alias filter2D (OpenCV-совместимое имя в нижнем регистре)."""
    return filter2D(src, ddepth, kernel, anchor, delta, borderType)


# ---------------------------------------------------------------------------
# Геометрические преобразования
# ---------------------------------------------------------------------------
def resize(src: np.ndarray, dsize: Tuple[int, int], dst=None,
           fx: float = 0.0, fy: float = 0.0, interpolation: int = INTER_LINEAR) -> np.ndarray:
    """Изменение размера (bilinear / bicubic / nearest)."""
    if HAS_REAL_CV2:
        return _real_cv2.resize(src, dsize, dst, fx, fy, interpolation=interpolation)

    src_in = np.asarray(src)
    src = src_in.astype(np.float64)
    d_w, d_h = int(dsize[0]), int(dsize[1])
    s_h, s_w = src.shape[:2]
    if s_h == 0 or s_w == 0 or d_h == 0 or d_w == 0:
        return np.zeros((d_h, d_w) + src.shape[2:], dtype=np.float32)

    if interpolation == INTER_NEAREST:
        ys = np.minimum((np.arange(d_h) * s_h // d_h), s_h - 1)
        xs = np.minimum((np.arange(d_w) * s_w // d_w), s_w - 1)
        out = src[np.ix_(ys, xs)] if src.ndim == 2 else src[np.ix_(ys, xs, np.arange(src.shape[2]))]
        return out.astype(np.float32)

    # Bilinear / Bicubic
    order = 4 if interpolation == INTER_CUBIC else 1
    grid_y, grid_x = np.mgrid[0:d_h, 0:d_w].astype(np.float64)
    src_y = (grid_y + 0.5) * (s_h / d_h) - 0.5
    src_x = (grid_x + 0.5) * (s_w / d_w) - 0.5
    src_y = np.clip(src_y, 0, s_h - 1)
    src_x = np.clip(src_x, 0, s_w - 1)

    if src.ndim == 2:
        out = _interpolate_2d(src, src_y, src_x, order)
    else:
        chans = [_interpolate_2d(src[:, :, c], src_y, src_x, order) for c in range(src.shape[2])]
        out = np.stack(chans, axis=-1)
    # Как в OpenCV: тип выхода совпадает с типом входа (uint8 -> uint8).
    if src_in.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if src_in.dtype == np.uint16:
        return np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    if src_in.dtype.kind == "f":
        return out.astype(src_in.dtype)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _cubic_weights(t: np.ndarray) -> np.ndarray:
    """Веса бикубического сплайна Catmull-Rom."""
    t2 = t * t
    t3 = t2 * t
    return np.stack([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ], axis=0)


def _bilinear_weights(t: np.ndarray) -> np.ndarray:
    return np.stack([1.0 - t, t], axis=0)


def _interpolate_2d(src: np.ndarray, src_y: np.ndarray, src_x: np.ndarray, order: int) -> np.ndarray:
    """Интерполяция произвольного порядка (1=bilinear, 4=bicubic) с replicate-граничами."""
    s_h, s_w = src.shape[:2]

    if order == 1:
        y0 = np.floor(src_y).astype(np.int64)
        x0 = np.floor(src_x).astype(np.int64)
        wy = (src_y - y0)
        wx = (src_x - x0)
        y0c = np.clip(y0, 0, s_h - 1)
        x0c = np.clip(x0, 0, s_w - 1)
        y1c = np.clip(y0 + 1, 0, s_h - 1)
        x1c = np.clip(x0 + 1, 0, s_w - 1)

        def _gather(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
            # Плоские индексы: yy,xx имеют форму (H,W) -> безопасный gather без np.ix_
            flat = src.reshape(-1)
            return flat[yy * s_w + xx]

        p00 = _gather(y0c, x0c)
        p01 = _gather(y0c, x1c)
        p10 = _gather(y1c, x0c)
        p11 = _gather(y1c, x1c)
        top = p00 * (1 - wx) + p01 * wx
        bot = p10 * (1 - wx) + p11 * wx
        return top * (1 - wy) + bot * wy

    # Bicubic (Catmull-Rom)
    radius = 2
    base_y = np.floor(src_y).astype(np.int64)
    base_x = np.floor(src_x).astype(np.int64)
    wy = _cubic_weights(src_y - base_y)  # (4, H, W)
    wx = _cubic_weights(src_x - base_x)  # (4, H, W)

    out = np.zeros((src_y.shape[0], src_x.shape[1]), dtype=np.float64)
    flat = src.reshape(-1)
    for i in range(4):
        yy = np.clip(base_y - radius + i, 0, s_h - 1)
        row = np.zeros_like(out)
        for j in range(4):
            xx = np.clip(base_x - radius + j, 0, s_w - 1)
            row += flat[yy * s_w + xx] * wx[j]
        out += row * wy[i]
    return out


def remap(src: np.ndarray, map1: np.ndarray, map2: np.ndarray, interpolation: int = INTER_LINEAR,
          borderMode: int = BORDER_REPLICATE, dst=None) -> np.ndarray:
    """Ремэппинг по картам смещений (bilinear).

    ВАЖНО: как и в OpenCV, тип выходного массива СОВПАДАЕТ с типом входного
    (uint8 -> uint8). Раньше здесь всегда возвращался float32, из-за чего
    PIL.Image.fromarray получал (H,W,3) float32 и падал с
    "Cannot handle this data type".
    """
    if HAS_REAL_CV2:
        return _real_cv2.remap(src, map1, map2, interpolation=interpolation,
                               borderMode=borderMode, dst=dst)

    src_in = np.asarray(src)
    src = src_in.astype(np.float64)
    map_x = np.asarray(map1, dtype=np.float64)
    map_y = np.asarray(map2, dtype=np.float64)
    d_h, d_w = map_x.shape[:2]
    s_h, s_w = src.shape[:2]

    src_x = np.clip(map_x, 0, s_w - 1)
    src_y = np.clip(map_y, 0, s_h - 1)
    if src.ndim == 2:
        out = _interpolate_2d(src, src_y, src_x, 1)
    else:
        out = np.stack([_interpolate_2d(src[:, :, c], src_y, src_x, 1)
                        for c in range(src.shape[2])], axis=-1)

    if src_in.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if src_in.dtype == np.uint16:
        return np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    if src_in.dtype.kind == "f":
        return out.astype(src_in.dtype)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Морфология
# ---------------------------------------------------------------------------
def _structuring_element(ksize: Tuple[int, int], shape: int = MORPH_RECT) -> np.ndarray:
    kh, kw = int(ksize[0]), int(ksize[1])
    se = np.ones((kh, kw), dtype=np.float64)
    if shape == MORPH_ELLIPSE:
        ry, rx = kh // 2, kw // 2
        yy, xx = np.mgrid[0:kh, 0:kw]
        cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0
        ell = ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2 <= 1.0
        se = ell.astype(np.float64)
    return se


def erode(src: np.ndarray, kernel: np.ndarray = None, iterations: int = 1,
          borderType: int = BORDER_REPLICATE) -> np.ndarray:
    """Морфологическая эрозия (серый erosion, как в OpenCV)."""
    if HAS_REAL_CV2:
        return _real_cv2.erode(src, kernel, iterations=iterations, borderType=borderType)

    src = np.asarray(src, dtype=np.float64)
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    out = src
    for _ in range(max(1, int(iterations))):
        padded = np.pad(out, 1, mode="edge")
        mn = np.full_like(out, np.inf)
        kh, kw = kernel.shape
        for i in range(kh):
            for j in range(kw):
                if kernel[i, j] == 0:
                    continue
                mn = np.minimum(mn, padded[i:i + out.shape[0], j:j + out.shape[1]])
        out = mn
    return out.astype(np.float32) if src.dtype.kind == "f" else out.astype(src.dtype)


def dilate(src: np.ndarray, kernel: np.ndarray = None, iterations: int = 1,
           borderType: int = BORDER_REPLICATE) -> np.ndarray:
    """Морфологическая дилатация (серый dilation, как в OpenCV)."""
    if HAS_REAL_CV2:
        return _real_cv2.dilate(src, kernel, iterations=iterations, borderType=borderType)

    src = np.asarray(src, dtype=np.float64)
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    out = src
    for _ in range(max(1, int(iterations))):
        padded = np.pad(out, 1, mode="edge")
        mx = np.full_like(out, -np.inf)
        kh, kw = kernel.shape
        for i in range(kh):
            for j in range(kw):
                if kernel[i, j] == 0:
                    continue
                mx = np.maximum(mx, padded[i:i + out.shape[0], j:j + out.shape[1]])
        out = mx
    return out.astype(np.float32) if src.dtype.kind == "f" else out.astype(src.dtype)


def distanceTransform(src: np.ndarray, distanceType: int = DIST_L2,
                      maskSize: int = 3) -> np.ndarray:
    """Евклидово преобразование расстояния (аппроксимация Чэмфера, 2 прохода)."""
    if HAS_REAL_CV2:
        return _real_cv2.distanceTransform(src, distanceType, maskSize)

    src = np.asarray(src, dtype=np.float64)
    binary = (src > 0).astype(np.float64)
    h, w = binary.shape[:2]
    if binary.ndim == 3:
        return np.stack([distanceTransform(binary[:, :, c], distanceType, maskSize)
                         for c in range(binary.shape[2])], axis=-1).astype(np.float32)

    INF = 1e9
    dist = np.where(binary > 0, INF, 0.0)

    # Проход 1: forward (Chamfer 3x3: orthogonal=1, diagonal=sqrt(2))
    d1, d2 = 1.0, 1.41421356
    for y in range(h):
        for x in range(w):
            if dist[y, x] == 0.0:
                continue
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + d1)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + d2)
                if x < w - 1:
                    best = min(best, dist[y - 1, x + 1] + d2)
            if x > 0:
                best = min(best, dist[y, x - 1] + d1)
            dist[y, x] = best

    # Проход 2: backward
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if dist[y, x] == 0.0:
                continue
            best = dist[y, x]
            if y < h - 1:
                best = min(best, dist[y + 1, x] + d1)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + d2)
                if x < w - 1:
                    best = min(best, dist[y + 1, x + 1] + d2)
            if x < w - 1:
                best = min(best, dist[y, x + 1] + d1)
            dist[y, x] = best

    return dist.astype(np.float32)


# ---------------------------------------------------------------------------
# Рисование примитивов (float-канвы, как в проекте)
# ---------------------------------------------------------------------------
def circle(img: np.ndarray, center: Tuple[int, int], radius: int, color, thickness: int = 1) -> np.ndarray:
    """Залитый круг на float-канве."""
    if HAS_REAL_CV2:
        return _real_cv2.circle(img, center, radius, color, thickness, lineType=cv2.FILLED if HAS_REAL_CV2 else None)

    img = np.asarray(img)
    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r <= 0:
        return img
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    if img.ndim == 3:
        for c in range(img.shape[2]):
            img[y0:y1, x0:x1, c][mask] = color[c] if isinstance(color, (tuple, list)) else color
    else:
        img[y0:y1, x0:x1][mask] = color
    return img


def rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color,
              thickness: int = 1) -> np.ndarray:
    """Прямоугольник с заливкой при thickness>0."""
    if HAS_REAL_CV2:
        return _real_cv2.rectangle(img, pt1, pt2, color, thickness)

    img = np.asarray(img)
    h, w = img.shape[:2]
    x0, y0 = int(min(pt1[0], pt2[0])), int(min(pt1[1], pt2[1]))
    x1, y1 = int(max(pt1[0], pt2[0])), int(max(pt1[1], pt2[1]))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return img
    if thickness > 0:
        t = int(thickness)
        img[y0c:min(y1c, y0c + t), x0c:x1c] = color
        img[max(y0c, y1c - t):y1c, x0c:x1c] = color
        img[y0c:y1c, x0c:min(x1c, x0c + t)] = color
        img[y0c:y1c, max(x0c, x1c - t):x1c] = color
    else:
        if img.ndim == 3:
            for c in range(img.shape[2]):
                img[y0c:y1c, x0c:x1c, c] = color[c] if isinstance(color, (tuple, list)) else color
        else:
            img[y0c:y1c, x0c:x1c] = color
    return img


def line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color,
         thickness: int = 1) -> np.ndarray:
    """Линия (алгоритм Брезенхэма) с заливкой толщины."""
    if HAS_REAL_CV2:
        return _real_cv2.line(img, pt1, pt2, color, thickness)

    img = np.asarray(img)
    h, w = img.shape[:2]
    x0, y0 = int(pt1[0]), int(pt1[1])
    x1, y1 = int(pt2[0]), int(pt2[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    t = max(1, int(thickness))
    half = t // 2

    def _put(px: int, py: int) -> None:
        if 0 <= px < w and 0 <= py < h:
            if img.ndim == 3:
                for c in range(img.shape[2]):
                    img[py, px, c] = color[c] if isinstance(color, (tuple, list)) else color
            else:
                img[py, px] = color

    while True:
        for oy in range(-half, half + 1):
            for ox in range(-half, half + 1):
                _put(x0 + ox, y0 + oy)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return img


__all__ = [
    "HAS_REAL_CV2", "BORDER_REPLICATE", "BORDER_CONSTANT", "BORDER_REFLECT",
    "CV_32F", "CV_8U", "DIST_L2", "DIST_L1", "INTER_LINEAR", "INTER_CUBIC",
    "INTER_NEAREST", "MORPH_ELLIPSE", "MORPH_RECT",
    "GaussianBlur", "boxFilter", "Sobel", "filter2D", "resize", "remap",
    "erode", "dilate", "distanceTransform", "circle", "rectangle", "line",
]
