"""
╔══════════════════════════════════════════════════════════════════╗
║          DYNAMIC VERIFICATION SYSTEM v3.0                        ║
║     Активная биометрическая верификация через анализ             ║
║     целостности текстур при перекрытии (Occlusion Analysis)      ║
║                                                                  ║
║     НОВОЕ: встроенный режим тестирования дипфейка                ║
║     [ T ] — загрузить фото-донора и включить face-swap           ║
╚══════════════════════════════════════════════════════════════════╝

Зависимости (base):
    pip install opencv-python mediapipe numpy scipy pillow

Дополнительно для режима "Тест дипфейк":
    pip install dlib
    + скачать модель 68-точек dlib:
      http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
      Распаковать рядом со скриптом.

Запуск:
    python dynamic_verification_v3.py
    python dynamic_verification_v3.py --camera 1
    python dynamic_verification_v3.py --donor face.jpg   # сразу с донором

Управление:
    SPACE — начать верификацию
    T     — выбрать фото-донора для теста дипфейка (открывается диалог)
    R     — сброс
    Q     — выход
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import sys
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ── Опциональный dlib для режима дипфейка ──────────────────────────
try:
    import dlib as _dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False

# ── Опциональный tkinter для диалога выбора файла ──────────────────
try:
    import tkinter as _tk
    from tkinter import filedialog as _fd
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# ШРИФТ — поиск системного TTF с поддержкой кириллицы
# ──────────────────────────────────────────────────────────────────

def _find_font() -> str:
    """Ищет первый доступный TTF-шрифт с поддержкой кириллицы."""
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None   # PIL будет использовать встроенный bitmap-шрифт


FONT_PATH = _find_font()

def _pil_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


# Кэш шрифтов по размеру
_font_cache: dict = {}

def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = _pil_font(size)
    return _font_cache[size]


# ──────────────────────────────────────────────────────────────────
# РЕНДЕРИНГ ТЕКСТА ЧЕРЕЗ PILLOW  (поддержка кириллицы)
# ──────────────────────────────────────────────────────────────────

def put_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    size: int,
    color: Tuple[int, int, int],          # RGB
    bold: bool = False,
    outline: bool = True,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Рисует Unicode/кириллический текст через Pillow поверх OpenCV-кадра.
    img  — BGR numpy array (изменяется на месте).
    pos  — (x, y) левый верхний угол текста.
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    f = font(size)

    x, y = pos
    if outline:
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2),
                       (-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((x + dx, y + dy), text, font=f, fill=outline_color)
    draw.text((x, y), text, font=f, fill=color)

    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def text_size(text: str, size: int) -> Tuple[int, int]:
    """Возвращает (ширину, высоту) строки в пикселях."""
    f = font(size)
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=f)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def put_text_centered(
    img: np.ndarray,
    text: str,
    cx: int, cy: int,
    size: int,
    color: Tuple[int, int, int],
    **kwargs,
) -> None:
    """Рисует текст с центром в (cx, cy)."""
    w, h = text_size(text, size)
    put_text(img, text, (cx - w // 2, cy - h // 2), size, color, **kwargs)


# ──────────────────────────────────────────────────────────────────
# ПРЯМОУГОЛЬНИК С ЗАКРУГЛЁННЫМИ УГЛАМИ + АЛЬФА
# ──────────────────────────────────────────────────────────────────

def draw_panel(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color_bgr: Tuple[int, int, int] = (10, 10, 40),
    alpha: float = 0.72,
    radius: int = 14,
) -> None:
    """Полупрозрачная закруглённая панель."""
    overlay = img.copy()
    r = radius
    # Центральный прямоугольник
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color_bgr, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color_bgr, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_progress_bar(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    progress: float,                            # 0.0 … 1.0
    color_fill: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (40, 40, 40),
    radius: int = 8,
) -> None:
    draw_panel(img, x, y, x + w, y + h, bg_color, alpha=1.0, radius=radius)
    filled = max(0, int(w * progress))
    if filled > radius * 2:
        draw_panel(img, x, y, x + filled, y + h, color_fill, alpha=1.0, radius=radius)



# ══════════════════════════════════════════════════════════════════
# FACE-SWAP ENGINE v2  —  InsightFace + Buffalo_l  (профессиональный)
# При недоступности InsightFace автоматически fallback на dlib
# ══════════════════════════════════════════════════════════════════

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model as _if_get_model
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False

# ── dlib fallback ─────────────────────────────────────────────────
_DF_MODEL_PATH   = "shape_predictor_68_face_landmarks.dat"
_DF_ALIGN_PTS    = list(range(17, 68))
_DF_OVERLAY_PTS  = [list(range(0, 68))]
_DF_LEFT_EYE     = list(range(36, 42))
_DF_RIGHT_EYE    = list(range(42, 48))
_DF_FEATHER      = 13

def _df_get_landmarks(detector, predictor, gray):
    faces = detector(gray, 1)
    if not faces:
        return None
    shape = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)

def _df_transform(src_pts, dst_pts):
    src = np.matrix(src_pts.astype(np.float64))
    dst = np.matrix(dst_pts.astype(np.float64))
    src -= src.mean(0); dst -= dst.mean(0)
    ss = np.std(src) + 1e-6; ds = np.std(dst) + 1e-6
    src /= ss; dst /= ds
    U, S, Vt = np.linalg.svd(src.T * dst)
    R = (U * Vt).T
    M = np.eye(3)
    M[:2, :2] = (ds / ss) * R
    M[:2, 2:] = np.matrix(dst.mean(0)).T - (ds/ss)*R*np.matrix(src.mean(0)).T
    return M

def _df_warp(src, M, shape):
    out = np.zeros(shape, dtype=src.dtype)
    cv2.warpAffine(src, np.linalg.inv(M)[:2], (shape[1], shape[0]),
                   dst=out, borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return out

def _df_mask(img, lm):
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    for g in _DF_OVERLAY_PTS:
        cv2.fillConvexPoly(mask, cv2.convexHull(lm[g].astype(np.int32)), 1)
    mask3 = np.stack([mask]*3, axis=2)
    k = _DF_FEATHER*2+1
    mask3 = (cv2.GaussianBlur(mask3,(k,k),0) > 0).astype(np.float64)
    return cv2.GaussianBlur(mask3,(k,k),0)

def _df_color_correct(src, dst, lm_dst):
    le = lm_dst[_DF_LEFT_EYE].mean(0).astype(int)
    re = lm_dst[_DF_RIGHT_EYE].mean(0).astype(int)
    k  = max(3, int(np.linalg.norm(re-le)*0.7)|1)
    db = np.clip(cv2.GaussianBlur(dst,(k,k),0).astype(np.float64),1,255)
    sb = np.clip(cv2.GaussianBlur(src,(k,k),0).astype(np.float64),1,255)
    return np.clip(src.astype(np.float64)*db/sb,0,255).astype(np.uint8)

def _df_swap_dlib(donor_img, donor_lm, target_img, target_lm):
    M         = _df_transform(donor_lm[_DF_ALIGN_PTS], target_lm[_DF_ALIGN_PTS])
    warped    = _df_warp(donor_img, M, target_img.shape)
    corrected = _df_color_correct(warped, target_img, target_lm)
    d_mask    = _df_mask(warped,     target_lm)
    t_mask    = _df_mask(target_img, target_lm)
    combined  = np.max([d_mask, t_mask], axis=0)
    result    = (target_img*(1-combined)+corrected*combined).astype(np.uint8)
    try:
        center = tuple(target_lm[list(range(17,68))].mean(0).astype(int))
        clone_mask = (combined[:,:,0]*255).astype(np.uint8)
        clone_mask = cv2.erode(clone_mask, np.ones((7,7),np.uint8), iterations=2)
        if clone_mask.sum() > 500:
            result = cv2.seamlessClone(corrected, target_img, clone_mask,
                                       center, cv2.NORMAL_CLONE)
    except Exception:
        pass
    return result


class DeepfakeTestMode:
    """
    Встроенный тест дипфейка — InsightFace (основной) / dlib (fallback).
    InsightFace даёт профессиональное качество: учитывает поворот головы,
    освещение, форму лица, делает Poisson blending автоматически.
    """

    def __init__(self):
        self.active       = False
        self.donor_img: Optional[np.ndarray] = None
        self.donor_thumb: Optional[np.ndarray] = None
        self.status_msg   = ""
        self.swap_ok      = False
        self.frame_ok     = 0
        self.frame_fail   = 0
        self.engine       = "none"   # "insightface" | "dlib" | "none"

        # InsightFace объекты
        self._if_app      = None
        self._if_swapper  = None
        self._donor_face  = None   # распознанное лицо донора

        # dlib fallback
        self.donor_lm     = None
        self._detector    = None
        self._predictor   = None

    # ── Инициализация InsightFace ──────────────────────────────────
    def _init_insightface(self) -> bool:
        if self._if_app is not None:
            return True
        if not _INSIGHTFACE_AVAILABLE:
            return False
        try:
            print("[DF] Загрузка InsightFace buffalo_l...")
            app = FaceAnalysis(name="buffalo_l",
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._if_app = app

            # Swapper модель — скачивается автоматически (~500MB, один раз)
            print("[DF] Загрузка inswapper_128.onnx (первый раз ~500MB)...")
            swapper = _if_get_model("inswapper_128.onnx",
                                    download=True, download_zip=True)
            swapper.prepare(ctx_id=0, det_thresh=0.5)
            self._if_swapper = swapper
            print("[DF] InsightFace готов!")
            return True
        except Exception as e:
            print(f"[DF] InsightFace недоступен: {e}")
            self._if_app = None
            self._if_swapper = None
            return False

    # ── Инициализация dlib fallback ────────────────────────────────
    def _init_dlib(self) -> bool:
        if self._detector is not None:
            return True
        if not _DLIB_AVAILABLE:
            self.status_msg = "Установите dlib: pip install dlib"
            return False
        if not os.path.isfile(_DF_MODEL_PATH):
            self.status_msg = f"Нет файла: {_DF_MODEL_PATH}"
            return False
        try:
            self._detector  = _dlib.get_frontal_face_detector()
            self._predictor = _dlib.shape_predictor(_DF_MODEL_PATH)
            return True
        except Exception as e:
            self.status_msg = f"Ошибка dlib: {e}"
            return False

    # ── Загрузка донора ────────────────────────────────────────────
    def load_donor(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is None:
            self.status_msg = f"Не удалось прочитать: {path}"
            return False

        # Пробуем InsightFace
        if self._init_insightface():
            faces = self._if_app.get(img)
            if faces:
                self.donor_img   = img
                self._donor_face = faces[0]
                self.engine      = "insightface"
                self._make_thumb(img, faces[0])
                self.active     = True
                self.status_msg = f"InsightFace: {os.path.basename(path)}"
                self.frame_ok = self.frame_fail = 0
                print(f"[DF] Донор загружен через InsightFace: {path}")
                return True
            else:
                print("[DF] InsightFace не нашёл лицо на фото, пробуем dlib...")

        # Fallback: dlib
        if self._init_dlib():
            gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            lm   = _df_get_landmarks(self._detector, self._predictor, gray)
            if lm is not None:
                self.donor_img  = img
                self.donor_lm   = lm
                self.engine     = "dlib"
                self._make_thumb(img, None, lm)
                self.active     = True
                self.status_msg = f"dlib: {os.path.basename(path)}"
                self.frame_ok = self.frame_fail = 0
                print(f"[DF] Донор загружен через dlib: {path}")
                return True

        self.status_msg = "Лицо на фото не найдено"
        return False

    def _make_thumb(self, img, face=None, lm=None):
        if face is not None:
            b = face.bbox.astype(int)
            x1,y1,x2,y2 = max(0,b[0]-10),max(0,b[1]-10),                           min(img.shape[1],b[2]+10),min(img.shape[0],b[3]+10)
        elif lm is not None:
            x1 = max(0,int(lm[:,0].min())-15)
            y1 = max(0,int(lm[:,1].min())-20)
            x2 = min(img.shape[1],int(lm[:,0].max())+15)
            y2 = min(img.shape[0],int(lm[:,1].max())+20)
        else:
            x1,y1,x2,y2 = 0,0,img.shape[1],img.shape[0]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        th = 80
        tw = max(1, int(crop.shape[1]*th/(crop.shape[0]+1e-6)))
        self.donor_thumb = cv2.resize(crop, (tw, th))

    # ── Диалог выбора файла ────────────────────────────────────────
    def open_file_dialog(self) -> Optional[str]:
        if not _TK_AVAILABLE:
            self.status_msg = "tkinter недоступен — используйте --donor"
            return None
        try:
            root = _tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = _fd.askopenfilename(
                title="Выберите фото донора",
                filetypes=[("Изображения","*.jpg *.jpeg *.png *.bmp *.webp"),
                           ("Все файлы","*.*")])
            root.destroy()
            return path if path else None
        except Exception as e:
            self.status_msg = f"Диалог: {e}"
            return None

    # ── Обработка кадра ───────────────────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.active or self.donor_img is None:
            return frame

        if self.engine == "insightface":
            return self._process_insightface(frame)
        elif self.engine == "dlib":
            return self._process_dlib(frame)
        return frame

    def _process_insightface(self, frame: np.ndarray) -> np.ndarray:
        try:
            faces = self._if_app.get(frame)
            if not faces:
                self.swap_ok = False
                self.frame_fail += 1
                return frame
            result = frame.copy()
            for face in faces:
                result = self._if_swapper.get(result, face,
                                              self._donor_face, paste_back=True)
            self.swap_ok = True
            self.frame_ok += 1
            return result
        except Exception as e:
            self.swap_ok = False
            self.frame_fail += 1
            return frame

    def _process_dlib(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray   = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            tgt_lm = _df_get_landmarks(self._detector, self._predictor, gray)
            if tgt_lm is None:
                self.swap_ok = False
                self.frame_fail += 1
                return frame
            result = _df_swap_dlib(self.donor_img, self.donor_lm, frame, tgt_lm)
            self.swap_ok = True
            self.frame_ok += 1
            return result
        except Exception:
            self.swap_ok = False
            self.frame_fail += 1
            return frame

    # ── UI overlay ────────────────────────────────────────────────
    def draw_overlay(self, display: np.ndarray):
        if not self.active:
            return
        h, w = display.shape[:2]

        # Миниатюра донора
        if self.donor_thumb is not None:
            th_h, th_w = self.donor_thumb.shape[:2]
            tx = w - th_w - 12
            ty = 60
            draw_panel(display, tx-6, ty-4, tx+th_w+6, ty+th_h+20,
                       (10,10,40), alpha=0.88)
            try:
                display[ty:ty+th_h, tx:tx+th_w] = self.donor_thumb
            except Exception:
                pass
            eng_label = "InsightFace" if self.engine=="insightface" else "dlib"
            put_text_centered(display, "ДОНОР",
                              tx+th_w//2, ty+th_h+8, 12, (160,160,160))
            put_text_centered(display, eng_label,
                              tx+th_w//2, ty+th_h+20, 11, (100,200,100))

        # Баннер
        if self.swap_ok:
            msg = "⚠ DEEPFAKE АКТИВЕН"
            col, bg = (220,60,220), (50,0,70)
        else:
            msg = "⏳ Deepfake: лицо не найдено"
            col, bg = (150,150,220), (20,20,50)

        pil = Image.new("RGB",(1,1)); dr=ImageDraw.Draw(pil)
        bb  = dr.textbbox((0,0), msg, font=_pil_font(18))
        mw  = bb[2]-bb[0]
        draw_panel(display, w//2-mw//2-14, 56,
                   w//2+mw//2+14, 86, bg, alpha=0.90)
        put_text_centered(display, msg, w//2, 71, 18, col)

        # Счётчик
        th_offset = (self.donor_thumb.shape[0] if self.donor_thumb is not None else 0)
        put_text(display,
                 f"swap {self.frame_ok} / fail {self.frame_fail}",
                 (w-175, 62+th_offset+24), 12, (130,130,160))


# ──────────────────────────────────────────────────────────────────
# КОНСТАНТЫ
# ──────────────────────────────────────────────────────────────────

THRESHOLD_REAL       = 0.60
THRESHOLD_FAKE       = 0.35
OCCLUSION_OVERLAP    = 0.12
MIN_OCCLUSION_FRAMES = 15
GESTURE_TIMEOUT      = 15.0
HISTORY_LEN          = 45

# Цвета (RGB для Pillow)
C_GREEN  = (60,  220, 80)
C_RED    = (220, 60,  60)
C_YELLOW = (255, 210, 0)
C_CYAN   = (0,   210, 255)
C_WHITE  = (255, 255, 255)
C_ORANGE = (255, 140, 30)
C_GRAY   = (160, 160, 160)

# Цвета (BGR для OpenCV)
BGR_CYAN   = (255, 210, 0)
BGR_GREEN  = (0,   220, 80)
BGR_ORANGE = (30,  140, 255)
BGR_DARK   = (40,  20,  10)

# ──────────────────────────────────────────────────────────────────
# ЖЕСТЫ
# ──────────────────────────────────────────────────────────────────

GESTURES = [
    {
        "id": "wave",
        "name": "Проведите ладонью перед лицом",
        "hint": "Медленно — слева направо",
        "min_x_travel": 0.25,
    },
    {
        "id": "nose",
        "name": "Потрите нос",
        "hint": "Прикоснитесь пальцем к носу",
    },
    {
        "id": "cheek",
        "name": "Прикоснитесь к щеке",
        "hint": "Положите ладонь на левую или правую щеку",
    },
    {
        "id": "forehead",
        "name": "Прикоснитесь ко лбу",
        "hint": "Положите ладонь на лоб",
    },
]

# ──────────────────────────────────────────────────────────────────
# ДАННЫЕ
# ──────────────────────────────────────────────────────────────────

@dataclass
class OcclusionMetrics:
    frame_count:        int   = 0
    temporal_variance:  float = 0.0
    gradient_coherence: float = 0.0
    flow_consistency:   float = 0.0
    spectral_entropy:   float = 0.0
    boundary_sharpness: float = 0.0

    def score(self) -> float:
        if self.frame_count < 5:
            return 0.5
        tv_score = np.clip(self.temporal_variance / 200.0, 0, 1)
        gc_score = np.clip(self.gradient_coherence / 0.7, 0, 1)
        fl_score = np.clip(self.flow_consistency, 0, 1)
        se_score = 1.0 - np.clip(self.spectral_entropy / 1.0, 0, 1)
        bs_score = np.clip(self.boundary_sharpness / 120.0, 0, 1)
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        scores  = [tv_score, gc_score, fl_score, se_score, bs_score]
        return float(np.dot(weights, scores))


@dataclass
class FrameState:
    face_bbox:   Optional[Tuple] = None
    hand_bbox:   Optional[Tuple] = None
    overlap_iou: float = 0.0
    is_occluded: bool  = False
    gray:        Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────────
# ГЕОМЕТРИЯ
# ──────────────────────────────────────────────────────────────────

def bbox_overlap_ratio(b1, b2) -> float:
    ax1, ay1 = b1[0], b1[1]
    ax2, ay2 = b1[0] + b1[2], b1[1] + b1[3]
    bx1, by1 = b2[0], b2[1]
    bx2, by2 = b2[0] + b2[2], b2[1] + b2[3]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / (union + 1e-6)


def intersect_region(b1, b2, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ──────────────────────────────────────────────────────────────────
# МЕТРИКИ
# ──────────────────────────────────────────────────────────────────

def compute_gradient_coherence(gray_patch: np.ndarray) -> float:
    if gray_patch.size < 16:
        return 0.0
    gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
    mag   = np.sqrt(gx**2 + gy**2) + 1e-6
    angle = np.arctan2(gy, gx)
    mean_cos = np.mean(np.cos(angle) * mag / mag.mean())
    mean_sin = np.mean(np.sin(angle) * mag / mag.mean())
    return float(np.sqrt(mean_cos**2 + mean_sin**2))


def compute_spectral_entropy(gray_patch: np.ndarray) -> float:
    if gray_patch.size < 16:
        return 0.5
    f  = np.fft.fft2(gray_patch.astype(np.float32))
    ps = np.abs(np.fft.fftshift(f))**2
    ps_norm = ps / (ps.sum() + 1e-9)
    entropy = -np.sum(ps_norm * np.log(ps_norm + 1e-12))
    return float(entropy / (np.log(ps_norm.size) + 1e-6))


def compute_boundary_sharpness(gray: np.ndarray, bbox_hand, bbox_face) -> float:
    region = intersect_region(bbox_hand, bbox_face, gray.shape)
    if region is None:
        return 0.0
    x1, y1, x2, y2 = region
    patch = gray[y1:y2, x1:x2]
    if patch.size < 9:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


# ──────────────────────────────────────────────────────────────────
# ОСНОВНОЙ КЛАСС
# ──────────────────────────────────────────────────────────────────

class DynamicVerification:

    def __init__(self, camera_index: int = 0, donor_path: str = ""):
        self.camera_index = camera_index
        self._donor_path  = donor_path

        mp_face  = mp.solutions.face_detection
        mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.face_detector = mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)

        self.state            = "INIT"
        self.current_gesture: Optional[dict] = None
        self.gesture_start    = 0.0
        self.occlusion_metrics = OcclusionMetrics()
        self.result: Optional[str] = None
        self.final_score      = 0.0

        self.pixel_buffer     = deque(maxlen=HISTORY_LEN)
        self.prev_gray: Optional[np.ndarray] = None
        self.flow_scores      = deque(maxlen=HISTORY_LEN)
        self.occlusion_progress = 0.0
        self.gesture_confirmed  = False
        self.hand_x_positions   = deque(maxlen=30)

        # ── Deepfake test mode ────────────────────────────────────
        self.df_mode = DeepfakeTestMode()
        if hasattr(self, '_donor_path') and self._donor_path:
            self.df_mode.load_donor(self._donor_path)

    # ──────────────────────── ГЛАВНЫЙ ЦИКЛ ────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[ОШИБКА] Не удалось открыть камеру.")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("[ Dynamic Verification v3.0 ]")
        print("  SPACE — начать / продолжить")
        print("  T     — загрузить фото-донора (тест дипфейка)")
        print("  R     — сброс")
        print("  Q     — выход")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # ── Deepfake: подменяем кадр перед верификацией ───────
            if self.df_mode.active:
                frame = self.df_mode.process(frame)

            display = frame.copy()

            if self.state == "INIT":
                self._render_init(display)
            elif self.state == "INSTRUCT":
                self._render_instruct(display)
            elif self.state == "GESTURE":
                fs = self._process_frame(frame, display)
                self._check_gesture_completion(fs, frame.shape)
                self._render_gesture_ui(display, fs)
            elif self.state == "ANALYZE":
                self._finalize_analysis()
                self.state = "RESULT"
            elif self.state == "RESULT":
                self._render_result(display)

            # ── Deepfake overlay поверх UI ────────────────────────
            self.df_mode.draw_overlay(display)

            # ── Подсказка [T] на INIT-экране ─────────────────────
            if self.state == "INIT" and not self.df_mode.active:
                h_d, w_d = display.shape[:2]
                put_text_centered(display, "[ T ] — тест дипфейк",
                                  w_d//2, h_d//2 + 80, 18, C_ORANGE)

            cv2.imshow("Dynamic Verification", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                self._handle_space()
            elif key == ord('r'):
                self._reset()
            elif key == ord('t'):
                self._open_deepfake_donor()

        cap.release()
        cv2.destroyAllWindows()
        self.face_detector.close()
        self.hand_detector.close()

    # ──────────────────────── DEEPFAKE TEST ──────────────────────

    def _open_deepfake_donor(self):
        """Открывает диалог выбора фото-донора и загружает его."""
        print("[DEEPFAKE TEST] Открываем диалог выбора фото...")
        path = self.df_mode.open_file_dialog()
        if path:
            ok = self.df_mode.load_donor(path)
            if ok:
                print(f"[DEEPFAKE TEST] Донор загружен: {path}")
                print("[DEEPFAKE TEST] Дипфейк ВКЛЮЧЁН. Система теперь видит подменённое лицо.")
                print("[DEEPFAKE TEST] Запустите верификацию — она должна обнаружить подделку.")
            else:
                print(f"[DEEPFAKE TEST] Ошибка: {self.df_mode.status_msg}")
        else:
            print("[DEEPFAKE TEST] Файл не выбран.")

    # ──────────────────────── ОБРАБОТКА КАДРА ─────────────────────

    def _process_frame(self, frame: np.ndarray, display: np.ndarray) -> FrameState:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fs   = FrameState(gray=gray)

        # Лицо
        face_res = self.face_detector.process(rgb)
        if face_res.detections:
            det = face_res.detections[0]
            bb  = det.location_data.relative_bounding_box
            fx  = max(0, int(bb.xmin * w))
            fy  = max(0, int(bb.ymin * h))
            fw  = min(int(bb.width * w), w - fx)
            fh  = min(int(bb.height * h), h - fy)
            fs.face_bbox = (fx, fy, fw, fh)
            cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), BGR_CYAN, 2)

        # Рука
        hand_res = self.hand_detector.process(rgb)
        if hand_res.multi_hand_landmarks:
            lm = hand_res.multi_hand_landmarks[0]
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            hx = max(0, int(min(xs)) - 15)
            hy = max(0, int(min(ys)) - 15)
            hw = min(int(max(xs) - min(xs)) + 30, w - hx)
            hh = min(int(max(ys) - min(ys)) + 30, h - hy)
            fs.hand_bbox = (hx, hy, hw, hh)
            self.hand_x_positions.append(np.mean(xs) / w)
            self.mp_draw.draw_landmarks(
                display, lm, mp.solutions.hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=BGR_GREEN, thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(0, 210, 255), thickness=2))

        # Перекрытие
        if fs.face_bbox and fs.hand_bbox:
            iou = bbox_overlap_ratio(fs.face_bbox, fs.hand_bbox)
            fs.overlap_iou = iou
            fs.is_occluded  = iou > OCCLUSION_OVERLAP
            if fs.is_occluded:
                self._collect_occlusion_metrics(gray, fs, display)

        if self.prev_gray is not None and fs.is_occluded and fs.face_bbox:
            self._compute_flow(gray, fs)

        self.prev_gray = gray.copy()
        return fs

    def _collect_occlusion_metrics(self, gray, fs, display):
        m = self.occlusion_metrics
        m.frame_count += 1

        region = intersect_region(fs.hand_bbox, fs.face_bbox, gray.shape)
        if region is None:
            return
        x1, y1, x2, y2 = region
        patch = gray[y1:y2, x1:x2]
        if patch.size < 16:
            return

        self.pixel_buffer.append(patch.copy())
        if len(self.pixel_buffer) >= 5:
            patches = list(self.pixel_buffer)[-10:]
            mh = min(p.shape[0] for p in patches)
            mw = min(p.shape[1] for p in patches)
            cps = [p[:mh, :mw].astype(np.float32) for p in patches
                   if p.shape[0] >= mh and p.shape[1] >= mw]
            if len(cps) >= 3:
                m.temporal_variance = float(np.mean(np.var(np.stack(cps, 0), 0)))

        m.gradient_coherence = 0.85 * m.gradient_coherence + 0.15 * compute_gradient_coherence(patch)
        m.spectral_entropy   = 0.85 * m.spectral_entropy   + 0.15 * compute_spectral_entropy(patch)
        m.boundary_sharpness = 0.85 * m.boundary_sharpness + 0.15 * compute_boundary_sharpness(gray, fs.hand_bbox, fs.face_bbox)
        self.occlusion_progress = min(1.0, m.frame_count / MIN_OCCLUSION_FRAMES)

        # Подсветка зоны перекрытия
        ov = display.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), BGR_ORANGE, -1)
        cv2.addWeighted(ov, 0.25, display, 0.75, 0, display)
        cv2.rectangle(display, (x1, y1), (x2, y2), BGR_ORANGE, 2)

    def _compute_flow(self, gray, fs):
        fx, fy, fw, fh = fs.face_bbox
        prev_p = self.prev_gray[fy:fy+fh, fx:fx+fw]
        curr_p = gray[fy:fy+fh, fx:fx+fw]
        if prev_p.shape != curr_p.shape or prev_p.size < 100:
            return
        flow = cv2.calcOpticalFlowFarneback(
            prev_p, curr_p, None, 0.5, 3, 10, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if mag.size > 0:
            cv_flow = np.std(mag) / (np.mean(mag) + 1e-6)
            self.flow_scores.append(float(np.exp(-cv_flow)))
            if len(self.flow_scores) > 3:
                self.occlusion_metrics.flow_consistency = float(np.mean(self.flow_scores))

    # ──────────────────────── ПРОВЕРКА ЖЕСТА ──────────────────────

    def _check_gesture_completion(self, fs, shape):
        if self.gesture_confirmed:
            return
        elapsed = time.time() - self.gesture_start
        if elapsed > GESTURE_TIMEOUT:
            if self.occlusion_metrics.frame_count >= MIN_OCCLUSION_FRAMES // 2:
                self.gesture_confirmed = True
                self.state = "ANALYZE"
            else:
                self._start_gesture()
            return
        if not self.current_gesture:
            return
        if self.occlusion_metrics.frame_count >= MIN_OCCLUSION_FRAMES:
            if self.current_gesture["id"] == "wave" and len(self.hand_x_positions) >= 10:
                travel = max(self.hand_x_positions) - min(self.hand_x_positions)
                if travel < self.current_gesture.get("min_x_travel", 0.15):
                    return
            self.gesture_confirmed = True
            self.state = "ANALYZE"

    # ──────────────────────── АНАЛИЗ ──────────────────────────────

    def _finalize_analysis(self):
        score = self.occlusion_metrics.score()
        self.final_score = score
        if score >= THRESHOLD_REAL:
            self.result = "REAL"
        elif score <= THRESHOLD_FAKE:
            self.result = "FAKE"
        else:
            self.result = "UNCERTAIN"

        print(f"\n{'='*48}")
        print(f"  РЕЗУЛЬТАТ ВЕРИФИКАЦИИ")
        print(f"{'='*48}")
        print(f"  Балл:               {score:.3f}")
        print(f"  Кадров перекрытия:  {self.occlusion_metrics.frame_count}")
        print(f"  ВЕРДИКТ:            {self.result}")
        print(f"{'='*48}\n")

    # ──────────────────────── НАВИГАЦИЯ ───────────────────────────

    def _handle_space(self):
        if self.state == "INIT":
            self.state = "INSTRUCT"
        elif self.state == "INSTRUCT":
            self._start_gesture()
        elif self.state == "RESULT":
            self._reset()

    def _start_gesture(self):
        self.current_gesture    = random.choice(GESTURES)
        self.gesture_start      = time.time()
        self.occlusion_metrics  = OcclusionMetrics()
        self.pixel_buffer.clear()
        self.flow_scores.clear()
        self.hand_x_positions.clear()
        self.occlusion_progress = 0.0
        self.gesture_confirmed  = False
        self.prev_gray          = None
        self.state              = "GESTURE"

    def _reset(self):
        self.state         = "INIT"
        self.result        = None
        self.final_score   = 0.0
        self.occlusion_metrics = OcclusionMetrics()
        self.gesture_confirmed = False

    # ══════════════════════════════════════════════════════════════
    #  РЕНДЕРИНГ ЭКРАНОВ
    # ══════════════════════════════════════════════════════════════

    def _render_init(self, display: np.ndarray):
        h, w = display.shape[:2]
        # Затемнение
        overlay = np.zeros_like(display)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        px, py = w//2 - 330, h//2 - 130
        draw_panel(display, px, py, px+660, py+270, (10, 10, 35), alpha=0.88)

        put_text_centered(display, "DYNAMIC VERIFICATION",
                          w//2, py + 50, 36, C_CYAN)
        put_text_centered(display, "Aktive Biometric Liveness Detection",
                          w//2, py + 95, 20, C_WHITE)

        put_text_centered(display, "Принцип работы:", w//2, py + 140, 18, C_YELLOW)
        put_text_centered(display,
                          "Анализ артефактов при перекрытии лица рукой",
                          w//2, py + 168, 17, C_WHITE)
        put_text_centered(display,
                          "Deepfake создаёт аномалии — система их обнаруживает",
                          w//2, py + 193, 17, C_WHITE)

        put_text_centered(display, "[ SPACE ] — начать",
                          w//2, py + 240, 22, C_GREEN)

    def _render_instruct(self, display: np.ndarray):
        h, w = display.shape[:2]
        overlay = np.zeros_like(display)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        px, py = w//2 - 350, h//2 - 165
        draw_panel(display, px, py, px+700, py+340, (10, 10, 35), alpha=0.90)

        put_text_centered(display, "ИНСТРУКЦИЯ", w//2, py + 40, 30, C_CYAN)

        lines = [
            ("1. Встаньте перед камерой, лицо должно быть видно",   C_WHITE),
            ("2. Система предложит случайный жест",                 C_WHITE),
            ("3. Выполните жест медленно и чётко",                  C_WHITE),
            ("4. Убедитесь, что рука пересекает область лица",       C_WHITE),
            ("", C_WHITE),
            ("Что анализируется:",                                   C_YELLOW),
            ("  • Временная дисперсия пикселей на границе",          C_GRAY),
            ("  • Когерентность градиентов перекрытия",              C_GRAY),
            ("  • Консистентность оптического потока",               C_GRAY),
            ("  • Спектральные артефакты (FFT) зоны перекрытия",     C_GRAY),
        ]
        for i, (line, color) in enumerate(lines):
            put_text(display, line, (px + 30, py + 82 + i * 24), 16, color)

        put_text_centered(display, "[ SPACE ] — готов",
                          w//2, py + 310, 22, C_GREEN)

    def _render_gesture_ui(self, display: np.ndarray, fs: FrameState):
        h, w = display.shape[:2]
        elapsed   = time.time() - self.gesture_start
        remaining = max(0.0, GESTURE_TIMEOUT - elapsed)
        gesture   = self.current_gesture or {}

        # ── Верхняя панель: задание ────────────────────────────────
        draw_panel(display, 8, 8, w - 8, 98, (8, 8, 30), alpha=0.80)
        put_text(display, "ЗАДАНИЕ:", (18, 14), 16, C_YELLOW)
        put_text(display, gesture.get("name", ""), (18, 38), 22, C_WHITE)
        put_text(display, gesture.get("hint", ""), (18, 70), 16, C_GRAY)

        # Таймер
        tcol = C_GREEN if remaining > 5 else C_RED
        ts   = f"{remaining:.1f} с"
        tw, _ = text_size(ts, 28)
        put_text(display, ts, (w - tw - 18, 30), 28, tcol)

        # ── Прогресс-бар ───────────────────────────────────────────
        bar_y = h - 68
        draw_panel(display, 8, bar_y, w - 8, bar_y + 30, (20, 20, 20), alpha=1.0, radius=8)
        pct = int(self.occlusion_progress * 100)
        if pct > 0:
            fill_color = BGR_GREEN if self.occlusion_progress < 0.99 else (255, 230, 0)
            filled_px  = max(0, int((w - 16) * self.occlusion_progress) - 16)
            if filled_px > 10:
                ov2 = display.copy()
                cv2.rectangle(ov2, (16, bar_y + 2), (16 + filled_px, bar_y + 28),
                              fill_color, -1)
                cv2.addWeighted(ov2, 1.0, display, 0.0, 0, display)
        put_text(display, f"Анализ перекрытия: {pct}%",
                 (20, bar_y + 6), 15, C_WHITE)

        # ── Статус ─────────────────────────────────────────────────
        if fs.is_occluded:
            status, scol = "◉  ПЕРЕКРЫТИЕ АКТИВНО — идёт анализ", C_ORANGE
        elif fs.face_bbox is None:
            status, scol = "⚠  Лицо не обнаружено — встаньте ближе", C_RED
        elif fs.hand_bbox is None:
            status, scol = "✋ Поднимите руку перед лицом", C_YELLOW
        else:
            status, scol = "↕  Поднесите руку ближе к лицу", C_YELLOW
        put_text(display, status, (18, h - 82), 17, scol)

        # ── Метрики (live) ─────────────────────────────────────────
        m = self.occlusion_metrics
        mx = w - 250
        draw_panel(display, mx - 4, 108, w - 8, 330, (8, 8, 30), alpha=0.78)
        put_text(display, "МЕТРИКИ", (mx, 116), 15, C_CYAN)
        rows = [
            ("Кадров",     f"{m.frame_count}"),
            ("Дисперсия",  f"{m.temporal_variance:.1f}"),
            ("Градиент",   f"{m.gradient_coherence:.3f}"),
            ("Поток",      f"{m.flow_consistency:.3f}"),
            ("Спектр",     f"{m.spectral_entropy:.3f}"),
            ("Резкость",   f"{m.boundary_sharpness:.1f}"),
        ]
        for i, (label, val) in enumerate(rows):
            put_text(display, f"{label}:", (mx, 140 + i * 28), 14, C_GRAY)
            vw, _ = text_size(val, 15)
            put_text(display, val, (w - vw - 14, 140 + i * 28), 15, C_WHITE)

        # ── Подсказка ──────────────────────────────────────────────
        put_text(display, "[ Q ] выход   [ R ] сброс",
                 (18, h - 12), 14, (120, 120, 120))

    def _render_result(self, display: np.ndarray):
        h, w = display.shape[:2]
        overlay = np.zeros_like(display)
        cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)

        verdict_map = {
            "REAL":      ("✔  ВЕРИФИКАЦИЯ ПРОШЛА",    C_GREEN,  "Лицо признано ЖИВЫМ"),
            "FAKE":      ("✘  ВЕРИФИКАЦИЯ ПРОВАЛЕНА",  C_RED,    "Обнаружен DEEPFAKE или ФОТО"),
            "UNCERTAIN": ("?  РЕЗУЛЬТАТ НЕОПРЕДЕЛЁН",  C_YELLOW, "Повторите верификацию"),
        }
        header, hcol, subtext = verdict_map.get(self.result,
                                                 ("ОШИБКА", C_WHITE, ""))

        py = h//2 - 180
        draw_panel(display, w//2 - 370, py, w//2 + 370, py + 365,
                   (8, 8, 28), alpha=0.92)

        put_text_centered(display, header,    w//2, py + 50,  32, hcol)
        put_text_centered(display, subtext,   w//2, py + 90,  20, C_WHITE)

        # ── Шкала балла ────────────────────────────────────────────
        bx, by_ = w//2 - 310, py + 120
        bw      = 620
        cv2.rectangle(display, (bx, by_), (bx + bw, by_ + 22), (45, 45, 45), -1)
        filled = int(bw * self.final_score)
        scol_bgr = BGR_GREEN if self.final_score >= THRESHOLD_REAL else (
                    (0, 60, 220) if self.final_score <= THRESHOLD_FAKE else (0, 180, 255))
        if filled > 0:
            cv2.rectangle(display, (bx, by_), (bx + filled, by_ + 22), scol_bgr, -1)

        # Пороги
        for thr, lbl in [(THRESHOLD_FAKE, "FAKE"), (THRESHOLD_REAL, "REAL")]:
            lx = bx + int(bw * thr)
            cv2.line(display, (lx, by_ - 4), (lx, by_ + 26), (220, 220, 220), 2)
            put_text(display, lbl, (lx - 12, by_ - 20), 13, C_GRAY)

        put_text_centered(display,
                          f"Балл реальности: {self.final_score:.3f}",
                          w//2, by_ + 42, 18, C_WHITE)

        # ── Детальные метрики ──────────────────────────────────────
        m = self.occlusion_metrics
        detail_rows = [
            f"Временная дисперсия:       {m.temporal_variance:.2f}",
            f"Когерентность градиентов:  {m.gradient_coherence:.3f}",
            f"Консистентность потока:    {m.flow_consistency:.3f}",
            f"Спектральная энтропия:     {m.spectral_entropy:.3f}",
            f"Резкость границы:          {m.boundary_sharpness:.2f}",
            f"Кадров проанализировано:   {m.frame_count}",
        ]
        for i, row in enumerate(detail_rows):
            put_text(display, row, (w//2 - 295, py + 172 + i * 28), 16, C_WHITE)

        put_text_centered(display,
                          "[ SPACE ] — повторить         [ Q ] — выход",
                          w//2, py + 340, 18, C_GREEN)


# ──────────────────────────────────────────────────────────────────
# ЗАПУСК
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Dynamic Verification v3 — биометрическая верификация + тест дипфейка",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Горячие клавиши:
  SPACE — начать / продолжить верификацию
  T     — загрузить фото-донора для теста дипфейка
  R     — сброс
  Q     — выход

Тест дипфейка:
  Нажмите T (или укажите --donor face.jpg) → система подменяет лицо →
  запустите верификацию → FAKE/UNCERTAIN — система поймала дипфейк!
        """)
    parser.add_argument("--camera", type=int, default=0,
                        help="Индекс камеры (по умолчанию 0)")
    parser.add_argument("--donor", type=str, default="",
                        help="Путь к фото-донору для автозапуска теста дипфейка")
    args = parser.parse_args()

    system = DynamicVerification(camera_index=args.camera,
                                 donor_path=args.donor)
    system.run()
