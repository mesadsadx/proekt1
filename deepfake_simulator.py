"""
╔══════════════════════════════════════════════════════════════════╗
║          DEEPFAKE SIMULATOR v1.0                                  ║
║     Симулятор подмены лица в режиме реального времени             ║
║     для тестирования системы Dynamic Verification                 ║
╚══════════════════════════════════════════════════════════════════╝

Назначение:
    Накладывает лицо с фотографии-донора на лицо перед камерой.
    Вывод идёт в виртуальную камеру (pyvirtualcam) или в окно.
    Верификатор (dynamic_verification_v2.py) запускается на
    виртуальной камере и должен распознать подделку.

Установка зависимостей:
    pip install opencv-python dlib numpy pillow pyvirtualcam scipy

    Для dlib (если нет prebuilt wheel):
        Windows: pip install dlib  (нужен CMake + Visual C++)
        Linux:   pip install dlib
        macOS:   brew install cmake && pip install dlib

    Модель 68-point landmarks (скачать один раз, ~100MB):
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        Распаковать в папку со скриптом.

    pyvirtualcam (виртуальная камера):
        Windows: установить OBS Virtual Camera
        Linux:   sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback
        macOS:   установить OBS Virtual Camera

Запуск:
    python deepfake_simulator.py --donor foto.jpg
    python deepfake_simulator.py --donor foto.jpg --camera 0 --virtual-cam
    python deepfake_simulator.py --donor foto.jpg --virtual-cam --cam-index 1

    Параллельно (в другом терминале):
    python dynamic_verification_v2.py --camera 1   # виртуальная камера
"""

import cv2
import dlib
import numpy as np
import argparse
import sys
import os
import time
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from scipy.spatial import Delaunay

# ──────────────────────────────────────────────────────────────────
# НАСТРОЙКИ
# ──────────────────────────────────────────────────────────────────

MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# Индексы ключевых точек лица (dlib 68-point)
JAW_POINTS        = list(range(0, 17))
EYEBROW_POINTS    = list(range(17, 27))
NOSE_POINTS       = list(range(27, 36))
LEFT_EYE_POINTS   = list(range(36, 42))
RIGHT_EYE_POINTS  = list(range(42, 48))
MOUTH_POINTS      = list(range(48, 68))
FACE_POINTS       = list(range(17, 68))

# Точки для выравнивания (alignment)
ALIGN_POINTS = (LEFT_EYE_POINTS + RIGHT_EYE_POINTS +
                EYEBROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Точки для построения маски
OVERLAY_POINTS = [FACE_POINTS + JAW_POINTS]

# Параметры сглаживания (чем больше — тем плавнее переход, но медленнее)
FEATHER_AMOUNT = 13
BLUR_AMOUNT    = 3

# Цвета UI (BGR)
C_CYAN   = (255, 210, 0)
C_GREEN  = (0, 220, 80)
C_RED    = (50,  50, 220)
C_ORANGE = (30, 140, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (20,  20,  40)

# ──────────────────────────────────────────────────────────────────
# ШРИФТ (Pillow, кириллица)
# ──────────────────────────────────────────────────────────────────

def _find_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

FONT_PATH = _find_font()
_font_cache = {}

def _get_font(size):
    if size not in _font_cache:
        if FONT_PATH:
            try:
                _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
                return _font_cache[size]
            except Exception:
                pass
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_text(img, text, pos, size, color_rgb, outline=True):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f = _get_font(size)
    x, y = pos
    if outline:
        for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2),(-2,0),(2,0),(0,-2),(0,2)]:
            draw.text((x+dx, y+dy), text, font=f, fill=(0,0,0))
    draw.text((x, y), text, font=f, fill=color_rgb)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def put_text_centered(img, text, cx, cy, size, color_rgb, **kw):
    pil = Image.new("RGB", (1,1))
    draw = ImageDraw.Draw(pil)
    f = _get_font(size)
    bbox = draw.textbbox((0,0), text, font=f)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    put_text(img, text, (cx - tw//2, cy - th//2), size, color_rgb, **kw)

def draw_panel(img, x1, y1, x2, y2, color_bgr=(10,10,40), alpha=0.75, r=12):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color_bgr, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color_bgr, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx,cy), r, color_bgr, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

# ──────────────────────────────────────────────────────────────────
# FACE SWAP — CORE
# ──────────────────────────────────────────────────────────────────

def get_landmarks(detector, predictor, img_gray):
    """Возвращает массив (68, 2) или None."""
    faces = detector(img_gray, 1)
    if not faces:
        return None
    shape = predictor(img_gray, faces[0])
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)


def transformation_from_points(src_pts, dst_pts):
    """Similarity transform (поворот + масштаб + перенос) методом Procrustes."""
    src = np.matrix(src_pts.astype(np.float64))
    dst = np.matrix(dst_pts.astype(np.float64))

    src -= src.mean(0)
    dst -= dst.mean(0)

    src_std = np.std(src)
    dst_std = np.std(dst)
    src /= (src_std + 1e-6)
    dst /= (dst_std + 1e-6)

    U, S, Vt = np.linalg.svd(src.T * dst)
    R = (U * Vt).T

    scale = dst_std / (src_std + 1e-6)
    T_pts = np.matrix(src.mean(0)).T
    D_pts = np.matrix(dst.mean(0)).T

    # 3x3 матрица трансформации
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2:] = D_pts - scale * R * T_pts
    return M


def warp_image(src, M, shape):
    """Применяет аффинное преобразование к изображению."""
    output = np.zeros(shape, dtype=src.dtype)
    M_inv = np.linalg.inv(M)
    cv2.warpAffine(src, M_inv[:2], (shape[1], shape[0]),
                   dst=output,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output


def get_face_mask(img, landmarks):
    """Создаёт сглаженную маску по контуру лица."""
    img_mask = np.zeros(img.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        pts = cv2.convexHull(landmarks[group].astype(np.int32))
        cv2.fillConvexPoly(img_mask, pts, 1)
    img_mask = np.array([img_mask, img_mask, img_mask]).transpose((1, 2, 0))
    img_mask = (cv2.GaussianBlur(img_mask, (FEATHER_AMOUNT*2+1, FEATHER_AMOUNT*2+1), 0) > 0) * 1.0
    img_mask = cv2.GaussianBlur(img_mask, (FEATHER_AMOUNT*2+1, FEATHER_AMOUNT*2+1), 0)
    return img_mask


def correct_colours(src, dst, landmarks_dst):
    """Коррекция цвета донора под цвет целевого лица."""
    blur_amount = BLUR_AMOUNT * 2 + 1
    left_eye_center  = landmarks_dst[LEFT_EYE_POINTS].mean(axis=0).astype(int)
    right_eye_center = landmarks_dst[RIGHT_EYE_POINTS].mean(axis=0).astype(int)
    eye_dist = np.linalg.norm(right_eye_center - left_eye_center)
    ksize = max(3, int(eye_dist * 0.7) | 1)  # нечётное

    dst_blur = cv2.GaussianBlur(dst, (ksize, ksize), 0).astype(np.float64)
    src_blur = cv2.GaussianBlur(src, (ksize, ksize), 0).astype(np.float64)

    src_blur = np.clip(src_blur, 1.0, 255.0)
    result = src.astype(np.float64) * dst_blur / src_blur
    return np.clip(result, 0, 255).astype(np.uint8)


def face_swap(donor_img, target_img,
              donor_lm, target_lm,
              seamless: bool = True):
    """
    Основная функция face-swap.
    donor_img  — BGR кадр с донорским лицом
    target_img — BGR кадр с целевым лицом (куда подставляем)
    Возвращает итоговое BGR изображение.
    """
    # 1. Similarity transform: donor → target
    M = transformation_from_points(
        donor_lm[ALIGN_POINTS],
        target_lm[ALIGN_POINTS]
    )

    # 2. Деформируем донора в систему координат цели
    warped_donor = warp_image(donor_img, M, target_img.shape)

    # 3. Коррекция цвета
    warped_corrected = correct_colours(warped_donor, target_img, target_lm)

    # 4. Маски
    donor_mask  = get_face_mask(warped_donor, target_lm)
    target_mask = get_face_mask(target_img,   target_lm)
    combined_mask = np.max([donor_mask, target_mask], axis=0)

    # 5. Смешиваем
    output = (target_img * (1.0 - combined_mask) +
              warped_corrected * combined_mask).astype(np.uint8)

    # 6. Seamless clone (Poisson blending) для максимальной реалистичности
    if seamless:
        try:
            center = target_lm[FACE_POINTS].mean(axis=0).astype(int)
            center = (int(center[0]), int(center[1]))
            # Маска для seamless clone
            clone_mask = (combined_mask[:, :, 0] * 255).astype(np.uint8)
            clone_mask = cv2.erode(clone_mask, np.ones((7,7), np.uint8), iterations=2)
            if clone_mask.sum() > 1000:
                output = cv2.seamlessClone(
                    warped_corrected, target_img, clone_mask,
                    center, cv2.NORMAL_CLONE)
        except Exception:
            pass  # если seamlessClone упал — используем blending

    return output


# ──────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ КЛАСС
# ──────────────────────────────────────────────────────────────────

class DeepfakeSimulator:

    def __init__(self,
                 donor_path: str,
                 camera_index: int = 0,
                 use_virtual_cam: bool = False,
                 virtual_cam_index: int = -1,
                 seamless: bool = True):

        # ── Проверка модели ────────────────────────────────────────
        if not os.path.isfile(MODEL_PATH):
            print(f"\n[ОШИБКА] Файл модели не найден: {MODEL_PATH}")
            print("Скачайте по адресу:")
            print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Распакуйте .bz2 и положите рядом со скриптом.\n")
            sys.exit(1)

        # ── Детекторы ─────────────────────────────────────────────
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)

        # ── Донорское изображение ──────────────────────────────────
        donor_bgr = cv2.imread(donor_path)
        if donor_bgr is None:
            print(f"[ОШИБКА] Не удалось прочитать фото: {donor_path}")
            sys.exit(1)

        donor_gray = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2GRAY)
        donor_lm   = get_landmarks(self.detector, self.predictor, donor_gray)
        if donor_lm is None:
            print("[ОШИБКА] Лицо на донорском фото не найдено.")
            print("Попробуйте другое фото — лицо должно быть анфас, хорошо освещено.")
            sys.exit(1)

        self.donor_bgr  = donor_bgr
        self.donor_lm   = donor_lm
        self.donor_thumb = self._make_donor_thumb(donor_bgr, donor_lm)

        # ── Параметры ─────────────────────────────────────────────
        self.camera_index     = camera_index
        self.use_virtual_cam  = use_virtual_cam
        self.virtual_cam_index = virtual_cam_index
        self.seamless         = seamless

        # ── Статистика ────────────────────────────────────────────
        self.fps_history    = []
        self.swap_active    = True   # включить/выключить подмену
        self.show_landmarks = False
        self.show_debug     = True
        self.frame_count    = 0
        self.swap_count     = 0
        self.fail_count     = 0

        print(f"[OK] Донор загружен: {donor_path}")
        print(f"[OK] Модель: {MODEL_PATH}")
        print(f"[  ] Виртуальная камера: {'включена' if use_virtual_cam else 'отключена'}")

    # ──────────────────────────────────────── ВСПОМОГАТЕЛЬНЫЕ ─────

    def _make_donor_thumb(self, img, lm):
        """Миниатюра донора с выделением лица."""
        x1 = max(0, int(lm[:, 0].min()) - 20)
        y1 = max(0, int(lm[:, 1].min()) - 30)
        x2 = min(img.shape[1], int(lm[:, 0].max()) + 20)
        y2 = min(img.shape[0], int(lm[:, 1].max()) + 20)
        crop = img[y1:y2, x1:x2]
        th   = 90
        tw   = int(crop.shape[1] * th / (crop.shape[0] + 1e-6))
        return cv2.resize(crop, (tw, th))

    # ──────────────────────────────────────── РЕНДЕРИНГ UI ────────

    def _draw_ui(self, display, target_lm, fps, swap_ok):
        h, w = display.shape[:2]

        # ── Верхняя панель ────────────────────────────────────────
        draw_panel(display, 0, 0, w, 52, (15, 8, 30), alpha=0.85, r=0)
        put_text(display, "DEEPFAKE SIMULATOR", (14, 10), 22, (255, 210, 0))
        status_txt = "● SWAP ON" if self.swap_active else "○ SWAP OFF"
        status_col = (60, 220, 80) if self.swap_active else (180, 180, 180)
        put_text(display, status_txt, (14, 30), 14, status_col)

        fps_txt = f"FPS: {fps:.0f}"
        pil = Image.new("RGB",(1,1)); d=ImageDraw.Draw(pil)
        bb = d.textbbox((0,0), fps_txt, font=_get_font(16))
        tw = bb[2]-bb[0]
        put_text(display, fps_txt, (w - tw - 14, 14), 16, (200,200,200))

        # ── Донор thumbnail (правый нижний угол) ──────────────────
        th_img = self.donor_thumb
        th_h, th_w = th_img.shape[:2]
        tx1, ty1 = w - th_w - 12, h - th_h - 58
        tx2, ty2 = tx1 + th_w, ty1 + th_h
        draw_panel(display, tx1-6, ty1-6, tx2+6, ty2+20, (10,10,35), alpha=0.9)
        try:
            display[ty1:ty2, tx1:tx2] = th_img
        except Exception:
            pass
        put_text_centered(display, "ДОНОР", tx1 + th_w//2, ty2 + 8, 13, (160,160,160))

        # ── Статус-бар swap ───────────────────────────────────────
        if self.swap_active:
            if swap_ok:
                bar_col  = (30, 180, 60)
                bar_text = f"✔ Подмена активна · {self.swap_count} кадров"
                bar_tcol = (60, 220, 80)
            else:
                bar_col  = (20, 20, 100)
                bar_text = "⚠ Лицо не найдено — встаньте перед камерой"
                bar_tcol = (100, 100, 220)
        else:
            bar_col  = (30, 30, 30)
            bar_text = "Подмена ОТКЛЮЧЕНА — показывается оригинал"
            bar_tcol = (180, 180, 180)

        draw_panel(display, 0, h-52, w, h, bar_col, alpha=0.85, r=0)
        put_text(display, bar_text, (14, h-38), 16, bar_tcol)

        # ── Подсказки управления ──────────────────────────────────
        hints = "[ S ] вкл/выкл подмену   [ L ] точки   [ D ] дебаг   [ Q ] выход"
        pil2 = Image.new("RGB",(1,1)); d2=ImageDraw.Draw(pil2)
        bb2 = d2.textbbox((0,0), hints, font=_get_font(13))
        hw = bb2[2]-bb2[0]
        put_text(display, hints, (w - hw - 10, h - 18), 13, (120,120,120))

        # ── Дебаг-панель (левый нижний) ───────────────────────────
        if self.show_debug and target_lm is not None:
            draw_panel(display, 8, h-200, 230, h-58, (8,8,30), alpha=0.82)
            rows = [
                f"Кадров: {self.frame_count}",
                f"Swap OK: {self.swap_count}",
                f"Без лица: {self.fail_count}",
                f"Seamless: {'вкл' if self.seamless else 'выкл'}",
                f"Виркамера: {'вкл' if self.use_virtual_cam else 'выкл'}",
            ]
            for i, row in enumerate(rows):
                put_text(display, row, (14, h-192 + i*25), 14, (200,200,200))

        # ── Landmarks ─────────────────────────────────────────────
        if self.show_landmarks and target_lm is not None:
            for i, (x, y) in enumerate(target_lm.astype(int)):
                cv2.circle(display, (x, y), 2, (0, 255, 200), -1)

        # ── Предупреждение о дипфейке (поверх всего) ─────────────
        if self.swap_active and swap_ok:
            warn = "⚠ DEEPFAKE АКТИВЕН"
            pw, ph = text_size_pil(warn, 18)
            draw_panel(display, w//2 - pw//2 - 12, 58,
                       w//2 + pw//2 + 12, 90, (20,0,80), alpha=0.90)
            put_text_centered(display, warn, w//2, 74, 18, (200,80,255))

    # ──────────────────────────────────────── ОСНОВНОЙ ЦИКЛ ───────

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не удалось открыть камеру {self.camera_index}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Реальное разрешение (камера может дать меньше)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Виртуальная камера ────────────────────────────────────
        vcam = None
        if self.use_virtual_cam:
            try:
                import pyvirtualcam
                vcam = pyvirtualcam.Camera(width=fw, height=fh, fps=30,
                                           device=self.virtual_cam_index
                                           if self.virtual_cam_index >= 0
                                           else None)
                print(f"[OK] Виртуальная камера: {vcam.device}")
                print(f"     Запустите верификатор с: --camera {vcam.device}")
            except ImportError:
                print("[WARN] pyvirtualcam не установлен. Работаем без виртуальной камеры.")
                print("       Установка: pip install pyvirtualcam")
                self.use_virtual_cam = False
            except Exception as e:
                print(f"[WARN] Виртуальная камера недоступна: {e}")
                self.use_virtual_cam = False

        print("\n[ DEEPFAKE SIMULATOR — управление ]")
        print("  S — вкл/выкл подмену лица")
        print("  L — показать landmark-точки")
        print("  D — дебаг-панель")
        print("  Q — выход\n")

        t_prev = time.time()
        fps    = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame   = cv2.flip(frame, 1)
            display = frame.copy()
            self.frame_count += 1

            # FPS
            t_now = time.time()
            fps   = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # ── Face swap ─────────────────────────────────────────
            swap_ok   = False
            out_frame = frame.copy()

            if self.swap_active:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Немного контраст — помогает детектору
                gray_eq = cv2.equalizeHist(gray)
                target_lm = get_landmarks(self.detector, self.predictor, gray_eq)

                if target_lm is not None:
                    try:
                        out_frame = face_swap(
                            self.donor_bgr,
                            frame,
                            self.donor_lm,
                            target_lm,
                            seamless=self.seamless
                        )
                        swap_ok = True
                        self.swap_count += 1
                    except Exception as e:
                        self.fail_count += 1
                        out_frame = frame.copy()
                else:
                    self.fail_count += 1
                    target_lm = None

                display = out_frame.copy()
            else:
                target_lm = None

            # ── UI поверх ─────────────────────────────────────────
            lm_for_ui = (get_landmarks(
                             self.detector,
                             self.predictor,
                             cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                         if (self.show_landmarks and not self.swap_active)
                         else (target_lm if self.swap_active else None))

            self._draw_ui(display, lm_for_ui, fps, swap_ok)

            # ── Виртуальная камера ────────────────────────────────
            if vcam is not None:
                try:
                    rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                    vcam.send(rgb)
                    vcam.sleep_until_next_frame()
                except Exception:
                    pass

            cv2.imshow("Deepfake Simulator", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self.swap_active = not self.swap_active
                print(f"  Подмена лица: {'ВКЛЮЧЕНА' if self.swap_active else 'ВЫКЛЮЧЕНА'}")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('d'):
                self.show_debug = not self.show_debug
            elif key == ord('p'):
                self.seamless = not self.seamless
                print(f"  Seamless clone: {'ВКЛЮЧЕН' if self.seamless else 'ВЫКЛЮЧЕН'}")

        cap.release()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()
        print(f"\n[ Итог ] Обработано кадров: {self.frame_count}")
        print(f"          Успешных swap: {self.swap_count}")
        print(f"          Без лица: {self.fail_count}")


def text_size_pil(text, size):
    pil = Image.new("RGB",(1,1))
    draw = ImageDraw.Draw(pil)
    bb = draw.textbbox((0,0), text, font=_get_font(size))
    return bb[2]-bb[0], bb[3]-bb[1]


# ──────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deepfake Simulator — подмена лица для тестирования верификации",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python deepfake_simulator.py --donor face.jpg
  python deepfake_simulator.py --donor face.jpg --camera 0 --virtual-cam
  python deepfake_simulator.py --donor face.jpg --no-seamless   (быстрее, менее реалистично)

Параллельный запуск с верификатором:
  Terminal 1: python deepfake_simulator.py --donor face.jpg --virtual-cam
  Terminal 2: python dynamic_verification_v2.py --camera 1
        """
    )
    parser.add_argument("--donor",       required=True,
                        help="Путь к фото донора (JPG/PNG, лицо анфас)")
    parser.add_argument("--camera",      type=int, default=0,
                        help="Индекс реальной камеры (по умолчанию 0)")
    parser.add_argument("--virtual-cam", action="store_true",
                        help="Выводить в виртуальную камеру (pyvirtualcam)")
    parser.add_argument("--cam-index",   type=int, default=-1,
                        help="Индекс виртуальной камеры (-1 = автовыбор)")
    parser.add_argument("--no-seamless", action="store_true",
                        help="Отключить Poisson blending (быстрее, менее реалистично)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Путь к .dat модели dlib (по умолчанию: {MODEL_PATH})")

    args = parser.parse_args()

    # Можно переопределить путь к модели
    MODEL_PATH = args.model

    sim = DeepfakeSimulator(
        donor_path        = args.donor,
        camera_index      = args.camera,
        use_virtual_cam   = args.virtual_cam,
        virtual_cam_index = args.cam_index,
        seamless          = not args.no_seamless,
    )
    sim.run()
