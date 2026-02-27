import functools
import logging
import threading
import time
import cv2
import keyboard
import numpy as np
import pyautogui
import pydirectinput
import pygame
from PIL import ImageGrab
import hashlib
import json
import re
import sqlite3
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Пути к файлам
sound_file_path = 'ASK.mp3'

# Инициализация звука (если не получится — просто отключим звук)
SOUND_ENABLED = True
try:
    pygame.mixer.init()
except Exception as e:
    SOUND_ENABLED = False
    logger.info(f"[WARN] Звук отключён (pygame.mixer.init не удался): {e}")


def prevent_reentry(method):
    """Не даёт методу запуститься повторно, пока предыдущий вызов не завершился."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        acquired, lock = self._acquire_method_lock(method.__name__)
        if not acquired:
            logger.info(f"Пропуск: {method.__name__} уже выполняется.")
            return None
        try:
            return method(self, *args, **kwargs)
        finally:
            lock.release()

    return wrapper

class FishingBot:
    def __init__(self):
        self.bot_running = False
        self.reward_button_template = 'knopkasebe.jpg'
        self.reward_button_name = "Забрать себе"
        self.post_cycle_reset_enabled = True
        self.cycle_limit = 6
        self.completed_cycles = 0
        self._method_locks = {}
        self.action_mode = 'take'  # take | release
        self.reset_first_click_coords = (1035, 962)
        self.reset_second_click_coords = [(1042, 748), (1034, 816)]
        self._reset_second_click_index = 0

    def set_action_mode(self, mode):
        if mode not in ('take', 'release'):
            logger.info(f"[WARN] Неизвестный режим действия: {mode}")
            return
        self.action_mode = mode
        logger.info(f"Режим действия переключен: {'ЗАБРАТЬ СЕБЕ' if mode == 'take' else 'ОТПУСТИТЬ'}")

    def _has_red_bar_in_roi(
            self,
            roi_rgb,
            bottom_part=0.35,  # анализируем нижние 35% ROI
            row_cov_thresh=0.50,  # в строке красный >= 50% ширины
            min_consecutive_rows=2,  # минимум 2 строки подряд
            ratio_thresh=0.002  # общая доля красного (0.2%) как антишум
    ):
        if roi_rgb is None or roi_rgb.size == 0:
            return False

        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        h, w = hsv.shape[:2]

        y0 = int(h * (1.0 - bottom_part))
        hsvb = hsv[y0:h, :]

        # Красный в HSV: две зоны (0..10) и (170..179)
        lower1 = np.array([0, 110, 110], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 110, 110], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsvb, lower1, upper1) | cv2.inRange(hsvb, lower2, upper2)

        # Чистим шум
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        red_ratio = cv2.countNonZero(mask) / mask.size
        if red_ratio < ratio_thresh:
            return False

        # Проверяем “полоску”: в некоторых строках красный занимает большую часть ширины
        row_frac = (mask > 0).mean(axis=1)  # доля красного по каждой строке
        good = row_frac >= row_cov_thresh

        best = cur = 0
        for v in good:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0

        return best >= min_consecutive_rows

    def _acquire_method_lock(self, method_name: str):
        """Защита от повторного запуска одного и того же метода поверх самого себя."""
        lock = self._method_locks.setdefault(method_name, threading.Lock())
        return lock.acquire(blocking=False), lock

    def set_reward_action(self, action: str):
        """Выбор кнопки после мини-игр: забрать себе / отпустить."""
        actions = {
            "take": ('knopkasebe.jpg', "Забрать себе"),
            "release": ('otpustit.png', "Отпустить"),
        }
        template, name = actions.get(action, actions["take"])
        self.reward_button_template = template
        self.reward_button_name = name
        logger.info(f"Выбрано действие после рыбалки: {name} ({template}).")

    def set_post_cycle_reset(self, enabled: bool):
        """Вкл/выкл последовательность клавиш после N циклов."""
        self.post_cycle_reset_enabled = bool(enabled)
        state = "включено" if self.post_cycle_reset_enabled else "выключено"
        logger.info(f"Сброс после {self.cycle_limit} циклов: {state}.")

    def set_cycle_limit(self, cycle_limit: int):
        """Изменение лимита циклов для последовательности сброса."""
        self.cycle_limit = max(1, int(cycle_limit))
        logger.info(f"Новый лимит циклов до сброса: {self.cycle_limit}")

    def find_object(self, template_path):
        """Поиск изображения на экране."""
        screenshot = np.array(ImageGrab.grab())
        screen_image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.info(f"Ошибка: файл {template_path} не найден.")
            return None
        result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        logger.info(f"Уровень совпадения для {template_path}: {max_val}")
        return max_loc if max_val > 0.8 else None

    def _extract_hw(self, image):
        """Безопасно получить (h, w) из numpy-изображения.
        Возвращает None, если кадр некорректный.
        """
        shape = getattr(image, 'shape', None)
        if not shape or len(shape) < 2:
            return None
        return int(shape[0]), int(shape[1])

    def find_image_on_screen(self, template_path, threshold=0.8, use_blur=False):
        """Универсальный поиск с опциональным размытием."""
        screenshot = np.array(ImageGrab.grab())
        screen_image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.info(f"Ошибка: файл {template_path} не найден.")
            return None
        if use_blur:
            screen_image = cv2.GaussianBlur(screen_image, (5, 5), 0)
            template = cv2.GaussianBlur(template, (5, 5), 0)
        result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_loc if max_val > threshold else None

    def find_green_zone(self, roi_rgb):
        """
        Возвращает (green_contours, mask)
        green_contours: список контуров зелёных зон (отфильтрованных), отсортирован по площади (больше -> раньше)
        """
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

        lower_green = np.array([27, 72, 110])
        upper_green = np.array([69, 195, 223])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], mask

        contours = [c for c in contours if cv2.contourArea(c) > 264]
        contours.sort(key=cv2.contourArea, reverse=True)

        return contours, mask

    def find_slider(self, roi_rgb):
        """
        Возвращает (slider_contours, mask)
        slider_contours: список контуров-кандидатов (обычно 1 лучший)
        """
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        s_max = 30
        v_min = 216

        mask = ((s <= s_max) & (v >= v_min)).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], mask

        rh, rw = mask.shape[:2]

        best = None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 30:
                continue

            x, y, w, h = cv2.boundingRect(c)

            if h < rh * 0.35:
                continue
            if w > rw * 0.25:
                continue

            score = h / (w + 1)
            if score > best_score:
                best_score = score
                best = c

        if best is None:
            return [], mask

        return [best], mask

    @prevent_reentry
    def start_fishing(self):
        """Основной цикл рыбалки."""
        while self.bot_running:

            self.completed_cycles += 1
            logger.info(f"Цикл завершён: {self.completed_cycles}/{self.cycle_limit}")
            if self.post_cycle_reset_enabled and self.completed_cycles >= self.cycle_limit:
                self.perform_cycle_reset_sequence()
                self.completed_cycles = 0

            logger.info("Ожидание перед первой мини-игрой...")
            time.sleep(5)

            logger.info("Запуск первой мини игры...")
            self.play_mini_game()

            logger.info("Запуск второй мини-игры...")
            self.second_mini_game()


            logger.info("Запуск третьей мини-игры...")
            track_result = self.track_image_movement()
            action_pressed_after_ad_disappear = False
            if track_result == 'ad_disappeared' and self.bot_running:
                action_name = "'Забрать себе'" if self.action_mode == 'take' else "'Отпустить'"
                logger.info(f"AD.png пропало в ROI -> пробуем нажать {action_name}...")
                ok = self.press_action_button()
                if not ok and self.bot_running:
                    logger.info("Не удалось нажать кнопку действия после пропажи AD.png -> возврат к первой мини-игре")
                    continue
                action_pressed_after_ad_disappear = True

            if not action_pressed_after_ad_disappear:
                action_name = "'Забрать себе'" if self.action_mode == 'take' else "'Отпустить'"
                logger.info(f"Запуск функции {action_name}...")
                self.press_action_button()

            logger.info("Возвращаемся к первой мини-игре...")
            time.sleep(3)

        logger.info("Цикл остановлен (bot_running = False).")

    @prevent_reentry
    def play_mini_game(self):
        """Первая мини-игра: жмём пробел чуть раньше входа ползунка в зелёную зону."""
        if not self.bot_running:
            return True

        ROI = (679, 878, 1243, 916)  # left, top, right, bottom
        # Чем больше скорость ползунка, тем раньше нажимаем.
        lead_base_px = 9
        lead_max_px = 36
        previous_slider_center = None

        while self.bot_running:
            if self.stop_bot_on_image('stop.png'):
                return True

            screenshot = np.array(ImageGrab.grab())  # RGB
            hw = self._extract_hw(screenshot)
            if hw is None:
                logger.info("[WARN] Некорректный кадр в play_mini_game, пропускаем итерацию")
                time.sleep(0.05)
                continue
            h, w = hw

            left, top, right, bottom = ROI
            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            roi = screenshot[top:bottom, left:right]
            if roi.size == 0:
                time.sleep(0.05)
                continue

            green_contours, _ = self.find_green_zone(roi)
            slider_contours, _ = self.find_slider(roi)

            if green_contours and slider_contours:
                # Берём самую крупную зелёную зону как основную цель.
                gx, _, gw, _ = cv2.boundingRect(green_contours[0])
                green_left = gx
                green_right = gx + gw

                slider = slider_contours[0]
                sx, _, sw, _ = cv2.boundingRect(slider)
                slider_center = sx + sw // 2

                velocity = 0
                if previous_slider_center is not None:
                    velocity = slider_center - previous_slider_center

                # Раннее нажатие: чем быстрее ползунок, тем раньше срабатывание.
                lead_px = min(lead_max_px, lead_base_px + abs(velocity) * 2)

                # Если уже в зоне — жмём сразу.
                #if green_left <= slider_center <= green_right:
                    #pyautogui.press('space')
                    #logger.info("Ползунок в зелёной зоне, нажат пробел.")
                    #return False

                # Предсказание через 1 кадр: если следующая позиция попадёт в зону,
                # жмём заранее до входа в зелёную маску.
                predicted_center = slider_center + velocity
                near_zone = min(abs(slider_center - green_left), abs(slider_center - green_right)) <= lead_px
                will_enter_zone = green_left <= predicted_center <= green_right

                if near_zone and will_enter_zone:
                    pyautogui.press('space')
                    logger.info(f"Раннее нажатие до входа в зелёную зону (lead={lead_px}px), нажат пробел.")
                    return False

                previous_slider_center = slider_center

            time.sleep(0.05)

    @prevent_reentry
    def second_mini_game(self, show_roi=False):
        bubbles_images = [
            cv2.imread('q11.png', cv2.IMREAD_GRAYSCALE),
            cv2.imread('q12.png', cv2.IMREAD_GRAYSCALE),
        ]

        if any(img is None for img in bubbles_images):
            logger.info("Ошибка загрузки одного из шаблонов пузырьков (q11/q12)!")
            return True

        if show_roi:
            cv2.namedWindow("ROI DEBUG", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROI DEBUG", 940, 540)

        try:
            while self.bot_running:
                full = np.array(ImageGrab.grab())  # RGB
                hw = self._extract_hw(full)
                if hw is None:
                    time.sleep(0.05)
                    continue
                height, width = hw

                left, right = 1325, 1528
                top, bottom = 822, 1004

                left, right = sorted((left, right))
                top, bottom = sorted((top, bottom))

                left = max(0, left)
                top = max(0, top)
                right = min(width, right)
                bottom = min(height, bottom)

                if right - left < 5 or bottom - top < 5:
                    time.sleep(0.05)
                    continue

                roi_img = full[top:bottom, left:right]  # RGB
                if roi_img.size == 0:
                    time.sleep(0.05)
                    continue

                if show_roi:
                    vis = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(vis, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.imshow("ROI DEBUG", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return True

                # ✅ ВМЕСТО q13/q14: если появился красный бар — жмём пробел
                if self._has_red_bar_in_roi(roi_img, bottom_part=0.35):
                    logger.info("Красная полоска найдена! Нажимаем пробел.")
                    pyautogui.press('space')
                    time.sleep(0.3)
                    return False

                # Старое: q11/q12 matchTemplate
                screen_gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)

                for bubble in bubbles_images:
                    th, tw = bubble.shape[:2]
                    rh, rw = screen_gray.shape[:2]
                    if rh < th or rw < tw:
                        continue

                    result = cv2.matchTemplate(screen_gray, bubble, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    if max_val > 0.8:
                        logger.info("Пузырьки найдены! Нажимаем пробел.")
                        pyautogui.press('space')
                        time.sleep(0.3)
                        return False

                time.sleep(0.05)

        finally:
            if show_roi:
                cv2.destroyWindow("ROI DEBUG")

    @prevent_reentry
    def track_image_movement(self):
        """Третья мини-игра: отслеживание движения изображения."""
        previous_frame = None
        current_key = None

        finish_template = 'EZEFISH.jpg' if self.action_mode == 'take' else 'otpustit.png'
        ad_bbox = (837, 1016, 912, 1057)
        ad_seen_in_roi = False
        ad_check_timeout = 30.0
        ad_check_deadline = time.time() + ad_check_timeout
        ad_timeout_logged = False
        flow_noise_threshold = 0.7

        i = 0
        check_every = 5  # проверять шаблон раз в 5 циклов (подстрой: 5/10/15/20)
        while self.bot_running:
            if self.stop_bot_on_image('stop.png'):
                return False

            ad_present = self._template_in_region('AD.png', bbox=ad_bbox, threshold=0.85)
            if ad_present:
                ad_seen_in_roi = True
            elif ad_seen_in_roi:
                logger.info("AD.png пропало в ROI (837, 1016, 912, 1057).")
                if current_key:
                    pydirectinput.keyUp(current_key)
                return 'ad_disappeared'
            elif time.time() > ad_check_deadline and not ad_timeout_logged:
                logger.info(f"Таймаут первичного ожидания AD.png: {ad_check_timeout} сек. Продолжаем без этой проверки.")
                ad_timeout_logged = True

            i += 1
            if i % check_every == 0:
                if self.find_object(finish_template):
                    logger.info(f"Уведомление о рыбе найдено ({finish_template}). Завершаем мини-игру.")
                    if current_key:
                        pydirectinput.keyUp(current_key)
                    return True

            screenshot = ImageGrab.grab()
            screen_np = np.array(screenshot)
            screen_resized = cv2.resize(screen_np, (640, 360))
            screen_gray = cv2.cvtColor(screen_resized, cv2.COLOR_RGB2GRAY)

            if previous_frame is None:
                previous_frame = screen_gray
                time.sleep(0.01)
                continue

            flow = cv2.calcOpticalFlowFarneback(  # type: ignore[arg-type]
                previous_frame, screen_gray, None,
                0.5, 3, 20, 3, 5, 1.2, 0
            )
            flow_x = np.mean(flow[..., 0])
            if flow_x > flow_noise_threshold:
                if current_key != 'd':
                    if current_key:
                        pydirectinput.keyUp(current_key)
                    pydirectinput.keyDown('d')
                    current_key = 'd'
                    logger.info("Движение вправо, зажимаем D")
            elif flow_x < -flow_noise_threshold:
                if current_key != 'a':
                    if current_key:
                        pydirectinput.keyUp(current_key)
                    pydirectinput.keyDown('a')
                    current_key = 'a'
                    logger.info("Движение влево, зажимаем A")
            else:
                if current_key:
                    pydirectinput.keyUp(current_key)
                    current_key = None

            previous_frame = screen_gray
            time.sleep(0.01)

        if current_key:
            pydirectinput.keyUp(current_key)
        return False

    def press_action_button(self, timeout=3.0, poll=1):
        """Нажатие кнопки действия в зависимости от режима (забрать/отпустить)."""
        template_path = 'knopkasebe.jpg' if self.action_mode == 'take' else 'otpustit.png'
        button_name = "'Забрать себе'" if self.action_mode == 'take' else "'Отпустить'"
        start = time.time()

        while self.bot_running and (time.time() - start) < timeout:
            loc = self.find_object(template_path)
            if loc:
                x, y = loc
                pyautogui.moveTo(x + 10, y + 10)
                pyautogui.click()
                logger.info(f"Кнопка {button_name} нажата.")
                return

            time.sleep(poll)

        logger.info(f"Кнопка {button_name} не найдена за {timeout} сек -> выходим (False)")

    def _press_game_key(self, key: str):
        """Надёжное нажатие клавиши в игре: сначала pydirectinput, затем fallback на pyautogui."""
        try:
            pydirectinput.keyDown(key)
            time.sleep(0.05)
            pydirectinput.keyUp(key)
            return
        except Exception as e:
            logger.info(f"[WARN] pydirectinput keyDown/keyUp для {key} не сработал: {e}")

        try:
            pydirectinput.press(key)
            return
        except Exception as e:
            logger.info(f"[WARN] pydirectinput.press для {key} не сработал: {e}")

        pyautogui.press(key)

    def press_game_key(self, key: str):
        """Надёжное нажатие клавиши в игре: сначала pydirectinput, затем fallback на pyautogui."""
        try:
            pydirectinput.keyDown(key)
            time.sleep(0.05)
            pydirectinput.keyUp(key)
            return
        except Exception as e:
            logger.info(f"[WARN] pydirectinput keyDown/keyUp для {key} не сработал: {e}")

        try:
            pydirectinput.press(key)
            return
        except Exception as e:
            logger.info(f"[WARN] pydirectinput.press для {key} не сработал: {e}")

        pyautogui.press(key)

    def perform_cycle_reset_sequence(self):
        """После N циклов: через 2с ESC, через 6с ESC, пауза 6с, затем E и ещё раз E через 6с."""
        if not self.bot_running:
            return

        logger.info(
            f"Достигнут лимит {self.cycle_limit} циклов. "
            f"Выполняем ESC(2с) -> CLICK1(6с) -> CLICK2(6с, чередование) -> E(6с)."
        )

        second_click_coords = self.reset_second_click_coords[self._reset_second_click_index]
        steps = [
            (2, 'key', 'esc'),
            (6, 'click', self.reset_first_click_coords),
            (6, 'click', second_click_coords),
            (6, 'key', 'e'),
        ]

        for wait_seconds, action_type, payload in steps:
            if not self.bot_running:
                return
            time.sleep(wait_seconds)
            if not self.bot_running:
                return
            if action_type == 'key':
                self._press_game_key(payload)
                logger.info(f"Нажата клавиша: {payload.upper()} (после {wait_seconds}с)")
            else:
                x, y = payload
                pyautogui.click(x=x, y=y)
                logger.info(f"Сделан клик мышью в ({x}, {y}) (после {wait_seconds}с)")

            self._reset_second_click_index = (self._reset_second_click_index + 1) % len(self.reset_second_click_coords)


    @prevent_reentry
    def press_knopkasebe_button(self, timeout=3.0, poll=1):
        """Совместимость со старым именем метода."""
        self.press_action_button(timeout=timeout, poll=poll)

    def _template_in_region(self, template_path, bbox, threshold=0.85):
        """
        bbox = (left, top, right, bottom) в координатах экрана.
        Возвращает True если найдено совпадение >= threshold.
        """
        tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            logger.info(f"Не удалось загрузить шаблон: {template_path}")
            return False

        roi = np.array(ImageGrab.grab(bbox=bbox))  # RGB
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        th, tw = tpl.shape[:2]
        rh, rw = roi_gray.shape[:2]
        if rh < th or rw < tw:
            return False

        res = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold

    def stop_bot_on_image(self, template_path, bbox=(306, 851, 363, 904), threshold=0.85):
        """
        Если картинка появилась в bbox — нажимает ESC, выключает бот и возвращает True.
        Иначе False.
        """
        if self._template_in_region(template_path, bbox, threshold):
            logger.info("Найдена стоп-картинка -> ESC и остановка бота.")
            pyautogui.press('esc')
            self.bot_running = False
            return True
        return False

    def stop_fishing(self):
        """Остановка бота."""
        self.bot_running = False


class BotController:
    def __init__(self):
        self.bot = FishingBot()
        self._lock = threading.Lock()
        self._thread = None

    def play_sound(self):
        if not SOUND_ENABLED:
            return
        try:
            pygame.mixer.music.load(sound_file_path)
            pygame.mixer.music.play()
        except Exception as e:
            logger.info(f"[WARN] Ошибка воспроизведения звука: {e}")

    def start(self):
        with self._lock:
            if self.bot.bot_running:
                logger.info("Бот уже запущен.")
                return
            self.play_sound()
            self.bot.completed_cycles = 0
            self.bot.bot_running = True
            self._thread = threading.Thread(target=self.bot.start_fishing, daemon=True)
            self._thread.start()
            logger.info("Бот запущен.")

    def stop(self):
        with self._lock:
            if not self.bot.bot_running:
                logger.info("Бот уже остановлен.")
                return
            self.bot.stop_fishing()
            self.play_sound()
            logger.info("Бот остановлен.")

    def press_esc(self):
        keyboard.press_and_release('esc')
        logger.info("Нажата клавиша Esc (через клавишу 0).")

    def set_take_mode(self):
        self.bot.set_action_mode('take')

    def set_release_mode(self):
        self.bot.set_action_mode('release')

    def exit_program(self):
        logger.info("Выход: останавливаем бота и закрываем программу...")
        self.stop()
        raise SystemExit


def run_hotkey_mode():
    ctl = BotController()

    keyboard.add_hotkey('+', ctl.start)
    keyboard.add_hotkey('-', ctl.stop)
    keyboard.add_hotkey('0', ctl.press_esc)

    # ESC = выйти из программы
    keyboard.add_hotkey('esc', ctl.exit_program)

    logger.info("Горячие клавиши активны:")
    logger.info("  +  -> старт")
    logger.info("  -  -> стоп")
    logger.info("  0  -> нажать Esc в игре")
    logger.info("  Esc -> выйти из программы")

    try:
        keyboard.wait()  # ждём любые события
    except SystemExit:
        pass


# ===== Integrated from license_manager.py =====

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Формат: FBOT-<7|14|30>-<B5>-<B8>-<B3>-<B9>-<checksum>
# B5/B8/B3/B9 — 4 цифры с проверкой кратности на 5/8/3/9 соответственно.
KEY_PATTERN = re.compile(r"^FBOT-(7|14|30)-(\d{4})-(\d{4})-(\d{4})-(\d{4})-([A-F0-9]{2})$")


@dataclass
class LicenseStatus:
    is_active: bool
    key_value: Optional[str]
    expires_at: Optional[datetime]

    @property
    def seconds_left(self) -> int:
        if not self.expires_at:
            return 0
        return max(0, int((self.expires_at - datetime.utcnow()).total_seconds()))


class LicenseManager:
    def __init__(self, db_path: str = "licenses.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS license_state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    active_key TEXT,
                    activated_at TEXT,
                    expires_at TEXT
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO license_state(id) VALUES (1)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS used_keys (
                    key_hash TEXT PRIMARY KEY,
                    used_at TEXT NOT NULL
                )
                """
            )

    def _checksum(self, base: str) -> str:
        return hashlib.sha256(base.encode("utf-8")).hexdigest().upper()[:2]

    def _key_hash(self, key_value: str) -> str:
        return hashlib.sha256(key_value.encode("utf-8")).hexdigest().upper()

    def validate_key_format(self, key_value: str) -> int:
        key_value = key_value.strip().upper()
        match = KEY_PATTERN.match(key_value)
        if not match:
            raise ValueError("Неверный формат ключа")

        duration = int(match.group(1))
        blocks = [int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))]
        checksum = match.group(6)

        divisors = [5, 8, 3, 9]
        for idx, (block, div) in enumerate(zip(blocks, divisors), start=1):
            if block % div != 0:
                raise ValueError(f"Блок {idx} должен быть кратен {div}")

        base = "-".join(key_value.split("-")[:-1])
        if self._checksum(base) != checksum:
            raise ValueError("Неверная контрольная сумма ключа")

        return duration

    def activate_with_key(self, key_value: str) -> LicenseStatus:
        key_value = key_value.strip().upper()
        duration = self.validate_key_format(key_value)
        key_hash = self._key_hash(key_value)

        now = datetime.utcnow()
        expires_at = now + timedelta(days=duration)

        with self._connect() as conn:
            used = conn.execute("SELECT 1 FROM used_keys WHERE key_hash = ?", (key_hash,)).fetchone()
            if used:
                raise ValueError("Этот ключ уже был использован")

            conn.execute(
                "INSERT INTO used_keys(key_hash, used_at) VALUES(?, ?)",
                (key_hash, now.isoformat()),
            )
            conn.execute(
                """
                UPDATE license_state
                SET active_key = ?, activated_at = ?, expires_at = ?
                WHERE id = 1
                """,
                (key_value, now.isoformat(), expires_at.isoformat()),
            )

        return LicenseStatus(True, key_value, expires_at)

    def get_status(self) -> LicenseStatus:
        with self._connect() as conn:
            row = conn.execute("SELECT active_key, expires_at FROM license_state WHERE id = 1").fetchone()

        if not row or not row["active_key"] or not row["expires_at"]:
            return LicenseStatus(False, None, None)

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() >= expires_at:
            self.deactivate()
            return LicenseStatus(False, None, None)

        return LicenseStatus(True, row["active_key"], expires_at)

    def deactivate(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE license_state SET active_key = NULL, activated_at = NULL, expires_at = NULL WHERE id = 1")


# ===== Integrated from launcher.py =====

CONFIG_PATH = "config.json"

logger = logging.getLogger(__name__)


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class LicensePanel(ttk.LabelFrame):
    def __init__(self, parent, license_manager: LicenseManager, on_status_change):
        super().__init__(parent, text="Лицензирование", padding=12, style="Card.TLabelframe")
        self.license_manager = license_manager
        self.on_status_change = on_status_change

        self.key_to_activate = tk.StringVar()
        self.status_var = tk.StringVar(value="Лицензия не активна")

        ttk.Label(self, textvariable=self.status_var, style="Card.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )

        ttk.Label(self, text="Ввести ключ:", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(self, textvariable=self.key_to_activate, width=30).grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Button(self, text="Активировать", command=self.activate_key, style="Accent.TButton").grid(
            row=1, column=2, padx=8, pady=(10, 0)
        )

        ttk.Button(self, text="Сбросить лицензию", command=self.deactivate_key).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        ttk.Label(
            self,
            text="Произойдет привязка пк.",
            style="Hint.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.columnconfigure(1, weight=1)

    def refresh_status(self):
        status = self.license_manager.get_status()
        if status.is_active and status.expires_at:
            left = timedelta(seconds=status.seconds_left)
            self.status_var.set(f"Активен ключ: {status.key_value} | осталось: {left}")
        else:
            self.status_var.set("Лицензия не активна")
        self.on_status_change(status.is_active)

    def activate_key(self):
        try:
            self.license_manager.activate_with_key(self.key_to_activate.get())
            self.key_to_activate.set("")
            self.refresh_status()
            messagebox.showinfo("ОК", "Лицензия активирована")
        except Exception as e:
            messagebox.showerror("Ошибка активации", str(e))

    def deactivate_key(self):
        self.license_manager.deactivate()
        self.refresh_status()


class LicenseActivationWindow(tk.Tk):
    def __init__(self, license_manager: LicenseManager):
        super().__init__()
        self.title("Активация лицензии")
        self.geometry("560x280")
        self.minsize(520, 260)

        self.license_manager = license_manager
        self.activated = False

        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        bg_main = "#0f131c"
        bg_card = "#171e2a"
        fg_primary = "#e7edf9"
        fg_secondary = "#a8b3c6"

        self.configure(bg=bg_main)

        style.configure("Main.TFrame", background=bg_main)
        style.configure("Card.TFrame", background=bg_card)
        style.configure("Header.TLabel", background=bg_main, foreground=fg_primary, font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background=bg_main, foreground=fg_secondary, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=bg_card, foreground=fg_primary)
        style.configure("Hint.TLabel", background=bg_card, foreground=fg_secondary)
        style.configure("Card.TLabelframe", background=bg_card, foreground=fg_primary)
        style.configure("Card.TLabelframe.Label", background=bg_card, foreground=fg_primary)
        style.configure("Accent.TButton", background="#4f8cff", foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", "#6ba0ff")])

    def _build_ui(self):
        root = ttk.Frame(self, padding=14, style="Main.TFrame")
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="FishingBot", style="Header.TLabel").pack(anchor="w")
        ttk.Label(root, text="Перед началом работы активируйте ключ", style="SubHeader.TLabel").pack(
            anchor="w", pady=(2, 10)
        )

        self.license_panel = LicensePanel(root, self.license_manager, self.on_license_change)
        self.license_panel.pack(fill="both", expand=True)
        self.license_panel.refresh_status()

    def on_license_change(self, active: bool):
        if active:
            self.activated = True
            messagebox.showinfo("ОК", "Лицензия активирована. Открываем основное окно.")
            self.destroy()

    def on_close(self):
        self.activated = False
        self.destroy()


class Launcher(tk.Tk):
    CYCLE_RESET_LIMIT = 7

    def __init__(self, license_manager: LicenseManager):
        super().__init__()
        self.title("Рыболовный помощник")
        self.geometry("560x360")
        self.minsize(540, 340)

        self._hotkey_ids = []
        self.ctl = BotController()
        self.license_manager = license_manager
        self.cfg = {}

        self.status_var = tk.StringVar(value="STOPPED")
        self.license_info_var = tk.StringVar(value="Лицензия: проверка...")
        self.reset_enabled_var = tk.BooleanVar(value=True)

        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.reload_config()
        self.sync_reset_options()

        self.after(200, self.poll_status)
        self.after(1000, self.poll_license)


    def _configure_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        bg_main = "#0f131c"
        bg_card = "#171e2a"
        fg_primary = "#e7edf9"
        fg_secondary = "#a8b3c6"

        self.configure(bg=bg_main)

        style.configure("Main.TFrame", background=bg_main)
        style.configure("Card.TFrame", background=bg_card)

        style.configure("Header.TLabel", background=bg_main, foreground=fg_primary, font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background=bg_main, foreground=fg_secondary, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#223149", foreground="#dce7ff", font=("Segoe UI", 10, "bold"))

        style.configure("Card.TLabel", background=bg_card, foreground=fg_primary)
        style.configure("Hint.TLabel", background=bg_card, foreground=fg_secondary)

        style.configure("Card.TLabelframe", background=bg_card, foreground=fg_primary)
        style.configure("Card.TLabelframe.Label", background=bg_card, foreground=fg_primary)

        style.configure("TButton", padding=8)
        style.configure("Accent.TButton", background="#4f8cff", foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", "#6ba0ff")])

        style.configure("TCheckbutton", background=bg_card, foreground=fg_primary)
        style.map("TCheckbutton", background=[("active", bg_card)])

    def _build_ui(self):
        root = ttk.Frame(self, padding=14, style="Main.TFrame")
        root.pack(fill="both", expand=True)
        self.content_root = root

        header = ttk.Frame(root, style="Main.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="FishingBot", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="Управление рыбалкой в один клик", style="SubHeader.TLabel").pack(side="left", padx=12)
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel", padding=(12, 6)).pack(side="right")

        ttk.Label(root, textvariable=self.license_info_var, style="SubHeader.TLabel").pack(anchor="w", pady=(6, 0))

        control_card = ttk.Frame(root, style="Card.TFrame", padding=12)
        control_card.pack(fill="x", pady=(12, 8))

        controls_row = ttk.Frame(control_card, style="Card.TFrame")
        controls_row.pack(fill="x")
        ttk.Button(controls_row, text="START", command=self.on_start, style="Accent.TButton").pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(controls_row, text="STOP", command=self.on_stop).pack(side="left", fill="x", expand=True, padx=(5, 0))

        modes = ttk.Frame(control_card, style="Card.TFrame")
        modes.pack(fill="x", pady=(10, 0))
        ttk.Button(modes, text="Забрать себе", command=self.ctl.set_take_mode).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(modes, text="Отпустить", command=self.ctl.set_release_mode).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        reset_box = ttk.LabelFrame(root, text="Сброс", padding=12, style="Card.TLabelframe")
        reset_box.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(
            reset_box,
            text=f"Включить замену наживки после {self.CYCLE_RESET_LIMIT} циклов",
            variable=self.reset_enabled_var,
            command=self.sync_reset_options,
        ).pack(anchor="w")

        ttk.Button(root, text="Reload config.json", command=self.on_reload).pack(fill="x", pady=(0, 10))

    def apply_config_to_bot(self, cfg: dict):
        global sound_file_path, SOUND_ENABLED

        sound = cfg.get("sound", {})
        if "file" in sound:
            sound_file_path = sound["file"]
        if "enabled" in sound:
            SOUND_ENABLED = bool(sound["enabled"])

        behavior = cfg.get("behavior", {})
        reward_action = behavior.get("reward_action", "take")
        if reward_action == "release":
            self.ctl.set_release_mode()
        else:
            self.ctl.set_take_mode()

    def setup_hotkeys(self, cfg: dict):
        self._clear_hotkeys()
        hk = cfg.get("hotkeys", {})
        self._safe_add_hotkey(hk.get("start", "+"), self.on_start)
        self._safe_add_hotkey(hk.get("stop", "-"), self.on_stop)
        self._safe_add_hotkey(hk.get("press_esc", "0"), self.ctl.press_esc)

    def _safe_add_hotkey(self, hotkey, callback):
        try:
            hk_id = keyboard.add_hotkey(hotkey, callback)
            self._hotkey_ids.append(hk_id)
        except Exception as e:
            logger.info(f"[WARN] hotkey {hotkey} disabled: {e}")

    def _clear_hotkeys(self):
        for hk_id in self._hotkey_ids:
            try:
                keyboard.remove_hotkey(hk_id)
            except Exception:
                pass
        self._hotkey_ids = []

    def reload_config(self):
        try:
            self.cfg = load_config()
            self.apply_config_to_bot(self.cfg)
            self.setup_hotkeys(self.cfg)
        except Exception as e:
            messagebox.showerror("Config error", str(e))

    def sync_reset_options(self):
        self.ctl.bot.set_post_cycle_reset(self.reset_enabled_var.get())
        self.ctl.bot.set_cycle_limit(self.CYCLE_RESET_LIMIT)

    def on_start(self):
        if not self.license_manager.get_status().is_active:
            messagebox.showwarning("Лицензия", "Сначала активируйте ключ доступа")
            return
        self.ctl.start()

    def on_stop(self):
        self.ctl.stop()

    def on_reload(self):
        self.reload_config()
        messagebox.showinfo("OK", "config.json перезагружен")

    def poll_status(self):
        self.status_var.set("RUNNING" if self.ctl.bot.bot_running else "STOPPED")
        self.after(200, self.poll_status)

    def poll_license(self):
        status = self.license_manager.get_status()
        if status.is_active and status.expires_at:
            left = timedelta(seconds=status.seconds_left)
            self.license_info_var.set(f"Лицензия активна, осталось: {left}")
        else:
            self.license_info_var.set("Лицензия неактивна")

        if not status.is_active:
            self.on_stop()
            messagebox.showwarning("Лицензия", "Лицензия неактивна. Приложение будет закрыто.")
            self.on_close()
            return
        self.after(1000, self.poll_license)

    def on_close(self):
        try:
            self.ctl.stop()
        finally:
            self._clear_hotkeys()
            self.destroy()


def run_gui_app():
    manager = LicenseManager()
    status = manager.get_status()

    if not status.is_active:
        activation = LicenseActivationWindow(manager)
        activation.mainloop()
        if not activation.activated:
            raise SystemExit

    app = Launcher(manager)
    app.mainloop()


if __name__ == "__main__":
    run_gui_app()
