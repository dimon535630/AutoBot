import functools
import threading
import time
import cv2
import keyboard
import numpy as np
import pyautogui
import pydirectinput
import pygame
from PIL import ImageGrab

# Пути к файлам
sound_file_path = 'ASK.mp3'

# Инициализация звука (если не получится — просто отключим звук)
SOUND_ENABLED = True
try:
    pygame.mixer.init()
except Exception as e:
    SOUND_ENABLED = False
    print(f"[WARN] Звук отключён (pygame.mixer.init не удался): {e}")


def prevent_reentry(method):
    """Не даёт методу запуститься повторно, пока предыдущий вызов не завершился."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        acquired, lock = self._acquire_method_lock(method.__name__)
        if not acquired:
            print(f"Пропуск: {method.__name__} уже выполняется.")
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
            print(f"[WARN] Неизвестный режим действия: {mode}")
            return
        self.action_mode = mode
        print(f"Режим действия переключен: {'ЗАБРАТЬ СЕБЕ' if mode == 'take' else 'ОТПУСТИТЬ'}")

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
        print(f"Выбрано действие после рыбалки: {name} ({template}).")

    def set_post_cycle_reset(self, enabled: bool):
        """Вкл/выкл последовательность клавиш после N циклов."""
        self.post_cycle_reset_enabled = bool(enabled)
        state = "включено" if self.post_cycle_reset_enabled else "выключено"
        print(f"Сброс после {self.cycle_limit} циклов: {state}.")

    def set_cycle_limit(self, cycle_limit: int):
        """Изменение лимита циклов для последовательности сброса."""
        self.cycle_limit = max(1, int(cycle_limit))
        print(f"Новый лимит циклов до сброса: {self.cycle_limit}")

    def find_object(self, template_path):
        """Поиск изображения на экране."""
        screenshot = np.array(ImageGrab.grab())
        screen_image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Ошибка: файл {template_path} не найден.")
            return None
        result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"Уровень совпадения для {template_path}: {max_val}")
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
            print(f"Ошибка: файл {template_path} не найден.")
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
            print(f"Цикл завершён: {self.completed_cycles}/{self.cycle_limit}")
            if self.post_cycle_reset_enabled and self.completed_cycles >= self.cycle_limit:
                self.perform_cycle_reset_sequence()
                self.completed_cycles = 0

            print("Ожидание перед первой мини-игрой...")
            time.sleep(5)

            print("Запуск первой мини игры...")
            self.play_mini_game()

            print("Запуск второй мини-игры...")
            self.second_mini_game()


            print("Запуск третьей мини-игры...")
            track_result = self.track_image_movement()
            action_pressed_after_ad_disappear = False
            if track_result == 'ad_disappeared' and self.bot_running:
                action_name = "'Забрать себе'" if self.action_mode == 'take' else "'Отпустить'"
                print(f"AD.png пропало в ROI -> пробуем нажать {action_name}...")
                ok = self.press_action_button()
                if not ok and self.bot_running:
                    print("Не удалось нажать кнопку действия после пропажи AD.png -> возврат к первой мини-игре")
                    continue
                action_pressed_after_ad_disappear = True

            if not action_pressed_after_ad_disappear:
                action_name = "'Забрать себе'" if self.action_mode == 'take' else "'Отпустить'"
                print(f"Запуск функции {action_name}...")
                self.press_action_button()

            print("Возвращаемся к первой мини-игре...")
            time.sleep(3)

        print("Цикл остановлен (bot_running = False).")

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
                print("[WARN] Некорректный кадр в play_mini_game, пропускаем итерацию")
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
                    #print("Ползунок в зелёной зоне, нажат пробел.")
                    #return False

                # Предсказание через 1 кадр: если следующая позиция попадёт в зону,
                # жмём заранее до входа в зелёную маску.
                predicted_center = slider_center + velocity
                near_zone = min(abs(slider_center - green_left), abs(slider_center - green_right)) <= lead_px
                will_enter_zone = green_left <= predicted_center <= green_right

                if near_zone and will_enter_zone:
                    pyautogui.press('space')
                    print(f"Раннее нажатие до входа в зелёную зону (lead={lead_px}px), нажат пробел.")
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
            print("Ошибка загрузки одного из шаблонов пузырьков (q11/q12)!")
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
                    print("Красная полоска найдена! Нажимаем пробел.")
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
                        print("Пузырьки найдены! Нажимаем пробел.")
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
                print("AD.png пропало в ROI (837, 1016, 912, 1057).")
                if current_key:
                    pydirectinput.keyUp(current_key)
                return 'ad_disappeared'
            elif time.time() > ad_check_deadline and not ad_timeout_logged:
                print(f"Таймаут первичного ожидания AD.png: {ad_check_timeout} сек. Продолжаем без этой проверки.")
                ad_timeout_logged = True

            i += 1
            if i % check_every == 0:
                if self.find_object(finish_template):
                    print(f"Уведомление о рыбе найдено ({finish_template}). Завершаем мини-игру.")
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
                    print("Движение вправо, зажимаем D")
            elif flow_x < -flow_noise_threshold:
                if current_key != 'a':
                    if current_key:
                        pydirectinput.keyUp(current_key)
                    pydirectinput.keyDown('a')
                    current_key = 'a'
                    print("Движение влево, зажимаем A")
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
                print(f"Кнопка {button_name} нажата.")
                return

            time.sleep(poll)

        print(f"Кнопка {button_name} не найдена за {timeout} сек -> выходим (False)")

    def _press_game_key(self, key: str):
        """Надёжное нажатие клавиши в игре: сначала pydirectinput, затем fallback на pyautogui."""
        try:
            pydirectinput.keyDown(key)
            time.sleep(0.05)
            pydirectinput.keyUp(key)
            return
        except Exception as e:
            print(f"[WARN] pydirectinput keyDown/keyUp для {key} не сработал: {e}")

        try:
            pydirectinput.press(key)
            return
        except Exception as e:
            print(f"[WARN] pydirectinput.press для {key} не сработал: {e}")

        pyautogui.press(key)

    def press_game_key(self, key: str):
        """Надёжное нажатие клавиши в игре: сначала pydirectinput, затем fallback на pyautogui."""
        try:
            pydirectinput.keyDown(key)
            time.sleep(0.05)
            pydirectinput.keyUp(key)
            return
        except Exception as e:
            print(f"[WARN] pydirectinput keyDown/keyUp для {key} не сработал: {e}")

        try:
            pydirectinput.press(key)
            return
        except Exception as e:
            print(f"[WARN] pydirectinput.press для {key} не сработал: {e}")

        pyautogui.press(key)

    def perform_cycle_reset_sequence(self):
        """После N циклов: через 2с ESC, через 6с ESC, пауза 6с, затем E и ещё раз E через 6с."""
        if not self.bot_running:
            return

        print(
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
                print(f"Нажата клавиша: {payload.upper()} (после {wait_seconds}с)")
            else:
                x, y = payload
                pyautogui.click(x=x, y=y)
                print(f"Сделан клик мышью в ({x}, {y}) (после {wait_seconds}с)")

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
            print(f"Не удалось загрузить шаблон: {template_path}")
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
            print("Найдена стоп-картинка -> ESC и остановка бота.")
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
            print(f"[WARN] Ошибка воспроизведения звука: {e}")

    def start(self):
        with self._lock:
            if self.bot.bot_running:
                print("Бот уже запущен.")
                return
            self.play_sound()
            self.bot.completed_cycles = 0
            self.bot.bot_running = True
            self._thread = threading.Thread(target=self.bot.start_fishing, daemon=True)
            self._thread.start()
            print("Бот запущен.")

    def stop(self):
        with self._lock:
            if not self.bot.bot_running:
                print("Бот уже остановлен.")
                return
            self.bot.stop_fishing()
            self.play_sound()
            print("Бот остановлен.")

    def press_esc(self):
        keyboard.press_and_release('esc')
        print("Нажата клавиша Esc (через клавишу 0).")

    def set_take_mode(self):
        self.bot.set_action_mode('take')

    def set_release_mode(self):
        self.bot.set_action_mode('release')

    def exit_program(self):
        print("Выход: останавливаем бота и закрываем программу...")
        self.stop()
        raise SystemExit


def main():
    ctl = BotController()

    keyboard.add_hotkey('+', ctl.start)
    keyboard.add_hotkey('-', ctl.stop)
    keyboard.add_hotkey('0', ctl.press_esc)

    # ESC = выйти из программы
    keyboard.add_hotkey('esc', ctl.exit_program)

    print("Горячие клавиши активны:")
    print("  +  -> старт")
    print("  -  -> стоп")
    print("  0  -> нажать Esc в игре")
    print("  Esc -> выйти из программы")

    try:
        keyboard.wait()  # ждём любые события
    except SystemExit:
        pass


if __name__ == '__main__':
    main()
