import sys
import numpy as np
import math
import random

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QDialog, QProgressDialog
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QImage
from PIL import Image

import cell_engine as ce
# import saving_widget

S: tuple[int] = (2, 3)
B: tuple[int] = (2, )
C: int = 9

rules = (S, B, C)

c1: tuple[int] = (255, 255, 0)
c2: tuple[int] = (230, 0, 0)
c3: tuple[int] = (50, 0, 0)

standart_colors = [c1, c2, c3]

def get_colors_by_rules_generations(rules, c1, c2, c3):
    base_grad = ce.get_color_gradient(c2, c3, rules[2] - 1)
    base_grad.insert(0, c1)
    base_grad.reverse()
    base_grad = np.array(base_grad)
    return base_grad


class GameAreaWidget(QWidget):
    MAX_SPEED = 500
    ZOOM_STEPS = {10: 0.5, 20: 2, 50: 3, 10000: 10}
    MAX_ZOOM = 100

    def __init__(
            self, parent,
            x: int, y: int, width: int, height: int,
            c_e: ce.CellEngine, speed: int):
        QWidget.__init__(self, parent)
        self.top: int = x
        self.left: int = y
        self.width: int = width
        self.height: int = height
        self.init_widget()

        self.cell_engine = c_e
        # self.cell_engine.set_square()

        self.iters = 0
        self.zoom = 1
        self.set_position_on_midle()

        self.last_x = None
        self.last_y = None

        self.press_pos = None

        self.is_mause_pos_labeld = False

        self.picktimer = QTimer()
        self.picktimer.timeout.connect(self.update_game)
        self.change_speed(speed)

        self.setMouseTracking(True)


    def set_iters(self, iters):
        self.iters = iters


    def set_cell_engine(self, c_e):
        self.cell_engine = c_e
        self.update()


    def set_position_on_midle(self):
        w, h = self.cell_engine.get_size()
        self.position = [
            w//2 - (self.width//2) / self.zoom,
            h//2 - (self.height//2) / self.zoom
        ]


    def count_cell_by_mouse(self, x, y):
        c_x = math.ceil(self.position[0] + x / self.zoom)
        c_y = math.ceil(self.position[1] + y / self.zoom)
        return [c_x, c_y]


    def set_mouse_labels(self, label_x, label_y):
        self.label_x = label_x
        self.label_y = label_y
        self.is_mause_pos_labeld = True
        self.fill_labels_none()


    def set_label_iter(self, label):
        self.label_iter = label
        label.picktimer = QTimer()
        func = lambda: label.setText(f'iterations: {self.iters}')
        label.picktimer.timeout.connect(func)
        label.picktimer.start()
        label.picktimer.setInterval(0)


    def update_labels(self, mouse_x, mouse_y):
        c_x, c_y = self.count_cell_by_mouse(mouse_x, mouse_y)

        w, h = self.cell_engine.get_size()
        if w >= c_x > 0 and h >= c_y > 0:
            self.label_x.setText(f'x: {c_x}')
            self.label_y.setText(f'y: {c_y}')
        else:
            self.fill_labels_none()


    def fill_labels_none(self):
        self.label_x.setText('x: ')
        self.label_y.setText('y: ')


    def remove_mose_labels(self):
        self.label_x = None
        self.label_y = None
        self.is_mause_pos_labeld = False


    def init_widget(self):
        self.move(self.top, self.left)
        self.setFixedSize(self.width, self.height)


    def is_run(self):
        return self.picktimer.isActive()


    def start_timer(self):
        if not self.is_run():
            self.picktimer.start()


    def stop_timer(self):
        if self.is_run():
            self.picktimer.stop()


    def count_delay_by_speed(self):
        delay = 0
        if self.speed != self.MAX_SPEED:
            delay = self.MAX_SPEED // self.speed
        return delay


    def change_speed(self, speed):
        self.speed = speed
        self.picktimer.setInterval(self.count_delay_by_speed())


    def update_game_map(self):
        self.cell_engine.game_step()
        self.iters += 1


    def update_game(self):
        self.update_game_map()
        self.update()


    def zoom_area(self, x, y, zoom):
        self.position = ce.count_zoom_position(self.position,
                                               (x, y), zoom, self.zoom)


    def get_pixmap(self, pos = (0, 0), zoom = 1):
        w, h = self.cell_engine.get_size()
        pixmap = np.zeros((h, w, 3), dtype=np.uint8)
        self.cell_engine.blit_on_array(pixmap, pos, zoom)
        return pixmap


    def draw_game(self, painter):
        pixmap = self.get_pixmap(self.position, self.zoom)
        w, h = self.cell_engine.get_size()
        bytes_per_line = w * 3
        paint_img = QImage(pixmap, w, h, bytes_per_line, QImage.Format_RGB888)
        painter.drawImage(0, 0, paint_img)


    def mouseMoveEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if event.buttons() == QtCore.Qt.NoButton:
            self.update_labels(event.x(), event.y())
        elif (event.buttons() == QtCore.Qt.LeftButton and
            (bool(modifiers == QtCore.Qt.ControlModifier))):
                if self.last_x is None or self.last_y is None:
                    self.last_x = event.x()
                    self.last_y = event.y()
                else:
                    delta_x = (event.x() - self.last_x) / self.zoom
                    delta_y = (event.y() - self.last_y) / self.zoom
                    self.last_x = event.x()
                    self.last_y = event.y()
                    self.position[0] -= delta_x
                    self.position[1] -= delta_y

                    if not self.is_run():
                        self.update()


    def mouseReleaseEvent(self, event):
        if (self.press_pos is not None and event.pos() in self.rect()):
            modifiers = QApplication.keyboardModifiers()
            c_x_l, c_y_l = self.count_cell_by_mouse(*self.press_pos)
            c_x, c_y = self.count_cell_by_mouse(event.x(), event.y())
            if event.button() == Qt.MiddleButton:
                self.set_position_on_midle()
                if not self.is_run():
                    self.update()
            elif (not self.is_run() and
                bool(modifiers == QtCore.Qt.NoModifier) and
                c_x_l == c_x and c_y_l == c_y):
                    if event.button() == Qt.RightButton:
                        self.cell_engine.remove_pixel(c_x - 1, c_y - 1)
                        self.update()
                    elif event.button() == Qt.LeftButton:
                        self.cell_engine.set_pixel(c_x - 1, c_y - 1)
                        self.update()

        self.press_pos = None


    def leaveEvent(self, event):
        self.fill_labels_none()


    def mousePressEvent(self, event):
        self.press_pos = (event.x(), event.y())
        if event.buttons() == QtCore.Qt.LeftButton:
            self.last_x = None
            self.last_y = None


    def wheelEvent(self, event):
        for zoom_step_range in self.ZOOM_STEPS:
            if self.zoom < zoom_step_range:
                zoom_step = self.ZOOM_STEPS[zoom_step_range]
                break
        steps = event.angleDelta().y() // 120 * zoom_step
        new_zoom = min(max(self.zoom + steps, 1), 100)
        normolize_x = event.x()
        normolize_y = event.y()
        self.zoom_area(normolize_x, normolize_y, new_zoom)
        self.zoom = new_zoom
        if not self.is_run():
            self.update()


    def paintEvent(self, event):
        qp = QPainter(self)
        self.draw_game(qp)
        qp.end()



class GameControllerUi(object):
    LIGHT_RED = '#DB4544'
    LIGHT_GREEN = '#44DB71'
    RED = '#8F1110'
    GREEN = '#259b48'


    def add_generations_linear_inputs(self,
            s_input_generations_linear,
            b_input_generations_linear,
            c_input_generations_linear):
        self.s_input_generations_linear = s_input_generations_linear
        self.b_input_generations_linear = b_input_generations_linear
        self.c_input_generations_linear = c_input_generations_linear


    def add_generations_standart_inputs(self,
            s_input_generations_standart,
            b_input_generations_standart,
            c_input_generations_standart):
        self.s_input_generations_standart = s_input_generations_standart
        self.b_input_generations_standart = b_input_generations_standart
        self.c_input_generations_standart = c_input_generations_standart


    def add_crystal_inputs(self, input_crystal):
        self.input_crystal = input_crystal


    def get_generations_linear_inputs(self):
        return (
            self.s_input_generations_standart,
            self.b_input_generations_standart,
            self.c_input_generations_standart
        )


    def get_generations_standart_inputs(self):
        return (
            self.s_input_generations_standart,
            self.b_input_generations_standart,
            self.c_input_generations_standart
        )


    def get_crystal_inputs(self):
        return (
            self.input_crystal,
        )


    def get_rules_generations_linear(self):
        s_data = self.s_input_generations_linear.text()
        b_data = self.b_input_generations_linear.text()
        c_data = self.c_input_generations_linear.text()

        s_list = list(map(int, s_data.split()))
        b_list = list(map(int, b_data.split()))
        c_res = int(c_data) + 1

        return (s_list, b_list, c_res)


    def get_rules_generations_standart(self):
        s_data = self.s_input_generations_standart.text()
        b_data = self.b_input_generations_standart.text()
        c_data = self.c_input_generations_standart.text()

        s_list = list(map(int, s_data.split()))
        b_list = list(map(int, b_data.split()))
        c_res = int(c_data) + 1

        return (s_list, b_list, c_res)


    def get_rules_crystal(self):
        data = int(self.input_crystal.text())
        n = 10
        rules = [0 for i in range(n)]
        iter = 0
        while data > 0:
            rules[iter] = data % 2
            data //= 2
            if iter >= n:
                break
            iter += 1

        rules.reverse()
        return rules


    def set_generations_linear_ui_from_rules(self, rules):
        self.s_input_generations_linear.setText(' '.join(map(str, rules[0])))
        self.b_input_generations_linear.setText(' '.join(map(str, rules[1])))
        self.c_input_generations_linear.setText(str(rules[2]-1))


    def set_generations_standart_ui_from_rules(self, rules):
        self.s_input_generations_standart.setText(' '.join(map(str, rules[0])))
        self.b_input_generations_standart.setText(' '.join(map(str, rules[1])))
        self.c_input_generations_standart.setText(str(rules[2]-1))


    def set_crystal_ui_from_rules(self, rules):
        n = len(rules)
        summ = 0
        for i in range(n):
            summ += rules[i] * (2 ** (n - 1 - i))

        self.input_crystal.setText(str(summ))


    def set_event_on_inputs_generations_linear(self, g_c, update_btn, clear_map_cb):
        def f_update():
            self.s_input_generations_linear.setStyleSheet(f"background-color: {self.GREEN};")
            self.b_input_generations_linear.setStyleSheet(f"background-color: {self.GREEN};")
            self.c_input_generations_linear.setStyleSheet(f"background-color: {self.GREEN};")
            rules = self.get_rules_generations_linear()
            pallet = g_c.CELL_ENGINES[g_c.GENERATIONS_LINEAR]['colors_pallet']
            colors = get_colors_by_rules_generations(rules, *pallet)
            dose_clear_map = clear_map_cb.isChecked()
            g_c.update_rules(rules, colors, dose_clear_map)

        update_btn.clicked.connect(f_update)
        s_i_l = lambda: self.s_input_generations_linear.setStyleSheet(f"background-color: {self.RED};")
        self.s_input_generations_linear.textChanged.connect(s_i_l)
        b_i_l = lambda: self.b_input_generations_linear.setStyleSheet(f"background-color: {self.RED};")
        self.b_input_generations_linear.textChanged.connect(b_i_l)
        c_i_l = lambda: self.c_input_generations_linear.setStyleSheet(f"background-color: {self.RED};")
        self.c_input_generations_linear.textChanged.connect(c_i_l)


    def set_event_on_inputs_generations_standart(self, g_c, update_btn, clear_map_cb):
        def f_update():
            self.s_input_generations_standart.setStyleSheet(f"background-color: {self.GREEN};")
            self.b_input_generations_standart.setStyleSheet(f"background-color: {self.GREEN};")
            self.c_input_generations_standart.setStyleSheet(f"background-color: {self.GREEN};")
            rules = self.get_rules_generations_standart()
            pallet = g_c.CELL_ENGINES[g_c.GENERATIONS_STANDART]['colors_pallet']
            colors = get_colors_by_rules_generations(rules, *pallet)
            dose_clear_map = clear_map_cb.isChecked()
            g_c.update_rules(rules, colors, dose_clear_map)

        update_btn.clicked.connect(f_update)
        s_i_l = lambda: self.s_input_generations_standart.setStyleSheet(f"background-color: {self.RED};")
        self.s_input_generations_standart.textChanged.connect(s_i_l)
        b_i_l = lambda: self.b_input_generations_standart.setStyleSheet(f"background-color: {self.RED};")
        self.b_input_generations_standart.textChanged.connect(b_i_l)
        c_i_l = lambda: self.c_input_generations_standart.setStyleSheet(f"background-color: {self.RED};")
        self.c_input_generations_standart.textChanged.connect(c_i_l)


    def set_event_on_inputs_crystal(self, g_c, update_btn, clear_map_cb):
        def f_update():
            self.input_crystal.setStyleSheet(f"background-color: {self.GREEN};")
            rules = self.get_rules_crystal()
            colors = g_c.CELL_ENGINES[g_c.CRYSTAL]['colors_pallet']
            dose_clear_map = clear_map_cb.isChecked()
            g_c.update_rules(rules, colors, dose_clear_map)

        update_btn.clicked.connect(f_update)
        s_i_l = lambda: self.input_crystal.setStyleSheet(f"background-color: {self.RED};")
        self.input_crystal.textChanged.connect(s_i_l)



class GameController(object):
    GENERATIONS_LINEAR = 'generations_linear'
    GENERATIONS_STANDART = 'generations_standart'
    CRYSTAL = 'crystal'


    def __init__(self, size_ce, game_widget: GameAreaWidget, ui: GameControllerUi):
        self.__init_cell_engines_config()

        self.default_size_ce = size_ce
        self.game_widget = game_widget
        self.ui_controller = ui
        self.generations_linear_ce = self.get_cell_engine_by_type(self.GENERATIONS_LINEAR, self.default_size_ce)
        self.generations_standart_ce = self.get_cell_engine_by_type(self.GENERATIONS_STANDART, self.default_size_ce)
        self.crystal_ce = self.get_cell_engine_by_type(self.CRYSTAL, self.default_size_ce)
        self.current_ce = self.generations_linear_ce
        self.current_ce_type = self.GENERATIONS_LINEAR


    def __init_cell_engines_config(self):
        self.CELL_ENGINES = {
            'generations_linear': {
                'id': 0,
                'default_rules': ([2, 3], [2], 9),
                'colors_pallet': (c1, c2, c3),
                'default_colors': get_colors_by_rules_generations(([2, 3], [2], 9), c1, c2, c3),
                'cell_engine_class': ce.CellEngine_GenerationsLinear,
                'set_function': self.set_generations_linear_ce,
                'saving_rules_func': None,
            },
            'generations_standart': {
                'id': 1,
                'default_rules': ([2, 3], [2], 9),
                'colors_pallet': ((255, 255, 255), c2, c3),
                'default_colors': get_colors_by_rules_generations(([2, 3], [2], 9), (255, 255, 255), c2, c3),
                'cell_engine_class': ce.CellEngine_GenerationsStandart,
                'set_function': self.set_generations_standart_ce,
                'saving_rules_func': None,
            },
            'crystal': {
                'id': 2,
                'default_rules': [0, 1, 0, 0, 0,  1, 1, 1, 1, 0],
                'colors_pallet': ((255, 255, 255), ),
                'default_colors': [(255, 255, 255)],
                'cell_engine_class': ce.CellEngine_Crystal,
                'set_function': self.set_crystal_ce,
                'saving_rules_func': None,
            },
        }

    def saving_rules_generations_linear(self, rules):
        pass


    def get_ce_type_by_id(self, ce_id):
        for i in self.CELL_ENGINES:
            if self.CELL_ENGINES[i]['id'] == ce_id:
                return i

        return ''


    def get_id_by_ce_type(self, ce_type):
        return self.CELL_ENGINES[ce_type]['id']


    def set_generations_linear_ce(self):
        self.current_ce = self.generations_linear_ce
        self.current_ce_type = self.GENERATIONS_LINEAR
        self.update_current_ce_in_game()
        self.ui_controller.set_generations_linear_ui_from_rules(self.current_ce.rules)


    def set_generations_standart_ce(self):
        self.current_ce = self.generations_standart_ce
        self.current_ce_type = self.GENERATIONS_STANDART
        self.update_current_ce_in_game()
        self.ui_controller.set_generations_standart_ui_from_rules(self.current_ce.rules)


    def set_crystal_ce(self):
        self.current_ce = self.crystal_ce
        self.current_ce_type = self.CRYSTAL
        self.update_current_ce_in_game()
        self.ui_controller.set_crystal_ui_from_rules(self.current_ce.rules)


    def set_ce_by_key(self, key):
        self.CELL_ENGINES[key]['set_function']()


    def get_current_ce(self):
        return self.current_ce


    def get_current_ce_type(self):
        return self.current_ce_type


    def update_current_ce_in_game(self):
        self.game_widget.set_iters(0)
        self.game_widget.stop_timer()
        self.game_widget.set_cell_engine(self.get_current_ce())


    # @abstractmethod
    def get_cell_engine_by_type(self, ce_type, size):
        ce_rules = self.CELL_ENGINES[ce_type]['default_rules']
        ce_colors = self.CELL_ENGINES[ce_type]['default_colors']
        ce_class = self.CELL_ENGINES[ce_type]['cell_engine_class']

        return ce_class(size, ce_rules, ce_colors)


    def save_game(self, fname):
        def num_to_color(num):
            c1 = num % 256
            num //= 256
            c2 = num % 256
            num //= 256
            c3 = num % 256
            return (c3, c2, c1)

        sw = QProgressDialog("Saving animation...", "Abort save", 0, 100)
        sw.open()

        pixmap = self.game_widget.get_pixmap()
        colors = self.game_widget.cell_engine.get_colors()
        add_rows = math.ceil(len(colors) / pixmap.shape[1])
        config_data = np.zeros((1 + add_rows, pixmap.shape[1], 3), dtype=np.uint8)
        h, w, _ = pixmap.shape

        sw.setValue(40)

        config_data[0][0] = num_to_color(h)
        config_data[0][1] = num_to_color(w)
        config_data[0][2] = num_to_color(self.game_widget.iters)
        config_data[0][3] = num_to_color(len(colors))
        config_data[0][4] = num_to_color(self.get_id_by_ce_type(self.get_current_ce_type()))
        for i in range(len(colors)):
            id_color_i = int(i // pixmap.shape[1])
            id_color_j = int(i % pixmap.shape[1])
            config_data[id_color_i + 1][id_color_j] = colors[i]

        sw.setValue(60)

        res_arr = np.vstack((config_data, pixmap))
        im = Image.fromarray(res_arr)
        im.save(fname)

        sw.setValue(100)
        sw.close()


    def pixmap_to_game_map(self, pixmap, colors):
        game_map = np.zeros((pixmap.shape[0], pixmap.shape[1]), dtype = np.int0)
        colors_e = lambda c1, c2: (c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2])
        for i in range(pixmap.shape[0]):
            for j in range(pixmap.shape[1]):
                for c_i in range(len(colors)):
                    # print(colors[c_i], pixmap[i][j])
                    if colors_e(colors[c_i], pixmap[i][j]):
                        game_map[i][j] = c_i
                        break

        return game_map


    def open_game(self, fname):
        def color_to_num(color):
            num = color[2]
            num += color[1] * 256
            num += color[0] * 256 * 256
            return num

        image = Image.open(fname)
        data = np.asarray(image)
        # im = Image.fromarray(data)
        # im.save('asas.bmp')
        h, w = color_to_num(data[0][0]), color_to_num(data[0][1])
        iters = color_to_num(data[0][2])
        colors_len = color_to_num(data[0][3])
        ce_id = color_to_num(data[0][4])
        ce_type = self.get_ce_type_by_id(ce_id)
        add_rows = math.ceil(colors_len / w)
        colors = list()
        print(h, w, data[0][0], data[0][1])
        print(iters, data[0][2])
        print(colors_len, data[0][3])
        print(ce_id, data[0][4])
        for i in range(colors_len):
            id_color_i = int(i // w)
            id_color_j = int(i % w)
            # print(i, id_color_i, id_color_j)
            # print(data[id_color_i + 1][id_color_j])
            temp_color = data[id_color_i + 1][id_color_j]
            colors.append(temp_color)

        ext_colors = [(0, 0, 0)]
        ext_colors.extend(colors)
        colors = np.array(colors)

        self.game_widget.stop_timer()
        self.set_ce_by_key(ce_type)
        self.game_widget.set_iters(iters)
        self.game_widget.cell_engine.colors = colors
        gm = self.pixmap_to_game_map(data[add_rows + 1:], ext_colors)
        # print(gm.shape)
        self.game_widget.cell_engine.game_map = gm

        # self.game_widget.cell_engine.rules = rules

    def save_animation(self, fname, iters=1000, step=50, duration=1000):
        sw = QProgressDialog("Saving animation...", "Abort save", 0, 1000)
        sw.open()

        self.game_widget.stop_timer()

        pixmap = self.game_widget.get_pixmap()
        render_images = [Image.fromarray(pixmap)]
        for i in range(iters):
            sw.setValue(i)
            if sw.wasCanceled():
                return
            if (i + 1) % step == 0:
                pixmap = self.game_widget.get_pixmap()
                im = Image.fromarray(pixmap)
                render_images.append(im)

            self.game_widget.update_game()


        render_images[0].save(fname, save_all=True, append_images=render_images[1:], duration=duration, loop=0)
        sw.close()


    def set_3_in_center(self):
        self.game_widget.set_iters(0)
        self.game_widget.cell_engine.clear_map()
        w, h = self.game_widget.cell_engine.get_size()
        self.game_widget.cell_engine.set_pixel(w//2, h//2)
        self.game_widget.cell_engine.set_pixel(w//2, h//2-1)
        self.game_widget.cell_engine.set_pixel(w//2+1, h//2)
        self.game_widget.update()


    def set_square(self):
        self.game_widget.set_iters(0)
        self.game_widget.cell_engine.clear_map()
        w, h = self.game_widget.cell_engine.get_size()
        self.game_widget.cell_engine.set_pixel(w//2, h//2)
        self.game_widget.cell_engine.set_pixel(w//2, h//2-1)
        self.game_widget.cell_engine.set_pixel(w//2+1, h//2)
        self.game_widget.cell_engine.set_pixel(w//2+1, h//2-1)
        self.game_widget.update()


    def set_big_square(self):
        self.game_widget.set_iters(0)
        self.game_widget.cell_engine.clear_map()
        w, h = self.game_widget.cell_engine.get_size()
        h_2 = h//2
        w_2 = w//2
        h_4 = h//4
        w_4 = w//4
        for i in range(w_2):
            self.game_widget.cell_engine.set_pixel(w_4+i, h_4)
            self.game_widget.cell_engine.set_pixel(w_4+i, h_4+h_2)
        for i in range(h_2):
            self.game_widget.cell_engine.set_pixel(w_4, h_4+i)
            self.game_widget.cell_engine.set_pixel(w_4+w_2, h_4+i)
        self.game_widget.update()


    def set_center_point(self):
        self.game_widget.set_iters(0)
        self.game_widget.cell_engine.clear_map()
        w, h = self.game_widget.cell_engine.get_size()
        self.game_widget.cell_engine.set_pixel(w//2, h//2)
        self.game_widget.update()


    def set_random(self):
        self.game_widget.set_iters(0)
        self.game_widget.cell_engine.clear_map()
        w, h = self.game_widget.cell_engine.get_size()
        for i in range(h):
            for j in range(w):
                if random.randint(0, 10) < 4:
                    self.game_widget.cell_engine.set_pixel(j, i)

        self.game_widget.update()


    def clear(self):
        self.game_widget.set_iters(0)
        self.game_widget.cell_engine.clear_map()
        self.game_widget.update()


    def update_rules(self, rules, colors, clear_map=True):
        # print('update_rules: ', rules)
        self.game_widget.stop_timer()
        self.game_widget.cell_engine.rules = rules
        self.game_widget.cell_engine.colors = colors
        if clear_map:
            self.clear()
