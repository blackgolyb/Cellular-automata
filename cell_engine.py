# import pygame
# from pygame.locals import *
import numba
from numba import njit#, prange
import random
import numpy as np
import time


def time_of_function(function):
    def wrapped(*args):
        start_time = time.perf_counter_ns()
        res = function(*args)
        # print(f'func_name: {function.__name__}', time.perf_counter_ns() - start_time)
        return res
    return wrapped
if __name__ == "__main__":
    pygame.init()

def square(surface, color, pos, dop_pos, length):
    pygame.draw.rect(screen, color,  list(map(lambda x: int(x), (pos[0]*length + dop_pos[0], pos[1]*length + dop_pos[1], length, length))) )

length = 1


#rule 235678/3468/9
S: tuple[int] = (2, 3, )#(2, 3, 5, 6, 7, 8, )
B: tuple[int] = (2, )#(3, 4, 6, 8, )
C: int = 8#9
c1: tuple[int] = (255, 255, 0)
c2: tuple[int] = (200, 0, 0)
c3: tuple[int] = (50, 0, 0)

rules = (S, B, C)

def get_color_gradient(
        start_color: tuple[int],
        end_color: tuple[int],
        iters: int) -> tuple[tuple[int]]:
    np_color1 = np.array(start_color)
    np_color2 = np.array(end_color)
    delta = (np_color2 - np_color1) / iters
    grad_arr = list()

    for i in range(iters):
        color_now = list(map(int, np_color1 + delta * i))
        grad_arr.append(color_now)

    # @njit
    # def grad(num):
    #     return grad_arr[num]

    return grad_arr

if __name__ == '__main__':
    temp_it = rules[2]//8
    base_grad1 = get_color_gradient(c1, c2, temp_it)
    base_grad2 = get_color_gradient(c2, c3, rules[2] - temp_it)
    base_grad = base_grad1[:]
    base_grad.extend(base_grad2)
    base_grad.reverse()
    base_grad = np.array(base_grad)
    print(base_grad)

    map_size: tuple[int] = (250, 300) # (w, h)
    if __name__ == "__main__":
        screen = pygame.display.set_mode((250, 300))

    game_map1: list[list[int]] = [
        [0 for j in range(map_size[0])] for i in range(map_size[1])
    ]

    game_map1: np.array = np.array(game_map1)

    game_map2: list[list[int]] = [
        [0 for j in range(map_size[0])] for i in range(map_size[1])
    ]

    game_map2: np.array = np.array(game_map2)

@njit
def count_neighbors(
        game_map: np.array,
        y: int, x: int,
        len_i: int, len_j:int,
        triger_point: int) -> int:
    result = 0
    s_i = -1 if y > 0 else 0
    e_i = 2 if y < len_i - 1 else 1
    s_j = -1 if x > 0 else 0
    e_j = 2 if x < len_j - 1 else 1
    for i in range(s_i, e_i):
        for j in range(s_j, e_j):
            if i == j == 0:
                continue
            if game_map[y + i][x + j] == triger_point:
                result += 1

    return result

# @time_of_function
@njit#(parallel=True)
def cell_game_engine_1_standart(
        game_map: np.array,
        rules: tuple[tuple[int] | int]) -> list[list[int]]:
    len_i = np.shape(game_map)[0]
    len_j = np.shape(game_map)[1]
    new_map = np.copy(game_map)
    c_n = lambda i, j: count_neighbors(game_map, i, j, len_i, len_j,  rules[2])
    # c_n(i, j)
    for i in range(len_i): # numba.prange
        for j in range(len_j):
            if game_map[i][j] == rules[2]:
                if c_n(i, j) not in rules[0]:
                    new_map[i][j] -= 1
            elif game_map[i][j]:
                new_map[i][j] -= 1
            elif not game_map[i][j]:
                if c_n(i, j) in rules[1]:
                    new_map[i][j] = rules[2]

    return new_map

@njit
def count_neighbors_crystal(
        game_map: np.array,
        y: int, x: int,
        len_i: int, len_j:int,) -> int:
    result = 0

    if y+1 < len_i:
        if game_map[y+1][x]:
            result += 1;
    if x+1 < len_j:
        if game_map[y][x+1]:
            result += 1;
    if y > 0:
        if game_map[y-1][x]:
            result += 1;
    if x > 0:
        if game_map[y][x-1]:
            result += 1;

    return result

# @njit#(paralle/l=True)
@njit
def cell_game_engine_crystal(
        game_map: np.array,
        rules: tuple[int]) -> list[list[int]]:
    len_i = np.shape(game_map)[0]
    len_j = np.shape(game_map)[1]
    new_map = np.copy(game_map)
    c_n = lambda i, j: count_neighbors_crystal(game_map, i, j, len_i, len_j)
    # c_n(i, j)
    for i in range(len_i): # numba.prange
        for j in range(len_j):
            neighbors_now = c_n(i, j)
            add_to = 5 if game_map[i][j] else 0
            new_map[i][j] = rules[add_to + neighbors_now]

    return new_map

@njit
def check_rules(game_map: np.array,
        i: int, j: int,
        rules: tuple[tuple[int] | int], neighbors: np.array): #  -> int
    if game_map[i][j] == rules[2]:
        if neighbors[j] not in rules[0]:
            game_map[i][j] -= 1
    elif game_map[i][j]:
        game_map[i][j] -= 1
    elif not game_map[i][j]:
        if neighbors[j] in rules[1]:
            game_map[i][j] = rules[2]


# @time_of_function
@njit
def cell_game_engine_1_liner(game_map: np.array,
        rules: tuple[tuple[int] | int]):
    len_i = np.shape(game_map)[0]
    len_j = np.shape(game_map)[1]
    prev_neighbors = None #[0 for i in range(len_j)]
    now_neighbors = np.zeros((len_j,), dtype=np.dtype('i4'))
    next_neighbors = np.zeros((len_j,), dtype=np.dtype('i4'))
    # c_n(i, j)

    # проверка первого элемента в первом ряду
    if game_map[0][0] == rules[2]:
        now_neighbors[1] += 1
        next_neighbors[0] += 1
        next_neighbors[1] += 1

    # проверка элементов первого ряда без первого и последнего
    for i in range(1, len_j - 1):
        if game_map[0][i] == rules[2]:
            now_neighbors[i-1] += 1
            now_neighbors[i+1] += 1
            next_neighbors[i-1] += 1
            next_neighbors[i] += 1
            next_neighbors[i+1] += 1

    # проверка поледнего элемента в первом ряду
    if game_map[0][-1] == rules[2]:
        now_neighbors[-2] += 1
        next_neighbors[-2] += 1
        next_neighbors[-1] += 1


    for i in range(1, len_i-1):
        prev_neighbors = now_neighbors
        now_neighbors = next_neighbors
        next_neighbors = np.zeros((len_j,), dtype=np.dtype('i4'))

        if game_map[i][0] == rules[2]:
            prev_neighbors[0] += 1
            prev_neighbors[1] += 1
            now_neighbors[1] += 1
            next_neighbors[0] += 1
            next_neighbors[1] += 1

        for j in range(1, len_j-1):
            if game_map[i][j] == rules[2]:
                prev_neighbors[j-1] += 1
                prev_neighbors[j] += 1
                prev_neighbors[j+1] += 1
                now_neighbors[j-1] += 1
                now_neighbors[j+1] += 1
                next_neighbors[j-1] += 1
                next_neighbors[j] += 1
                next_neighbors[j+1] += 1

            # game_map[i-1][j-1] =
            check_rules(game_map, i-1, j-1, rules, prev_neighbors)

        if game_map[i][-1] == rules[2]:
            prev_neighbors[-2] += 1
            prev_neighbors[-1] += 1
            now_neighbors[-2] += 1
            next_neighbors[-2] += 1
            next_neighbors[-1] += 1

        # game_map[i-1][-2] =
        check_rules(game_map, i-1, -2, rules, prev_neighbors)
        # game_map[i-1][-1] =
        check_rules(game_map, i-1, -1, rules, prev_neighbors)


    if game_map[-1][0] == rules[2]:
        now_neighbors[0] += 1
        now_neighbors[1] += 1
        next_neighbors[1] += 1

    for i in range(1, len_j-1):
        if game_map[-1][i] == rules[2]:
            now_neighbors[i-1] += 1
            now_neighbors[i] += 1
            now_neighbors[i+1] += 1
            next_neighbors[i-1] += 1
            next_neighbors[i+1] += 1

        # game_map[-2][i-1] =
        check_rules(game_map, -2, i-1, rules, now_neighbors)
        # game_map[-1][i-1] =
        check_rules(game_map, -1, i-1, rules, next_neighbors)

    if game_map[-1][-1] == rules[2]:
        now_neighbors[-2] += 1
        now_neighbors[-1] += 1
        next_neighbors[-2] += 1

    # game_map[-2][-2] =
    check_rules(game_map, -2, -2, rules, now_neighbors)
    # game_map[-2][-1] =
    check_rules(game_map, -2, -1, rules, now_neighbors)
    # game_map[-1][-2] =
    check_rules(game_map, -1, -2, rules, next_neighbors)
    # game_map[-1][-1] =
    check_rules(game_map, -1, -1, rules, next_neighbors)


@njit
def get_area(
        pos: list[float],
        size: list[int], map_size: list[int],
        zoom: int) -> list[int]:
    c_size_w = size[0] / zoom
    c_size_h = size[1] / zoom
    s_x = min(map_size[0]-1, max(0, int(pos[0])-1))
    s_y = min(map_size[1]-1, max(0, int(pos[1])-1))
    e_x = max(0, min(int(pos[0] + 1 + c_size_w), map_size[0]))
    e_y = max(0, min(int(pos[1] + 1 + c_size_w), map_size[1]))

    return [s_x, e_x, s_y, e_y]


@njit
def blit_game_on_array(pixmap, game_map, pos, window_size, zoom, color_grad):
    area = get_area(pos, np.shape(game_map), window_size, zoom)
    # print_map = game_map[area[2] : area[3]][area[0] : area[1]]
    cor_pos = [-pos[0] * zoom, -pos[1] * zoom]
    dpx = cor_pos[0] % zoom
    dpy = cor_pos[1] % zoom
    # pixmap[:, :area[0]] = np.array([0, 0, 255])
    for i in range(area[2], area[3]):
        for j in range(area[0], area[1]):
            if game_map[i][j]:
                s_y = min(window_size[1], max(int(i*zoom+cor_pos[1]), 0))
                s_x = min(window_size[0], max(int(j*zoom+cor_pos[0]), 0))
                e_y = max(0, min(int((i+1)*zoom+cor_pos[1]), window_size[1]))
                e_x = max(0, min(int((j+1)*zoom+cor_pos[0]), window_size[0]))
                pixmap[s_y:e_y, s_x:e_x] = color_grad[game_map[i][j] - 1]


@njit
def blit_game_on_array_crystal(pixmap, game_map, pos, window_size, zoom, color):
    area = get_area(pos, np.shape(game_map), window_size, zoom)
    color = np.array(color)
    cor_pos = [-pos[0] * zoom, -pos[1] * zoom]
    dpx = cor_pos[0] % zoom
    dpy = cor_pos[1] % zoom
    for i in range(area[2], area[3]):
        for j in range(area[0], area[1]):
            if game_map[i][j]:
                s_y = min(window_size[1], max(int(i*zoom+cor_pos[1]), 0))
                s_x = min(window_size[0], max(int(j*zoom+cor_pos[0]), 0))
                e_y = max(0, min(int((i+1)*zoom+cor_pos[1]), window_size[1]))
                e_x = max(0, min(int((j+1)*zoom+cor_pos[0]), window_size[0]))
                pixmap[s_y:e_y, s_x:e_x] = color


@njit
def count_zoom_position(
        pos: list[float], mpos: list[int],
        nzoom: float, lzoom: float) -> list[float]:
    zoom_prop = (nzoom - lzoom) / (lzoom * nzoom)
    dposx = mpos[0] * zoom_prop + pos[0]
    dposy = mpos[1] * zoom_prop + pos[1]
    res = [dposx, dposy]

    return res

def draw_map(game_map, pos, size):
    cor_pos = [-pos[0] * size, -pos[1] * size]
    for i in range(len(game_map)):
        for j in range(len(game_map[i])):
            if game_map[i][j]:
                # print(game_map[i][j] - 1)
                square(
                    screen, base_grad[game_map[i][j] - 1], (j, i), cor_pos, size
                )

class CellEngine(object):
    def __init__(self, size, rules, colors):
        self.game_map: np.array = np.zeros((size[1], size[0]), dtype=np.int0)
        self.rules = rules
        self.colors = colors

    def get_size(self):
        return [self.game_map.shape[1], self.game_map.shape[0]]

    def get_colors(self):
        return self.colors

    def blit_on_array(self, pixmap, position, zoom):
        blit_game_on_array(pixmap, self.game_map, position, (pixmap.shape[1], pixmap.shape[0]), zoom, self.colors)

    def clear_map(self):
        self.game_map = np.zeros(self.game_map.shape, dtype=np.int0)

    def game_step(self):
        pass

    def set_pixel(self, x, y):
        pass

    def remove_pixel(self, x, y):
        pass



class CellEngine_Blur(object):
    def __init__(self, size):
        self.size = size


    def get_size(self):
        return self.size


class CellEngine_Crystal(CellEngine):
    def set_pixel(self, x, y):
        self.game_map[y][x] = 1


    def remove_pixel(self, x, y):
        self.game_map[y][x] = 0


    def game_step(self):
        self.game_map = cell_game_engine_crystal(self.game_map, self.rules)

    def blit_on_array(self, pixmap, position, zoom):
        blit_game_on_array_crystal(pixmap, self.game_map, position, (pixmap.shape[1], pixmap.shape[0]), zoom, self.colors[0])



class CellEngine_Generations(CellEngine):
    def set_pixel(self, x, y):
        self.game_map[y][x] = self.rules[2]

    def remove_pixel(self, x, y):
        self.game_map[y][x] = 0



class CellEngine_GenerationsLinear(CellEngine_Generations):
    def game_step(self):
        cell_game_engine_1_liner(self.game_map, self.rules)



class CellEngine_GenerationsStandart(CellEngine_Generations):
    def game_step(self):
        self.game_map = cell_game_engine_1_standart(self.game_map, self.rules)

# game_map = gen_random_map(game_map)

#glider spawn for test
# game_map[100][100] = C - 1
# game_map[100-1][100] = C - 1
# game_map[100][100+1] = C - 1

# game_map[100][100] = C
# game_map[100-1][100] = C
# game_map[100][100+1] = C

# game_map1[100][100] = C
# game_map1[100-1][100] = C
# game_map1[100][100+1] = C
# for i in (50, 150):
#     game_map1[50][i] = C
# for i in (50, 150):
#     game_map1[150][i] = C
# for i in (50, 150):
#     game_map1[i][50] = C
# for i in (50, 150):
#     game_map1[i][150] = C

# game_map2[100][100] = C
# game_map2[100-1][100] = C
# game_map2[100][100+1] = C
# game_map[100+1][100+1] = C
# game_map[100][100] = 1

#game_map[50][50] = 1

# rules = [0, 0, 1, 0, 1, 1, 0, 1]

if __name__ == '__main__':
    zoom = 1
    position = [0, 0]
    print(K_PLUS, K_MINUS)
    print(K_UP, K_DOWN, K_RIGHT, K_LEFT)
    is_pause = False
    iters = 0
    run = True
    next_iter = False
    is_draw = False
    surf = pygame.Surface(map_size)
    while True:
        screen.fill((0, 0, 0))

        pygame.event.get()
        keys = pygame.key.get_pressed()
        mpos = pygame.mouse.get_pos()

        if keys[K_KP_PLUS]:
            new_zoom = zoom + 1
            position = count_zoom_position(position, mpos, new_zoom, zoom)
            zoom = new_zoom
        elif keys[K_KP_MINUS]:
            if zoom > 0:
                new_zoom = zoom - 1
                position = count_zoom_position(position, mpos, new_zoom, zoom)
                zoom = new_zoom

        if keys[K_UP]:
            position[1] -= 10
        elif keys[K_DOWN]:
            position[1] += 10

        if keys[K_RIGHT]:
            # is_pause = True
            position[0] += 10
        elif keys[K_LEFT]:
            position[0] -= 10

        if keys[K_SPACE]:
            print('pause')
            is_pause = not is_pause
            next_iter = True

        # cor_pos1 = [-position[0] * zoom, -position[1] * zoom]
        # cor_pos2 = [-position[0] * zoom + 200, -position[1] * zoom]
        # if is_draw:
        #     draw_map(game_map1, cor_pos1, zoom)
        #     draw_map(game_map2, cor_pos2, zoom)



        game_engine_liner(game_map1, rules)
        pixmap = np.full((map_size[0], map_size[1], 3), np.array((0, 0, 0)), dtype=np.dtype('i4'))
        blit_game_on_array(pixmap, game_map1, position, map_size, zoom, base_grad)
        # pixmap = np.transpose(pixmap, (1, 0, 2))
        # print(pixmap)
        surf.fill((0, 0, 0))
        pygame.surfarray.blit_array(surf, pixmap)
        screen.blit(surf, (0,0))



        # fast_cell_boy(game_map1, rules)
        # draw_map(game_map1, position, zoom)
        # print(1)
        # game_map1 = cell_engine_map_crystal(game_map1, rules)



        # pygame.pixelcopy.array_to_surface(screen, draw_arr)
        # print(np.shape(draw_arr))

        # if run or next_iter:
        #     game_map2 = update_map(game_map2, rules)
        #     next_iter = False
        # if iters == 101:
        #     run = False
        # print(iters)
        # if iters > 300:
        #     derep = 0
        #     is_draw = True
        #     hui = []
        #     for i in range(np.shape(game_map2)[0]):
        #         for j in range(np.shape(game_map2)[1]):
        #             if game_map2[i][j] != game_map1[i][j]:
        #                 run = False
        #
        #                 derep += 1
        #                 hui.append([i, j, game_map2[i][j], game_map1[i][j]])
        #     print(iters, derep, hui)
        #
        # iters += 1
        # if is_pause:
        #     # print('not paused')
        #     # game_map = update_map(game_map, rules)
        #     fast_cell_boy(game_map, rules)
        #     is_pause = False
        # print(1)
        # neighbors_map = check_map(game_map)
        # game_map = set_item_in_map(game_map, neighbors_map)

        pygame.display.update()
        pygame.time.delay(50)
