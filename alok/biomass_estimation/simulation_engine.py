from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image, ImageDraw
import random
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
import uuid
sm = cm.ScalarMappable(cmap=cm.get_cmap('Reds'), norm=Normalize(vmin=0.3, vmax=3.0))


AVG_WEIGHT = 4.0
COV = 0.2
SPEED_FACTOR = 0.9
SPEED_FACTOR_STD = 0.05
MIN_DEPTH = 0.3
MAX_DEPTH = 3.0


def length_from_weight(weight):
    return (weight ** (1 / 3.0)) / 2.36068 * random.gauss(1.0, 0.05)


class Fish:

    def __init__(self, weight_mean, weight_std, speed_factor_mean, speed_factor_std,
                 min_depth, max_depth, max_y_coordinate=3.0, yaw_std=15):
        self.id = uuid.uuid4()
        self.weight = max(random.gauss(weight_mean, weight_std), 0.1)
        self.length = length_from_weight(self.weight)
        self.height = 0.3 * self.length
        self.depth = random.uniform(min_depth, max_depth)
        self.speed = self.length * random.gauss(speed_factor_mean, speed_factor_std)
        self.is_sampled = False
        self.position = [-10, self.depth, random.uniform(-max_y_coordinate, max_y_coordinate)]
        self.yaw = np.random.normal(0, yaw_std) * np.pi / 180.0

    def update_position(self, delta_t):
        delta_x = self.speed * delta_t * np.cos(self.yaw)
        delta_y = self.speed * delta_t * np.sin(self.yaw)
        self.position[0] += delta_x
        self.position[1] += delta_y

    def get_position(self):
        return self.position


class Camera:

    def __init__(self, position, fov_degrees, aspect_ratio=0.75):
        self.position = position
        self.fov = fov_degrees * np.pi / 180.0
        self.vfov = 2 * np.arctan(np.tan(self.fov / 2) * aspect_ratio)
        self.pixel_width = 1000
        self.pixel_height = int(self.pixel_width * aspect_ratio)
        self.focal_length_pixel = (self.pixel_width / 2) / np.tan(self.fov / 2)

    def contains(self, fish):
        # determine if fish is inside HFOV
        fish_position = fish.get_position()
        fish_segment_at_depth = (fish_position[0] - fish.length * np.cos(fish.yaw) / 2.0,
                                 fish_position[0] + fish.length * np.cos(fish.yaw) / 2.0)
        field_size = 2 * fish_position[1] * np.tan(self.fov / 2.0)
        field_center = self.position[0]
        field_segment_at_depth = (field_center - field_size / 2.0, field_center + field_size / 2.0)
        inside_horizontal_field = (fish_segment_at_depth[0] > field_segment_at_depth[0]) and \
                                  (fish_segment_at_depth[1] < field_segment_at_depth[1])

        # determine if fish is inside VFOV
        vertical_fish_segment_at_depth = (fish_position[2] - fish.height / 2.0, fish_position[2] + fish.height / 2.0)
        vertical_field_segment_at_depth = (-fish_position[1] * np.tan(self.vfov / 2.0), fish_position[1] * np.tan(self.vfov / 2.0))
        inside_vertical_field = (vertical_fish_segment_at_depth[0] >
                                 vertical_field_segment_at_depth[0]) and \
                                (vertical_fish_segment_at_depth[1] <
                                 vertical_field_segment_at_depth[1])

        return inside_horizontal_field and inside_vertical_field


def is_detected(fish, a=1.5, b=2.5, default_p=1.0):
    depth = fish.get_position()[1]
    if depth < a:
        p = default_p
    else:
        p = max(default_p * (b - depth) / (b - a), 0)

    return random.random() < p


def get_pixel_bbox(fish, camera):
    x_pixel = fish.position[0] * camera.focal_length_pixel / fish.position[
        1] + camera.pixel_width / 2.0
    y_pixel = -(fish.position[2] * camera.focal_length_pixel / fish.position[
        1]) + camera.pixel_height / 2.0
    length_pixel = fish.length * camera.focal_length_pixel / fish.position[1]
    height_pixel = fish.height * camera.focal_length_pixel / fish.position[1]
    bbox = [x_pixel - length_pixel / 2.0, y_pixel - height_pixel / 2.0,
            x_pixel + length_pixel / 2.0, y_pixel + height_pixel / 2.0]
    return [int(x) for x in bbox]


def get_ellipse_equation(bbox):
    center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    a = 0.5 * (bbox[2] - bbox[0])
    b = 0.5 * (bbox[3] - bbox[1])
    x = Symbol('x')
    y = Symbol('y')
    equation = (x - center[0])**2 / a**2 + (y - center[1]**2) / b**2 - 1
    return equation


def overlap(fish_1, fish_2, camera):
    bbox_1 = get_pixel_bbox(fish_1, camera)
    bbox_2 = get_pixel_bbox(fish_2, camera)
    eq1 = get_ellipse_equation(bbox_1)
    eq2 = get_ellipse_equation(bbox_2)
    solutions = solve([eq1, eq2], (Symbol('x'), Symbol('y')))
    for sol in solutions:
        if sol[0].is_real and sol[1].is_real:
            return True
    return False


def get_nonoccluded_fishes(fishes, left_camera, right_camera):
    fishes = sorted(fishes, key=lambda x: x.depth)
    occluded_fishes = []
    for i in range(len(fishes)):
        for j in range(i):
            is_left_occluded = overlap(fishes[i], fishes[j], left_camera)
            is_right_occluded = overlap(fishes[i], fishes[j], right_camera)
            if is_left_occluded or is_right_occluded:
                occluded_fishes.append(fishes[i])
                break

    nonoccluded_fishes = []
    for fish in fishes:
        if fish.id not in [fish.id for fish in occluded_fishes]:
            nonoccluded_fishes.append(fish)

    return nonoccluded_fishes


def draw_frame(fishes, left_camera):
    im = Image.new('RGB', (left_camera.pixel_width, left_camera.pixel_height))
    draw = ImageDraw.Draw(im)
    for fish in reversed(sorted(fishes, key=lambda x: x.depth)):
        bbox = get_pixel_bbox(fish, left_camera)
        color = sm.to_rgba(fish.depth, bytes=True)
        draw.ellipse(tuple(bbox), fill=color[:3])
        if fish.is_sampled:
            draw.rectangle(tuple(bbox), fill=(0, 0, 255, 100))
    return np.array(im)


def spawn_fish(fishes):
    fish = Fish(AVG_WEIGHT, AVG_WEIGHT * COV, SPEED_FACTOR, SPEED_FACTOR_STD, MIN_DEPTH, MAX_DEPTH)
    fishes.append(fish)


def move_fish(t, t_new, fishes):
    delta_t = t_new - t
    for fish in fishes:
        fish.update_position(delta_t)

    fishes = [fish for fish in fishes if fish.get_position()[0] < 10.0]
    return fishes


def trigger_capture(fishes, sampled_fishes, left_camera, right_camera):
    detected_fishes = []
    for fish in fishes:
        is_left_detected = left_camera.contains(fish) and is_detected(fish)
        is_right_detected = right_camera.contains(fish) and is_detected(fish)
        if is_left_detected and is_right_detected and not fish.is_sampled:
            detected_fishes.append(fish)

    nonoccluded_fishes = get_nonoccluded_fishes(detected_fishes, left_camera, right_camera)
    for fish in fishes:
        if fish.id in [fish.id for fish in nonoccluded_fishes]:
            sampled_fishes.append(fish)
            fish.is_sampled = True

    return fishes
