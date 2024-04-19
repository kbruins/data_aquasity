import nidaqmx
import nidaqmx.system
from math import ceil, pi, sqrt
import pygame
from pygui.elements import *
from pygui.functions import *
from pygui.elements import K_ALIGN_CENTER, K_ALIGN_LEFT, K_ALIGN_RIGHT, K_ALIGN_TOP, K_ALIGN_BOTTOM, K_TOP_LEFT, K_TOP_RIGHT, K_BOTTOM_LEFT, K_BOTTOM_RIGHT
from pygui import events
from collections.abc import Sequence
import math
import numpy as np
import scipy

ndarray = np.ndarray
pygame.init()
TEXTURES = "assets"
FPS_TARGET = 20  # fps
SAMPLES_TARGET = 50
MS_PER_FRAME = 1000.0/FPS_TARGET
MS_PER_SAMPLE = 1000.0/SAMPLES_TARGET
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (27, 0, 91)
LIGHT_BLUE = (82, 237, 255)
PI2 = pi/2
RAD_TO_DEG = 90/PI2
BUTTON_FONT = pygame.font.SysFont("Arial", 30)
DEFAULT_FONT = pygame.font.SysFont("Arial", 30)
UPDATE_SCREEN = pygame.event.custom_type()
SAMPLE_DATA = pygame.event.custom_type()
clamp = pygame.math.clamp
clock = pygame.time.Clock()


class Object3D:
    def __init__(self, vertices: list[pygame.Vector3], faces: list[Sequence[int]], normals: list[pygame.Vector3],
                 position: Sequence[int | float] | pygame.Vector3 = (0, 0, 0),
                 orientation: Sequence[int | float] = (0, 0, 0)):
        self.vertices = vertices
        self.faces_index = faces
        self.faces: list[list[pygame.Vector3]] = []
        self.normals = normals
        self.position = pygame.Vector3(position)
        self.orientation = list(orientation)  # x, y, z euler angles
        # stores data for the original position
        self.positions: tuple[tuple[float, float, float], ...] = tuple([tuple(vertex) for vertex in vertices])
        self.index_faces()

    def index_faces(self):
        self.faces.clear()
        for face in self.faces_index:
            self.faces.append([self.vertices[i] for i in face])

    def update_positions(self):
        self.positions = tuple([tuple(vertex) for vertex in self.vertices])

    def reset_vertices(self):
        for i, vertex in enumerate(self.vertices):
            vertex.update(self.positions[i])


class Object2D:
    def __init__(self, vertices: list[pygame.Vector2], faces: list[Sequence[int]],
                 position: Sequence[int | float] | pygame.Vector2 = (0, 0), orientation: int | float = 0):
        self.vertices = vertices
        self.faces_index = faces
        self.faces: list[list[pygame.Vector2]] = []
        self.position = pygame.Vector2(position)
        self.orientation = orientation  # degrees
        # stores data for the original position
        self.positions: tuple[tuple[float, float], ...] = tuple([tuple(vertex) for vertex in vertices])
        self.index_faces()

    def index_faces(self):
        self.faces.clear()
        for face in self.faces_index:
            self.faces.append([self.vertices[i] for i in face])

    def update_positions(self):
        self.positions = tuple([tuple(vertex) for vertex in self.vertices])

    def reset_vertices(self):
        for i, vertex in enumerate(self.vertices):
            vertex.update(self.positions[i])


class MyDaq:
    def __init__(self, name: str = "Dev1"):
        self.name = name
        self.task: nidaqmx.Task = nidaqmx.Task()
        self.connected = False
        #self.data_array = np.empty(3, "float")
        #self.stream = nidaqmx.stream_readers.AnalogMultiChannelReader(self.task)
        self.data = [0.0, 0.0, 0.0]

    def init_task(self):
        """start the task"""
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(f"{self.name}/ao0", min_val=0.0, max_val=3.3)
                task.write(3.3)
            self.task.ai_channels.add_ai_voltage_chan(f"{self.name}/ai0")
            self.task.ai_channels.add_ai_voltage_chan(f"{self.name}/ai1")
            self.task.ai_channels.add_ai_voltage_chan(f"{self.name}/ai2")
            self.connected = True

        except nidaqmx.DaqError:
            self.connected = False

    def close_taks(self):
        self.task.close()
        self.connected = False

    def read_data(self):
        """read the data of the task. initializing if it's not initialized"""
        if (not self.connected) and (self.name in nidaqmx.system.System.local().devices):
            self.init_task()

        try:
            self.data = self.task.read()
            #self.stream.read_one_sample(self.data_array)
            #self.data = list(self.data_array)

            self.connected = True
        except nidaqmx.DaqError:
            self.connected = False


class Pendulum:
    def __init__(self, mydaq: MyDaq):
        self.mydaq: MyDaq = mydaq

        self.raw_pod: float = 0.0
        self.raw_accelerometer: pygame.Vector2 = pygame.Vector2(0.0, 0.0)
        self.normalized_accelerometer: pygame.Vector2 = pygame.Vector2(0.0, 0.0)  # gravitational space
        self.angle_pod: float = 0.0
        self.angle_accelerometer: float = 0.0

        self.buffer: list[list[float]] = [[], [], []]
        self.sampling_buffer: list[list[float]] = [[], [], []]
        self.data_samples: dict[int, list[tuple[float, float, float]]] = {}
        self.buffer_size: int = 0
        self.sample_size: int = 50
        self.sampling: int | None = None
        self.set_buffer_size(10)

        self.pod_start: float = 0.0
        self.pod_slope: float = 1.0
        self.acc_x_start = 0.0
        self.acc_x_slope = 1.0
        self.acc_y_start = 0.0
        self.acc_y_slope = 1.0

    def acquire(self):
        """get a single sample from all sensors"""
        self.mydaq.read_data()
        if not self.mydaq.connected:
            return

        if self.sampling is not None:  # add data if sampling
            for i, data in enumerate(self.mydaq.data):
                self.sampling_buffer[i].append(data)
            if len(self.sampling_buffer[0]) >= self.sample_size:
                self.finish_sampling()

        for i, buffer in enumerate(self.buffer):  # shift data and insert new data
            buffer[1:] = buffer[:-1]
            buffer[0] = self.mydaq.data[i]

        # update the raw values by taking the average of the current buffer.
        self.raw_pod = sum(self.buffer[0]) / len(self.buffer[0])
        self.raw_accelerometer.x = sum(self.buffer[1]) / len(self.buffer[1])
        self.raw_accelerometer.y = sum(self.buffer[2]) / len(self.buffer[2])

    def set_buffer_size(self, buffer_size: int):
        """set the size of the buffer used for smoothing the real time values"""
        if buffer_size < 1:
            raise ValueError("buffer size cannot be less then 1")
        self.buffer_size = buffer_size
        for i in range(3):
            self.buffer[i] = [0.0]*buffer_size

    def start_sampling(self, angle: int):
        """start sampling at the given angle"""
        self.sampling = angle
        self.sampling_buffer[0].clear()
        self.sampling_buffer[1].clear()
        self.sampling_buffer[2].clear()

    def finish_sampling(self):
        """called when sampling is complete.
        adds the sample buffer data to the overall data samples and updates the regression."""
        if self.sampling not in self.data_samples:  # create a new entry if it does not exist yet
            self.data_samples[self.sampling] = []
        for i in range(len(self.sampling_buffer[0])):  # add new samples to the overall samples
            self.data_samples[self.sampling].append((self.sampling_buffer[0][i], self.sampling_buffer[1][i], self.sampling_buffer[2][i]))
        self.update_regression()
        self.sampling = None

    def update_regression(self):
        if len(self.data_samples.keys()) < 2:
            return

        pod_array, acc_x_array, acc_y_array = self.get_sample_arrays()
        self.pod_slope, self.pod_start, _, _, _ = scipy.stats.linregress(pod_array)
        self.acc_x_slope, self.acc_x_start, _, _, _ = scipy.stats.linregress(acc_x_array)
        self.acc_y_slope, self.acc_y_start, _, _, _ = scipy.stats.linregress(acc_y_array)

        print(f"raw_pod: angle = V*{self.pod_slope} + {self.pod_start}")
        print(f"acc: x = V*{self.acc_x_slope} + {self.acc_x_start}")
        print(f"acc: y = V*{self.acc_y_slope} + {self.acc_y_start}")

    def get_sample_arrays(self) -> tuple[ndarray, ndarray, ndarray]:
        """converts the current data samples to a set of arrays used for calculating regression."""
        pod_samples: list[tuple[float, float]] = []
        acc_x_samples: list[tuple[float, float]] = []
        acc_y_samples: list[tuple[float, float]] = []
        for key, values in self.data_samples.items():
            key_x = math.cos(math.radians(key))
            key_y = -math.sin(math.radians(key))
            for pod, acc_x, acc_y in values:
                pod_samples.append((pod, float(key)))
                acc_x_samples.append((acc_x, float(key_x)))
                acc_y_samples.append((acc_y, float(key_y)))

        return np.asarray(pod_samples), np.asarray(acc_x_samples), np.asarray(acc_y_samples)

    def pod_angle(self):
        """update the current potentiometers angle in degrees."""
        self.angle_pod = self.raw_pod * self.pod_slope + self.pod_start

    def accelerometer_angle(self):
        """updates the gravitaional, force and angle values of the accelerometer."""
        normal_x = self.raw_accelerometer.x * self.acc_x_slope + self.acc_x_start
        normal_y = self.raw_accelerometer.y * self.acc_y_slope + self.acc_y_start
        self.normalized_accelerometer.update(normal_x, normal_y)
        self.angle_accelerometer = self.normalized_accelerometer.angle_to((1, 0))


class Renderer:
    def __init__(self, screen_size: Sequence[int | float, int | float], fov=90.0, fov_axis_horizontal=True,
                 perspective=True, light=pygame.Vector3(0.4, -0.4, 0.8), colorkey=(255, 255, 255)):
        # camera looks from the top down. coordinate system = z up right-handed (like blender)
        # rotation = YXZ Eular (roll, pitch, yaw)
        self.fov = fov
        self.fov_axis = fov_axis_horizontal  # true when horizontal
        self.perspective = perspective
        self.light_vector = light.normalize()
        self.colorkey = colorkey
        self.view_size = (0.0, 0.0)
        self.rect = pygame.Rect((0, 0), screen_size)
        self.angle_to_pixel = (0, 0)
        self.update_camera(screen_size, fov, fov_axis_horizontal)

    def update_camera(self, viewport_size: Sequence[int | float, int | float] | None = None, fov: float | None = None,
                      fov_axis: bool | None = None):
        if viewport_size is not None:
            self.rect.size = tuple(viewport_size)
        if fov is not None:
            self.fov = fov
        if fov_axis is not None:
            self.fov_axis = fov_axis

        if self.fov_axis:
            width = math.tan(self.fov/RAD_TO_DEG/2)
            height = width*self.rect.height/self.rect.width
        else:
            height = math.tan(self.fov/RAD_TO_DEG/2)
            width = height*self.rect.width/self.rect.height
        self.view_size = (width, height)
        self.angle_to_pixel = (self.rect.centerx / self.view_size[0], self.rect.centery / self.view_size[1])

    def pixel_shader(self, normal_vector: pygame.math.Vector3, roughness: float, rect: pygame.rect.Rect,
                     surface: pygame.surface.Surface, mask: pygame.mask.Mask, normals: Sequence[pygame.math.Vector3],
                     pixel_positions: Sequence[pygame.math.Vector2]):
        """shades the light per pixel for a more realistic appearance.
        want to lower your fps? great! then this is the function for you"""
        surfarray = pygame.surfarray.array3d(surface)
        spec_array = np.empty(surfarray.shape[0:2], dtype='f4')
        smoothness2 = pow(roughness, 2)
        nom = smoothness2-1
        for y in range(rect.height):
            for x in range(rect.width):
                if mask.get_at((x, y)):

                    distances = np.asarray([pixel_pos.distance_to((x+rect.left, y+rect.top))
                                            for pixel_pos in pixel_positions])
                    total_distance = distances.sum()
                    distances /= total_distance
                    np.subtract(1, distances, out=distances)

                    camera_vector = pygame.math.Vector3(0.0, 0.0, 0.0)
                    for i, normal in enumerate(normals):
                        camera_vector += normal*distances[i]

                    halfway_vec = (self.light_vector+camera_vector.normalize()).normalize()
                    spec_array[x, y] = max((normal_vector*-1).dot(halfway_vec), 0.0)
                else:
                    spec_array[x, y] = 0.0

        np.power(spec_array, 2.0, spec_array)
        spec_array *= nom
        spec_array += 1
        np.power(spec_array, 2.0, spec_array)
        spec_array *= pi
        np.divide(smoothness2, spec_array, spec_array)
        spec_array *= 254.0
        spec_array.clip(0.0, 254.0, spec_array)
        spec_array = np.where(pygame.surfarray.array_colorkey(surface) != 0, spec_array, 255.0)
        np.copyto(surfarray, np.expand_dims(spec_array, 2), casting="unsafe")
        return pygame.surfarray.make_surface(surfarray)

    def get_lighting(self, normal: pygame.math.Vector3, roughness: float = 0.5):
        smoothness2 = pow(roughness, 2)
        halfway_vec = (self.light_vector+pygame.math.Vector3(0.0, 0.0, 1.0)).normalize()

        specular = max((normal*-1).dot(halfway_vec), 0.0)
        specular = pow(specular, 2) * (smoothness2-1) + 1
        specular = pow(specular, 2) * pi

        return smoothness2 / specular

    @staticmethod
    def vertex_shader(origen: pygame.math.Vector3, orientation: Sequence[float, float, float],
                      vertices: Sequence[pygame.math.Vector3], normal: pygame.math.Vector3 | None = None):
        for vertex in vertices:
            vertex.rotate_y_ip(orientation[1])
            vertex.rotate_x_ip(orientation[0])
            vertex.rotate_z_ip(orientation[2])
            vertex += origen

        if normal is not None:
            normal.rotate_y_ip(orientation[1])
            normal.rotate_x_ip(orientation[0])
            normal.rotate_z_ip(orientation[2])
            return vertices, normal

        return vertices

    def get_pixel_positions(self, model: Object3D, offset=(0, 0)):
        pitch, roll, yaw = model.orientation
        vertices = model.vertices
        origen = model.position
        offset_x, offset_y = offset

        for vertex in vertices:
            vertex.rotate_y_ip(roll)
            vertex.rotate_x_ip(pitch)
            vertex.rotate_z_ip(yaw)
            vertex += origen
            # angle from the camera to the vertex
            angle_x = vertex.xz.angle_to((0.0, 1.0))
            angle_y = vertex.yz.angle_to((0.0, 1.0))
            # pixel position on the screen of the vertex. remember. pygames y on screen is inverted
            vertex.x = self.rect.centerx + (math.tan(angle_x/RAD_TO_DEG) * self.angle_to_pixel[0]) - offset_x
            vertex.y = self.rect.centery - (math.tan(angle_y/RAD_TO_DEG) * self.angle_to_pixel[1]) - offset_y

    def draw_object(self, model: Object3D, color: pygame.color.Color, backface_culling=True, vertex_shading=False,
                    roughness=0.5, surface: pygame.Surface | None = None, offset=(0, 0)) -> pygame.Surface:
        normals = [vec.copy() for vec in model.normals]
        pitch, roll, yaw = model.orientation
        vertices = model.vertices
        faces = model.faces
        origen = model.position
        offset_x, offset_y = offset

        if surface is None:
            surface = colored_rect(self.colorkey, self.rect.size)
        for vertex in vertices:
            vertex.rotate_y_ip(roll)
            vertex.rotate_x_ip(pitch)
            vertex.rotate_z_ip(yaw)
            vertex += origen
        face_pos = [vec[0].copy() for vec in faces]
        for vertex in vertices:
            # angle from the camera to the vertex
            angle_x = vertex.xz.angle_to((0.0, 1.0))
            angle_y = vertex.yz.angle_to((0.0, 1.0))
            # pixel position on the screen of the vertex. remember. pygames y on screen is inverted
            vertex.x = self.rect.centerx + (math.tan(angle_x/RAD_TO_DEG) * self.angle_to_pixel[0]) - offset_x
            vertex.y = self.rect.centery - (math.tan(angle_y/RAD_TO_DEG) * self.angle_to_pixel[1]) - offset_y

        for i, normal in enumerate(normals):
            normal.rotate_y_ip(roll)
            normal.rotate_x_ip(pitch)
            normal.rotate_z_ip(yaw)
            if normal.angle_to(face_pos[i]) < 90 and backface_culling:  # facing away from camera. skip rendering
                continue
            if vertex_shading:
                color_mult = self.get_lighting(normals[i], roughness)
                pygame.draw.polygon(surface, [int(clamp(channel*color_mult, 0, 255)) for channel in color],
                                    [vertex.xy for vertex in faces[i]])
            else:
                pygame.draw.polygon(surface, color, [vertex.xy for vertex in faces[i]])

        return surface

    def draw_2d(self, model: Object2D, color: pygame.Color, surface: pygame.Surface | None = None) -> pygame.Surface:

        yaw = model.orientation
        vertices = model.vertices
        faces = model.faces
        origen = model.position

        if surface is None:
            surface = colored_rect(self.colorkey, self.rect.size)
        for vertex in vertices:
            vertex.rotate_ip(yaw)
            vertex += origen

        for face in faces:
            pygame.draw.polygon(surface, color, face)

        return surface

    def draw_ngon(self, origen: pygame.math.Vector3, orientation: Sequence[float, float, float],
                  points: list[pygame.math.Vector3], surface: pygame.Surface | None = None, color=BLACK,
                  normal_vector: Sequence[float, float, float] | pygame.Vector3 = (0.0, 0.0, -1.0), roughness=0.5,
                  vertex_shading=False, pixel_shader=False, offset: tuple[int, int] = (0, 0)):

        normal = pygame.math.Vector3(normal_vector)
        vertices = points.copy()
        offset_x, offset_y = offset

        if surface is None:
            surface = colored_rect(self.colorkey, self.rect.size)
        for vertex in vertices:
            vertex.rotate_y_ip(orientation[1])
            vertex.rotate_x_ip(orientation[0])
            vertex.rotate_z_ip(orientation[2])
            vertex += origen
            # angle from the camera to the object
            angle_x = vertex.xz.angle_to((0.0, 1.0))
            angle_y = vertex.yz.angle_to((0.0, 1.0))
            # pixel position on the screen of the vertex
            pixel_x = round(math.tan(angle_x/RAD_TO_DEG) / self.view_size[0] * self.rect.centerx + self.rect.centerx)
            pixel_y = round(math.tan(angle_y/RAD_TO_DEG) / self.view_size[1] * self.rect.centery + self.rect.centery)
            # noinspection PyTypeChecker
            vertex.update(pixel_x-offset_x, pixel_y-offset_y, vertex.z)  # change to a 2d screen position. keeps depth.

        normal.rotate_y_ip(orientation[1])
        normal.rotate_x_ip(orientation[0])
        normal.rotate_z_ip(orientation[2])

        if vertex_shading:
            color_mult = self.get_lighting(normal, roughness)
            pygame.draw.polygon(surface, [int(clamp(channel*color_mult, 0, 255)) for channel in color],
                                [vertex.xy for vertex in points])
        else:
            pygame.draw.polygon(surface, color, [vertex.xy for vertex in points])
            if pixel_shader:
                pass
                # todo: fix this
                # mask = pygame.mask.from_surface(surface)
                # apply shader

        return surface

    def draw_square(self, origen: pygame.math.Vector3, orientation: Sequence[float, float, float],
                    size: Sequence[float, float], color=BLACK, smoothness=0.5):
        # corner vertices
        bottem_right = pygame.math.Vector3(size[0]*0.5, size[1]*0.5, 0.0)
        top_left = pygame.math.Vector3(-bottem_right.x, -bottem_right.y, 0.0)
        top_right = pygame.math.Vector3(bottem_right.x, top_left.y, 0.0)
        bottem_left = pygame.math.Vector3(top_left.x, bottem_right.y, 0.0)

        return self.draw_ngon(origen, orientation, [top_left, top_right, bottem_right, bottem_left], color,
                              False, roughness=smoothness)

    def draw_cylinder(self, radius: float, length: float, origen: pygame.math.Vector3,
                      orientation: Sequence[float, float, float], smoothness=0.5):
        pass


class Menu:
    def __init__(self, gui: GUI, background: GUISprite):
        self.gui = gui
        self.background_sprite = background

    def center_background(self):
        self.background_sprite.viewport_to_pixels((0.5, 0.5), self.gui.image.get_size())
        self.gui.add_objects(sprites=(self.background_sprite, ))
        self.gui.bake_background()

    def minimize(self):
        self.remove_outline()
        pygame.display.iconify()

    def basic_button(self, pos: tuple[int | float, int | float], text: str, action, use_viewport=True,
                     alignment=K_ALIGN_CENTER, surface: pygame.Surface | None = None,
                     color=DARK_BLUE, font=BUTTON_FONT, priority=15, name="Button"):
        if surface is None:
            surface = get_img("button", TEXTURES)
        return Button(pos, center_text(text, surface, font, color), action, priority, name, use_viewport, alignment,
                      wip, self.show_outline, self.remove_outline)

    def show_outline(self):
        active = self.gui.get_focus()
        rect = active.get_global_rect()
        rect.inflate_ip(8, 8)
        surface = colored_rect(WHITE, rect.size, True).convert_alpha()

        # put the corner in the left corner and then size/rotate 2 lines to get a quarter image
        outline_part = get_img("outline_corner", TEXTURES)
        surface.blit(outline_part, (0, 0))
        surface.blit(pygame.transform.flip(outline_part, True, False), (rect.w-16, 0))
        outline_part = get_img("outline_line", TEXTURES)
        surface.blit(pygame.transform.scale(outline_part, (rect.w-32, 4)), (16, 0))
        surface.blit(pygame.transform.scale(pygame.transform.rotate(outline_part, 90), (4, rect.h-32)), (0, 16))

        surface.blit(pygame.transform.flip(surface, True, True), (0, 0))
        sprite = GUISprite(rect.topleft, surface, 26, "outline", use_viewport=False, alignment=K_TOP_LEFT)
        self.gui.add_objects(sprites=(sprite, ))

    def remove_outline(self):
        self.gui.remove_objects(sprites=(self.gui.get_sprite({"outline"})))

    def baked_menu(self, last_menu=None, min_btn=True, close_btn=True):
        x, y = self.gui.rect.size
        quit_button = Button((x - 5, 5), get_img("button_close", TEXTURES),
                             lambda: pygame.event.post(pygame.event.Event(pygame.QUIT)),
                             use_viewport=False, alignment=K_TOP_RIGHT,
                             press=self.show_outline, unfocused=self.remove_outline)
        minimize_button = Button((x - 50, 5), get_img("button_minimize", TEXTURES), self.minimize,
                                 use_viewport=False, alignment=K_TOP_RIGHT,
                                 press=self.show_outline, unfocused=self.remove_outline)
        if min_btn:
            self.gui.add_objects(buttons=(minimize_button, ))
        if close_btn:
            self.gui.add_objects(buttons=(quit_button, ))
        if last_menu is not None:
            self.gui.add_objects(buttons=(Button((5, 5), get_img("wing_base_small", TEXTURES), last_menu,
                                                 use_viewport=False, alignment=K_TOP_LEFT), ))
        self.gui.bake_background()


class MainMenu(Menu):
    def __init__(self, gui: GUI, pendulum: Pendulum, renderer: Renderer, background: GUISprite):

        super().__init__(gui, background)
        self.gui = gui
        self.pendulum = pendulum
        self.renderer = renderer
        self.sidebar = "pendulum"
        self.sampling = False
        self.sample_size = 50
        self.sample_buffers: list[list] = [[], [], []]

        self.avg_pod = 0.0
        self.avg_acc_x = 0.0
        self.avg_acc_y = 0.0
        self.std_pod = 0.0
        self.std_acc_x = 0.0
        self.std_acc_y = 0.0

        self.err_pod = 0.0
        self.err_acc_x = 0.0
        self.err_acc_y = 0.0
        self.err_pod_angle = 0.0
        self.err_acc_angle = 0.0

        # main objects
        self.input_smooth_buffer = TextBox((0.3, 0.1), comp_text_box(200), self.end_buffer_input, name="samples", text="buffer: None",
                                           press=self.start_buffer_input, whitelist={"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})

        self.volt_pod = TextBox((0.3, 0.2), comp_text_box(300), None, name="pot", text="")
        self.volt_x = TextBox((0.3, 0.3), comp_text_box(300), None, name="acc_x", text="")
        self.volt_y = TextBox((0.3, 0.4), comp_text_box(300), None, name="acc_y", text="")

        self.angle_pod = TextBox((0.3, 0.5), comp_text_box(300), None, name="pod_angle", text="")
        self.angle_acc = TextBox((0.3, 0.6), comp_text_box(300), None, name="acc_angle", text="")
        self.force_acc = TextBox((0.3, 0.7), comp_text_box(300), None, name="acc_length", text="")

        self.sample_min90 = self.basic_button((0.1, 0.2), "sample -90°", lambda angle=-90: self.start_sampling(angle))
        self.sample_0 = self.basic_button((0.1, 0.3), "sample 0°", lambda angle=0: self.start_sampling(angle))
        self.sample_90 = self.basic_button((0.1, 0.4), "sample 90°", lambda angle=90: self.start_sampling(angle))
        self.sample_180 = self.basic_button((0.1, 0.5), "sample 180°", lambda angle=180: self.start_sampling(angle))
        self.sample_angle = TextBox((0.1, 0.6), comp_text_box(300), self.update_stats, text="0")
        self.btn_start_sampling = self.basic_button((0.1, 0.7), "start sampling", self.custom_sample)
        self.btn_clear_data = self.basic_button((0.1, 0.8), "clear data", self.clear_sample_data)

        self.show_pendulum = self.basic_button((0.5, 0.2), "pendulum", self.sidebar_pendulum)
        self.show_graph = self.basic_button((0.7, 0.2), "graph [WIP]", self.sidebar_graph)
        self.show_statistics = self.basic_button((0.9, 0.2), "statistics", self.sidebar_stats)

        # stats objects
        self.input_sample_size = TextBox((0.7, 0.3), comp_text_box(200), self.end_stats_input, text="size: None",
                                         press=self.start_stats_input, whitelist={"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
        self.text_avg_pod = TextBox((0.6, 0.4), comp_text_box(300), None, name="stats raw_pod", text="")
        self.text_avg_x = TextBox((0.6, 0.5), comp_text_box(300), None, name="stats acc y", text="")
        self.text_avg_y = TextBox((0.6, 0.6), comp_text_box(300), None, name="stats acc x", text="")
        self.text_std_pod = TextBox((0.8, 0.4), comp_text_box(300), None, name="std raw_pod", text="")
        self.text_std_x = TextBox((0.8, 0.5), comp_text_box(300), None, name="std acc y", text="")
        self.text_std_y = TextBox((0.8, 0.6), comp_text_box(300), None, name="std acc x", text="")
        self.file_name = TextBox((0.7, 0.7), comp_text_box(700), None, name="file name", text="file name")
        self.btn_save_file = self.basic_button((0.55, 0.8), "save to file", self.save_file, name="save")
        self.btn_load_file = self.basic_button((0.7, 0.8), "load file", self.load_file, name="acquire")
        self.btn_print_console = self.basic_button((0.85, 0.8), "print to console", self.print_stats, name="print")

        self.sub_menu_sprites = [self.text_avg_pod, self.text_avg_x, self.text_avg_y,
                                 self.text_std_pod, self.text_std_x, self.text_std_y]
        self.sub_menu_buttons = [self.file_name, self.btn_save_file, self.btn_load_file, self.btn_print_console,
                                 self.input_sample_size]

        # pendulum model
        self.pendulum_model = Object3D([], [], [])

        verts = 32
        cube_size = 0.04
        pendulem_length = 0.5
        base_vector = pygame.Vector3(0.02, 0, 0)
        base_normal = pygame.Vector3(1, 0, 0).rotate_y(360.0/verts*0.5)
        base_normal.normalize_ip()

        cube_vertices = [pygame.Vector3(cube_size, cube_size, -cube_size), pygame.Vector3(-cube_size, cube_size, -cube_size),
        pygame.Vector3(cube_size, -cube_size, -cube_size), pygame.Vector3(-cube_size, -cube_size, -cube_size),
        pygame.Vector3(cube_size, cube_size, cube_size), pygame.Vector3(-cube_size, cube_size, cube_size),
        pygame.Vector3(cube_size, -cube_size, cube_size), pygame.Vector3(-cube_size, -cube_size, cube_size)]

        for vert in cube_vertices:
            vert.y += cube_size+pendulem_length
            self.pendulum_model.vertices.append(vert)

        self.pendulum_model.faces_index.extend([(0, 1, 3, 2), (2, 3, 7, 6)])
        self.pendulum_model.normals.extend([pygame.Vector3(0, 0, -1), pygame.Vector3(0, -1, 0)])
        for angle in range(0, verts):
            rotated_vector = base_vector.rotate_y(angle*360.0/verts)
            rotated_normal = base_normal.rotate_y(angle*360.0/verts)
            self.pendulum_model.vertices.append(pygame.Vector3(rotated_vector.x, 0, rotated_vector.z))
            self.pendulum_model.vertices.append(pygame.Vector3(rotated_vector.x, pendulem_length, rotated_vector.z))
            self.pendulum_model.faces_index.append((angle*2+8, angle*2+9, ((angle*2+3) % verts) + 8, ((angle*2+2) % verts) + 8))
            self.pendulum_model.normals.append(pygame.Vector3(rotated_normal))

        self.pendulum_model.index_faces()
        self.pendulum_model.update_positions()

    def show(self):
        """shows the menu screen"""
        self.gui.name = "Main"

        self.input_smooth_buffer.text = f"buffer: {self.pendulum.buffer_size}"

        self.center_background()
        self.gui.clear()
        # gui.add_objects(buttons=())
        self.baked_menu()
        self.gui.add_objects(sprites=(self.volt_pod, self.volt_x, self.volt_y,
                                      self.angle_pod, self.angle_acc, self.force_acc),
                             buttons=(self.input_smooth_buffer, self.sample_0, self.sample_90,
                                      self.sample_180, self.sample_min90, self.show_pendulum, self.show_graph, self.show_statistics,
                                      self.sample_angle, self.btn_start_sampling, self.btn_clear_data))

        original_focus = self.gui.focus
        try:
            self.gui.focus = self.show_pendulum
            self.show_outline()
            {"": wip, "pendulum": self.sidebar_pendulum, "stats": self.sidebar_stats}[self.sidebar]()

        except KeyError:
            print(f"{self.sidebar} not in functions list")
            self.remove_outline()
        self.gui.focus = original_focus
        self.draw()

    def draw(self):
        """update the screen"""
        self.update_text()
        try:
            {"": self.gui.filled_surface, "pendulum": self.draw_pendulum, "stats": self.draw_stats}[self.sidebar]()
        except KeyError:
            print(f"{self.sidebar} not in functions list")
            self.gui.filled_surface()
        pygame.display.flip()

    def sidebar_pendulum(self):
        self.gui.remove_objects(sprites=self.sub_menu_sprites, buttons=self.sub_menu_buttons)
        self.gui.remove_objects(sprites=(self.gui.get_sprite("selected")))
        self.sidebar = "pendulum"
        self.gui.get_sprite("outline")[0].name = "selected"

    def sidebar_graph(self):
        self.gui.remove_objects(sprites=self.sub_menu_sprites, buttons=self.sub_menu_buttons)
        self.gui.remove_objects(sprites=(self.gui.get_sprite("selected")))
        self.sidebar = ""
        self.gui.get_sprite("outline")[0].name = "selected"

    def sidebar_stats(self):
        self.gui.remove_objects(sprites=self.sub_menu_sprites, buttons=self.sub_menu_buttons)
        self.gui.remove_objects(sprites=(self.gui.get_sprite("selected")))
        self.sidebar = "stats"
        self.gui.get_sprite("outline")[0].name = "selected"

        self.input_sample_size.text = f"size: {self.sample_size}"
        self.gui.add_objects(sprites=(self.text_avg_pod, self.text_avg_x, self.text_avg_y,
                                      self.text_std_pod, self.text_std_x, self.text_std_y),
                             buttons=(self.btn_load_file, self.input_sample_size, self.file_name,
                                      self.btn_save_file, self.btn_print_console))

    def draw_pendulum(self):
        self.gui.filled_surface()

        def draw(model: Object3D, origen, rotation, color, roughness, dest_sprite: GUISprite, offset):
            model.position = origen
            model.orientation = rotation
            model.reset_vertices()
            self.renderer.draw_object(model, color, True, True, roughness, dest_sprite.image, offset)
        screen_size = self.gui.rect.size
        position = (-0.3*screen_size[0], 0.0*screen_size[1])
        model_color = pygame.Color(225, 230, 240)
        draw(self.pendulum_model, (0, 0.2, 1.5), (0, 0, self.pendulum.angle_pod+180), model_color, 0.7, self.gui, position)

    def draw_stats(self):
        if self.sampling:
            self.sample_buffers[0].append(self.pendulum.raw_pod)
            self.sample_buffers[1].append(self.pendulum.raw_accelerometer.x)
            self.sample_buffers[2].append(self.pendulum.raw_accelerometer.y)
            if len(self.sample_buffers[0]) >= self.sample_size:
                self.remove_outline()
                self.sampling = False
                self.btn_load_file.image = center_text("acquire", get_img("button", TEXTURES), BUTTON_FONT, DARK_BLUE)
                if (angle := self.get_sample_angle()) is not None:
                    self.calculate_stats(angle)
        self.text_avg_pod.text = f"avg pod: {self.avg_pod:.4f}"
        self.text_avg_x.text = f"avg x: {self.avg_acc_x:.4f}"
        self.text_avg_y.text = f"avg y: {self.avg_acc_y:.4f}"
        self.text_std_pod.text = f"std pod: {self.std_pod:.4f}"
        self.text_std_x.text = f"std x: {self.std_acc_x:.4f}"
        self.text_std_y.text = f"std y: {self.std_acc_y:.4f}"
        self.gui.filled_surface()

    def clear_sample_data(self):
        self.remove_outline()
        self.pendulum.data_samples.clear()

    def get_sample_angle(self) -> int | None:
        try:
            angle: int = int(self.sample_angle.text)
        except ValueError:
            print(f"{self.sample_angle.text} is not a valid value")
            self.sample_angle.text = "not an int"
            return None
        if angle < -180 or angle > 180:
            print(f"{angle} is outside range -180 to 180")
            self.sample_angle.text = "outside range"
            return None

        return angle

    def custom_sample(self):
        self.remove_outline()
        try:
            angle: int = int(self.sample_angle.text)
        except ValueError:
            print(f"{self.sample_angle.text} is not a valid value")
            self.sample_angle.text = "not an int"
            return
        if angle < -180 or angle > 180:
            print(f"{angle} is outside range -180 to 180")
            self.sample_angle.text = "outside range"
            return

        self.pendulum.start_sampling(angle)

    def start_sampling(self, angle: int):
        self.pendulum.start_sampling(angle)
        self.remove_outline()

    def start_buffer_input(self):
        self.input_smooth_buffer.text = str(self.pendulum.buffer_size)

    def end_buffer_input(self):
        if self.input_smooth_buffer.text:
            self.pendulum.set_buffer_size(int(clamp(int(self.input_smooth_buffer.text), 1, 50)))
        else:
            self.pendulum.set_buffer_size(10)
        self.input_smooth_buffer.text = f"buffer: {self.pendulum.buffer_size}"

    def start_stats_input(self):
        self.input_sample_size.text = str(self.sample_size)

    def end_stats_input(self):
        if self.input_sample_size.text:
            self.sample_size = int(clamp(int(self.input_sample_size.text), 30, 1000))
        else:
            self.sample_size = 50
        self.input_sample_size.text = f"size: {self.sample_size}"

    def update_stats(self):
        if (angle := self.get_sample_angle()) is not None:
            self.calculate_stats(angle)
        self.text_avg_pod.text = f"raw_pod: {self.avg_pod:.4f}"
        self.text_avg_x.text = f"x: {self.avg_acc_x:.4f}"
        self.text_avg_y.text = f"y: {self.avg_acc_y:.4f}"
        self.text_std_pod.text = f"raw_pod std: {self.std_pod:.4f}"
        self.text_std_x.text = f"x std: {self.std_acc_x:.4f}"
        self.text_std_y.text = f"y std: {self.std_acc_y:.4f}"
        self.gui.filled_surface()

    def calculate_stats(self, angle: int) -> bool:
        if angle not in self.pendulum.data_samples:
            print(f"angle {angle} has not been sampled yet")
            return False
        stats_average = [0.0, 0.0, 0.0]
        stats_std = [0.0, 0.0, 0.0]
        data_points = [[], [], []]
        for buffer in self.pendulum.data_samples[angle]:
            data_points[0].append(buffer[0])
            data_points[1].append(buffer[1])
            data_points[2].append(buffer[2])

        for i, buffer in enumerate(data_points):

            length = len(buffer)
            average = sum(buffer) / length
            std = sum([pow(number - average, 2) for number in buffer])
            std /= length - 1
            std = sqrt(std)
            stats_average[i] = average
            stats_std[i] = std

        # error potentiometer
        min_pod = min(data_points[0])
        max_pod = max(data_points[0])
        self.err_pod = (max_pod - min_pod) / 2
        self.err_pod_angle = self.err_pod * abs(self.pendulum.pod_slope)

        # error accelerometer.
        min_x, min_y = min(data_points[1]), min(data_points[2])
        max_x, max_y = max(data_points[1]), max(data_points[2])
        average_x, average_y = stats_average[1:3]
        self.err_acc_x, self.err_acc_y = (max_x - min_x) / 2, (max_y - min_y) / 2
        err_x, err_y = self.err_acc_x * abs(self.pendulum.acc_x_slope), self.err_acc_y * abs(self.pendulum.acc_y_slope)
        corrected_x = average_x * self.pendulum.acc_x_slope + self.pendulum.acc_x_start
        corrected_y = average_y * self.pendulum.acc_y_slope + self.pendulum.acc_y_start
        average_division = corrected_x / corrected_y
        delta_div = abs(average_division) * sqrt((err_x / corrected_x)**2 + (err_y / corrected_y)**2)
        self.err_acc_angle = (1/(1+math.degrees(math.tan(average_division))**2)) * delta_div

        self.avg_pod, self.avg_acc_x, self.avg_acc_y = stats_average
        self.std_pod, self.std_acc_x, self.std_acc_y = stats_std

        return True

    def spreadsheet_string(self) -> str:
        result_string = "data points\t\t\t\t\tstats\tvolt pod\t\t\tvolt acc x\t\t\t volt acc y\t\t\terror pod\terror acc\n"
        result_string += "angle\tpod\tacc_x\tacc_y\t\tangle\taverage\tstd\terror\taverage\tstd\terror\taverage\tstd\terror\tdegrees\tdegrees\n"
        sampled_angles = list(self.pendulum.data_samples.keys())
        value_buffer = self.avg_pod, self.std_pod, self.err_pod, self.avg_acc_x, self.std_acc_x, self.err_acc_x, self.avg_acc_y, self.std_acc_y, self.err_acc_y, self.err_pod_angle, self.err_acc_angle
        sampled_angles.sort()
        data_entries = []
        stat_entries = []
        for angle in sampled_angles:
            self.calculate_stats(angle)
            stat_entry = f"{angle}\t{self.avg_pod:.8}\t{self.std_pod:.8}\t{self.err_pod:.8}"
            stat_entry += f"\t{self.avg_acc_x:.8}\t{self.std_acc_x:.8}\t{self.err_acc_x:.8}"
            stat_entry += f"\t{self.avg_acc_y:.8}\t{self.std_acc_y:.8}\t{self.err_acc_y:.8}"
            stat_entry += f"\t{self.err_pod_angle:.8}\t{self.err_acc_angle:.8}"
            stat_entries.append(stat_entry)
            for entry in self.pendulum.data_samples[angle]:
                data_entries.append(f"{angle}\t{entry[0]:.8}\t{entry[1]:.8}\t{entry[2]:.8}")

        self.avg_pod, self.std_pod, self.err_pod, self.avg_acc_x, self.std_acc_x, self.err_acc_x, self.avg_acc_y, self.std_acc_y, self.err_acc_y, self.err_pod_angle, self.err_acc_angle = value_buffer

        for i, second_part in enumerate(stat_entries):
            data_entries[i] = data_entries[i] + "\t\t" + second_part

        result_string += "\n".join(data_entries)
        return result_string

    def print_stats(self):
        self.remove_outline()
        if (angle := self.get_sample_angle()) is None:
            return
        if not self.calculate_stats(angle):
            print("specified angle is not yet sampled")
            return

        print(f"\nshowing stats for angle: {angle}")
        print("\navarages (Volt):")
        print(f"\t-potentiometer: {self.avg_pod:.4f}")
        print(f"\t-accelerometer x: {self.avg_acc_x:.4f}")
        print(f"\t-accelerometer y: {self.avg_acc_y:.4f}")
        print("\nstandard deviation (Volt):")
        print(f"\t-potentiometer: {self.std_pod:.4f}")
        print(f"\t-accelerometer x: {self.std_acc_x:.4f}")
        print(f"\t-accelerometer y: {self.std_acc_y:.4f}")
        print("\nerror (Volt):")
        print(f"\t-potentiometer: {self.err_pod:.4f}")
        print(f"\t-accelerometer x: {self.err_acc_x:.4f}")
        print(f"\t-accelerometer y: {self.err_acc_y:.4f}")
        print("\nerror (angle):")
        print(f"\t-potentiometer: {self.err_pod_angle:.4f}")
        print(f"\t-accelerometer: {self.err_acc_angle:.4f}")

    def save_file(self):
        self.remove_outline()
        if not self.pendulum.data_samples:
            print("no data")
            return
        if self.file_name.text == "file name" or not self.file_name.text:
            print("provide a file name")
            return
        print("saving...")
        with open("data\\" + self.file_name.text, "w") as f:
            f.write(self.spreadsheet_string())
        print(f"saved file as {self.file_name.text}")

    def load_file(self):
        try:
            with open("data\\" + self.file_name.text, "r") as f:
                file_contents = f.read()
        except IOError:
            print("could not open or read the given file")
            self.remove_outline()
            return
        file_lines = file_contents.split("\n")
        if len(file_lines) < 3:
            print("could not load file")
            self.remove_outline()
            return
        for line in file_lines[2:]:
            try:
                angle, pod, x, y = line.split("\t")[:4]
            except IndexError:
                continue
            try:
                angle_int = int(angle)
                if angle_int not in self.pendulum.data_samples:
                    self.pendulum.data_samples[angle_int] = []
                self.pendulum.data_samples[angle_int].append((float(pod), float(x), float(y)))
            except ValueError:
                continue
        self.pendulum.update_regression()
        self.remove_outline()
        print("file loaded")

    def update_text(self):
        if self.pendulum.mydaq.connected:
            self.volt_pod.text = "volt pod: " + f"{self.pendulum.raw_pod:.4f}"
            self.volt_x.text = "volt x: " + f"{self.pendulum.raw_accelerometer.x:.4f}"
            self.volt_y.text = "volt y: " + f"{self.pendulum.raw_accelerometer.y:.4f}"
            self.pendulum.pod_angle()
            self.pendulum.accelerometer_angle()
            self.angle_pod.text = "angle pod: " + f"{self.pendulum.angle_pod:.4f}"
            self.angle_acc.text = "angle acc: " + f"{self.pendulum.angle_accelerometer:.4f}"
            self.force_acc.text = "force acc: " + f"{self.pendulum.normalized_accelerometer.length():.4f}"
        else:
            self.volt_pod.text = "volt pod: disconnected"
            self.volt_x.text = "volt x: disconnected"
            self.volt_y.text = "volt y: disconnected"
            self.angle_pod.text = "angle pod: disconnected"
            self.angle_acc.text = "angle acc: disconnected"
            self.force_acc.text = "force acc: disconnected"


def wip(*_args):
    pass


def main():
    def create_display(size, flags) -> pygame.Surface:
        if pygame.display.get_num_displays() > 1:
            return pygame.display.set_mode(size, flags=flags, display=1)
        else:
            return pygame.display.set_mode(size, flags=flags)

    def draw_screen(_event=None):
        if gui.name == "Main":
            main_menu.draw()

    def center_background():
        background_sprite.viewport_to_pixels((0.5, 0.5), window.get_size())
        gui.add_objects(sprites=(background_sprite, ))
        gui.bake_background()

    def key_press(event: pygame.event.Event):
        nonlocal fullscreen, window, small_window_size
        focus = gui.get_focus()
        if focus is not None and focus.active:
            events.on_key_press(event)
            pygame.event.post(pygame.event.Event(events.DRAW_SCREEN))
            return

        key = event.key
        if key == pygame.K_F11:
            fullscreen = not fullscreen
            if fullscreen:
                small_window_size = pygame.display.get_window_size()
                window = pygame.display.set_mode(flags=pygame.FULLSCREEN)
            else:
                window = pygame.display.set_mode(small_window_size, flags=pygame.RESIZABLE)

            gui.image = window
            window_changed_size()

    def key_release(event: pygame.event.Event):
        key = event.key

    def mouse_press(event: pygame.event.Event):
        events.on_mouse_press(event)

    def mouse_release(event: pygame.event.Event):
        events.on_mouse_release(event)

    def mouse_move(event: pygame.event.Event):
        events.on_mouse_move(event)

    def mouse_scroll(event: pygame.event.Event):
        events.on_scroll(event)

    def window_changed_size(_event=None):
        renderer.update_camera(pygame.display.get_window_size())
        redraw_menus()

    def redraw_menus():
        try:
            {"Main": main_menu.show}[gui.name]()
        except KeyError:
            print(f"cant resize menu {gui.name}")

    def close(_event=None):
        nonlocal run
        pygame.display.set_caption("closing")
        pygame.display.iconify()
        mydaq.close_taks()
        run = False

    def main_loop():
        while run:
            clock.tick(FPS_TARGET)
            events.handle_events()
            draw_screen()

    pygame.event.set_blocked(None)
    run = True
    fullscreen = True
    window = create_display((0, 0), pygame.FULLSCREEN) if fullscreen else create_display((640, 480), pygame.RESIZABLE)
    small_window_size = pygame.display.get_window_size()
    if fullscreen:
        small_window_size = (small_window_size[0], ceil(small_window_size[1]*0.95))
    gui = events.init(window)
    background_sprite = GUISprite((0, 0), get_img("background_stainlesssteel4", TEXTURES, False), 1)
    gui.source_image = background_sprite.image
    gui.clear()
    center_background()
    renderer = Renderer(pygame.display.get_window_size(), fov=60)
    mydaq = MyDaq("Dev1")
    pendulum = Pendulum(mydaq)
    main_menu = MainMenu(gui, pendulum, renderer, background_sprite)

    draw_screen()
    devices = nidaqmx.system.System.local().devices
    print([str(device) for device in devices])

    mydaq.init_task()

    event_functions = {pygame.NOEVENT: wip,
                       pygame.KEYDOWN: key_press, pygame.KEYUP: key_release,
                       pygame.MOUSEBUTTONDOWN: mouse_press, pygame.MOUSEBUTTONUP: mouse_release,
                       pygame.MOUSEMOTION: mouse_move, pygame.MOUSEWHEEL: mouse_scroll,
                       pygame.WINDOWFOCUSGAINED: wip, pygame.WINDOWRESIZED: window_changed_size,
                       pygame.WINDOWMOVED: window_changed_size,
                       pygame.QUIT: close, UPDATE_SCREEN: draw_screen,
                       events.DRAW_SCREEN: draw_screen, SAMPLE_DATA: lambda _event=None: pendulum.acquire()}

    events.event_functions.update(event_functions)
    pygame.event.clear()
    pygame.event.set_allowed((pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.KEYUP, pygame.KEYDOWN,
                              pygame.WINDOWRESIZED, pygame.QUIT, pygame.WINDOWMOVED, events.DRAW_SCREEN,
                              pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED, pygame.JOYAXISMOTION,
                              pygame.MOUSEWHEEL, UPDATE_SCREEN, SAMPLE_DATA))

    main_menu.show()
    pygame.time.set_timer(UPDATE_SCREEN, int(MS_PER_FRAME))

    pygame.time.set_timer(SAMPLE_DATA, int(MS_PER_SAMPLE))
    main_loop()


if __name__ == "__main__":
    main()
