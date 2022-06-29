#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import csv
import weakref

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from agents.navigation.controller import VehiclePIDController

from buffered_saver_lc import BufferedImageSaver
from agents.tools.misc import get_speed

import random
from PIL import Image
import scipy.misc as misc
import math
import logging

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_m
    from pygame.locals import K_o
    from pygame.locals import K_a
    from pygame.locals import K_l
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_j
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_t
    from pygame.locals import K_u
    from pygame.locals import K_b
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, vehicle, autopilot_enabled=True):
        self.vehicle = vehicle
        self.autopilot_enabled = autopilot_enabled
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0
        self.left_lane_change_activated = 0
        self.right_lane_change_activated = 0
        self.lane_change_second_half = 0
        self.start_data_collection = False
        self.force_left_lane_change = False
        self.force_right_lane_change = False
        self.junk = 0
        self.get_waypoint = False
        self.spawn_static_object = False
        self.destroy_static_object = False

    def parse_events(self, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == K_m:
                    self.autopilot_enabled = False
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    print('Autopilot Off')
                if event.key == K_o:
                    self.autopilot_enabled = True
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    print('Autopilot On; Lane change deactivated!')
                if event.key == K_l:
                    self.left_lane_change_activated = 1
                    self.lane_change_second_half = -1
                    print('Left lane change activated')
                if event.key == K_r:
                    self.right_lane_change_activated = 1
                    self.lane_change_second_half = -1
                    print('Right lane change activated')
                if event.key == K_s:
                    self.lane_change_second_half = 1
                    print('Second half of lane change')
                if event.key == K_c:
                    self.start_data_collection = True
                    print('Starting data collection')
                if event.key == K_p:
                    self.start_data_collection = False
                    print('Pausing data collection')
                if event.key == K_a:
                    print("Forcing left lane change")
                    self.force_left_lane_change = True
                if event.key == K_d:
                    print("Forcing right lane change")
                    self.force_right_lane_change = True
                if event.key == K_q:
                    print("lane change over")
                    self.left_lane_change_activated = 0
                    self.right_lane_change_activated = 0
                    self.lane_change_second_half = 0
                if event.key == K_t:
                    print("spawn static object")
                    self.spawn_static_object = True
                if event.key == K_u:
                    print("destroying static object")
                    self.destroy_static_object = True
                if event.key == K_g:
                    print("getting waypoint")
                    self.get_waypoint = True

            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

        if not self.autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self.control.reverse = self.control.gear < 0          
            self.vehicle.apply_control(self.control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self.control.throttle = 1.0 if keys[K_UP] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[K_RIGHT]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 1)
        self.control.brake = 1.0 if keys[K_DOWN] else 0.0
        self.control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        #self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        #self.hud.notification('Crossed line %s' % ' and '.join(text))
        print('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))
    return image.raw_data, array

def draw_image2(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    """
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (800, 0))
    """
    return image.raw_data, array


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.
        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=10.0)


class polyline():
    def __init__(self, loc1, loc2, loc3, wps, distance=10):
        
        self.distance = distance
        self.locs_list = [loc1, loc2, loc3]       
        for i in range(len(wps)):
            self.locs_list.append(wps[i].transform.location)
        
        self.nxt_pointer = 1
        self.crossed_pointer = 0

    def translate(self, loc, cu_loc):
        return loc - cu_loc

    def transform(self, loc, cu_loc, yaw):
        # first translate
        tr_loc = self.translate(loc, cu_loc)

        # yaw is angle in positive clockwise directions when looking
        # in the positive direction of the Z-axis.
        cos_t = math.cos(math.radians(yaw))
        sin_t = math.sin(math.radians(yaw))
     
        tmp_x = tr_loc.x * cos_t + tr_loc.y * sin_t
        tmp_y = - tr_loc.x * sin_t + tr_loc.y * cos_t
        tr_loc.x = tmp_x
        tr_loc.y = tmp_y
        return tr_loc

    def transform_locs_list(self, cu_loc, yaw):
        tr_locs_list = []
        for i in range(len(self.locs_list)):
            loc = self.transform(self.locs_list[i], cu_loc, yaw)
            tr_locs_list.append(loc)
        return tr_locs_list
   
    def compute_distance(self, loc1, loc2):
        vector = loc1 - loc2
        # Only distance in x-y axis
        distance = np.sqrt(vector.x**2 + vector.y**2)
        return distance

    def compute_polyline_distance(self, cu_loc, yaw, cr_pt, nxt_pt):
        tr_cu_loc = self.transform(cu_loc, cu_loc, yaw)
        tr_locs_list = self.transform_locs_list(cu_loc, yaw)

        if cr_pt + 1 == nxt_pt:
            d = self.compute_distance(tr_cu_loc, tr_locs_list[nxt_pt])
        else:
            d = self.compute_distance(tr_cu_loc, tr_locs_list[cr_pt+1])
            for pt_idx in range(cr_pt+1, nxt_pt):
                d += self.compute_distance(tr_locs_list[pt_idx], tr_locs_list[pt_idx+1])
        return d

    def find_x_image_on_line(self, cu_loc, crossed_loc, next_loc):
        # returns in world coordinates
        m = float(crossed_loc.y - next_loc.y)/float(crossed_loc.x - next_loc.x)
        c = float(crossed_loc.x * next_loc.y - next_loc.x * crossed_loc.y)/float(crossed_loc.x - next_loc.x)
        x = cu_loc.x
        y = m * x + c
        return (x, y)

    def find_point_on_line(self, cu_loc, yaw, initial_loc, terminal_loc, distance):
        tr_initial_loc = self.transform(initial_loc, cu_loc, yaw)
        tr_terminal_loc = self.transform(terminal_loc, cu_loc, yaw)
        v = np.array([tr_initial_loc.x, tr_initial_loc.y], dtype=float)
        u = np.array([tr_terminal_loc.x, tr_terminal_loc.y], dtype=float)
        n = v - u
        n /= np.linalg.norm(n, 2)
        point = v - distance * n
        return tuple(point)


def main():
    data_path = '/home/apoorva/data/lc-town03-polyline/'
    lane_change_number = 10
    BIS = BufferedImageSaver(data_path, 300, 800, 600, 3, 'CameraRGB', lane_change_number)

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()# load_world('Town05') #
    tm = client.get_trafficmanager(3000)
    tm.set_synchronous_mode(True)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    tot_target_reached = 0
    num_min_waypoints = 5
    obstacle = None

    try:
        m = world.get_map()
        spawn_points = m.get_spawn_points()
        start_pose = spawn_points[243] #141
        end_location = spawn_points[83].location 
        #random.choice(spawn_points).location 
        #carla.Location(x=-74.650337, y=141.064636, z=0.000000)
        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.a2')),
            start_pose)
        actor_list.append(vehicle)
        world.player = vehicle
        tm.ignore_lights_percentage(vehicle,100)
        tm.auto_lane_change(vehicle, False)
        agent = BehaviorAgent(vehicle, ignore_traffic_light=True, behavior='cautious')     
        agent.set_destination(agent.vehicle.get_location(), end_location, clean=True)
        #lane_invasion_sensor = LaneInvasionSensor(vehicle)
        #actor_list.append(lane_invasion_sensor.sensor)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=2.5, z=1.5)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_top_view = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8),
                            carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_top_view)

        controller = KeyboardControl(world, vehicle, True)
        polyline_controller = False

        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_top_view, fps=30) as sync_mode:
            while True:
                if controller.parse_events(clock):
                    return
                agent.update_information(world)
                clock.tick()
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_topview = sync_mode.tick(timeout=2.0)
                # Draw the display.
                raw, img = draw_image2(display, image_rgb) #draw_image2
                raw2, img2 = draw_image(display, image_topview)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        traffic_light.set_green_time(10.0)

                # Set new destination when target has been reached
                if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                    print("ReRouting")

                if controller.spawn_static_object:
                    controller.spawn_static_object = False
                    ego_vehicle_loc = vehicle.get_location()
                    ego_vehicle_wp = m.get_waypoint(ego_vehicle_loc)
                    obstacle_wp = list(ego_vehicle_wp.next(30))[0]
                    obstacle_location = obstacle_wp.transform.location
                    obstacle = world.spawn_actor(
                        random.choice(blueprint_library.filter('vehicle.audi.a2')),
                        obstacle_wp.transform)
                    obstacle.set_location(obstacle_location)
                    obstacle.set_simulate_physics(False)
                    actor_list.append(obstacle)
                if obstacle:
                    ego_vehicle_loc = vehicle.get_location()
                    vector = ego_vehicle_loc - obstacle_location
                    distance = np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
                    if distance > 27 and distance < 28:
                        print(distance)
                else:
                    distance = 1000

                if controller.destroy_static_object:
                    controller.destroy_static_object = False
                    obstacle.destroy()
                    obstacle = None

 
                if controller.force_left_lane_change:
                    print("Left Here")
                    controller.force_left_lane_change = False
                    ego_vehicle_loc = vehicle.get_location()
                    ego_vehicle_wp = m.get_waypoint(ego_vehicle_loc)
                    ego_vehicle_nxt_wp = list(ego_vehicle_wp.next(10))[0]
                    left_nxt_wpt = ego_vehicle_nxt_wp.get_left_lane()
                    left_nxt_nxt_wpt = list(left_nxt_wpt.next(15))[0]
                      
                    left_nxt_nxt_nxt_wpts = []
                    for wpt_i in range(60):
                        if wpt_i == 0:
                            tmp_wpt = list(left_nxt_nxt_wpt.next(1))[0]
                        else:
                            tmp_wpt = list(tmp_wpt.next(1))[0]
                        left_nxt_nxt_nxt_wpts.append(tmp_wpt)
                    len_nxt_nxt_nxt_wpts = min(60, len(left_nxt_nxt_nxt_wpts))
                    left_nxt_nxt_nxt_wpts = left_nxt_nxt_nxt_wpts[:len_nxt_nxt_nxt_wpts]
                    lc_waypoints_count = 3 + len_nxt_nxt_nxt_wpts  
                    wps = [ego_vehicle_wp, ego_vehicle_nxt_wp, left_nxt_nxt_wpt,
                           left_nxt_nxt_nxt_wpts[-1]]                     
                    if left_nxt_wpt is not None:
                        pl = polyline(ego_vehicle_loc,
                                      ego_vehicle_nxt_wp.transform.location,
                                      left_nxt_nxt_wpt.transform.location,
                                      left_nxt_nxt_nxt_wpts,
                                      distance=10)
                        #draw_waypoints(world, wps , z=0.0)
                        polyline_controller = True

                if controller.force_right_lane_change:
                    print("Right Here")
                    controller.force_right_lane_change = False
                    ego_vehicle_loc = vehicle.get_location()
                    ego_vehicle_wp = m.get_waypoint(ego_vehicle_loc)
                    ego_vehicle_nxt_wp = list(ego_vehicle_wp.next(10))[0]
                    right_nxt_wpt = ego_vehicle_nxt_wp.get_right_lane()
                    right_nxt_nxt_wpt = list(right_nxt_wpt.next(15))[0]
                    right_nxt_nxt_nxt_wpts = []
                    for wpt_i in range(60):
                        if wpt_i == 0:
                            tmp_wpt = list(right_nxt_nxt_wpt.next(1))[0]
                        else:
                            tmp_wpt = list(tmp_wpt.next(1))[0]
                        right_nxt_nxt_nxt_wpts.append(tmp_wpt)                        
                    len_nxt_nxt_nxt_wpts = min(60, len(right_nxt_nxt_nxt_wpts))
                    right_nxt_nxt_nxt_wpts = right_nxt_nxt_nxt_wpts[:len_nxt_nxt_nxt_wpts]
                    lc_waypoints_count = 3 + len_nxt_nxt_nxt_wpts                        
                    wps = [ego_vehicle_wp, ego_vehicle_nxt_wp,
                           right_nxt_nxt_wpt, right_nxt_nxt_nxt_wpts[-1]]
                    if right_nxt_wpt is not None:
                        pl = polyline(ego_vehicle_loc,
                                      ego_vehicle_nxt_wp.transform.location,
                                      right_nxt_nxt_wpt.transform.location,
                                      right_nxt_nxt_nxt_wpts,
                                      distance=10)
                        #draw_waypoints(world, wps , z=0.0)
                        polyline_controller = True
                            

                if controller.autopilot_enabled:
                    if polyline_controller == True:
                        cu_tr = vehicle.get_transform()
                        cu_loc = cu_tr.location # world co-ordinates
                        yaw = cu_tr.rotation.yaw
                        local_cu_loc = pl.transform(cu_loc, cu_loc, yaw)
                        local_locs_list = pl.transform_locs_list(cu_loc, yaw)

                        # TODO verify this logic                            
                        if local_locs_list[pl.crossed_pointer + 1].x <= 0:   
                            pl.crossed_pointer += 1
                        d1 = 0
                        while pl.compute_polyline_distance(
                            cu_loc, yaw, pl.crossed_pointer, pl.nxt_pointer) <= pl.distance:
                            if not pl.nxt_pointer == lc_waypoints_count - 1:
                                # d1 is distance along polyline
                                d1 = pl.compute_polyline_distance(
                                    cu_loc, yaw, pl.crossed_pointer, pl.nxt_pointer)
                                pl.nxt_pointer += 1
                                    
                            else:
                                d1 = pl.compute_polyline_distance(
                                    cu_loc, yaw, pl.crossed_pointer, pl.nxt_pointer)
                                polyline_controller = False
                                break
                        # The ego vehicle may not lie exactly on line
                        gt_point = pl.find_x_image_on_line(
                            cu_loc,
                            pl.locs_list[pl.crossed_pointer],
                            pl.locs_list[pl.crossed_pointer+1])

                        if d1 == 0:
                            point = pl.find_point_on_line(cu_loc, yaw, cu_loc,
                                                          pl.locs_list[pl.nxt_pointer],
                                                          pl.distance)
                        else:
                            point = pl.find_point_on_line(cu_loc, yaw,
                                                          pl.locs_list[pl.nxt_pointer - 1],
                                                          pl.locs_list[pl.nxt_pointer],
                                                          pl.distance - d1)

                        dy =  local_cu_loc.y - point[1]
                        steering =  - dy * 0.05
                        control = agent.run_step()
                        control.steer = steering
                        vehicle.apply_control(control)
                    else:
                        control = agent.run_step()
                        vehicle.apply_control(control)
                           

                if controller.get_waypoint:
                    location = vehicle.get_location()
                    ego_vehicle_wp = m.get_waypoint(location)
                    next_loc = list(ego_vehicle_wp.next(10))[0].transform.location
                    vehicle.set_location(next_loc)
                    controller.get_waypoint = False
                v = vehicle.get_velocity()   
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)), (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)), (8, 28))
                display.blit(
                    font.render('% 5f speed (ego-car)' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)), True, (255, 255, 255)), (8, 48))
                display.blit(
                    font.render('% 5f steering ' % control.steer, True, (255, 255, 255)), (8, 68))
                pygame.display.flip()
    

                if controller.start_data_collection:
                    if BIS.index % 50 == 0:
                        print(BIS.index)
                    BIS.add_image(raw,
                                  control.steer,
                                  controller.left_lane_change_activated,
                                  controller.right_lane_change_activated,
                                  controller.lane_change_second_half,
                                  controller.junk,
                                  distance,
                                  'CameraRGB')

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
