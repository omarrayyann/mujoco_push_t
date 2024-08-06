import mujoco
import mujoco.viewer
import numpy as np
import time
from Utils.utils import *
from controller import OpsaceController
from mujoco_ar import MujocoARConnector
import random
import rerun as rr

class Simulation:
    
    def __init__(self):
        """
        Initialize the simulation with configurations and setup.
        """
        # Configs
        self.scene_path = 'Environment/scene.xml'
        self.mjmodel = mujoco.MjModel.from_xml_path(self.scene_path)
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.dt = 0.002
        self.button = 0
        self.placement_time = -1
        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer.enable_depth_rendering()
        self.cameras = ["front_camera","side_camera","top_camera"]
        self.T_object = "T"
        self.T_joint = "T_joint"
        self.T_outline_object = "T_outline"
        self.mjmodel.opt.timestep = self.dt
        self.frequency = 1000
        self.target_pos = np.array([0.5, 0.0, 0.23])
        self.target_rot = rotation_matrix_x(np.pi) @ np.identity(3)
        self.pos_origin = self.target_pos.copy()
        self.rot_origin = self.target_rot.copy()
        self.target_quat = np.zeros(4)
        self.eef_site_name = 'eef'
        self.site_id = self.mjmodel.site(self.eef_site_name).id
        self.camera_data = None
        self.joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
        self.max_ori_error = 0.04
        self.max_pos_error = 0.015

        # Controller
        self.controller = OpsaceController(self.mjmodel, self.joint_names, self.eef_site_name)
        self.q0 = np.array([0, 0.4335, 0, -1.7657, 0, 0.9424, 0])
        self.dof_ids = np.array([self.mjmodel.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.mjmodel.actuator(name).id for name in self.joint_names])
        self.mjdata.qpos[self.actuator_ids] = self.q0

        # MujocoAR (pip install mujoco_ar)
        self.mujocoAR = MujocoARConnector(controls_frequency=10,mujoco_model=self.mjmodel,mujoco_data=self.mjdata)

        self.mujocoAR.link_site(
            name="eef_target",
            scale=2.0,
            translation=self.pos_origin,
            button_fn=lambda: (self.random_placement(), setattr(self, 'placement_time', time.time())) if time.time() - self.placement_time > 2.0 else None,
            disable_rot=True,
        )

        # Rerun
        # rr.init("Mujoco_push_t", spawn=True)
    
    def send_rr(self) -> dict:
        data = {}    
        for camera in self.cameras:
            self.rgb_renderer.update_scene(self.mjdata, camera)
            self.depth_renderer.update_scene(self.mjdata, camera)
            data[camera+"_rgb"] = self.rgb_renderer.render()
            data[camera+"_depth"] = self.depth_renderer.render()
            rr.log(camera+"_rgb", rr.Image(data[camera+"_rgb"]).compress(jpeg_quality=95))
        return data
    
    def is_valid_position(self, pos1, pos2, min_dist, max_dist):
        """
        Check if the distance between two positions is at least min_dist.
        """
        if pos1 is None or pos2 is None:
            return False
        distance = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2])) 
        return distance >= min_dist and distance <= max_dist
    
    def get_pos_from_range(self, range):
        """
        Generate a random position within the given range.
        """
        return np.array([random.uniform(range[0,0], range[0,1]), random.uniform(range[1,0], range[1,1]), range[2,0]])

    def done(self):
        """
        Check if the task is done based on the positions of T_object and T_outline_object.
        """
        T_pos = self.mjdata.body(self.T_object).xpos.copy()
        T_outline_pos = self.mjdata.body(self.T_outline_object).xpos.copy()

        T_quat = self.mjdata.body(self.T_object).xquat.copy()
        T_outline_quat = self.mjdata.body(self.T_outline_object).xquat.copy()
        T_euler = quat_to_euler(T_quat)
        T_outline_euler = quat_to_euler(T_outline_quat)

        if np.linalg.norm(np.array(T_pos)[0:2] - np.array(T_outline_pos)[0:2]) <= self.max_pos_error and np.linalg.norm(np.array(T_euler)[2] - np.array(T_outline_euler)[2]) < self.max_ori_error:
            return True
        
    def fell(self):
        """
        Check if T_object has fallen based on its position.
        """
        T_pos = self.mjdata.body(self.T_object).xpos.copy()
        if T_pos[2] < 0.1 or T_pos[2] > 2 or abs(T_pos[0]) > 1.0 or abs(T_pos[1]) > 1.0:
            return True

    def random_placement(self, min_seperation=0.1, max_seperation=0.3):
        """
        Randomly place the T_object and T_outline_object in the scene while maintaining minimum separation.
        """
        T_outline_range = np.array([[0.42, 0.64], [-0.2, 0.2], [0.21, 0.21]])
        T_range = np.array([[0.42, 0.64], [-0.2, 0.2], [0.25, 0.25]])

        T_outline_pos, T_pos = None, None
        while not self.is_valid_position(T_outline_pos, T_pos, min_seperation, max_seperation):
            T_outline_pos = self.get_pos_from_range(T_outline_range)
            T_pos = self.get_pos_from_range(T_range)
        
        pick_euler = np.array([0,0,random.uniform(0, 3.14)])
        place_euler = np.array([0,0,random.uniform(0, 3.14)])

        self.mujocoAR.pause_updates()
        self.mujocoAR.reset_position()
        self.mjdata.qpos[self.actuator_ids] = self.q0
        mujoco.mj_step(self.mjmodel, self.mjdata)
        self.mjdata.joint(self.T_joint).qvel = np.zeros(6)
        T_quat = np.zeros(4)
        mujoco.mju_euler2Quat(T_quat, pick_euler.flatten(), "xyz")
        self.mjdata.joint(self.T_joint).qpos = np.block([T_pos, T_quat])        
        set_body_pose(self.mjmodel, self.T_outline_object, T_outline_pos, euler=place_euler)
        mujoco.mj_step(self.mjmodel, self.mjdata)

        self.mujocoAR.resume_updates()

    def start(self):
        """
        Start the simulation.
        """
        self.mujocoAR.start()
        self.mac_launch()

    def mac_launch(self):
        """
        Launch the MuJoCo viewer and control loop for the simulation.
        """
        with mujoco.viewer.launch_passive(self.mjmodel, self.mjdata, show_left_ui=False, show_right_ui=False) as viewer:

            self.random_placement()
            while viewer.is_running():       

                step_start = time.time()

                tau = self.controller.get_tau(self.mjmodel, self.mjdata, self.target_pos, self.target_rot)
                self.mjdata.ctrl[self.actuator_ids] = tau[self.actuator_ids]

                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()

                if (self.done() or self.fell() or self.button) and time.time() - self.placement_time > 2.0:
                    self.random_placement()
                    self.placement_time = time.time()
                
                self.target_pos[:2] = self.mjdata.site("eef_target").xpos[:2]
                    
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    # Initialize and start the simulation
    sim = Simulation()
    sim.start()

rr.log("world/camera/image/rgb", rr.Image(img_rgb).compress(jpeg_quality=95))
