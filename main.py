import mujoco
import mujoco.viewer
import numpy as np
import time
from Utils.utils import *
from controller import OpsaceController
from mujoco_ar import MujocoARConnector
import random
import threading

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

        # Recording and Policy Related
        self.record = False
        self.run_policy = False
        self.recording_frequency = 10
        self.last_recording_time = -1

        # Controller
        self.controller = OpsaceController(self.mjmodel, self.joint_names, self.eef_site_name)
        self.q0 = np.array([0, 0.4335, 0, -1.7657, 0, 0.9424, 0])
        self.dof_ids = np.array([self.mjmodel.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.mjmodel.actuator(name).id for name in self.joint_names])
        self.mjdata.qpos[self.actuator_ids] = self.q0

        # MujocoAR (pip install mujoco_ar)
        self.mujocoAR = MujocoARConnector(mujoco_model=self.mjmodel,mujoco_data=self.mjdata)

        self.mujocoAR.link_site(
            name="eef_target",
            scale=2.0,
            position_origin=self.pos_origin,
            button_fn=lambda: (self.random_placement(), setattr(self, 'placement_time', time.time()), self.reset_data()) if time.time() - self.placement_time > 2.0 else None,
            disable_rot=True,
        )

    
    def get_camera_data(self) -> dict:
        data = {}    
        for camera in self.cameras:
            self.rgb_renderer.update_scene(self.mjdata, camera)
            self.depth_renderer.update_scene(self.mjdata, camera)
            data[camera+"_rgb"] = self.rgb_renderer.render()
            data[camera+"_depth"] = self.depth_renderer.render()
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
        threading.Thread(target=self.mac_launch).start()

    def mac_launch(self):
        """
        Launch the MuJoCo viewer and control loop for the simulation.
        """
        with mujoco.viewer.launch_passive(self.mjmodel, self.mjdata, show_left_ui=False, show_right_ui=False) as viewer:

            self.random_placement()
            self.reset_data()

            while viewer.is_running():       

                step_start = time.time()
                self.record_data()

                tau = self.controller.get_tau(self.mjmodel, self.mjdata, self.target_pos, self.target_rot)
                self.mjdata.ctrl[self.actuator_ids] = tau[self.actuator_ids]

                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()

                if (self.done() or self.fell() or self.button) and time.time() - self.placement_time > 2.0:
                    if not self.fell() and not self.button:
                        self.record_data()
                        self.save_data()
                    self.reset_data()
                    self.random_placement()
                    self.placement_time = time.time()
                
                self.target_pos[:2] = self.mjdata.site("eef_target").xpos[:2]
                    
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def record_data(self):

        if not self.record or self.camera_data is None:
            return

        if self.camera_data is not None and self.last_recording_time != -1 and time.time()-self.last_recording_time < (1/self.recording_frequency):
            return
        
        if self.record_start_time == None:
            self.record_start_time = time.time()
            time_diff = 0.0
        else:
            time_diff = time.time() - self.record_start_time

        pose = np.identity(4)
        pose[0:3,3] = self.mjdata.site(self.site_id).xpos.copy()
        pose[0:3,0:3] = self.mjdata.site(self.site_id).xmat.copy().reshape((3,3))

        q = self.mjdata.qpos[self.dof_ids].copy()
        dq = self.mjdata.qvel[self.dof_ids].copy()
        
        camera1_rgb = self.camera_data[self.cameras[0]+"_rgb"]
        camera1_depth = self.camera_data[self.cameras[0]+"_depth"]
        camera2_rgb = self.camera_data[self.cameras[1]+"_rgb"]
        camera2_depth = self.camera_data[self.cameras[1]+"_depth"]
        camera3_rgb = self.camera_data[self.cameras[2]+"_rgb"]
        camera3_depth = self.camera_data[self.cameras[2]+"_depth"] 

        self.camera1_rgbs.append(camera1_rgb)
        self.camera1_depths.append(camera1_depth)
        self.camera2_rgbs.append(camera2_rgb)
        self.camera2_depths.append(camera2_depth)
        self.camera3_rgbs.append(camera3_rgb)
        self.camera3_depths.append(camera3_depth)
        self.poses.append(pose)
        self.times.append(time_diff)
        self.q.append(q)
        self.dq.append(dq)

        self.last_recording_time = time.time()

    def save_data(self):

        if not self.record:
            return

        new_file_name = "Data/" + str(get_latest_number("Data")+1)+".npz"
        camera1_rgbs = np.array(self.camera1_rgbs)
        camera1_depths = np.array(self.camera1_depths)
        camera2_rgbs = np.array(self.camera2_rgbs)
        camera2_depths = np.array(self.camera2_depths)
        camera3_rgbs = np.array(self.camera3_rgbs)
        camera3_depths = np.array(self.camera3_depths)
        poses = np.array(self.poses)
        times = np.array(self.times)
        q = np.array(self.q)
        dq = np.array(self.dq)

        np.savez(new_file_name, camera1_rgbs=camera1_rgbs, camera1_depths=camera1_depths, camera2_rgbs=camera2_rgbs, camera2_depths=camera2_depths, camera3_rgbs=camera3_rgbs, camera3_depths=camera3_depths, poses=poses, times=times, q=q, dq=q)

    def reset_data(self):
        self.camera1_rgbs = []
        self.camera1_depths = []
        self.camera2_rgbs = []
        self.camera2_depths = []
        self.camera3_rgbs = []
        self.camera3_depths = []
        self.poses = []
        self.times = []
        self.q = []
        self.dq = []
        self.record_start_time = time.time()

    def run_poses_from_npz(self, npz_file_path):

        data = np.load(npz_file_path)
        poses = data['poses']
        times = data['times']

        self.random_placement()
        with mujoco.viewer.launch_passive(self.mjmodel, self.mjdata, show_left_ui=False, show_right_ui=False) as viewer:
          
            start_time = time.time()
            data_time = times[0]

            i = 1

            while i<len(poses)-1:

                step_start = time.time()
                
                # Set the pose
                if (time.time()-start_time) - (times[i]-data_time) >= 0:
                    print(i)
                    self.target_pos = poses[i][0:3,3]
                    set_site_pose(self.mjmodel,"eef_target",poses[i][0:3,3])
                    i += 1
                
                tau = self.controller.get_tau(self.mjmodel,self.mjdata,self.target_pos,self.target_rot,False)
                self.mjdata.ctrl[self.actuator_ids] = tau[self.actuator_ids]

                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()
                

            while True:
                tau = self.controller.get_tau(self.mjmodel,self.mjdata,self.target_pos,self.target_rot,False)
                self.mjdata.ctrl[self.actuator_ids] = tau[self.actuator_ids]
                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()

if __name__ == "__main__":

    sim = Simulation()
    sim.start()

    while True:
        sim.camera_data = sim.get_camera_data()