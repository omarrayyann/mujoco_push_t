# MuJoCo PushT Task
**Part of the [MujocoAR](https://github.com/omarrayyann/MujocoAR) package demos**

A MuJoCo simulation environment of the pushT task. The goal is to push the T over the T outline. The simulation includes an operational space controller to handle the movement of the KUKA-iiwa14 arm with a tool at the end.


<table>
<!--   <tr>
    <th><code>camera_name="whole_view"</code></th>
    <th><code>camera_name="top_view"</code></th>
    <th><code>camera_name="side_view"</code></th>
    <th><code>camera_name="front_view"</code></th>
  </tr> -->
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7a00b71e-7d81-40bd-aee6-709feb0e9a82" width="800px" /></td>
    <td><img src="https://github.com/user-attachments/assets/c1e927c5-a4af-4c95-a6d0-fe7f8a026c34" width="800px" /></td>
    <td><img src="https://github.com/user-attachments/assets/a58ed764-4e05-40a5-b26a-5bd896584f34" width="800px" /></td>
  </tr>
</table>


## MuJoCo AR Setup

```python
# Initializing MuJoCo AR
self.mujocoAR = MujocoARConnector(mujoco_model=self.mjmodel,mujoco_data=self.mjdata)

# Linking a Target Site with the AR Position
self.mujocoAR.link_site(
  name="eef_target",
  scale=2.0,
  translation=self.pos_origin,
  button_fn=lambda: (self.random_placement(), setattr(self, 'placement_time', time.time()), self.reset_data()) if time.time() - self.placement_time > 2.0 else None,
  disable_rot=True,
)

# Start!
self.mujocoAR.start()
```

## Usage Guide

1. **Clone the repository**:

   ```bash
   git clone https://github.com/omarrayyann/mujoco_push_t.git
   cd mujoco_push_t
   
3. **Install MujocoAR and othe Requirements**:
   ```bash
   
   pip install requirements.txt
   
4. **Download the [MuJoCo AR App](https://apps.apple.com/jo/app/past-code/id1551535957) from the App Store.**
   
5. **Run the application**:

   ```bash
   mjpython main.py
   
6. **Enter the IP and Port shown into the app's start screen to start. Make sure to be connected to the same Wi-Fi network as the device. Incase of a latency, I recommend connecting to your phone's hotspot.**

## Author

Omar Rayyan (olr7742@nyu.edu)
