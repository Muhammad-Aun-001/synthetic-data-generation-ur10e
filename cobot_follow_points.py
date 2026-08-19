import omni.usd
from isaacsim.core.prims import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import time
import math

robot = Articulation(prim_paths_expr="/World/ur10e_physics", name="UR10e")
stage = omni.usd.get_context().get_stage()

def euler_to_quaternion(roll, pitch, yaw):
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)
    
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_q = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y_q, z])

def get_cube_position():
    cube_prim = stage.GetPrimAtPath("/World/RedCube")
    if not cube_prim.IsValid():
        return np.array([0, 0, 0])
    xform = UsdGeom.Xformable(cube_prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])

def set_cube_position(x, y, z):
    cube_prim = stage.GetPrimAtPath("/World/RedCube")
    if cube_prim.IsValid():
        xform = UsdGeom.XformCommonAPI(cube_prim)
        xform.SetTranslate((x, y, z))

# Pattern points for spheres (UPDATED)
pattern_points = [
   (0.819, 0.378, 0.275),
   (0.645, 0.378, 0.209),
   (0.645, 0.388, 0.209),
   (0.645, 0.471, 0.269),
   (0.645, 0.420, 0.352),
   (0.645, 0.651, 0.420)
]

# Cube points - 6 points (UPDATED)
cube_points = [
   (0.819, 0.178, 0.275),
   (0.645, 0.178, 0.209),
   (0.445, 0.388, 0.209),
   (0.445, 0.471, 0.269),
   (0.445, 0.420, 0.352),
   (0.445, 0.651, 0.420)
]

# Create red cube
cube = UsdGeom.Cube.Define(stage, "/World/RedCube")
cube.CreateSizeAttr(0.01)
xform_cube = UsdGeom.XformCommonAPI(cube)
xform_cube.SetScale((0.01, 0.01, 0.01))
xform_cube.SetTranslate(cube_points[0])
cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

FIXED_ROLL = -180
FIXED_PITCH = -180

URDF_PATH = "/home/aun/Desktop/ur10e_moveit/ur10e_clean_isaac.urdf"
YAML_PATH = "/home/aun/Desktop/ur10e_moveit/ur10e_follow.yaml"

solver = LulaKinematicsSolver(
    urdf_path=URDF_PATH,
    robot_description_path=YAML_PATH
)

# TIMING - Each segment takes 1 second
move_time = 0.1  # Time per segment in seconds
points_between = 150  # Number of interpolation steps

timeline = omni.timeline.get_timeline_interface()
timeline.play()
time.sleep(1)

try:
    # Create spheres along the pattern points path
    all_sphere_positions = []
    
    for i in range(len(pattern_points) - 1):
        p1 = pattern_points[i]
        p2 = pattern_points[i + 1]
        
        for j in range(points_between + 1):
            t = j / points_between
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            z = p1[2] + (p2[2] - p1[2]) * t
            all_sphere_positions.append((x, y, z))
    
    # Create spheres with scale 0.5 on all axes
    for idx, pos in enumerate(all_sphere_positions):
        sphere_path = f"/World/StaticGreenSphere_{idx}"
        new_sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        new_sphere.CreateRadiusAttr(0.055)
        xform_new = UsdGeom.XformCommonAPI(new_sphere)
        xform_new.SetScale((0.055, 0.055, 0.055))
        xform_new.SetTranslate((pos[0], pos[1], pos[2]))
        new_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
    
    current_yaw = 170
    
    # Loop through each segment (5 segments between 6 points)
    for i in range(len(cube_points) - 1):
        start = cube_points[i]
        end = cube_points[i + 1]
        
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        delta_z = end[2] - start[2]
        
        steps = points_between
        step_duration = move_time / steps
        
        step_x = delta_x / steps
        step_y = delta_y / steps
        step_z = delta_z / steps
        
        # Yaw settings
        if i == 0:
            yaw_start = 170
            yaw_end = 170
        elif i == 1:
            yaw_start = 170
            yaw_end = 85
        else:
            yaw_start = 85
            yaw_end = 85
        
        yaw_step = (yaw_end - yaw_start) / steps
        
        current_x, current_y, current_z = start
        current_yaw = yaw_start
        
        print(f"Moving from point {i} to point {i+1} in {move_time} seconds")
        segment_start = time.time()
        
        for step in range(steps):
            step_start = time.time()
            
            current_x += step_x
            current_y += step_y
            current_z += step_z
            current_yaw += yaw_step
            
            # Move the red cube
            set_cube_position(current_x, current_y, current_z)
            
            # Compute IK and move robot
            current_orientation = euler_to_quaternion(FIXED_ROLL, FIXED_PITCH, current_yaw)
            cube_pos = get_cube_position()
            joints = solver.compute_inverse_kinematics(
                target_position=cube_pos,
                target_orientation=current_orientation,
                frame_name="wrist_2_link"
            )
            
            if joints is not None:
                robot.set_joint_positions(joints[0])
            
            # Precise timing
            elapsed = time.time() - step_start
            if elapsed < step_duration:
                time.sleep(step_duration - elapsed)
            
            omni.kit.app.get_app().update()
        
        segment_end = time.time()
        print(f"Segment {i} ACTUALLY took: {segment_end - segment_start:.3f} seconds")
    
    print("All movements complete! Holding final position...")
    
    # Hold final position
    while True:
        omni.kit.app.get_app().update()
        cube_pos = get_cube_position()
        current_orientation = euler_to_quaternion(FIXED_ROLL, FIXED_PITCH, 85)
        joints = solver.compute_inverse_kinematics(
            target_position=cube_pos,
            target_orientation=current_orientation,
            frame_name="wrist_2_link"
        )
        if joints is not None:
            robot.set_joint_positions(joints[0])
        time.sleep(0.05)
        
except KeyboardInterrupt:
    pass