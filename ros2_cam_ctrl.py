#!/usr/bin/env python3
"""
Complete Camera Controller + LightBeam Sensor + 3D Point Calculator
- Uses QUATERNIONS directly from Isaac Sim for computation (no Euler conversion)
- Display shows Euler angles (for user reference only)
- Red button to capture 3D points where beam hits
- Press 'M' to maximize window, 'X' to quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from simulation_interfaces.srv import GetEntityState, SetEntityState
import cv2
import numpy as np
import time
import math


class CameraControlViewer(Node):
    def __init__(self):
        super().__init__('camera_control_viewer')
        
        # === ROS2 Subscribers and Clients ===
        self.subscription = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.distance_sub = self.create_subscription(Float32, '/lightbeam/distance', self.distance_callback, 10)
        self.hit_sub = self.create_subscription(Bool, '/lightbeam/hit', self.hit_callback, 10)
        
        self.get_state_client = self.create_client(GetEntityState, '/get_entity_state')
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        # Wait for services
        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            pass
        
        self.set_state_client.wait_for_service(timeout_sec=10.0)
        
        # === Camera Configuration ===
        self.camera_entity = '/World/Camera'
        self.translation_step = 0.02
        self.orientation_step = 1.0  # degrees
        
        # Current camera state
        self.current_position = None  # (x, y, z)
        
        # Store RAW quaternion directly from Isaac Sim (for computation)
        self.current_quat = None  # (x, y, z, w) - EXACT values from Transform
        
        # Store Euler for display only (converted from quaternion, no modifications)
        self.current_euler_display = None  # (roll, pitch, yaw) in degrees
        
        # LightBeam data
        self.beam_distance = 0.0
        self.beam_hit = False
        
        # Captured points storage
        self.captured_points = []
        self.button_rect = None
        
        # === FPS Counter ===
        self.fps = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # UI Configuration
        self.button_x_offset = 250
        self.button_y_offset = 20
        self.button_diameter = 50
        self.panel_width_multiplier = 4
        self.panel_height = 50
        
        # === Get initial camera pose ===
        self.get_camera_state()
        
        # === OpenCV Window ===
        self.window_name = 'Camera Control'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.display_instructions()
        self.keyboard_loop()
    
    def distance_callback(self, msg):
        self.beam_distance = msg.data
    
    def hit_callback(self, msg):
        self.beam_hit = msg.data
    
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (degrees) - for DISPLAY only"""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (degrees) to quaternion"""
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        
        return (qx, qy, qz, qw)
    
    def get_camera_state(self):
        """Get current camera position and RAW quaternion from Isaac Sim"""
        request = GetEntityState.Request()
        request.entity = self.camera_entity
        
        try:
            future = self.get_state_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            response = future.result()
            if response and response.result:
                # Get position
                pos = response.state.pose.position
                self.current_position = (pos.x, pos.y, pos.z)
                
                # Get RAW quaternion directly from Isaac Sim (for computation)
                quat = response.state.pose.orientation
                self.current_quat = (quat.x, quat.y, quat.z, quat.w)
                
                # Convert to Euler for display only (no modifications)
                self.current_euler_display = self.quaternion_to_euler(
                    quat.x, quat.y, quat.z, quat.w
                )
                
                return True
            return False
        except Exception as e:
            return False
    
    def set_camera_state(self, new_position=None, new_quat=None):
        """Update camera position and/or orientation in Isaac Sim using quaternion"""
        request = SetEntityState.Request()
        request.entity = self.camera_entity
        
        # Set position
        if new_position:
            request.state.pose.position.x = float(new_position[0])
            request.state.pose.position.y = float(new_position[1])
            request.state.pose.position.z = float(new_position[2])
            self.current_position = new_position
        else:
            request.state.pose.position.x = float(self.current_position[0])
            request.state.pose.position.y = float(self.current_position[1])
            request.state.pose.position.z = float(self.current_position[2])
        
        # Set orientation using RAW quaternion
        if new_quat:
            request.state.pose.orientation.x = float(new_quat[0])
            request.state.pose.orientation.y = float(new_quat[1])
            request.state.pose.orientation.z = float(new_quat[2])
            request.state.pose.orientation.w = float(new_quat[3])
            self.current_quat = new_quat
            # Update display Euler from new quaternion
            self.current_euler_display = self.quaternion_to_euler(
                new_quat[0], new_quat[1], new_quat[2], new_quat[3]
            )
        else:
            request.state.pose.orientation.x = float(self.current_quat[0])
            request.state.pose.orientation.y = float(self.current_quat[1])
            request.state.pose.orientation.z = float(self.current_quat[2])
            request.state.pose.orientation.w = float(self.current_quat[3])
        
        try:
            future = self.set_state_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            response = future.result()
            return response and response.result
        except Exception as e:
            return False
    
    def rotate_by_euler_delta(self, delta_roll, delta_pitch, delta_yaw):
        """
        Apply delta rotation (in degrees) by converting current quaternion,
        adding delta in Euler space, converting back to quaternion.
        This keeps computation stable and maintains exact local axis control.
        """
        # Convert current quaternion to Euler
        current_euler = self.quaternion_to_euler(
            self.current_quat[0], self.current_quat[1], 
            self.current_quat[2], self.current_quat[3]
        )
        
        # Add delta to each axis
        new_roll = current_euler[0] + delta_roll
        new_pitch = current_euler[1] + delta_pitch
        new_yaw = current_euler[2] + delta_yaw
        
        # Convert back to quaternion
        new_quat = self.euler_to_quaternion(new_roll, new_pitch, new_yaw)
        
        # Send to Isaac Sim
        return self.set_camera_state(new_quat=new_quat)
    
    def translate_x(self, direction):
        new_x = self.current_position[0] + (direction * self.translation_step)
        new_position = (new_x, self.current_position[1], self.current_position[2])
        self.set_camera_state(new_position=new_position)
        return True
    
    def translate_y(self, direction):
        new_y = self.current_position[1] + (direction * self.translation_step)
        new_position = (self.current_position[0], new_y, self.current_position[2])
        self.set_camera_state(new_position=new_position)
        return True
    
    def translate_z(self, direction):
        new_z = self.current_position[2] + (direction * self.translation_step)
        new_position = (self.current_position[0], self.current_position[1], new_z)
        self.set_camera_state(new_position=new_position)
        return True
    
    def rotate_roll(self, direction):
        """Rotate around X axis (Roll) - ONLY affects roll"""
        delta = direction * self.orientation_step
        return self.rotate_by_euler_delta(delta, 0, 0)
    
    def rotate_pitch(self, direction):
        """Rotate around Y axis (Pitch) - ONLY affects pitch"""
        delta = direction * self.orientation_step
        return self.rotate_by_euler_delta(0, delta, 0)
    
    def rotate_yaw(self, direction):
        """Rotate around Z axis (Yaw) - ONLY affects yaw"""
        delta = direction * self.orientation_step
        return self.rotate_by_euler_delta(0, 0, delta)
    
    def calculate_3d_point(self):
        """
        Calculate 3D point using RAW quaternion directly from Isaac Sim.
        NO Euler conversion for computation - uses quaternion rotation matrix.
        """
        tx, ty, tz = self.current_position
        distance = self.beam_distance
        
        # Get RAW quaternion from Isaac Sim (exactly as it is)
        qx, qy, qz, qw = self.current_quat
        
        # Normalize quaternion
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm
        
        # Build rotation matrix directly from quaternion
        # This gives the exact orientation as in Isaac Sim Transform section
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        xw = qx * qw
        yz = qy * qz
        yw = qy * qw
        zw = qz * qw
        
        R00 = 1 - 2 * (yy + zz)
        R01 = 2 * (xy - zw)
        R02 = 2 * (xz + yw)
        
        R10 = 2 * (xy + zw)
        R11 = 1 - 2 * (xx + zz)
        R12 = 2 * (yz - xw)
        
        R20 = 2 * (xz - yw)
        R21 = 2 * (yz + xw)
        R22 = 1 - 2 * (xx + yy)
        
        # Camera's forward direction is along -Z (in camera local coordinates)
        # Apply rotation matrix to get world direction
        dx = R02 * (-1)  # -Z in camera space = forward in world space
        dy = R12 * (-1)
        dz = R22 * (-1)
        
        # Calculate 3D point
        point_x = tx + distance * dx
        point_y = ty + distance * dy
        point_z = tz + distance * dz
        
        return (round(point_x, 3), round(point_y, 3), round(point_z, 3))
    
    def capture_point(self):
        if self.beam_hit and self.beam_distance > 0:
            point = self.calculate_3d_point()
            self.captured_points.append(point)
            # Print in simple format: X Y Z
            print(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}")
    
    def draw_center_circle(self, img):
        height, width = img.shape[:2]
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 140
        if radius < 2:
            radius = 2
        dark_pink = (147, 20, 255)
        cv2.circle(img, (center_x, center_y), radius, dark_pink, 2)
        return img
    
    def display_instructions(self):
        print("\n" + "="*60)
        print("CAMERA CONTROL - QUATERNION-BASED COMPUTATION")
        print("="*60)
        print("\nTRANSLATION (Step: 2cm):")
        print("  Z (forward/back): Q / A")
        print("  Y (up/down):      W / S")
        print("  X (left/right):   E / D")
        print("\nORIENTATION (Step: 1 degree, local axes):")
        print("  Yaw (rotate):     Y / H")
        print("  Pitch (tilt):     U / J")
        print("  Roll (side):      I / K")
        print("\nCONTROLS:")
        print("  Red Circle:       Click to capture 3D point")
        print("  M:                Maximize window")
        print("  X:                Quit")
        print("="*60 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.button_rect:
            bx, by, br = self.button_rect
            if (x - bx) ** 2 + (y - by) ** 2 <= br ** 2:
                self.capture_point()
    
    def image_callback(self, msg):
        self.fps_counter += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = now
        
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            h, w = img.shape[:2]
            
            # Draw center circle
            img = self.draw_center_circle(img)
            
            # Draw red capture button
            button_radius = self.button_diameter // 2
            button_center_x = w - self.button_x_offset
            button_center_y = self.button_y_offset + button_radius
            self.button_rect = (button_center_x, button_center_y, button_radius)
            
            cv2.circle(img, (button_center_x, button_center_y), button_radius, (255,255,255), 2)
            cv2.circle(img, (button_center_x, button_center_y), button_radius - 3, (0,0,255), -1)
            
            # Panel for points display
            panel_width = self.button_diameter * self.panel_width_multiplier
            panel_x = button_center_x + button_radius + 10
            panel_y = button_center_y - button_radius
            panel_h = self.button_diameter
            
            img[panel_y:panel_y+panel_h, panel_x:panel_x+panel_width] = 0
            cv2.rectangle(img, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_h), (255,255,255), 1)
            cv2.putText(img, "POINTS", (panel_x+5, panel_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            
            # Display captured points
            for i, p in enumerate(self.captured_points[-4:]):
                idx = len(self.captured_points) - 4 + i + 1 if len(self.captured_points) > 4 else i + 1
                text = f"{idx}:({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
                cv2.putText(img, text, (panel_x+5, panel_y+35+i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            
            # Display camera info
            y_offset = 30
            line_height = 28
            
            cv2.putText(img, f'FPS: {self.fps}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            y_offset += line_height
            
            if self.current_position:
                pos_text = f'Pos: X={self.current_position[0]:.2f} Y={self.current_position[1]:.2f} Z={self.current_position[2]:.2f}'
                cv2.putText(img, pos_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            y_offset += line_height
            
            # Display Euler angles (from quaternion - NO modifications)
            if self.current_euler_display:
                euler_text = f'Angles: Roll={self.current_euler_display[0]:.1f} Pitch={self.current_euler_display[1]:.1f} Yaw={self.current_euler_display[2]:.1f}'
                cv2.putText(img, euler_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            y_offset += line_height
            
            # Beam distance
            if self.beam_hit:
                beam_text = f'Distance: {self.beam_distance:.3f} m'
                cv2.putText(img, beam_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            else:
                cv2.putText(img, 'Distance: NO HIT', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
            
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)
            
        except Exception as e:
            pass
    
    def keyboard_loop(self):
        while rclpy.ok():
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('x') or key == ord('X'):
                break
            
            if key == ord('m') or key == ord('M'):
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Translation
            if key == ord('q'): self.translate_z(+1)
            if key == ord('a'): self.translate_z(-1)
            if key == ord('w'): self.translate_y(+1)
            if key == ord('s'): self.translate_y(-1)
            if key == ord('e'): self.translate_x(+1)
            if key == ord('d'): self.translate_x(-1)
            
            # Orientation
            if key == ord('y'): self.rotate_yaw(+1)
            if key == ord('h'): self.rotate_yaw(-1)
            if key == ord('u'): self.rotate_pitch(+1)
            if key == ord('j'): self.rotate_pitch(-1)
            if key == ord('i'): self.rotate_roll(+1)
            if key == ord('k'): self.rotate_roll(-1)
            
            self.get_camera_state()
            rclpy.spin_once(self, timeout_sec=0.01)
        
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraControlViewer()
    rclpy.spin(node)


if __name__ == '__main__':
    main()