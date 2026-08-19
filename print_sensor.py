import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
import omni.timeline
from isaacsim.sensors.physx import _range_sensor
import asyncio

# Initialize ROS2 only once
if not rclpy.ok():
    rclpy.init()

class LightBeamRos2Bridge(Node):
    def __init__(self):
        super().__init__('lightbeam_bridge')
        self.distance_pub = self.create_publisher(Float32, '/lightbeam/distance', 10)
        self.hit_pub = self.create_publisher(Bool, '/lightbeam/hit', 10)
        self.ls = _range_sensor.acquire_lightbeam_sensor_interface()
        self.timeline = omni.timeline.get_timeline_interface()
        self.sensor_path = "/World/Camera/LightBeam_Sensor"
        print('LightBeam Bridge Started - Publishing to /lightbeam/distance and /lightbeam/hit')
        
    async def run(self):
        while True:
            if self.timeline.is_playing():
                try:
                    depth = self.ls.get_linear_depth_data(self.sensor_path)
                    hit = self.ls.get_beam_hit_data(self.sensor_path).astype(bool)
                    if len(depth) > 0:
                        self.distance_pub.publish(Float32(data=float(depth[0])))
                        self.hit_pub.publish(Bool(data=bool(hit[0])))
                except:
                    pass
            await asyncio.sleep(0.05)

# Create and run node
node = LightBeamRos2Bridge()
asyncio.ensure_future(node.run())
print("Bridge running. Press PLAY to start publishing.")