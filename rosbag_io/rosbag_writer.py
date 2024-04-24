import rosbag2_py
from rclpy.serialization import serialize_message

from cv_bridge import CvBridge
import cv2 as cv

from rosbag_io.rosbag_common import get_rosbag_options, create_topic

class RosbagWriter():
    def __init__(self, bag_path: str, write_compressed: bool, storage_id: str) -> None:
        self.bag_path = bag_path
        self.write_compressed = write_compressed

        self.storage_id = storage_id
        
        storage_options, converter_options = get_rosbag_options(self.bag_path, self.storage_id)
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(storage_options, converter_options)

        self.type_map = {}

        self.bride = CvBridge()

    def __dell__(self):
        self.writer.close()

    def write_image(self, image, topic_name, timestamp):
        if topic_name not in self.type_map:
            create_topic(self.writer, topic_name, 'sensor_msgs/msg/Image' if not self.write_compressed else 'sensor_msgs/msg/CompressedImage')
            self.type_map[topic_name] = 'sensor_msgs/msg/Image' if not self.write_compressed else 'sensor_msgs/msg/CompressedImage'
        
        if self.write_compressed:
            image_msg = self.bride.cv2_to_compressed_imgmsg(image)
            self.writer.write(topic_name, serialize_message(image_msg), timestamp)
        else:
            image_msg = self.bride.cv2_to_imgmsg(image)
            self.writer.write(topic_name, serialize_message(image_msg), timestamp)
    def write_any(self, msg, msg_type, topic_name, timestamp):
        if topic_name not in self.type_map:
            create_topic(self.writer, topic_name, msg_type)
            
        self.writer.write(topic_name, serialize_message(msg), timestamp)