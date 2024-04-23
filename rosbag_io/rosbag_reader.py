import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image

from rosbag_io.rosbag_common import get_rosbag_options, wait_for, RosMessage

class RosbagReader():
    def __init__(self, bag_path) -> None:
        self.bag_path = bag_path
        self.storage_id = 'mcap' if bag_path.endswith('.mcap') else 'sqlite3'
        
        storage_options, converter_options = get_rosbag_options(self.bag_path, self.storage_id)
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)

        self.type_map = self.create_type_map()

    def __dell__(self):
        self.reader.close()

    def __iter__(self):
        return self
    
    def __next__(self) -> RosMessage:
        if self.reader.has_next():
            (topic, data, t) = self.reader.read_next()
            if self.type_map[topic] == 'sensor_msgs/msg/CompressedImage':
                return RosMessage(topic, self.type_map[topic], deserialize_message(data, get_message(self.type_map[topic])), t)
            elif self.type_map[topic] == 'sensor_msgs/msg/Image':
                return RosMessage(topic, self.type_map[topic], deserialize_message(data, get_message(self.type_map[topic])), t)
            else:
                return self.__next__()
        else:
            raise StopIteration
    
    def __enter__(self):
        return self
    
    def create_type_map(self):
        topic_types = self.reader.get_all_topics_and_types()
        return {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    
    def get_type_map(self):
        return self.type_map
