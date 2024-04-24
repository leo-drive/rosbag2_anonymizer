import time
from typing import Callable

from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
import rosbag2_py

class RosMessage:
    def __init__(self, topic, type, data, timestamp):
        self.topic = topic
        self.type = type
        self.data = data
        self.timestamp = timestamp

    def __repr__(self) -> str:
        f"topic: {self.topic} | type: {self.type} | timestamp: {self.timestamp} | data: {self.data}"

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def wait_for(
    condition: Callable[[], bool],
    timeout: Duration,
    sleep_time: float = 0.1,
):
    clock = Clock(clock_type=ClockType.STEADY_TIME)
    start = clock.now()
    while not condition():
        if clock.now() - start > timeout:
            return False
        time.sleep(sleep_time)
        return True
    
def create_topic(writer, topic_name, topic_type, serialization_format='cdr'):
    """
    Create a new topic.

    :param writer: writer instance
    :param topic_name:
    :param topic_type:
    :param serialization_format:
    :return:
    """
    topic_name = topic_name
    topic = rosbag2_py.TopicMetadata(name=topic_name, type=topic_type,
                                     serialization_format=serialization_format)

    writer.create_topic(topic)