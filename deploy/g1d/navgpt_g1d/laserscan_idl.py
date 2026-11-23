"""sensor_msgs/msg/LaserScan for CycloneDDS (unitree_sdk2py ships no LaserScan IDL).

Field order and types follow ROS 2 sensor_msgs/msg/LaserScan, so samples from a
ROS 2 publisher such as the Slamware node decode directly.
"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LaserScan_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.LaserScan_"):
    header: 'unitree_sdk2py.idl.std_msgs.msg.dds_.Header_'
    angle_min: types.float32
    angle_max: types.float32
    angle_increment: types.float32
    time_increment: types.float32
    scan_time: types.float32
    range_min: types.float32
    range_max: types.float32
    ranges: types.sequence[types.float32]
    intensities: types.sequence[types.float32]
