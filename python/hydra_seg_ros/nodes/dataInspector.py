#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from hydra_msgs.msg import HydraVisionPacket

class DataInspector:
    def __init__(self):
        rospy.init_node('data_inspector_node', anonymous=True)
        rospy.loginfo("Starting Data Inspector Node.")
        
        # Subscribe to the output of your perception node
        self.packet_sub = rospy.Subscriber(
            "/mask2former_ros_node/vision_packet", # Adjust topic name as needed
            HydraVisionPacket, 
            self.packet_callback
        )
        
        self.bridge = CvBridge()
        self.received_count = 0

    def packet_callback(self, packet_msg: HydraVisionPacket):
        self.received_count += 1
        rospy.loginfo(f"--- Packet Received (#{self.received_count}) ---")
        
        # Extract the panoptic map from the 'label' field
        panoptic_label_msg = packet_msg.label
        
        # Convert the ROS Image message back to a NumPy array
        try:
            panoptic_map = self.bridge.imgmsg_to_cv2(panoptic_label_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # --- THIS IS YOUR DATA VERIFICATION ---
        # Perform the same inspection we did in the debug script
        print(f"[INFO] Received Panoptic Map:")
        print(f"  - Shape: {panoptic_map.shape}")
        print(f"  - Data Type: {panoptic_map.dtype}")
        unique_ids = np.unique(panoptic_map)
        print(f"  - Unique IDs Present: {unique_ids}")
        print("-" * 20)

def main():
    DataInspector()
    rospy.spin()
