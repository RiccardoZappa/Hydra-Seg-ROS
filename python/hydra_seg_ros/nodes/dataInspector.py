#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from hydra_msgs.msg import HydraVisionPacket

COCO_PANOPTIC_CLASSES = {
    # --- "Things" - Objects with Instances ---
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
    # --- "Stuff" - Background Regions ---
    80: 'banner',
    81: 'blanket',
    82: 'bridge',
    83: 'cardboard',
    84: 'counter',
    85: 'curtain',
    86: 'door-stuff',
    87: 'floor-wood',
    88: 'flower',
    89: 'fruit',
    90: 'gravel',
    91: 'house',
    92: 'light',
    93: 'mirror-stuff',
    94: 'net',
    95: 'pillow',
    96: 'platform',
    97: 'playingfield',
    98: 'railroad',
    99: 'river',
    100: 'road',
    101: 'roof',
    102: 'sand',
    103: 'sea',
    104: 'shelf',
    105: 'snow',
    106: 'stairs',
    107: 'tent',
    108: 'towel',
    109: 'wall-brick',
    110: 'wall-stone',
    111: 'wall-tile',
    112: 'wall-wood',
    113: 'water-other',
    114: 'window-blind',
    115: 'window-other',
    116: 'tree-merged',
    117: 'fence-merged',
    118: 'ceiling-merged',
    119: 'sky-other-merged',
    120: 'cabinet-merged',
    121: 'table-merged',
    122: 'floor-other-merged',
    123: 'pavement-merged',
    124: 'mountain-merged',
    125: 'grass-merged',
    126: 'dirt-merged',
    127: 'paper-merged',
    128: 'food-other-merged',
    129: 'building-other-merged',
    130: 'rock-merged',
    131: 'wall-other-merged',
    132: 'rug-merged',
}


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
        self.panoptic_id_multiplier = 1000 

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


        print(f"[INFO] Received Panoptic Map:")
        print(f"  - Shape: {panoptic_map.shape}")
        print(f"  - Data Type: {panoptic_map.dtype}") # Should be int32
        
        unique_ids = np.unique(panoptic_map)
        print(f"  - Raw Panoptic IDs found: {unique_ids}")
        
        print("\n[INFO] Decoding Panoptic IDs into Semantic and Instance info:")
        for panoptic_id in unique_ids:
            if panoptic_id == 0:
                print("  - ID 0: Background")
                continue
            
            # --- THE DECODING LOGIC ---
            # This is the inverse of the formula from your perception node.
            # This is exactly what Hydra's C++ code will do.
            semantic_id = (panoptic_id // self.panoptic_id_multiplier) - 1
            instance_id = panoptic_id % self.panoptic_id_multiplier
            
            # Look up the class name for readability
            class_name = COCO_PANOPTIC_CLASSES.get(semantic_id, f"Unknown_Class_{semantic_id}")
            
            print(f"  - ID {panoptic_id}: Decodes to -> Semantic '{class_name}' (ID: {semantic_id}), Instance: {instance_id}")

        print("-" * 40)

def main():
    DataInspector()
    rospy.spin()
