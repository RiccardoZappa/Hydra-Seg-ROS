import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from sensor_msgs.msg import Image

# This is the same full COCO class dictionary. It's important to have a
# consistent color mapping for your debugging.
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


class PanopticVisualizer:
    def __init__(self):
        rospy.init_node('panoptic_visualizer_node', anonymous=True)
        rospy.loginfo("Starting Panoptic Visualizer Node.")
        
        # --- Parameters ---
        self.panoptic_id_multiplier = 1000
        # Create a color map for visualization
        self.color_map = self.create_color_map()

        # --- Subscriber ---
        # Subscribes to the machine-readable panoptic map
        self.panoptic_sub = rospy.Subscriber(
            "/mask2former_ros_node/panoptic_label", # This must match the publisher topic
            Image, 
            self.map_callback
        )
        
        # --- Publisher ---
        # Publishes the human-readable color image for RViz
        self.viz_pub = rospy.Publisher("~visualization_image", Image, queue_size=10)
        
        self.bridge = CvBridge()

    def create_color_map(self):
        # Creates a consistent color for every possible semantic class
        np.random.seed(42)
        num_classes = len(COCO_PANOPTIC_CLASSES)
        # We make it large enough for any potential ID
        color_map = np.random.randint(0, 255, size=(num_classes * 2, 3), dtype=np.uint8)
        color_map[0] = [0, 0, 0] # Ensure background is black
        return color_map

    def map_callback(self, panoptic_msg: Image):
        try:
            # Convert the incoming 32SC1 message to an int32 NumPy array
            panoptic_map = self.bridge.imgmsg_to_cv2(panoptic_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # --- The Visualization Logic ---
        
        # Step 1: Decode the semantic ID from the panoptic ID for coloring
        # We use integer division to get the semantic part of the ID.
        semantic_map = (panoptic_map // self.panoptic_id_multiplier)

        # Step 2: Convert the ID map to a color image
        # This is a powerful NumPy feature. For each pixel in semantic_map, it looks up
        # the corresponding color in self.color_map and builds a new 3-channel image.
        color_image = self.color_map[semantic_map]

        # Step 3: Publish the new color image
        try:
            # Convert the colorized NumPy array back to a ROS message with 'bgr8' encoding for RViz
            viz_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            viz_msg.header = panoptic_msg.header # Keep the same timestamp and frame
            self.viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

def main():
    PanopticVisualizer()
    rospy.spin()

if __name__ == '__main__':
    main()
