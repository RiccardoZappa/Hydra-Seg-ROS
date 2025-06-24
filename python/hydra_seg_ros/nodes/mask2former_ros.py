# Standard Imports
import message_filters
import rospy
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
import numpy as np
import cv2
from scipy.special import softmax

import onnxruntime as ort

# ROS Message Imports
from sensor_msgs.msg import Image, CameraInfo
from hydra_msgs.msg import HydraVisionPacket, Masks 

# Utility Imports
from hydra_seg_ros.utils import ros_utils

class Mask2FormerRosNode:
    """
    A ROS node to perform panoptic segmentation using a Mask2Former model
    and publish the results for a mapping framework like Hydra.
    """
    def __init__(self):
        rospy.init_node("mask2former_ros_node")
        self.init_ros()
        rospy.loginfo("Starting Mask2FormerRosNode.")

    def init_ros(self):
        # --- MODEL CONFIGURATION ---
        # Get parameters from the ROS Parameter Server
        self.model_path = rospy.get_param(
            "~model_path", "/path/to/your/mask2former_panoptic_tiny.onnx"
        )
        self.custom_op_path = rospy.get_param(
            "~custom_op_path", "/path/to/your/libmmdeploy_onnxruntime_ops.so"
        )
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.9)

        # --- LOAD THE ONNX MODEL ---
        rospy.loginfo(f"Loading ONNX model from: {self.model_path}")
        so = ort.SessionOptions()
        so.register_custom_ops_library(self.custom_op_path)
        self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=['CUDAExecutionProvider'])
        
        # Get model input/output details once
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        _, _, self.input_height, self.input_width = self.session.get_inputs()[0].shape
        rospy.loginfo("Model loaded successfully.")

        # --- ROS SUBSCRIBERS ---
        # These subscribe to the synchronized camera topics
        self.cam_info_sub = message_filters.Subscriber("~cam_info", CameraInfo)
        self.color_sub = message_filters.Subscriber("~colors", Image)
        self.depth_sub = message_filters.Subscriber("~depth", Image)

        # --- ROS PUBLISHERS ---
        # This now publishes our machine-readable panoptic map for debugging in RViz
        self.panoptic_label_pub = rospy.Publisher("~panoptic_label", Image, queue_size=10)
        # This publishes the full data packet for Hydra
        self.vision_packet_pub = rospy.Publisher(
            "~vision_packet", HydraVisionPacket, queue_size=10
        )
        self.cam_info_pub = rospy.Publisher("~camera_info", CameraInfo, queue_size=10)

        # Counters
        self.map_view_cnt: int = 0
        
        # --- SYNCHRONIZER AND BRIDGE  ---
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.cam_info_sub, self.color_sub, self.depth_sub],
            10,
            0.1,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(self.vision_callback)
        self.bridge = CvBridge()
        
        rospy.loginfo("ROS publishers and subscribers initialized.")

    def _extract_panoptic_data(self, class_logits, mask_logits, original_image_shape):

        class_probs = softmax(class_logits, axis=-1)
        scores = np.max(class_probs, axis=-1)
        class_ids = np.argmax(class_probs, axis=-1)
        
        confident_detections = scores > self.conf_threshold
        
        if not np.any(confident_detections):
            return {"panoptic_map": np.zeros(original_image_shape, dtype=np.uint16)}

        scores = scores[confident_detections]
        class_ids = class_ids[confident_detections]
        mask_logits = mask_logits[confident_detections]

        full_res_masks = np.zeros((len(scores), original_image_shape[0], original_image_shape[1]), dtype=np.bool_)
        for i in range(len(scores)):
            mask = cv2.resize(mask_logits[i], (original_image_shape[1], original_image_shape[0]), interpolation=cv2.INTER_LINEAR)
            full_res_masks[i] = mask > 0

        sorted_indices = np.argsort(scores)[::-1]
        panoptic_map = np.zeros(original_image_shape, dtype=np.uint16)
        
        for i in sorted_indices:
            panoptic_map[full_res_masks[i]] = class_ids[i] + 1
            
        return {"panoptic_map": panoptic_map}

    def vision_callback(
        self, cam_info_msg: CameraInfo, color_msg: Image, depth_msg: Image
    ):
        # --- 1. Get Image from ROS ---
        color_cv = self.bridge.imgmsg_to_cv2(color_msg)
        original_shape = color_cv.shape[:2]

        # --- 2. Pre-process Image ---
        # Resize, normalize, and change layout for the model
        input_size = (self.input_width, self.input_height)
        preprocessed_img = cv2.resize(color_cv, input_size)
        preprocessed_img = preprocessed_img.astype(np.float32) / 255.0
        preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1))
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        
        # --- 3. Run Inference ---
        raw_results = self.session.run(self.output_names, {self.input_name: preprocessed_img})
        # Assuming the first two outputs are class and mask logits.
        cls_logits = raw_results[0][0]
        mask_logits = raw_results[1][0]
        
        # --- 4. Post-Process Data ---
        extracted_data = self._extract_panoptic_data(cls_logits, mask_logits, original_shape)
        panoptic_map = extracted_data["panoptic_map"]
        
        # --- 5. Prepare and Publish ROS Message ---
        try:

            panoptic_label_msg = self.bridge.cv2_to_imgmsg(panoptic_map, encoding="16UC1")
            panoptic_label_msg.header = color_msg.header # Use same timestamp and frame
            
            empty_masks_msg = Masks()
            
            cam_info_msg_pub, vision_packet_msg = ros_utils.pack_vision_msgs(
                self.map_view_cnt, 
                cam_info_msg, 
                color_msg, 
                depth_msg, 
                panoptic_label_msg,
                empty_masks_msg
            )
            
            self.cam_info_pub.publish(cam_info_msg_pub)
            self.vision_packet_pub.publish(vision_packet_msg)
            self.panoptic_label_pub.publish(panoptic_label_msg)
            
            self.map_view_cnt += 1
            
        except CvBridgeError as e:
            rospy.logerr(e)

def main():
    Mask2FormerRosNode()
    rospy.spin()