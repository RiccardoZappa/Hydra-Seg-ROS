#!/usr/bin/env python
import hydra_seg_ros.nodes.yolo_ros as yolo_ros
import hydra_seg_ros.nodes.mask2former_ros as mask2former_ros

if __name__ == "__main__":
    mask2former_ros.main()
    # yolo_ros.main()
