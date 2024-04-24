import os
import yaml

from tqdm import tqdm

import cv2
import cv_bridge
from PIL import Image
import supervision as sv

import numpy as np

import torch
import torchvision

from rosbag_io.rosbag_reader import RosbagReader
from rosbag_io.rosbag_writer import RosbagWriter

from model.open_clip import OpenClipModel
from model.grounding_dino import GroundingDINO
from model.sam import SAM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == '__main__':
    reader = RosbagReader(config['rosbag']['input_bag_path'])
    writer = RosbagWriter(config['rosbag']['output_bag_paht'], config['rosbag']['output_save_compressed_image'], config['rosbag']['output_storage_id'])

    # GroundingDINO parameters
    GROUNDING_DINO_CONFIG_PATH = config['grounding_dino']['config_path']
    GROUNDING_DINO_CHECKPOINT_PATH = config['grounding_dino']['checkpoint_path']

    CLASSES = config['grounding_dino']['classes']
    BOX_THRESHOLD = config['grounding_dino']['box_threshold']
    TEXT_THRESHOLD = config['grounding_dino']['text_threshold']
    NMS_THRESHOLD = config['grounding_dino']['nms_threshold']

    # Segment-Anything parameters
    SAM_ENCODER_VERSION = config['segment_anything']['encoder_version']
    SAM_CHECKPOINT_PATH = config['segment_anything']['checkpoint_path']

    # OpenClip parameters
    VALIDATION_CLASSES = config['openclip']['classes']

    
    # Grounding DINO
    grounding_dino = GroundingDINO(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)

    # Segment-Anything
    sam = SAM(SAM_ENCODER_VERSION, SAM_CHECKPOINT_PATH, DEVICE)

    # Openclip
    open_clip = OpenClipModel("ViT-B-32", "laion2b_s34b_b79k")

    for i, (msg, is_image) in enumerate(reader):
        if not is_image:
            writer.write_any(msg.data, msg.type, msg.topic, msg.timestamp)
        else:
            image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg.data)

            # Run DINO
            detections = grounding_dino(
                image=image,
                classes=CLASSES,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            # Remove class_id if it is 'None'
            not_nons = [index for index, (_, _, _, class_id, _) in  enumerate(detections) if class_id is not None]
            detections.xyxy = detections.xyxy[not_nons]
            detections.confidence = detections.confidence[not_nons]
            detections.class_id = detections.class_id[not_nons]

            # NMS
            nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # Run open clip
            valid_ids = []
            invalid_ids = []
            for index, (xyxy, mask, confidence, class_id, _) in enumerate(detections):
                detection_image = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                pil_image = Image.fromarray(detection_image)

                scores = open_clip(pil_image, VALIDATION_CLASSES)
                if scores.numpy().tolist()[0][class_id] > config['openclip']['score_threshold']:
                    valid_ids.append(index)
                else:
                    invalid_ids.append(index)

                # Debug ------------------
                # cv2.imshow("as", detection_image)
                # cv2.waitKey(2000)
                # Debug ------------------
            detections.xyxy = detections.xyxy[valid_ids]
            detections.confidence = detections.confidence[valid_ids]
            detections.class_id = detections.class_id[valid_ids]

            # Run SAM
            detections = sam(image=image, detections=detections)

            # Blur detections
            blurred_img = cv2.GaussianBlur(image, (config['blur']['kernel_size'], config['blur']['kernel_size']), config['blur']['sigma_x'])
            output = image.copy()
            for xyxy, mask, confidence, class_id, _ in detections:
                output[mask] = blurred_img[mask]

            writer.write_image(output, msg.topic, msg.timestamp)

            # Debug ------------------
            # box_annotator = sv.BoxAnnotator()
            # mask_annotator = sv.MaskAnnotator()
            # labels = [
            #     f"{CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _
            #     in detections]
            # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            # annotated_image = box_annotator.annotate(scene=output, detections=detections, labels=labels)
            # cv2.imwrite(f"/home/bzeren/projects/labs/rosbag2_annoymizer/output/{image_path.split('/')[-1]}", annotated_image)
            # Debug ------------------

