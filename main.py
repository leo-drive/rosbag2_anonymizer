import yaml
import json

import cv2
import cv_bridge
from PIL import Image

import supervision as sv

import torch
import torchvision

from common import create_classes, calculate_iou, bbox_check, blur_detections

from rosbag_io.rosbag_reader import RosbagReader
from rosbag_io.rosbag_writer import RosbagWriter

from model.open_clip import OpenClipModel
from model.grounding_dino import GroundingDINO
from model.sam import SAM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('validation.json', 'r') as json_file:
    json_data = json.load(json_file)

if __name__ == '__main__':
    reader = RosbagReader(config['rosbag']['input_bag_path'])
    writer = RosbagWriter(config['rosbag']['output_bag_paht'], config['rosbag']['output_save_compressed_image'], config['rosbag']['output_storage_id'])

    # Define classes
    DETECTION_CLASSES, CLASSES, CLASS_MAP = create_classes(json_data=json_data)

    # GroundingDINO parameters
    GROUNDING_DINO_CONFIG_PATH = config['grounding_dino']['config_path']
    GROUNDING_DINO_CHECKPOINT_PATH = config['grounding_dino']['checkpoint_path']

    BOX_THRESHOLD = config['grounding_dino']['box_threshold']
    TEXT_THRESHOLD = config['grounding_dino']['text_threshold']
    NMS_THRESHOLD = config['grounding_dino']['nms_threshold']

    # Segment-Anything parameters
    SAM_ENCODER_VERSION = config['segment_anything']['encoder_version']
    SAM_CHECKPOINT_PATH = config['segment_anything']['checkpoint_path']

    # OpenClip parameters
    OPENCLIP_MODEL_NAME = config['openclip']['model_name']
    OPENCLIP_PRETRAINED_MODEL = config['openclip']['pretrained_model']

    #Validation
    IOU_THRESHOLD = config['bbox_validation']['iou_threshold']

    # Grounding DINO
    grounding_dino = GroundingDINO(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)

    # Segment-Anything
    sam = SAM(SAM_ENCODER_VERSION, SAM_CHECKPOINT_PATH, DEVICE)

    # Openclip
    open_clip = OpenClipModel(OPENCLIP_MODEL_NAME, OPENCLIP_PRETRAINED_MODEL)

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

            # Validation
            valid_ids = []
            invalid_ids = []
            for index, (xyxy, mask, confidence, class_id, _) in enumerate(detections):
                if CLASSES[class_id] in DETECTION_CLASSES:
                    # Run OpenClip
                    # and accept as a valid object if the score is greater than 0.9
                    detection_image = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                    pil_image = Image.fromarray(detection_image)
                    scores = open_clip(pil_image, CLASSES)
                    if scores.numpy().tolist()[0][class_id] > config['openclip']['score_threshold']:
                        valid_ids.append(index)
                        continue
                    
                    # Bbox validation
                    # If the object is within the 'should_inside' object 
                    # and if the score is the highest among the scores, 
                    # or greater than 0.4.
                    if bbox_check(xyxy, class_id, detections, IOU_THRESHOLD, CLASSES, CLASS_MAP) and (max(scores.numpy().tolist()[0]) == scores.numpy().tolist()[0][class_id] or scores.numpy().tolist()[0][class_id] > 0.3):
                        valid_ids.append(index)
                    else:
                        invalid_ids.append(index)
                else:
                    invalid_ids.append(index)
            # valid_detections = sv.Detections(xyxy=detections.xyxy[valid_ids], confidence=detections.confidence[valid_ids], class_id=detections.class_id[valid_ids])
            # invalid_detections = sv.Detections(xyxy=detections.xyxy[invalid_ids], confidence=detections.confidence[invalid_ids], class_id=detections.class_id[invalid_ids])
            detections.xyxy = detections.xyxy[valid_ids]
            detections.confidence = detections.confidence[valid_ids]
            detections.class_id = detections.class_id[valid_ids]

            # Run SAM
            detections = sam(image=image, detections=detections)

            # Blur detections
            output = blur_detections(image, detections, config['blur']['kernel_size'], config['blur']['sigma_x'])

            # Write blured image to rosbag
            writer.write_image(output, msg.topic, msg.timestamp)

            # Debug ------------------
            # box_annotator = sv.BoxAnnotator()
            # labels = [
            #     f"{CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _
            #     in detections]
            # annotated_image = box_annotator.annotate(scene=output, detections=detections, labels=labels)

            # invalid_box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#FF0000']))
            # invalid_labels = [
            #     f"{CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _
            #     in invalid_detections]
            # annotated_image = invalid_box_annotator.annotate(scene=output, detections=invalid_detections, labels=invalid_labels)

            # valid_box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#008000']))
            # valid_labels = [
            #     f"{CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _
            #     in valid_detections]
            # annotated_image = valid_box_annotator.annotate(scene=output, detections=valid_detections, labels=valid_labels)

            # height, width = image.shape[:2]
            # annotated_image = cv2.resize(annotated_image, (width // 2, height // 2))
        
            # cv2.imshow('test', annotated_image)
            # cv2.waitKey(1)
            # Debug ------------------

