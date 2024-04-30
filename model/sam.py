from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
import numpy as np


class SAM:
    def __init__(self, encoder_version, checkpoint_path, device) -> None:
        self.sam_predictor = self.load_sam_predictor(
            encoder_version, checkpoint_path, device
        )

    def load_sam_predictor(
        self, encoder_version, checkpoint_path, device
    ) -> SamPredictor:
        sam = sam_model_registry[encoder_version](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor

    def __call__(self, image, detections: sv.Detections) -> sv.Detections:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        detections.mask = np.array(result_masks)
        return detections
