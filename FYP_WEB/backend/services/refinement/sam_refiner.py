import cv2
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from segment_anything.utils.amg import batched_mask_to_box
from backend.services.registry.model_registry import ModelRegistry

logger = logging.getLogger("SAMBoxRefiner")

class SAMBoxRefiner:
    def __init__(self, device: torch.device):
        self.device = device
        self.registry = ModelRegistry()
        self.sam = self.registry.load_sam(device)
        self.transform = self.registry.get_model("sam_transform")

    def refine_boxes(
        self,
        image_path: str,
        boxes: torch.Tensor,
        already_refined: Optional[torch.Tensor] = None,
        batch_size: int = 50
    ) -> torch.Tensor:
        """
        Refine candidate bounding boxes using SAM prompt mask decoding.
        """
        if len(boxes) == 0:
            return boxes

        # Ensure input coordinates are float on target device
        boxes = boxes.float().to(self.device)

        if already_refined is None:
            already_refined = torch.zeros(len(boxes), dtype=torch.bool, device=self.device)
        else:
            already_refined = already_refined.to(self.device)

        # Initialize output with original boxes
        sam_refined = boxes.clone()
        
        to_refine_mask = ~already_refined
        if not to_refine_mask.any():
            return sam_refined

        try:
            # Read image and format for SAM
            curr_image = cv2.imread(image_path)
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            img_shape = curr_image.shape[:2]

            curr_image_resized = self.transform.apply_image(curr_image)
            curr_image_tensor = torch.as_tensor(curr_image_resized, device=self.sam.device).permute(2, 0, 1).contiguous()

            boxes_to_refine = boxes[to_refine_mask]
            sam_box_prompts = self.transform.apply_boxes_torch(boxes_to_refine, img_shape)

            sam_refined_list = []
            for k in range(0, len(sam_box_prompts), batch_size):
                batch = sam_box_prompts[k : k + batch_size]
                if len(batch) == 0:
                    continue

                batched_input = [{
                    "image": curr_image_tensor,
                    "boxes": batch,
                    "original_size": img_shape
                }]

                # Run SAM inference
                with torch.no_grad():
                    out = self.sam(batched_input, multimask_output=False)
                    masks = out[0]['masks'].clone().detach()
                    
                    # Convert binary masks back to tight bounding boxes
                    refined_boxes = batched_mask_to_box(masks).squeeze(1)
                    sam_refined_list.append(refined_boxes.to(self.device))

            if sam_refined_list:
                sam_refined[to_refine_mask] = torch.cat(sam_refined_list, dim=0)

            return sam_refined

        except Exception as e:
            logger.error(f"SAM refinement failed: {e}. Returning original boxes.")
            return boxes
