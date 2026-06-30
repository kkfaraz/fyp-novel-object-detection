import re
from detectron2.structures import Instances, Boxes
import torch
from torchvision.ops import batched_nms

def postprocess_instances(instances, lvis_classes, known_ids):
    if instances is None or len(instances) == 0:
        return instances
        
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes
    
    keep_idxs = []
    new_classes = []
    
    for i in range(len(boxes)):
        cls_id = int(classes[i].item())
        score = float(scores[i].item())
        
        # known vs unknown
        det_type = "known" if (cls_id + 1) in known_ids else "unknown"
        
        # 1. Strict Confidence Filtering
        if det_type == "known" and score < 0.5:
            continue
        if det_type == "unknown" and score < 0.6:
            continue
        if score <= 0.0:
            continue
            
        # 3. Remove Background Predictions Completely
        label = lvis_classes[cls_id] if lvis_classes and 0 <= cls_id < len(lvis_classes) else ""
        label_lower = label.lower()
        if not label or label_lower in ["background", "null", "invalid", "bg"]:
            continue
            
        # 4. Clean Unknown Labeling
        if det_type == "unknown":
            words = label.split()
            if len(words) > 2:
                # Limit caption length, single noun extraction
                # fallback to just "Unknown Object" if too long
                label = "Unknown Object"
        
        keep_idxs.append(i)
        
    if len(keep_idxs) == 0:
        empty_inst = Instances(instances.image_size)
        empty_inst.pred_boxes = Boxes(torch.empty(0, 4))
        empty_inst.scores = torch.empty(0)
        empty_inst.pred_classes = torch.empty(0, dtype=torch.int64)
        return empty_inst
        
    keep_tensor = torch.tensor(keep_idxs, dtype=torch.long, device=boxes.device)
    
    filtered_boxes = boxes[keep_tensor]
    filtered_scores = scores[keep_tensor]
    filtered_classes = classes[keep_tensor]
    
    # 2. Apply Proper NMS
    # Class-wise NMS
    nms_keep = batched_nms(filtered_boxes.float(), filtered_scores.float(), filtered_classes, iou_threshold=0.5)
    
    final_inst = Instances(instances.image_size)
    final_inst.pred_boxes = Boxes(filtered_boxes[nms_keep])
    final_inst.scores = filtered_scores[nms_keep]
    final_inst.pred_classes = filtered_classes[nms_keep]
    
    return final_inst
