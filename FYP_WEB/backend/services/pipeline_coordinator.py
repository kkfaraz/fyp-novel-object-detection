import os
import time
import base64
import cv2
import torch
import logging
import numpy as np
import traceback
from PIL import Image
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Environment & Registry
from backend.services.utils.env_manager import EnvironmentManager
from backend.services.registry.model_registry import ModelRegistry
from backend.services.engine.execution_engine import ExecutionEngine

# Strategies & Adapters
from backend.services.strategies.coco_strategy import COCOStrategy
from backend.services.strategies.voc_strategy import VOCStrategy
from backend.services.strategies.lvis_strategy import LVISStrategy
from backend.services.adapters.dataset_adapter import DatasetAdapter

# Calibration, Fusion, Refinement
from backend.services.calibration.calibrator import Calibrator
from backend.services.fusion.proposal_fusion import ProposalFusion
from backend.services.refinement.semantic_verifier import CLIPSemanticVerifier
from backend.services.refinement.sam_refiner import SAMBoxRefiner
from backend.services.label_mapping.label_mapper import LabelMapper

# Visualizer & utils
from backend.services.utils.visualizer_utils import BBoxVisualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
import detectron2.data.transforms as T

logger = logging.getLogger("PipelineCoordinator")

class CooperativeDetectionPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = ModelRegistry()
        self.engine = ExecutionEngine()
        self.label_mapper = LabelMapper()
        
        # Load strategies
        self.strategies = {
            "coco": COCOStrategy(),
            "voc": VOCStrategy(),
            "lvis": LVISStrategy()
        }
        
        # Instantiate detectors & downstream components
        self.gdino_detector = GroundingDINODetector(self.device)
        self.maskrcnn_detector = MaskRCNNDetector(self.device)
        self.clip_verifier = CLIPSemanticVerifier(self.device)
        self.sam_refiner = SAMBoxRefiner(self.device)
        
        self.fusion = ProposalFusion(device=self.device)
        logger.info("✓ CooperativeDetectionPipeline successfully initialized.")

    def _get_strategy(self, dataset_name: str):
        dataset_name = dataset_name.lower().strip()
        if dataset_name not in self.strategies:
            logger.warning(f"Unknown dataset strategy '{dataset_name}'. Falling back to LVIS.")
            return self.strategies["lvis"]
        return self.strategies[dataset_name]

    def _filter_background_proposals(self, bg_boxes: torch.Tensor, bg_scores: torch.Tensor, gdino_boxes: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(bg_boxes) == 0 or len(gdino_boxes) == 0:
            return bg_boxes, bg_scores
        
        from torchvision.ops import box_iou
        ious = box_iou(bg_boxes, gdino_boxes)
        max_ious, _ = ious.max(dim=1)
        keep_mask = max_ious < iou_threshold
        return bg_boxes[keep_mask], bg_scores[keep_mask]

    def _instances_to_detections(self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, strategy_classes: List[str], known_ids: set, source_contrib: torch.Tensor = None) -> List[Dict[str, Any]]:
        detections = []
        for idx in range(len(boxes)):
            box = boxes[idx].cpu().tolist()
            score = float(scores[idx].item())
            lbl_idx = int(labels[idx].item())

            if 0 <= lbl_idx < len(strategy_classes):
                label_name = strategy_classes[lbl_idx]
            else:
                label_name = f"class_{lbl_idx}"

            # Map the category name to a clean, consistent format
            mapped_label = self.label_mapper.map(label_name)

            # Determine if it's a known or unknown class
            # (In COCO OVD baseline, COCO known class IDs are 0 to 79)
            det_type = "known" if lbl_idx in known_ids or (lbl_idx < 80 and len(strategy_classes) > 80) else "unknown"

            # Determine source details
            det_source = "pipeline"
            if source_contrib is not None:
                contrib_list = source_contrib[idx].cpu().tolist()
                max_contrib = max(contrib_list)
                max_src_idx = contrib_list.index(max_contrib)
                sources = {0: "maskrcnn_known", 1: "gdino", 2: "background_verifier"}
                det_source = sources.get(max_src_idx, "pipeline")

            detections.append({
                "id": idx,
                "label": str(mapped_label),
                "confidence": float(f"{score:.4f}"),
                "type": str(det_type),
                "bbox": [float(f"{v:.1f}") for v in box],
                "detector_source": det_source,
            })
        return detections

    def _generate_visualization(self, image_path: str, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, strategy_classes: List[str]) -> str:
        """Draws bounding boxes and labels onto the image, returning base64 encoded JPG."""
        try:
            curr_image = cv2.imread(image_path)
            h, w = curr_image.shape[:2]
            
            # Setup dummy detectron2 metadata catalog for BBoxVisualizer
            meta = MetadataCatalog.get("_temp_vis")
            meta.set(thing_classes=strategy_classes)
            
            visualizer = BBoxVisualizer(curr_image[:, :, ::-1], meta, scale=1.2)
            
            inst = Instances((h, w))
            inst.pred_boxes = Boxes(boxes.cpu())
            inst.scores = scores.cpu()
            inst.pred_classes = labels.cpu().to(torch.int64)
            
            vis_img = visualizer.draw_instance_predictions(inst)
            vis_np = vis_img.get_image()[:, :, ::-1] # RGB -> BGR
            
            _, buffer = cv2.imencode('.jpg', vis_np)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            return ""

    def run(self, image_path: str, dataset_name: str = "lvis", **kwargs) -> Dict[str, Any]:
        """
        Runs the full multi-stage cooperative detection pipeline.
        """
        t_start = time.time()
        
        # 1. Resolve strategy & configuration
        strategy = self._get_strategy(dataset_name)
        thresholds = strategy.get_thresholds()
        clip_threshold = kwargs.get("clip_threshold", thresholds.get("text_threshold", 0.25))
        wbf_iou_threshold = kwargs.get("wbf_iou_threshold", thresholds.get("nms_threshold", 0.50))
        
        # Create adapter and calibrator
        adapter = DatasetAdapter(strategy)
        calibrator = Calibrator(strategy.get_calibration_config())
        
        # 2. Preprocess image for Detectron2 Mask R-CNN
        img = cv2.imread(image_path)
        orig_h, orig_w = img.shape[:2]
        
        data_dict = {
            "file_name": os.path.abspath(image_path),
            "height": orig_h,
            "width": orig_w,
            "image_id": 0,
        }
        
        # Detectron2 scaling
        augmentations = T.AugmentationList([
            T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
        ])
        aug_input = T.AugInput(img, sem_seg=None)
        augmentations(aug_input)
        data_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))
        )
        inputs = [data_dict]

        # 3. STAGE 1: Parallel/Sequential Detector Execution
        t_s1_start = time.time()
        
        task_known = lambda: self.maskrcnn_detector.detect_known(inputs)
        task_gdino = lambda: self.gdino_detector.detect(inputs, strategy)
        task_rpn = lambda: self.maskrcnn_detector.extract_rpn_proposals(image_path)
        
        # Run tasks using ExecutionEngine
        known_raw, gdino_raw, rpn_raw = self.engine.run_tasks([task_known, task_gdino, task_rpn])
        stage1_time = time.time() - t_s1_start
        
        # Adapt and standardize Stage 1 outputs
        known_adapted = adapter.adapt_maskrcnn_known(known_raw)
        gdino_adapted = adapter.adapt_gdino(gdino_raw)
        rpn_adapted = adapter.adapt_rpn(rpn_raw)
        
        # Generate Stage 1 visualization (Knowns + GDINO)
        s1_vis_boxes = torch.cat([known_adapted["boxes"], gdino_adapted["boxes"]]) if len(known_adapted["boxes"]) or len(gdino_adapted["boxes"]) else torch.empty(0, 4, device=self.device)
        s1_vis_scores = torch.cat([known_adapted["scores"], gdino_adapted["scores"]]) if len(known_adapted["scores"]) or len(gdino_adapted["scores"]) else torch.empty(0, device=self.device)
        s1_vis_labels = torch.cat([known_adapted["labels"], gdino_adapted["labels"]]) if len(known_adapted["labels"]) or len(gdino_adapted["labels"]) else torch.empty(0, dtype=torch.long, device=self.device)
        stage1_b64 = self._generate_visualization(image_path, s1_vis_boxes, s1_vis_scores, s1_vis_labels, strategy.get_classes())

        # 4. STAGE 2: Unknown Object Classification using CLIP
        t_s2_start = time.time()
        
        # Filter background proposals that overlap with GDINO detections to focus on novel objects
        bg_boxes, bg_scores = self._filter_background_proposals(
            rpn_adapted["boxes"],
            rpn_adapted["scores"],
            gdino_adapted["boxes"]
        )
        
        # Run CLIP semantic verifier on remaining background proposals
        stage2_out = self.clip_verifier.verify(image_path, bg_boxes, strategy, clip_threshold=clip_threshold)
        stage2_time = time.time() - t_s2_start
        
        # Generate Stage 2 visualization (CLIP verified background crops)
        stage2_b64 = self._generate_visualization(image_path, stage2_out["boxes"], stage2_out["scores"], stage2_out["labels"], strategy.get_classes())

        # 5. STAGE 3: Proposal Fusion, SAM Refinement, and Calibration
        t_s3_start = time.time()
        
        # Compile multi-source proposals
        proposals = {
            0: known_adapted,              # Mask R-CNN known classes
            1: gdino_adapted,              # Grounding DINO detections
            2: stage2_out                  # CLIP verified background detections
        }
        
        # Calibrate confidence scores
        for src_id in proposals:
            proposals[src_id]["scores"] = calibrator.calibrate(proposals[src_id]["scores"])
            
        # Fuse proposals using WBF
        fused = self.fusion.fuse(proposals, num_classes=len(strategy.get_classes()), use_soft_nms=False)
        
        # Refine fused box coordinates using SAM
        # Track which boxes were already refined in GDINO (Source 1)
        source_contrib = fused["source_contributions"]
        already_refined = source_contrib[:, 1] > 0.5 if len(source_contrib) > 0 else torch.zeros(len(fused["boxes"]), dtype=torch.bool, device=self.device)
        
        refined_boxes = self.sam_refiner.refine_boxes(image_path, fused["boxes"], already_refined)
        stage3_time = time.time() - t_s3_start
        
        t_total = time.time() - t_start
        
        # Build clean JSON detections for the frontend
        known_ids = set(range(80)) if dataset_name != "lvis" else set(range(1203)) # Overwritten based on classes
        if dataset_name == "voc":
            known_ids = set(range(20))
            
        detections = self._instances_to_detections(
            refined_boxes,
            fused["scores"],
            fused["classes"],
            strategy.get_classes(),
            known_ids,
            source_contrib
        )
        
        # Generate Stage 3 visualization (Final refined results)
        stage3_b64 = self._generate_visualization(image_path, refined_boxes, fused["scores"], fused["classes"], strategy.get_classes())
        
        known_count = sum(1 for d in detections if d["type"] == "known")
        unknown_count = sum(1 for d in detections if d["type"] == "unknown")
        
        return {
            "detections": detections,
            "annotated_image": stage3_b64,
            "stage_images": {
                "stage1": stage1_b64,
                "stage2": stage2_b64,
                "stage3": stage3_b64
            },
            "stage_times": {
                "stage1": float(f"{stage1_time:.2f}"),
                "stage2": float(f"{stage2_time:.2f}"),
                "stage3": float(f"{stage3_time:.2f}"),
                "total": float(f"{t_total:.2f}")
            },
            "metrics": {
                "total_detections": len(detections),
                "known_count": known_count,
                "novel_count": unknown_count,
                "processing_time": float(f"{t_total:.2f}"),
            },
            "known_objects": [d for d in detections if d["type"] == "known"],
            "novel_objects": [d for d in detections if d["type"] == "unknown"],
            "debug": {
                "image_size": [orig_w, orig_h],
                "pipeline_version": "cooperative-nod-v1.0",
                "dataset_knowledge": f"{dataset_name.upper()} Cooperative Detection",
            }
        }

# Inject GroundingDINODetector & MaskRCNNDetector imports after definitions
from backend.services.detectors.gdino_detector import GroundingDINODetector
from backend.services.detectors.maskrcnn_detector import MaskRCNNDetector
