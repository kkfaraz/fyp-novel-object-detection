from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry
import torch
import sys
import os
import pickle

# Add parent scripts directory for SAEG module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    
    from vild_templates import get_vild_templates
    SAEG_AVAILABLE = True
except ImportError:
    SAEG_AVAILABLE = False
    print("[SAEG] Module not available - using simple templates")


def load_fully_supervised_trained_model(cfg_file, weight_dir):
    # Load the model weights of supevised training phase
    opts = [f'train.output_dir={weight_dir}', f'train.init_checkpoint={weight_dir}/model_final.pth']

    cfg = LazyConfig.load(cfg_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    default_setup(cfg, None)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    return model, cfg



def load_sam_model(device, sam_checkpoint):
    # Try loading SAM3 first
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Search upwards for the workspace root containing 'sam3' and 'scripts'
        proj_root = current_dir
        for _ in range(5):
            if os.path.exists(os.path.join(proj_root, "sam3")) and os.path.exists(os.path.join(proj_root, "scripts")):
                break
            proj_root = os.path.dirname(proj_root)
        
        # Add paths for sam3 namespace package
        sys.path.insert(0, os.path.join(proj_root, "sam3"))
        sys.path.insert(0, os.path.join(proj_root, "scripts"))
        
        from sam3.model_builder import build_sam3_image_model
        from sam3_helper import SAM3CallableWrapper, download_sam3_checkpoint
        
        print("[SAM3 Loader] Attempting to load SAM3...")
        checkpoint_dir = os.path.join(proj_root, "SAM_weights")
        target_path = os.path.join(checkpoint_dir, "sam3.pt")
        
        # Auto-download if missing
        if not os.path.exists(target_path):
            print(f"[SAM3 Loader] Checkpoint not found at {target_path}. Downloading...")
            download_sam3_checkpoint(checkpoint_dir)
            
        print(f"[SAM3 Loader] Loading SAM3 model on {device} using checkpoint {target_path}...")
        sam3_model = build_sam3_image_model(
            device=device,
            load_from_HF=False,
            checkpoint_path=target_path,
            enable_inst_interactivity=True
        )
        print("[SAM3 Loader] Wrapping SAM3 model in callable wrapper...")
        sam = SAM3CallableWrapper(sam3_model, device)
        print("[SAM3 Loader] ✓ SAM3 successfully loaded and wrapped!")
        return sam
    except Exception as e:
        import traceback
        print(f"[SAM3 Loader] ⚠ Failed to load SAM3: {e}")
        print("[SAM3 Loader] Traceback details:")
        traceback.print_exc()
        print("[SAM Loader] Falling back to standard SAM (v1)...")
        try:
            from segment_anything import sam_model_registry
            fallback_checkpoint = sam_checkpoint
            if "sam3" in sam_checkpoint or not os.path.exists(sam_checkpoint):
                possible_paths = [
                    "SAM_weights.pth",
                    "../SAM_weights.pth",
                    "../../SAM_weights.pth",
                    "/home/faraz/FYP/NOD/FYP_CAP_2/SAM_weights.pth"
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        fallback_checkpoint = p
                        break
            
            print(f"[SAM Loader] Loading standard SAM from {fallback_checkpoint}...")
            sam = None
            for model_type in ["vit_l", "vit_h"]:
                try:
                    sam = sam_model_registry[model_type](checkpoint=fallback_checkpoint)
                    print(f"[SAM Loader] ✓ Loaded standard SAM using model type: {model_type}")
                    break
                except Exception as load_err:
                    print(f"[SAM Loader] Tried loading as {model_type} but failed: {load_err}")
            if sam is None:
                raise RuntimeError("Failed to load SAM with vit_l and vit_h configurations.")
            sam.to(device)
            sam.eval()
            return sam
        except Exception as fallback_err:
            print(f"[SAM Loader] ⚠ Fallback to standard SAM also failed: {fallback_err}")
            raise RuntimeError(f"Failed to load both SAM3 and standard SAM: {fallback_err}") from fallback_err


def load_maskrcnn_model(cfg_file, weight_dir, device):
    """
    Load TorchVision Mask-RCNN model for RPN proposal generation.
    
    CRITICAL: Uses TorchVision's pretrained Mask-RCNN (COCO weights) because
    the custom checkpoint is in TorchVision format, not Detectron2 format.
    
    The RPN in TorchVision Mask-RCNN generates class-agnostic proposals
    which we extract for Stage-2 background object discovery.
    """
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    import os
    
    print(f"[Mask-RCNN] Loading TorchVision pretrained model (COCO weights)...")
    
    # Load TorchVision's pretrained Mask-RCNN
    # This model uses ResNet50-FPN backbone and is pretrained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Configure RPN for maximum proposal generation
    # Access the RPN through model.rpn
    if hasattr(model, 'rpn'):
        rpn = model.rpn
        
        # Configure RPN head to be very permissive
        rpn.score_thresh = 0.0  # No filtering by score
        rpn.nms_thresh = 0.7  # Less aggressive NMS
        
        # Pre-NMS and Post-NMS limits
        rpn._pre_nms_top_n = {'training': 6000, 'testing': 6000}
        rpn._post_nms_top_n = {'training': 2000, 'testing': 2000}
        
        # Minimum box size  
        rpn.min_size = 0  # Accept even tiny proposals
        
        print("[Mask-RCNN] RPN configured: score_thresh=0.0, nms_thresh=0.7")
        print("[Mask-RCNN] RPN configured: pre_nms=6000, post_nms=2000")
    
    # Also try to load custom weights if available
    weight_path = os.path.join(weight_dir, "MaskRCNN_v2.pt")
    if os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load, ignore mismatches
            model.load_state_dict(state_dict, strict=False)
            print(f"[Mask-RCNN] Loaded custom weights from {weight_path}")
        except Exception as e:
            print(f"[Mask-RCNN] Could not load custom weights: {e}")
            print("[Mask-RCNN] Using pretrained COCO weights only")
    else:
        print("[Mask-RCNN] Using pretrained COCO weights (no custom checkpoint)")
    
    model.to(device)
    model.eval()
    
    print("[Mask-RCNN] Model loaded successfully (TorchVision format)")
    print("[Mask-RCNN] RPN will generate ~1000+ proposals per image")
    
    return model, None


def load_coco_text_features_saeg(clip_model, tokenizer, coco_classes, device):
    """
    Generate COCO class text features using SAEG (synonym averaging + templates).
    Falls back to simple templates if synonyms not available.
    
    Args:
        clip_model: CLIP model
        tokenizer: CLIP tokenizer  
        coco_classes: List of 80 COCO class names
        device: 'cuda' or 'cpu'
        
    Returns:
        text_features: Tensor (80, embedding_dim) normalized CLIP embeddings
    """
    if not SAEG_AVAILABLE:
        print("[SAEG] Not available - using simple templates")
        # Fallback to simple template
        with torch.no_grad():
            class_prompts = [f"a photo of a {cls}" for cls in coco_classes]
            class_tokens = tokenizer(class_prompts).to(device)
            coco_text_features = clip_model.encode_text(class_tokens)
            coco_text_features = coco_text_features / coco_text_features.norm(dim=-1, keepdim=True)
        return coco_text_features
    
    # Try to load COCO synonyms
    synonyms_path = 'coco_ovd_class_to_synonyms.pkl'
    if not os.path.exists(synonyms_path):
        print(f"[SAEG] Synonyms not found at {synonyms_path} - using simple templates")
        with torch.no_grad():
            class_prompts = [f"a photo of a {cls}" for cls in coco_classes]
            class_tokens = tokenizer(class_prompts).to(device)
            coco_text_features = clip_model.encode_text(class_tokens)
            coco_text_features = coco_text_features / coco_text_features.norm(dim=-1, keepdim=True)
        return coco_text_features
    
    print("[SAEG] Generating COCO text features with synonym averaging...")
    
    # Load synonyms
    with open(synonyms_path, 'rb') as f:
        class_to_synonyms = pickle.load(f)
    
    # Get ViLD templates
    templates = get_vild_templates()
    
    # Use SAEG module
    saeg = SAEGTextFeatureGenerator(
        clip_model=clip_model,
        tokenizer=tokenizer,
        templates=templates,
        class_to_synonyms=class_to_synonyms,
        device=device
    )
    
    coco_text_features = saeg.generate_text_features(coco_classes)
    
    print(f"[SAEG] ✓ Generated features: {coco_text_features.shape}")
    return coco_text_features
