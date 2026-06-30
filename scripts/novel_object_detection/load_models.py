import open_clip
import torch
import pickle
import sys
import os

# Add parent scripts directory to path for SAEG/SRM modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detectron2.data import MetadataCatalog

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry
from utils import article, processed_name

# SAEG imports for synonym-averaged text features

from vild_templates import get_vild_templates

def load_fully_supervised_trained_model(cfg_file, weight_dir, device="cuda"):
    model_final_path = os.path.join(weight_dir, "model_final.pth")
    
    # Priority 1: Prefer COCO-pretrained weights (80 classes, matches config architecture)
    if not os.path.exists(model_final_path):
        model_final_path = os.path.join("model weights/MaskRCNN_COCO_OVD", "model_final.pth")
    if not os.path.exists(model_final_path):
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        model_final_path = os.path.join(proj_path, "model weights/MaskRCNN_COCO_OVD/model_final.pth")

    # Priority 2: Check for LVIS weights MaskRCNNLVIS.pkl in weight_dir
    lvis_weight_path = os.path.join(weight_dir, "MaskRCNNLVIS.pkl")
    if os.path.exists(lvis_weight_path) and not os.path.exists(model_final_path):
        model_final_path = lvis_weight_path
    elif not os.path.exists(model_final_path):
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        sibling_lvis_path = os.path.join(proj_path, "scripts/novel_object_detection/rcnn_weight_dir/MaskRCNNLVIS.pkl")
        if os.path.exists(sibling_lvis_path):
            model_final_path = sibling_lvis_path
        
    print(f"[Model Loader] Loading Detectron2 weights from: {model_final_path}")
    
    # If weight_dir is a file path, use its parent directory for detectron2 output
    output_dir = weight_dir
    if weight_dir.endswith('.pkl') or weight_dir.endswith('.pth'):
        output_dir = os.path.dirname(weight_dir)
        if not os.path.isdir(output_dir):
            output_dir = "/tmp"
    
    opts = [f'train.output_dir={output_dir}', f'train.init_checkpoint={model_final_path}', f'train.device={device}']

    cfg = LazyConfig.load(cfg_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    default_setup(cfg, None)

    # Detect num_classes and mask_classes from checkpoint weights dynamically
    num_classes = 80
    mask_classes = 80
    if hasattr(cfg.model.roi_heads.mask_head, 'num_classes'):
        mask_classes = cfg.model.roi_heads.mask_head.num_classes

    if model_final_path.endswith(".pkl") or model_final_path.endswith(".pth"):
        try:
            if model_final_path.endswith(".pkl"):
                with open(model_final_path, "rb") as f:
                    checkpoint = pickle.load(f)
            else:
                checkpoint = torch.load(model_final_path, map_location="cpu")
            
            state_dict = checkpoint.get("model", checkpoint)
            
            # Detect box predictor classes
            weight_key = "roi_heads.box_predictor.cls_score.weight"
            if weight_key in state_dict:
                cls_weight = state_dict[weight_key]
                num_classes = cls_weight.shape[0] - 1
                print(f"[Model Loader] Detected {num_classes} classes from box predictor weights.")
                
            # Detect mask predictor classes
            mask_weight_key = "roi_heads.mask_head.predictor.mask_fcn_logits.weight"
            if mask_weight_key in state_dict:
                mask_weight = state_dict[mask_weight_key]
                mask_classes = mask_weight.shape[0]
                print(f"[Model Loader] Detected {mask_classes} mask classes from mask head weights.")
        except Exception as e:
            print(f"[Model Loader] Could not detect class count from weights: {e}")

    cfg.model.roi_heads.box_predictor.num_classes = num_classes
    if hasattr(cfg.model.roi_heads.mask_head, 'num_classes'):
        cfg.model.roi_heads.mask_head.num_classes = mask_classes
        
    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    DetectionCheckpointer(model).load(model_final_path)
    
    return model, cfg


def load_clip_model(data_split, device):
    print("[SigLIP] Loading ViT-SO400M-14-SigLIP (webli pretrained)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')

    clip_model = clip_model.to(device)


    lvis_metadata = MetadataCatalog.get(data_split)
    lvis_classes = lvis_metadata.get("thing_classes")

    with open('lvis_original_class_to_synonyms.pkl', 'rb') as f:
        class_names_to_synonyms = pickle.load(f)

    from vild_templates import get_vild_templates
    templates = get_vild_templates()
    print(f"[SigLIP] Using {len(templates)} ViLD templates for text embedding")

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = []
        for classname in lvis_classes:
            syn_features = []
            for syn in class_names_to_synonyms[classname]:
                syn_clean = processed_name(syn, rm_dot=True)
                texts = [template.format(syn_clean) for template in templates]
                texts = tokenizer(texts, context_length=clip_model.context_length).to(device)
                syn_embeddings = clip_model.encode_text(texts)
                syn_embeddings /= syn_embeddings.norm(dim=-1, keepdim=True)
                syn_embedding = syn_embeddings.mean(dim=0)
                syn_embedding /= syn_embedding.norm()
                syn_features.append(syn_embedding)

            syn_features = torch.stack(syn_features, dim=0).to(device)
            syn_feature = syn_features.mean(dim=0)
            syn_feature /= syn_feature.norm()
            text_features.append(syn_feature)

        text_features = torch.stack(text_features, dim=0).to(device)
    
    return clip_model, preprocess, text_features, lvis_classes


def load_clip_model_saeg(data_split, device):
    """
    Load CLIP/SigLIP model with SAEG (Synonym Averaged Embedding Generator).
    
    This function implements the full SAEG algorithm from the paper:
    - Uses 64 ViLD templates instead of 60 custom templates
    - Averages embeddings across templates for each synonym (Eq. 1)
    - Averages embeddings across synonyms for each class (Eq. 2)
    - Produces final text feature matrix (Eq. 3)
    
    Args:
        data_split: Dataset split name (e.g., 'lvis_v1_val')
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        clip_model: Loaded CLIP/SigLIP model
        preprocess: Image preprocessing function
        text_features: SAEG-generated text features (num_classes, embedding_dim)
        lvis_classes: List of class names
    
    Reference:
        Paper Section 3.2 "Unknown Object Labelling" - SAEG component
    """
    print("[SAEG] Loading ViT-SO400M-14-SigLIP (webli pretrained)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
        
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Get class names from dataset metadata
    lvis_metadata = MetadataCatalog.get(data_split)
    lvis_classes = lvis_metadata.get("thing_classes")
    
    # Load synonym dictionary
    with open('lvis_original_class_to_synonyms.pkl', 'rb') as f:
        class_names_to_synonyms = pickle.load(f)
    
    # Load ViLD templates (64 templates from the paper)
    vild_templates = get_vild_templates()
    
    print(f"[SAEG] Initializing Synonym Averaged Embedding Generator")
    print(f"[SAEG] Classes: {len(lvis_classes)}, Templates: {len(vild_templates)}")
    
    # Initialize SAEG generator
    saeg = SAEGTextFeatureGenerator(
        clip_model=clip_model,
        tokenizer=tokenizer,
        templates=vild_templates,
        class_to_synonyms=class_names_to_synonyms,
        device=device,
        show_progress=True
    )
    
    # Generate text features using SAEG (Equations 1-3)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = saeg.generate_text_features(
            lvis_classes, 
            clear_cache_every=50
        )
    
    print(f"[SAEG] ✓ Text features generated successfully")
    print(f"[SAEG] Shape: {text_features.shape} (expected: {len(lvis_classes)} x embedding_dim)")
    
    return clip_model, preprocess, text_features, lvis_classes


def load_torchvision_maskrcnn(device):
    """
    Load TorchVision pretrained Mask-RCNN for RPN proposal extraction.
    
    Loaded on `device` (GPU) for fast Stage-1 inference.
    After Stage 1 completes, main.py offloads it to CPU via .cpu()
    to reclaim VRAM for the Stage-2 VLRM+CLIP models.
    
    RPN settings:
    - score_thresh = 0.0 (no score filtering)
    - nms_thresh = 0.7 (less aggressive NMS)
    - pre_nms_top_n = 6000, post_nms_top_n = 2000
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    
    print(f"[TorchVision Mask-RCNN] Loading pretrained model (COCO weights) on {device}...")
    
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)  # On GPU during Stage 1; offloaded after in main.py
    
    # Configure RPN for maximum proposal generation
    model.rpn.score_thresh = 0.0
    model.rpn.nms_thresh = 0.7
    model.rpn._pre_nms_top_n = {'training': 6000, 'testing': 6000}
    model.rpn._post_nms_top_n = {'training': 2000, 'testing': 2000}
    model.rpn.min_size = 0
    
    print(f"[TorchVision Mask-RCNN] ✓ Loaded on {device} (RPN: pre_nms=6000, post_nms=2000)")
    
    return model


def load_sam_model(device, sam_checkpoint):
    # Clear GPU memory before loading SAM
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
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
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../SAM_weights.pth")
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


def load_clip_model_caption_based(data_split, device, text_refiner=None):
    """
    FYP: Load CLIP with caption-based text features instead of handcrafted prompt templates.
    
    ============================================================================
    ORIGINAL APPROACH (60+ Handcrafted Templates):
    ============================================================================
    The original load_clip_model() uses ~60 manually crafted prompt templates like:
    - "There is {article} {} in the scene."
    - "a photo of {article} {}."
    - "a good photo of the {}."
    - "a bad photo of the {}."
    - "a bright photo of the {}."
    ... (60+ templates × class synonyms = ~1200+ embeddings per class)
    
    ============================================================================
    FYP APPROACH (Caption-Based / Prompt-Independent):
    ============================================================================
    This function uses simple universal templates:
    - "a photo of a {class}"
    - "a photo of the {class}"
    
    Optionally, can use LLM (GPT-3.5) to generate semantic descriptions:
    - Input: "A {class} is"
    - Output: "A cat is a small furry domestic animal with whiskers..."
    
    ============================================================================
    ARCHITECTURE NOTE - BERT Integration:
    ============================================================================
    - BLIP Captioner uses BERT internally for caption generation
    - GroundingDINO uses BERT for text-to-region grounding
    - SigLIP/CLIP uses ViT for text embedding (not BERT, but similar transformer)
    
    This eliminates the need for manual prompt engineering by leveraging
    the semantic understanding capabilities of these BERT-based models.
    
    Args:
        data_split: Dataset split name (e.g., 'lvis_v1_val')
        device: CUDA device ('cuda' or 'cpu')
        text_refiner: Optional TextRefiner for LLM-enhanced descriptions
    
    Returns:
        clip_model: SigLIP model
        preprocess: Image preprocessing function
        text_features: Tensor of shape (num_classes, embedding_dim)
        class_names: List of class names
    """
    # Clear GPU memory before loading CLIP
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("[Caption-Based SigLIP] Loading ViT-SO400M-14-SigLIP (webli pretrained)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
        
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Get class names from dataset metadata
    metadata = MetadataCatalog.get(data_split)
    class_names = metadata.get("thing_classes")
    
    # FYP: Simple universal templates (replacing 60+ handcrafted templates)
    simple_templates = [
        "a photo of a {}",
        "a photo of the {}",
    ]
    
    print(f"[Caption-Based CLIP] Computing text features for {len(class_names)} classes...")
    if text_refiner:
        print(f"[Caption-Based CLIP] LLM refinement: ENABLED (Ollama)")
    else:
        print(f"[Caption-Based CLIP] LLM refinement: DISABLED")
    
    with torch.no_grad():
        text_features = []
        
        # Proper CUDA initialization (no warmup - causes CUBLAS errors)
        if device == "cuda":
            try:
                torch.cuda.init()
                torch.cuda.empty_cache()
            except:
                pass  # Already initialized
        
        for idx, classname in enumerate(class_names):
            # Process class name: replace underscores, lowercase
            processed = classname.replace('_', ' ').lower()
            
            # FYP: Prompt-Independent - Use LLM to generate semantic description
            if text_refiner is not None:
                try:
                    refined = text_refiner.inference(
                        f"A {processed} is", 
                        {"length": 10}
                    )
                    description = refined.get('caption', processed)
                    # Truncate to max 10 words
                    words = description.split()[:10]
                    description = ' '.join(words)
                    texts = [description]
                    if idx < 5:
                        print(f"   Class '{processed}' -> LLM: '{description}'")
                except:
                    texts = [f"a {processed}"]
            else:
                texts = [f"a {processed}"]
            
            # Clear CUDA cache frequently to prevent OOM
            if idx % 50 == 0 and idx > 0:
                torch.cuda.empty_cache()
            
            # Encode text on GPU, store on CPU
            try:
                tokens = tokenizer(texts, context_length=clip_model.context_length).to(device)
                with torch.cuda.amp.autocast():
                    embeddings = clip_model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embedding = embeddings.mean(dim=0)
                embedding = embedding / embedding.norm()
                text_features.append(embedding.cpu())
            except:
                embed_dim = getattr(clip_model, 'output_dim', 768)
                text_features.append(torch.zeros(embed_dim))
            
            if (idx + 1) % 200 == 0:
                print(f"[Caption-Based CLIP] Processed {idx + 1}/{len(class_names)} classes")
        
        # Move all features to GPU at the end
        text_features = torch.stack(text_features, dim=0).to(device)
    
    print(f"[Caption-Based CLIP] Text features shape: {text_features.shape}")
    print(f"[Caption-Based CLIP] Done! Class features generated")
    
    return clip_model, preprocess, text_features, class_names


def load_glip_model(device, proj_path):
    """
    Load GLIP-Large model for Stage 2 proposal scoring.
    
    GLIP provides grounded language-image pre-training which enables
    better region-text alignment than standard CLIP. This is used as
    an alternative/comparison scorer in Stage 2.
    
    Args:
        device: CUDA device
        proj_path: Path to project root containing GLIP directory
        
    Returns:
        glip_demo: GLIPDemo instance for inference
    """
    import sys
    import os
    
    # Add GLIP to path
    glip_path = os.path.join(proj_path, "GLIP")
    if glip_path not in sys.path:
        sys.path.insert(0, glip_path)
    
    print("[GLIP] Loading GLIP-Large model for Stage 2 scoring...")
    
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
    
    # GLIP-Large config and weights
    config_path = os.path.join(glip_path, "configs/pretrain/glip_Swin_L.yaml")
    weight_path = os.path.join(glip_path, "MODEL/glip_large_model.pth")
    
    if not os.path.exists(weight_path):
        print(f"[GLIP] WARNING: Weights not found at {weight_path}")
        print("[GLIP] Download from: https://huggingface.co/GLIPModel/GLIP/blob/main/glip_large_model.pth")
        return None
    
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHT = weight_path
    cfg.MODEL.DEVICE = "cuda" if device == "cuda" else "cpu"
    cfg.MODEL.DYHEAD.SCORE_AGG = "MEAN"
    cfg.TEST.IMS_PER_BATCH = 1
    
    try:
        glip_demo = GLIPDemo(
            cfg,
            confidence_threshold=0.3,
            min_image_size=800,
        )
        print(f"[GLIP] Loaded GLIP-Large successfully on {device}")
        return glip_demo
    except Exception as e:
        print(f"[GLIP] ERROR loading GLIP: {e}")
        return None