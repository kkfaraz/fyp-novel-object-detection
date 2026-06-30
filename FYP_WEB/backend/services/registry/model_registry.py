import os
import sys
import gc
import json
import logging
import pickle
import torch
from pathlib import Path
from groundingdino.util.inference import load_model as load_gdino_model
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import open_clip

logger = logging.getLogger("ModelRegistry")
logging.basicConfig(level=logging.INFO)

# Apply monkeypatches for newer transformers compatibility and PyTorch 2.6 legacy weights
def apply_compatibility_patches():
    try:
        import transformers
        from transformers import BertModel, BertPreTrainedModel
        
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            return [None] * num_hidden_layers
            
        for target in [BertModel, BertPreTrainedModel]:
            if not hasattr(target, "get_head_mask"):
                target.get_head_mask = get_head_mask
        
        try:
            import transformers.models.bert.modeling_bert as bert_mod
            if hasattr(bert_mod, "BertModel") and not hasattr(bert_mod.BertModel, "get_head_mask"):
                bert_mod.BertModel.get_head_mask = get_head_mask
        except:
            pass
        
        if not hasattr(BertModel, "_old_geam"):
            BertModel._old_geam = BertModel.get_extended_attention_mask
            def patched_geam(self, attention_mask, input_shape, device=None, dtype=None):
                if device is not None and not isinstance(device, torch.dtype):
                    target_dtype = dtype if isinstance(dtype, torch.dtype) else self.dtype
                    return self._old_geam(attention_mask.to(target_dtype), input_shape, dtype=target_dtype)
                return self._old_geam(attention_mask, input_shape, dtype=device if dtype is None else dtype)
            BertModel.get_extended_attention_mask = patched_geam
        
        # Patch torch.load for compatibility
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            try:
                return original_load(*args, **kwargs)
            except Exception as e:
                # If magic number error or similar pickle loading error occurs, attempt pickle load fallback
                err_str = str(e)
                if "magic number" in err_str or "unpickling" in err_str.lower() or "pickle" in err_str.lower():
                    import pickle
                    file_arg = args[0]
                    try:
                        if isinstance(file_arg, (str, Path)):
                            with open(file_arg, "rb") as f:
                                return pickle.load(f, encoding="latin1")
                        elif hasattr(file_arg, "read"):
                            try:
                                file_arg.seek(0)
                            except Exception:
                                pass
                            return pickle.load(file_arg, encoding="latin1")
                    except Exception:
                        pass
                raise e
        torch.load = patched_load
        
        logger.info("✓ Compatibility patches successfully applied.")
    except Exception as e:
        logger.warning(f"Failed to apply compatibility patches: {e}")

class ModelRegistry:
    _instance = None
    _models = {}
    _configs = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            apply_compatibility_patches()
        return cls._instance

    @staticmethod
    def discover_checkpoint(model_name: str) -> Path:
        """
        Dynamically search for weights in:
        1. Environment variables
        2. checkpoints/ folder inside project root
        3. Adjacent workspaces (CAP_2 and cooperative-foundational-models)
        """
        env_var = os.environ.get(f"{model_name.upper()}_WEIGHTS")
        if env_var and Path(env_var).exists():
            return Path(env_var)

        from backend.services.utils.env_manager import EnvironmentManager
        root = EnvironmentManager.get_project_root()
        
        search_paths = [
            root / "checkpoints" / model_name,
            root / "checkpoints",
            root.parent / "CAP_2",
            root.parent / "CAP_2" / "model weights" / "MaskRCNN_COCO_OVD",
            root.parent / "CAP_2" / "SAM_weights",
            root.parent / "cooperative-foundational-models",
        ]

        file_patterns = {
            "gdino": ["GDINO_weights.pth", "groundingdino_swint_ogc.pth"],
            "maskrcnn": ["model_final.pth", "MaskRCNNLVIS.pkl"],
            "torchvision_maskrcnn": ["MaskRCNN_v2.pt"],
            "sam": ["SAM_weights.pth", "sam_vit_h_4b8939.pth", "sam3.pt"]
        }
        
        patterns = file_patterns.get(model_name.lower(), [])
        for sp in search_paths:
            if not sp.exists():
                continue
            for pattern in patterns:
                # Check directly in the search path
                direct = sp / pattern
                if direct.exists() and direct.is_file():
                    return direct
                # Recursive glob search (maxdepth 2)
                try:
                    for found in sp.rglob(pattern):
                        if found.is_file():
                            return found
                except Exception:
                    pass

        # Final default fallback inside checkpoints/
        default_path = root / "checkpoints" / model_name / file_patterns[model_name.lower()][0]
        return default_path

    def load_all_models(self, device: torch.device):
        """
        Loads all required models once.
        """
        self.load_gdino(device)
        self.load_maskrcnn(device)
        self.load_sam(device)
        self.load_clip(device)

    def load_gdino(self, device: torch.device):
        if "gdino" in self._models:
            return self._models["gdino"]

        from backend.services.utils.env_manager import EnvironmentManager
        root = EnvironmentManager.get_project_root()
        
        config_path = root / "backend" / "config" / "cfg" / "GroundingDINO" / "GDINO.py"
        if not config_path.exists():
            config_path = root / "backend" / "config" / "cfg" / "GroundingDINO" / "GroundingDINO_SwinT_OGC.py"

        checkpoint_path = self.discover_checkpoint("gdino")
        logger.info(f"Loading Grounding DINO from config: {config_path}, weights: {checkpoint_path}")
        
        try:
            model = load_gdino_model(str(config_path), str(checkpoint_path))
            model = model.to(device)
            model.eval()
            self._models["gdino"] = model
            logger.info("✓ Grounding DINO model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
            raise

    def load_maskrcnn(self, device: torch.device):
        if "maskrcnn" in self._models:
            return self._models["maskrcnn"]

        from backend.services.utils.env_manager import EnvironmentManager
        root = EnvironmentManager.get_project_root()
        
        cfg_file = root / "backend" / "config" / "cfg" / "MaskRCNN_R101-FPN-New-Baseline" / "R101-FPN-New-Baseline.py"
        weight_path = self.discover_checkpoint("maskrcnn")

        logger.info(f"Loading Mask R-CNN from config: {cfg_file}, weights: {weight_path}")
        
        try:
            # Need to temporarily append detectron2 path to sys.path if not present
            sys_paths = list(sys.path)
            sys.path.insert(0, str(root.parent / "CAP_2" / "detectron2"))
            sys.path.insert(0, str(root.parent / "CAP_2" / "scripts"))
            
            from detectron2.config import LazyConfig, instantiate
            from detectron2.engine import default_setup
            from detectron2.checkpoint import DetectionCheckpointer
            
            # Recreate load_fully_supervised_trained_model logic
            output_dir = str(weight_path.parent)
            opts = [f'train.output_dir={output_dir}', f'train.init_checkpoint={weight_path}', f'train.device={device}']
            
            cfg = LazyConfig.load(str(cfg_file))
            cfg = LazyConfig.apply_overrides(cfg, opts)
            
            # Temporarily redirect stdout during setup
            import contextlib
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                default_setup(cfg, None)
                
            num_classes = 80
            mask_classes = 80
            if hasattr(cfg.model.roi_heads.mask_head, 'num_classes'):
                mask_classes = cfg.model.roi_heads.mask_head.num_classes

            if str(weight_path).endswith(".pkl") or str(weight_path).endswith(".pt") or str(weight_path).endswith(".pth"):
                checkpoint = None
                if str(weight_path).endswith(".pkl"):
                    with open(weight_path, "rb") as f:
                        checkpoint = pickle.load(f)
                else:
                    try:
                        checkpoint = torch.load(weight_path, map_location="cpu")
                    except Exception:
                        try:
                            with open(weight_path, "rb") as f:
                                checkpoint = pickle.load(f)
                        except Exception as pe:
                            logger.warning(f"Could not load checkpoint {weight_path} with torch or pickle: {pe}")
                            checkpoint = {}
                
                state_dict = checkpoint.get("model", checkpoint) if checkpoint else {}
                weight_key = "roi_heads.box_predictor.cls_score.weight"
                if weight_key in state_dict:
                    cls_weight = state_dict[weight_key]
                    num_classes = cls_weight.shape[0] - 1
                
                mask_weight_key = "roi_heads.mask_head.predictor.mask_fcn_logits.weight"
                if mask_weight_key in state_dict:
                    mask_weight = state_dict[mask_weight_key]
                    mask_classes = mask_weight.shape[0]

            cfg.model.roi_heads.box_predictor.num_classes = num_classes
            if hasattr(cfg.model.roi_heads.mask_head, 'num_classes'):
                cfg.model.roi_heads.mask_head.num_classes = mask_classes
                
            model = instantiate(cfg.model)
            model.to(device)
            DetectionCheckpointer(model).load(str(weight_path))
            model.eval()

            self._models["maskrcnn"] = model
            self._configs["maskrcnn_cfg"] = cfg
            sys.path = sys_paths # Restore sys.path
            logger.info("✓ Mask R-CNN model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Mask R-CNN: {e}")
            raise

    def load_torchvision_maskrcnn(self, device: torch.device):
        if "torchvision_maskrcnn" in self._models:
            return self._models["torchvision_maskrcnn"]
        
        logger.info(f"Loading TorchVision Mask-RCNN on {device}...")
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            model = maskrcnn_resnet50_fpn(pretrained=True)
            
            # Try to load custom weights
            try:
                weight_path = self.discover_checkpoint("torchvision_maskrcnn")
                if weight_path and weight_path.exists():
                    logger.info(f"Loading custom TorchVision weights from {weight_path}")
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                logger.warning(f"Could not load custom TorchVision weights: {e}. Using COCO weights.")
                
            model.eval()
            model.to(device)
            
            # Configure RPN for maximum proposal generation
            model.rpn.score_thresh = 0.0
            model.rpn.nms_thresh = 0.7
            model.rpn._pre_nms_top_n = {'training': 6000, 'testing': 6000}
            model.rpn._post_nms_top_n = {'training': 2000, 'testing': 2000}
            model.rpn.min_size = 0
            
            self._models["torchvision_maskrcnn"] = model
            logger.info("✓ TorchVision Mask-RCNN model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load TorchVision Mask-RCNN: {e}")
            raise

    def load_sam(self, device: torch.device):
        if "sam" in self._models:
            return self._models["sam"]

        checkpoint_path = self.discover_checkpoint("sam")
        logger.info(f"Loading SAM from weights: {checkpoint_path}")
        
        try:
            # We try to load using standard sam_model_registry
            sam = None
            for model_type in ["vit_l", "vit_h"]:
                try:
                    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
                    logger.info(f"✓ Loaded standard SAM using model type: {model_type}")
                    break
                except Exception as load_err:
                    pass
            
            if sam is None:
                raise RuntimeError("Failed to load SAM with vit_l and vit_h configurations.")
                
            sam.to(device)
            sam.eval()
            self._models["sam"] = sam
            self._models["sam_transform"] = ResizeLongestSide(sam.image_encoder.img_size)
            logger.info("✓ SAM model loaded successfully.")
            return sam
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            raise

    def load_clip(self, device: torch.device):
        if "clip" in self._models:
            return self._models["clip"]

        logger.info("Loading ViT-SO400M-14-SigLIP (webli pretrained)...")
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
            clip_model = clip_model.to(device)
            clip_model.eval()
            
            self._models["clip"] = clip_model
            self._models["clip_preprocess"] = preprocess
            logger.info("✓ CLIP model loaded successfully.")
            return clip_model
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise

    def get_model(self, name: str):
        return self._models.get(name.lower())

    def get_config(self, name: str):
        return self._configs.get(name.lower())
