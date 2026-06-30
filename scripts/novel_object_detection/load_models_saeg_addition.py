import open_clip
import torch
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detectron2.data import MetadataCatalog
from vild_templates import get_vild_templates
from saeg_module import SAEGTextFeatureGenerator


def load_clip_model_saeg(data_split, device):
    """
    Load CLIP model with SAEG (Synonym Averaged Embedding Generator).

    This function implements the full SAEG algorithm from the paper:
    - Uses 64 ViLD templates instead of 60 custom templates
    - Averages embeddings across templates for each synonym (Eq. 1)
    - Averages embeddings across synonyms for each class (Eq. 2)
    - Produces final text feature matrix (Eq. 3)

    Args:
        data_split: Dataset split name (e.g., 'lvis_v1_val')
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        clip_model: Loaded SigLIP model
        preprocess: Image preprocessing function
        text_features: SAEG-generated text features (num_classes, embedding_dim)
        lvis_classes: List of class names

    Reference:
        Paper Section 3.2 "Unknown Object Labelling" - SAEG component
    """
    # Load the SigLIP model
    print("[SAEG] Loading ViT-SO400M-14-SigLIP (webli pretrained)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP', pretrained='webli'
    )
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
