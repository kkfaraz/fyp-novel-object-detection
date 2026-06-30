"""
ViLD Prompt Templates
=====================

This module provides the 64 prompt templates from the ViLD paper, which are
used by the SAEG (Synonym Averaged Embedding Generator) component.

Reference:
    ViLD: Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation
    Paper: https://arxiv.org/abs/2104.13921
    Templates adapted from: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/vild/configs/vild_ensemble_prompts.py
"""

def get_vild_templates():
    """
    Returns the 64 prompt templates used in ViLD for ensemble text embeddings.
    
    These templates are designed to provide diverse linguistic contexts for
    class names, improving the robustness of CLIP text embeddings.
    
    Returns:
        list: 64 template strings with {} placeholder for class name
    
    Example:
        >>> templates = get_vild_templates()
        >>> len(templates)
        64
        >>> templates[0].format("cat")
        'a photo of a cat.'
    """
    return [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "a blurry photo of the {}.",
        "a black and white photo of the {}.",
        "a low contrast photo of the {}.",
        "a high contrast photo of the {}.",
        "a bad photo of the {}.",
        "a good photo of the {}.",
        "a photo of the small {}.",
        "a photo of the big {}.",
        "a photo of a {}.",
        "a photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "a cropped photo of a {}.",
        "a cropped photo of the {}.",
        "a jpeg corrupted photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of a {}.",
        "a blurry photo of the {}.",
        "a pixelated photo of a {}.",
        "a pixelated photo of the {}.",
        "a rendition of a {}.",
        "a rendition of the {}.",
        "a photo of a clean {}.",
        "a photo of the clean {}.",
        "a photo of a dirty {}.",
        "a photo of the dirty {}.",
        "a photo of a nice {}.",
        "a photo of the nice {}.",
        "a photo of a weird {}.",
        "a photo of the weird {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a painting of a {}.",
        "a painting of the {}.",
        "There is a {} in the scene.",
        "There is the {} in the scene.",
        "This is a {} in the scene.",
        "This is the {} in the scene.",
        "This is one {} in the scene.",
        "a {} in a video game.",
        "a {} in the video game.",
        "itap of a {}.",
        "itap of my {}.",
        "itap of the {}.",
        "a plushie {}.",
        "the plushie {}.",
    ]


def get_template_count():
    """Returns the number of templates."""
    return len(get_vild_templates())


if __name__ == "__main__":
    # Simple test
    templates = get_vild_templates()
    print(f"Loaded {len(templates)} ViLD templates")
    print("\nFirst 5 templates:")
    for i, template in enumerate(templates[:5], 1):
        print(f"  {i}. {template}")
    print("\nExample with 'cat':")
    for template in templates[:3]:
        print(f"  - {template.format('cat')}")
