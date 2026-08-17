"""
Prompt Templates
=================

Combined template set from the baseline paper (55 templates) + ViLD additions.
Supports article handling ({article}) and "This is " prefix matching the
original paper's text feature computation.

Reference:
    Enhancing Novel Object Detection via Cooperative Foundational Models
    Bharadwaj et al., WACV 2025

    ViLD: Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation
    Paper: https://arxiv.org/abs/2104.13921
"""


def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'


def processed_name(name, rm_dot=False):
    res = name.replace('_', ' ').replace('/', ' or ').lower()
    if rm_dot:
        res = res.rstrip('.')
    return res


def get_vild_templates():
    """
    Returns prompt templates used for ensemble text embeddings.
    Uses {article} as placeholder for article (a/an) and {} for the class name.

    Returns:
        list: template strings
    """
    return [
        "There is {article} {} in the scene.",
        "There is the {} in the scene.",
        "a photo of {article} {} in the scene.",
        "a photo of the {} in the scene.",
        "a photo of one {} in the scene.",
        "itap of {article} {}.",
        "itap of my {}.",
        "itap of the {}.",
        "a photo of {article} {}.",
        "a photo of my {}.",
        "a photo of the {}.",
        "a photo of one {}.",
        "a photo of many {}.",
        "a good photo of {article} {}.",
        "a good photo of the {}.",
        "a bad photo of {article} {}.",
        "a bad photo of the {}.",
        "a photo of a nice {}.",
        "a photo of the nice {}.",
        "a photo of a cool {}.",
        "a photo of the cool {}.",
        "a photo of a weird {}.",
        "a photo of the weird {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of a clean {}.",
        "a photo of the clean {}.",
        "a photo of a dirty {}.",
        "a photo of the dirty {}.",
        "a bright photo of {article} {}.",
        "a bright photo of the {}.",
        "a dark photo of {article} {}.",
        "a dark photo of the {}.",
        "a photo of a hard to see {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of {article} {}.",
        "a low resolution photo of the {}.",
        "a cropped photo of {article} {}.",
        "a cropped photo of the {}.",
        "a close-up photo of {article} {}.",
        "a close-up photo of the {}.",
        "a jpeg corrupted photo of {article} {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of {article} {}.",
        "a blurry photo of the {}.",
        "a pixelated photo of {article} {}.",
        "a pixelated photo of the {}.",
        "a black and white photo of the {}.",
        "a black and white photo of {article} {}.",
        "a plastic {}.",
        "the plastic {}.",
        "a toy {}.",
        "the toy {}.",
        "a plushie {}.",
        "the plushie {}.",
        "a cartoon {}.",
        "the cartoon {}.",
        "an embroidered {}.",
        "the embroidered {}.",
        "a painting of the {}.",
        "a painting of a {}.",
    ]


def get_template_count():
    """Returns the number of templates."""
    return len(get_vild_templates())


def format_class_prompts(class_name, templates=None, add_this_is=True):
    """
    Format all templates for a given class name with article handling
    and optional 'This is ' prefix.

    Args:
        class_name: Raw class name (e.g., "tuxedo", "apple")
        templates: Template list (defaults to get_vild_templates())
        add_this_is: Whether to prepend 'This is ' to prompts starting
                     with 'a ' or 'the '

    Returns:
        list of formatted prompt strings
    """
    if templates is None:
        templates = get_vild_templates()
    name = processed_name(class_name, rm_dot=True)
    art = article(name)

    prompts = []
    for template in templates:
        prompt = template.format(name, article=art)
        if add_this_is and (prompt.startswith('a ') or prompt.startswith('the ')):
            prompt = 'This is ' + prompt
        prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    templates = get_vild_templates()
    print(f"Loaded {len(templates)} templates")
    print("\nExample with 'cat':")
    for p in format_class_prompts("cat")[:5]:
        print(f"  - {p}")
    print("\nExample with 'apple':")
    for p in format_class_prompts("apple")[:5]:
        print(f"  - {p}")
