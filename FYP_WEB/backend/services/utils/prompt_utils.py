import random
import torch
from typing import List, Dict, Any, Tuple

def build_captions_and_token_span(cat_list: List[str], force_lowercase: bool = True) -> Tuple[str, Dict[str, List[List[int]]]]:
    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan

def create_positive_map_from_span(tokenized, token_span, max_text_len: int = 256) -> torch.Tensor:
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def get_text_prompt_list_for_g_dino(classes: List[str], tokenizer, class_len_per_prompt: int) -> Tuple[List[str], List[torch.Tensor]]:
    classes = [i.lower() for i in classes]
    classes = [s.replace("_", " ") for s in classes]

    classes_split = [classes[i:i + class_len_per_prompt] for i in range(0, len(classes), class_len_per_prompt)]
    
    text_prompt_list = []
    positive_map_list = []
    for classes_subset in classes_split:
        captions, cat2tokenspan = build_captions_and_token_span(classes_subset, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in classes_subset if cat in cat2tokenspan]
        
        # In case some class names are skipped
        if len(tokenspanlist) == 0:
            continue
            
        positive_map = create_positive_map_from_span(tokenizer(captions), tokenspanlist)
        positive_map_list.append(positive_map)
        text_prompt_list.append(captions)

    return text_prompt_list, positive_map_list
