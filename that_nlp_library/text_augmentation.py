# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_text_augmentation.ipynb.

# %% ../nbs/01_text_augmentation.ipynb 3
from __future__ import annotations
import unidecode
import random

# %% auto 0
__all__ = ['remove_vnmese_accent', 'fill_mask_augmentation']

# %% ../nbs/01_text_augmentation.ipynb 4
def remove_vnmese_accent(sentence:str, # Input sentence
                         prob=1, # Probability that this function is applied to the text
                        ):
    "Perform Vietnamese accent removal"
    return unidecode.unidecode(sentence) if random.random()<prob else sentence

# %% ../nbs/01_text_augmentation.ipynb 9
def fill_mask_augmentation(sentence:str, # Input Sentence,
                           fillmask_pipeline, # HuggingFace fill-mask pipeline
                           prob=1, # Probability that this function is applied to the text
                           random_top_k=1, # To select output randomly from top k mask filled
                          ):
    # References: https://huggingface.co/docs/datasets/v2.14.1/en/process#data-augmentation
    if random.random()>=prob: return sentence
    mask_token = fillmask_pipeline.tokenizer.mask_token
    words = sentence.split(' ')
    K = random.randint(1, len(words)-1)
    masked_sentence = " ".join(words[:K]  + [mask_token] + words[K+1:])
    predictions = fillmask_pipeline(masked_sentence,top_k = random_top_k)
    weights = [p['score'] for p in predictions]
    sentences = [p['sequence'] for p in predictions]
    return random.choices(sentences, weights = weights, k = 1)[0]
    
