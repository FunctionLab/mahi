# esm_model.py
import numpy as np
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pickle
import sys
import torch

print('starting to generate embeddings from ESM!', flush=True)

def load_esm_model(model_name="esmc_600m", device=None):
    # load the pretrained ESM model
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = ESMC.from_pretrained(model_name).to(device)
    
    return client

def get_esm_embeddings(model, protein_sequence):
    # return the CLS token embeddings as vector

    protein = ESMProtein(sequence=protein_sequence)
    protein_tensor = model.encode(protein)
    logits_output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
    cls_embedding = logits_output.embeddings[:, 0, :].squeeze(0).cpu()
    
    return cls_embedding