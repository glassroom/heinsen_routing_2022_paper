# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
from heinsen_routing import EfficientVectorRouting as Routing

from data import Modality


def _get_pretrained_transformer(modality):
    if (modality == Modality.TEXT) or (modality == Modality.TEXT.name):
        config = { 'name': 'roberta-large', 'revision': '5069d8a', 'd_depth': 25, 'chunk_len': 512, }
        tokenizer = AutoTokenizer.from_pretrained(config['name'], revision=config['revision'])
        transformer = AutoModelForMaskedLM.from_pretrained(config['name'], output_hidden_states=True, revision=config['revision'])
    elif (modality == Modality.IMGS) or (modality == Modality.IMGS.name):
        config = {
            'name': 'microsoft/beit-large-patch16-224', 'revision': '0bd443c', 'd_depth': 25,
        }
        tokenizer = None
        transformer = AutoModelForImageClassification.from_pretrained(config['name'], output_hidden_states=True, revision=config['revision'])
    else:
        raise ValueError(f'Modality not recognized: "{modality}".')
    return config, tokenizer, transformer


class RoutingHead(nn.Module):
    """Route [n pos, d_depth, d_inp] to [n_out, d_out] ([n_out] if d_out is 1)."""

    def __init__(self, transformer_config, kwds_by_routing):
        super().__init__()
        d_depth, d_inp = (transformer_config['d_depth'], kwds_by_routing[0]['d_inp'])
        self.normalize = nn.LayerNorm(d_inp, elementwise_affine=False)
        self.W = nn.Parameter(torch.ones(d_depth, d_inp))
        self.B = nn.Parameter(torch.zeros(d_depth, d_inp))
        self.route = nn.Sequential(*[Routing(**kwds) for kwds in kwds_by_routing])

    def forward(self, x):
        x = self.normalize(x)      # [..., n pos, d_depth, d_inp]
        x = x * self.W + self.B    # [..., n pos, d_depth, d_inp]
        x = x.flatten(-3,-2)       # [..., n_inp, d_inp]
        x = self.route(x)          # [..., n_out, d_out]
        return x.squeeze(-1)       # if d_out is 1, remove it


class MultiModalClassifier(nn.Module):
    """Apply a transformer to chunks of tokens; join if necessary; classify them."""

    def __init__(self, modality, n_classes, n_hid=64):
        super().__init__()
        self.modality = modality if modality in list(Modality) else getattr(Modality, modality)
        self.transformer_config, self.tokenizer, self.transformer = _get_pretrained_transformer(self.modality)

        # Create 'pad chunk' for padding sequences processed in chunks by the transformer:
        if self.modality == Modality.TEXT:
            self.transformer.eval()
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    n_pad = self.transformer_config['chunk_len'] - 2
                    bos_id, eos_id, pad_id = (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
                    pad_chunk = torch.stack(self.transformer(
                        input_ids=torch.tensor([bos_id, eos_id] + [pad_id]*n_pad)[None, :],
                        attention_mask=torch.tensor([1, 1] + [0]*n_pad)[None, :],
                    ).hidden_states, dim=-2)  # [1, n pos, depth, d_emb]
            self.register_buffer('pad_chunk', pad_chunk)  # [1, chunk_len, depth, d_emb]
        else:  # Modality.IMGS
            self.pad_chunk = None

        # Create classification head:
        d_emb = self.transformer.config.hidden_size
        d_hid, n_cls = (d_emb, n_classes)
        self.head = RoutingHead(self.transformer_config, kwds_by_routing=[
            { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
            { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
            { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
        ])

    def body(self, x):
        if self.modality == Modality.TEXT:
            input_ids, attn_mask, chunk_to_seq_id = (x['input_ids'], x['attention_mask'], x['overflow_to_sample_mapping'])  # x is a dict computed by tokenizer
            x = torch.stack(self.transformer(input_ids=input_ids, attention_mask=attn_mask).hidden_states, dim=-2)  # [chunks in batch, n pos, d_depth, d_emb]
            n_seqs = chunk_to_seq_id.max().item() + 1  # number of seqs in batch; each seq may span more than one chunk
            if n_seqs < len(x):
                bool_idxs_to_chunks = (torch.arange(n_seqs, device=chunk_to_seq_id.device)[:, None] == chunk_to_seq_id)  # [n_seqs, n chunks in batch]
                x = [x[idx] for idx in bool_idxs_to_chunks]  # [[1st sample's chunks, n pos, depth, d_emb], [2nd sample's chunks, n pos, d_depth, d_emb], ...]
                n_per_sample = max(len(chunks) for chunks in x)
                x = [chunks if (len(chunks) == n_per_sample) else torch.cat([chunks] + [self.pad_chunk]*(n_per_sample - len(chunks)), dim=0) for chunks in x]
                x = torch.stack([chunks.view(-1, *chunks.shape[-2:]) for chunks in x], dim=0)  # [batch sz, padded & concatenated n pos, d_depth, d_emb]
        else:  # Modality.IMGS
            x = torch.stack(self.transformer(x).hidden_states, dim=-2)  # [batch sz, n pos, d_depth, d_emb]
        return x

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x