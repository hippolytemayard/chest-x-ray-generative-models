from collections import namedtuple
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import einsum, nn

from src.utils.helpers import once

# constants

AttentionConfig = namedtuple("AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])

# helpers

print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once("Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda")
            self.cuda_config = AttentionConfig(False, True, True)

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
