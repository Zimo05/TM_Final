import torch.nn as nn

import sys
from pathlib import Path

_TRANSFORMER_DIR = str(Path(__file__).resolve().parent)
if _TRANSFORMER_DIR not in sys.path:
    sys.path.insert(0, _TRANSFORMER_DIR)

from TransformerModel.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # mask after self attention: 防止 padding 位置的注意力输出传播
        # mask after FFN: 确保前馈网络不会在 padding 位置产生非零值
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
