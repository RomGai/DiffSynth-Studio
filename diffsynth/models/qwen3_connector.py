from typing import Optional

import torch
from torch import nn


class Qwen3TokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        heads_num: int,
        depth: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.context_proj = nn.Linear(in_channels, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads_num,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        if mask is not None:
            key_padding_mask = mask == 0
        else:
            key_padding_mask = None

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            mask_float = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        context = self.context_proj(pooled)

        time_input = t.float().view(-1, 1)
        time_embed = self.time_mlp(time_input)

        x = x + (context + time_embed).unsqueeze(1)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


class Qwen3Connector(nn.Module):
    def __init__(
        self,
        in_channels: int = 4096,
        hidden_size: int = 3584,
        heads_num: int = 28,
        depth: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.refiner = Qwen3TokenRefiner(
            in_channels=in_channels,
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
        ).to(**factory_kwargs)
        self.global_proj_out = nn.Linear(in_channels, 768, **factory_kwargs)
        self.scale_factor = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.scale_factor.data += -(1 - 0.09)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_float = mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device) if mask is not None else None
        if mask_float is None:
            x_mean = x.mean(dim=1)
        else:
            x_mean = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        x_mean = x_mean * (1 + self.scale_factor.to(dtype=x.dtype, device=x.device))
        global_out = self.global_proj_out(x_mean)
        encoder_hidden_states = self.refiner(x, t, mask)
        return encoder_hidden_states, global_out
