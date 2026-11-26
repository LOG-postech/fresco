import torch
import torch.nn as nn

from .util import Block

class StageFirst(nn.Module):

    def __init__(self, n_embd: int = 384, n_head: int = 6, block_size: int = 256, vocab_size: int = 1000, dropout: float = 0.2, n_layer: int = 1) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input0: torch.Tensor) -> torch.Tensor:
        idx = input0.clone()
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)

        return x
