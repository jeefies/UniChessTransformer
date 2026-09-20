import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class RelativeAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_tokens=65):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
        # Learned 2D relative position bias for all tokens (65x65)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, num_tokens, num_tokens))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias[:, :N, :N]
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_tokens=65, use_swiglu=True, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RelativeAttention(d_model, num_heads, num_tokens)
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = int(d_model * mlp_ratio)
        if use_swiglu:
            # SwiGLU: 2/3 of 4x to match standard MLP parameter budget
            d_ff_swi = int(2 * d_ff / 3)
            self.mlp = SwiGLU(d_model, d_ff_swi)
        else:
            self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BilinearPolicyHead(nn.Module):
    """
    Policy Head Option A:
    Q = W_q(x) in R^{B x 64 x D_p}, K = W_k(x) in R^{B x 64 x D_p}
    logits = (Q K^T) / sqrt(D_p) + static_bias_{64 x 64} -> flatten to 4096.
    """
    def __init__(self, d_model, d_p=64):
        super().__init__()
        self.d_p = d_p
        self.scale = d_p ** -0.5
        self.wq = nn.Linear(d_model, d_p)
        self.wk = nn.Linear(d_model, d_p)
        self.static_bias = nn.Parameter(torch.zeros(64, 64))
        nn.init.trunc_normal_(self.static_bias, std=0.02)

    def forward(self, square_tokens):
        # square_tokens: (B, 64, d_model)
        Q = self.wq(square_tokens)  # (B, 64, d_p)
        K = self.wk(square_tokens)  # (B, 64, d_p)
        logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale + self.static_bias  # (B, 64, 64)
        return logits.flatten(1)  # (B, 4096)

class ConvPolicyHead(nn.Module):
    """
    Policy Head Option B:
    ResNet-style 1x1 conv / projection head.
    Tokens reshaped to (B, d_model, 8, 8) -> Conv 1x1 to 64 channels -> (B, 64, 8, 8) -> 4096.
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv2d(d_model, 64, kernel_size=1)

    def forward(self, square_tokens):
        # square_tokens: (B, 64, d_model)
        B, N, C = square_tokens.shape
        x = square_tokens.transpose(1, 2).contiguous().view(B, C, 8, 8)
        logits = self.conv(x)  # (B, 64, 8, 8)
        return logits.reshape(B, -1)  # (B, 4096)

class ChessTransformer(nn.Module):
    def __init__(
        self,
        in_channels=19,
        d_model=256,
        num_layers=8,
        num_heads=8,
        stem_type="conv",  # "conv" or "linear"
        policy_head_type="bilinear",  # "bilinear" or "conv"
        use_swiglu=True,
        d_p=64,
    ):
        super().__init__()
        self.d_model = d_model
        self.stem_type = stem_type
        self.policy_head_type = policy_head_type

        # Stem
        if stem_type == "conv":
            self.stem = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)
        else:
            self.stem = nn.Linear(in_channels, d_model)

        # 2D Positional Embeddings for the 64 squares
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks (65 tokens: 1 CLS + 64 squares)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_tokens=65, use_swiglu=use_swiglu)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Heads
        if policy_head_type == "bilinear":
            self.policy_head = BilinearPolicyHead(d_model, d_p=d_p)
        else:
            self.policy_head = ConvPolicyHead(d_model)

        # Promotion Head: MLP from CLS -> 4 logits (Q, R, B, N)
        self.promo_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )

        # Value Head: CLS -> Linear(d_model, 128) -> ReLU -> Linear(128, 3) (WDL)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # Input shape: (B, 19, 8, 8)
        B = x.shape[0]
        if self.stem_type == "conv":
            feat = self.stem(x)  # (B, d_model, 8, 8)
            tokens = feat.flatten(2).transpose(1, 2)  # (B, 64, d_model)
        else:
            # Linear stem expects (B, 64, 19)
            x_flat = x.flatten(2).transpose(1, 2)
            tokens = self.stem(x_flat)  # (B, 64, d_model)

        tokens = tokens + self.pos_embed

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls, tokens], dim=1)  # (B, 65, d_model)

        for block in self.blocks:
            x_seq = block(x_seq)

        x_seq = self.norm(x_seq)

        cls_out = x_seq[:, 0]          # (B, d_model)
        sq_out = x_seq[:, 1:]          # (B, 64, d_model)

        # Compute heads
        policy_logits = self.policy_head(sq_out)   # (B, 4096)
        promo_logits = self.promo_head(cls_out)    # (B, 4)
        value_wdl = self.value_head(cls_out)       # (B, 3)

        return policy_logits, promo_logits, value_wdl

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_profile(model, name, device, dtype=torch.bfloat16):
    print("=" * 80)
    print(f"Profiling Model: {name}")
    param_cnt = count_parameters(model)
    print(f"Parameters: {param_cnt:,} ({param_cnt/1e6:.2f}M)")
    model = model.to(device=device, dtype=dtype)
    model.train()

    # Training benchmark (forward + backward)
    train_batch_sizes = [512, 1024]
    for bs in train_batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            inp = torch.randn(bs, 19, 8, 8, device=device, dtype=dtype)
            p_logits, pr_logits, v_wdl = model(inp)
            loss = p_logits.sum() + pr_logits.sum() + v_wdl.sum()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        start = time.perf_counter()
        iters = 20
        for _ in range(iters):
            optimizer.zero_grad()
            inp = torch.randn(bs, 19, 8, 8, device=device, dtype=dtype)
            p_logits, pr_logits, v_wdl = model(inp)
            loss = p_logits.sum() + pr_logits.sum() + v_wdl.sum()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = (bs * iters) / elapsed
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"  [Train Fwd+Bwd] BS={bs:4d} | Throughput: {throughput:8.1f} pos/sec | Peak Mem: {peak_mem_mb:6.1f} MB | Time: {elapsed*1000/iters:.2f} ms/step")

    # Inference Latency benchmark (for MCTS)
    model.eval()
    infer_batch_sizes = [64, 128]
    with torch.no_grad():
        for bs in infer_batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Warmup
            for _ in range(20):
                inp = torch.randn(bs, 19, 8, 8, device=device, dtype=dtype)
                _ = model(inp)

            torch.cuda.synchronize()
            start = time.perf_counter()
            iters = 100
            for _ in range(iters):
                inp = torch.randn(bs, 19, 8, 8, device=device, dtype=dtype)
                _ = model(inp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            avg_latency_ms = (elapsed * 1000) / iters
            infer_throughput = (bs * iters) / elapsed
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  [Inference]     BS={bs:4d} | Latency: {avg_latency_ms:6.2f} ms | Throughput: {infer_throughput:8.1f} pos/sec | Peak Mem: {peak_mem_mb:6.1f} MB")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    configs = [
        # (name, d_model, layers, heads, stem, policy_head, swiglu)
        ("ChessTransformer-8L-256D-ConvStem-BilinearPolicy", 256, 8, 8, "conv", "bilinear", True),
        ("ChessTransformer-8L-256D-ConvStem-ConvPolicy",     256, 8, 8, "conv", "conv", True),
        ("ChessTransformer-10L-384D-ConvStem-BilinearPolicy", 384, 10, 12, "conv", "bilinear", True),
        ("ChessTransformer-12L-384D-ConvStem-BilinearPolicy", 384, 12, 12, "conv", "bilinear", True),
        ("ChessTransformer-8L-256D-LinearStem-BilinearPolicy", 256, 8, 8, "linear", "bilinear", True),
        ("ChessTransformer-8L-256D-ConvStem-BilinearPolicy-GELU", 256, 8, 8, "conv", "bilinear", False),
    ]

    for name, d_model, layers, heads, stem, pol_head, swiglu in configs:
        model = ChessTransformer(
            in_channels=19,
            d_model=d_model,
            num_layers=layers,
            num_heads=heads,
            stem_type=stem,
            policy_head_type=pol_head,
            use_swiglu=swiglu,
            d_p=64,
        )
        benchmark_profile(model, name, device, dtype=torch.bfloat16)

