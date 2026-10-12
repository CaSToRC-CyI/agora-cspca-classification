############ OMNIRAD FINE-TUNING HELPER FUNCTIONS #############
# ── helpers ──────────────────────────────────────────────────────────────────


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# ── Option A: last N blocks + final norm ─────────────────────────────────────


def unfreeze_last_n_blocks(model, n=4):
    freeze_all(model)
    total_blocks = len(model.blocks)  # 12 for this model
    for i in range(total_blocks - n, total_blocks):
        unfreeze_module(model.blocks[i])
    unfreeze_module(model.norm)  # final LayerNorm before head


# ── Option B: last N blocks + patch embedding + final norm ───────────────────


def unfreeze_last_n_blocks_and_embed(model, n=4):
    freeze_all(model)
    total_blocks = len(model.blocks)
    for i in range(total_blocks - n, total_blocks):
        unfreeze_module(model.blocks[i])
    unfreeze_module(model.norm)
    unfreeze_module(model.patch_embed)  # Conv2d projection


# ── Option C: attention sublayers only (all blocks) ──────────────────────────


def unfreeze_attention_only(model):
    freeze_all(model)
    for block in model.blocks:
        unfreeze_module(block.attn)
        # also unfreeze the pre-attention LayerNorm
        unfreeze_module(block.norm1)
    unfreeze_module(model.norm)


# ── Option D: LayerScale only (ultra-lightweight) ────────────────────────────


def unfreeze_layerscale_only(model):
    freeze_all(model)
    for block in model.blocks:
        unfreeze_module(block.ls1)
        unfreeze_module(block.ls2)
    unfreeze_module(model.norm)


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# Map versions to functions
# Lambda helps accepting 'n' and ignoring if not needed.
FINETUNE_MAP = {
    "v0": lambda model, n=4: freeze_all(model),  # Transfer Learning
    "v1": lambda model, n=4: unfreeze_last_n_blocks(model, n=n),
    "v2": lambda model, n=4: unfreeze_last_n_blocks_and_embed(model, n=n),
    "v3": lambda model, n=4: unfreeze_attention_only(model),
    "v4": lambda model, n=4: unfreeze_layerscale_only(model),
    "v5": lambda model, n=12: unfreeze_all(model),
}


# ── Usage Example ─────────────────────────────────────────────────────────────────────

# unfreeze_last_n_blocks(encoder, n=4)
# count_trainable(encoder)

# # then pass only requires_grad=True params to your optimizer
# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, encoder.parameters()),
#     lr=1e-4
# )
