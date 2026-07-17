import torch

from models.backbone.transformer.config import ToyBackboneConfig
from models.backbone.transformer.model import ToyBackbone
from models.depth_transformer.config import ToyDepthConfig
from models.depth_transformer.model import ToyDepthTransformer
from models.embeddings.codec_embeddings import MimiCodebookEmbeddings
from training.losses.ce_loss import backbone_loss, depth_loss
from training.optimizers.build_optimizer import build_optimizer


def _build_models(hidden_size=32, text_vocab_size=20):
    codec_embeddings = MimiCodebookEmbeddings(hidden_size=hidden_size)
    backbone = ToyBackbone(
        ToyBackboneConfig(
            text_vocab_size=text_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
        ),
        codec_embeddings,
    )
    depth = ToyDepthTransformer(
        ToyDepthConfig(backbone_hidden_size=hidden_size, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32),
        codec_embeddings,
    )
    return backbone, depth


def test_forward_shapes():
    backbone, depth = _build_models()
    b, tt, ta = 2, 4, 5
    text_ids = torch.randint(0, 20, (b, tt))
    audio_ids = torch.randint(0, 2048, (b, ta))
    hidden, logits = backbone(text_ids, audio_ids)
    assert hidden.shape == (b, ta, 32)
    assert logits.shape == (b, ta, 2048)

    n = b * ta
    depth_logits = depth(hidden.reshape(n, 32), torch.randint(0, 2048, (n, 6)))
    assert depth_logits.shape == (n, 7, 2048)


def test_optimizer_dedupes_shared_embeddings():
    backbone, depth = _build_models()
    optimizer = build_optimizer([backbone, depth], lr=1e-3)
    seen = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in seen
            seen.add(id(p))


def test_loss_decreases_on_repeated_batch():
    backbone, depth = _build_models()
    optimizer = build_optimizer([backbone, depth], lr=1e-2)

    b, tt, ta = 2, 4, 6
    text_ids = torch.randint(0, 20, (b, tt))
    codes = torch.randint(0, 2048, (b, 8, ta))
    audio_mask = torch.ones((b, ta), dtype=torch.long)

    def step():
        hidden, logits0 = backbone(text_ids, codes[:, 0, :])
        loss_bb = backbone_loss(logits0, codes[:, 0, :], audio_mask)
        flat_hidden = hidden.reshape(b * ta, -1)
        flat_targets = codes[:, 1:8, :].permute(0, 2, 1).reshape(b * ta, 7)
        flat_input = codes[:, 1:7, :].permute(0, 2, 1).reshape(b * ta, 6)
        flat_mask = audio_mask.reshape(b * ta)
        depth_logits = depth(flat_hidden, flat_input)
        loss_depth = depth_loss(depth_logits, flat_targets, flat_mask)
        loss = loss_bb + loss_depth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    first = step()
    for _ in range(20):
        last = step()
    assert last < first
