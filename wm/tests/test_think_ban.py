"""Thinking-token yasağı (CPU)."""
import torch

from wm.sampler import (make_ban_processor, make_vllm_logits_processor,
                        make_vllm_request_processor)

KEY = 0x92E5A7C8A419E4D1
V = 300
BAN = [7, 11]
g = torch.Generator().manual_seed(3)

# 1) ban boşsa: eski davranışla bire bir aynı
ref = make_vllm_logits_processor(KEY, 1.2, 0.95, k=2)
new = make_vllm_request_processor(KEY, 1.2, 0.95, k=2, ban_ids=())
out_r, out_n = [], []
for _ in range(40):
    base = torch.randn(V, generator=g) * 3
    a = ref([1, 2, 3] + out_r, base.clone())
    row = base.clone()
    b = new([1, 2, 3], out_n, row)
    assert torch.equal(a.to(row.dtype), row)
    out_r.append(int(a.argmax())); out_n.append(int(b.argmax()))
print("ok  ban=() → eski davranış birebir")

# 2) yasaklı token, modelin açık ara 1. tercihi olsa bile asla seçilmez (watermark'lı)
proc = make_vllm_request_processor(KEY, 1.2, 0.95, k=2, ban_ids=BAN)
out = []
for step in range(300):
    base = torch.randn(V, generator=g)
    base[BAN[step % 2]] += 50.0                 # yasaklı token baskın
    row = base.clone()
    proc([1, 2, 3], out, row)
    t = int(row.argmax())
    assert t not in BAN, (step, t)
    assert torch.isinf(row[BAN]).all() and (row[BAN] < 0).all()
    out.append(t)
print("ok  watermark'lı: yasaklı token 300 adımda hiç seçilmedi (baskın olsa bile)")

# 3) yalnız-yasak işlemcisi (watermark'sız): diğer logit'lere DOKUNMAZ
bp = make_ban_processor(BAN)
base = torch.randn(V, generator=g)
row = base.clone()
bp([1], [], row)
keep = [i for i in range(V) if i not in BAN]
assert torch.equal(row[keep], base[keep]) and torch.isinf(row[BAN]).all()
print("ok  watermark'sız: yalnız yasaklı logit'ler -inf, gerisi aynen")
print("BÜTÜN YASAK TESTLERİ GEÇTİ")
