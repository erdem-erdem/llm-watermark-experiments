"""vLLM V1 istek-düzeyi işlemci sarmalayıcısı (CPU; vllm gerekmez).

make_vllm_request_processor(prompt_ids, output_ids, logits) ==
make_vllm_logits_processor(prompt+output, logits) ve sonuç YERİNDE yazılır.
"""
import torch

from wm.sampler import make_vllm_logits_processor, make_vllm_request_processor

KEY = 0x92E5A7C8A419E4D1  # 63. bit açık: 64-bit anahtar yolu
V, STEPS = 400, 60


def run():
    g = torch.Generator().manual_seed(1)
    prompt = [5, 9, 11, 3]
    ref = make_vllm_logits_processor(KEY, 0.8, 0.95, k=2)
    new = make_vllm_request_processor(KEY, 0.8, 0.95, k=2)
    out_ref, out_new = [], []
    for _ in range(STEPS):
        base = torch.randn(V, generator=g) * 3
        # küçük sözlükte tekrar pencereleri de oluşsun diye bazı adımlarda
        # tek bir token baskın
        if len(out_ref) % 7 == 3:
            base[7] += 20
        a = ref(prompt + out_ref, base.clone())
        row = base.clone()
        b = new(prompt, out_new, row)
        assert b is row, "sonuç aynı tensör olmalı (yerinde)"
        assert torch.equal(row, a.to(row.dtype)), "sonuç referansla aynı olmalı"
        ta, tb = int(torch.argmax(a)), int(torch.argmax(b))
        assert ta == tb
        out_ref.append(ta)
        out_new.append(tb)
    # sağlama: 64-bit anahtar hex gidiş-dönüşü
    assert int(f"{KEY:016x}", 16) == KEY
    print(f"ok  istek-düzeyi işlemci == referans ({STEPS} adım, yerinde yazım, tekrar pencereleri dahil)")


if __name__ == "__main__":
    run()
    print("BÜTÜN VLLM-ADAPTÖR TESTLERİ GEÇTİ")
