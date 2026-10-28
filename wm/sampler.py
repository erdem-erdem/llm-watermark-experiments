"""Örnekleyici modülü — SPEC.md Bölüm 3.2.

Gumbel-max: her adımda T ve top-p uygulanmış dağılım p üzerinde
  argmax_v [ log p_v - log(-log R_v) ]
seçilir (argmax R^(1/p) ile eşdeğerdir: exponential-race formu; P(v)=p_v).

Tekrar kuralı: pencere bu cevapta daha önce görüldüyse watermark atlanır,
aynı formül anahtardan bağımsız tohumlu tekdüze U ile kullanılır (normal
örneklemeye denktir).

Motor entegrasyonu:
- vLLM: SamplingParams.logits_processors'a verilen callable; greedy
  (temperature=0) çözümleme ile kullanılır, T/top-p bu işlemcide uygulanır.
- HF: transformers LogitsProcessor alt sınıfı; do_sample=False ile kullanılır.

Duruş (state): pencere ve tekrar kümesi üretilen token ID'lerinin sonundan
hesaplanır; callable'a gelen token_ids (prompt+üretim) her çağrıda tam
geçtiği için state'siz yeniden hesaplama mümkündür (O(n·k); cevap ~512
token, k=2 için ihmal edilebilir).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from wm.prf import R_vec


def top_p_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p filtresi: kümülatif olasılık top_p'i aşan kuyruk maskelenir (p=0).

    logits: (V,) ya da (B,V). Dönüş aynı biçimde boolean maske (True=koru).
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # İlk top_p'i aşan ilk token korunur (klasik kural: cum - sorted_probs < top_p)
    keep_sorted = (cum - sorted_probs) < top_p
    keep_sorted[..., 0] = True
    mask = torch.zeros_like(keep_sorted)
    mask.scatter_(-1, sorted_idx, keep_sorted)
    return mask


def gumbel_max_logits(
    logits: torch.Tensor,
    key: int,
    window: Sequence[int],
    temperature: float,
    top_p: float,
    skip_watermark: bool,
    rng_state: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, int, int, int]:
    """Gumbel-max logits'i üret.

    Dönüş: (islenmis_logits, yeni_rng_a, yeni_rng_b, yeni_rng_c)
    rng_state: anahtardan bağımsız tekrar penceresi örneklemesi için
    tohumlanmış PCG benzeri durum (a,b,c). None ise 0'lardan başlar.

    Isınmış yol: log p - log(-log R) ; maskeleme sonrası p=0 olan yerlerde
    -inf yazılır. Tekrar penceresi: R yerine tohumlu U kullanılır.
    """
    V = logits.shape[-1]
    device = logits.device
    # 1) sıcaklık + top-p dağılımı
    z = logits / max(temperature, 1e-6)
    mask = top_p_mask(z, top_p)
    logp = torch.log_softmax(z.masked_fill(~mask, float("-inf")), dim=-1)

    a, b, c = (rng_state or (0, 0, 0))
    if skip_watermark:
        # Anahtardan bağımsız tekdüze U: tohumlanmış üçlü LCG-vari karışım
        a, b, c = _next_rng(a, b, c)
        seed = (a ^ b ^ c) & 0xFFFFFFFFFFFF
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        U = torch.rand(V, generator=gen, dtype=torch.float64).to(device)
        Rv = U
    else:
        Rv = R_vec(key, window, V, device=device).to(device)

    with torch.no_grad():
        # Gumbel gürültüsü: G = -log(-log U); skor = log p + G
        #              = log p - log(-log R)   (doküman ipucu formülü)
        neglogR = torch.log(-torch.log(torch.clamp(Rv, min=1e-300)))
        out = logp - neglogR
        out = torch.where(mask, out, torch.full_like(out, float("-inf")))
    return out, a, b, c


def _next_rng(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Basit anahtarsız tohum akışı (64-bit)."""
    a = (a * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    b = (b * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    c = ((c ^ (c >> 12)) * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)
    return a, b, c


def windows_seen(answer_ids: Sequence[int], k: int, start: int = 0):
    """Cevap içindeki (start>0 ise prompt dahil tam dizide) pencerelerin
    geçtiği pozisyonları döndürür: i, window=ids[i-k:i] tanımlı."""
    for i in range(max(k, start), len(answer_ids)):
        yield i


def is_repeated_window(full_ids: Sequence[int], pos: int, k: int,
                       seen: set[tuple[int, ...]] | None = None) -> bool:
    """pos konumundaki pencere (son k token) bu cevapta daha önce geçti mi?
    seen kümesi verilirse hem kontrol hem kayıt yapılır."""
    window = tuple(int(t) for t in full_ids[pos - k:pos])
    if window in seen:
        return True
    if seen is not None:
        seen.add(window)
    return False


# ---------------------------------------------------------------------------
# vLLM entegrasyonu
# ---------------------------------------------------------------------------

def make_vllm_logits_processor(
    key: int,
    temperature: float,
    top_p: float,
    prompt_lens: dict[int, int] | None = None,
    watermark: bool = True,
    k: int = 2,
):
    """vLLM SamplingParams.logits_processors için callable üretir.

    vLLM callable imzası: fn(input_ids: list[int], logits: Tensor) -> Tensor
    input_ids, prompt + o ana kadar üretilen tokenları içerir. Greedy
    çözümleme (temperature=0) ile kullanılmalıdır; T/top-p burada uygulanır.

    Not: vLLM her istek için ayrı callable örneği önerilir (istek başına
    cevap-uzunluğu durumu isteğin kendi callable'ında tutulur).
    """
    seen_windows: set[tuple[int, ...]] = set()
    rng_state = [0, 0, 0]

    def processor(input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if not watermark:
            # Watermark'sız yol: logit'e dokunma; vLLM'in standart örnekleyicisi
            # (T, top-p) ile örnekler. Ancak tutarlılık için aynı kod yolunu
            # kullanıyoruz: tekrar penceresi mantığıyla tohumlu örnekleme.
            # (Bu callable yalnızca greedy çağrıda kullanılır; watermark'sız
            # üretimlerde processor hiç verilmez — gen/generate.py'a bak.)
            return logits
        pos = len(input_ids)
        # pencere: tam dizinin son k tokenı (Bölüm 3.1)
        window = tuple(int(t) for t in input_ids[-k:]) if pos >= k else tuple(input_ids)
        repeat = window in seen_windows
        if not repeat:
            seen_windows.add(window)
        out, a, b, c = gumbel_max_logits(
            logits, key, window, temperature, top_p,
            skip_watermark=repeat, rng_state=tuple(rng_state))
        rng_state[0], rng_state[1], rng_state[2] = a, b, c
        return out.to(logits.dtype)

    return processor


def ban_tokens_(logits: torch.Tensor, ban_ids) -> torch.Tensor:
    """Yasaklı token'ların logit'ini -inf yap (YERİNDE). Watermark'tan ÖNCE
    uygulanır: dağılım p yasaklıları içermez, Gumbel-max kalanlar üzerinden
    çalışır (tespit yalnız üretilen token'ı puanladığı için etkilenmez)."""
    if ban_ids:
        logits[list(ban_ids)] = float("-inf")
    return logits


def make_ban_processor(ban_ids):
    """Yalnızca yasaklama (watermark'sız istekler). vLLM'in standart
    örnekleyicisi (T/top-p/tohum) sonrasında aynen çalışır; -inf sıcaklıktan
    sonra da -inf kalır."""
    ban = tuple(int(i) for i in ban_ids)

    def processor(prompt_ids: list[int], output_ids: list[int],
                  logits: torch.Tensor) -> torch.Tensor:
        return ban_tokens_(logits, ban)

    return processor


def make_vllm_request_processor(key: int, temperature: float, top_p: float,
                                k: int = 2, ban_ids=()):
    """vLLM V1 AdapterLogitsProcessor için istek düzeyi callable.

    İmza: (prompt_ids, output_ids, logits[V]) -> logits[V]. Pencere, TAM
    dizinin (prompt + üretim) son k tokenıdır — tespit tarafıyla aynı
    (wm.detect). Matematik make_vllm_logits_processor ile birebir aynıdır
    (o sarılır). Sonuç logits satırına YERİNDE yazılır ve aynı tensör
    döndürülür (adaptörün atama/yerinde varsayımlarının ikisinde de doğru).
    Her istek için yeni örnek üretilir (tekrar-pencere durumu istek başınadır).
    """
    inner = make_vllm_logits_processor(key, temperature, top_p, k=k)
    ban = tuple(int(i) for i in ban_ids)

    def processor(prompt_ids: list[int], output_ids: list[int],
                  logits: torch.Tensor) -> torch.Tensor:
        ban_tokens_(logits, ban)
        out = inner(list(prompt_ids) + list(output_ids), logits)
        if out is not logits:
            logits.copy_(out.to(logits.dtype))
        return logits

    return processor


# ---------------------------------------------------------------------------
# HF transformers entegrasyonu
# ---------------------------------------------------------------------------

def make_hf_logits_processor(key: int, temperature: float, top_p: float,
                             prompt_len: int, watermark: bool = True, k: int = 2):
    """HF generate için LogitsProcessor alt sınıfı üretir (do_sample=False).

    Her adımda input_ids tam dizidir; pencere son k token'dan alınır.
    """
    from transformers import LogitsProcessor as HFLogitsProcessor

    seen_windows: set[tuple[int, ...]] = set()
    rng_state = [0, 0, 0]

    class _Proc(HFLogitsProcessor):
        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            # batch=1 varsayımı (eğitim dışı üretimde batch=1 kullanılır)
            ids = input_ids[0].tolist()
            if not watermark:
                # watermark'sız: standart örnekleme yolunda bu processor
                # hiç eklenmez; greedy karşılaştırmalar için burada dokunma
                return scores
            pos = len(ids)
            window = tuple(ids[-k:]) if pos >= k else tuple(ids)
            repeat = window in seen_windows
            if not repeat:
                seen_windows.add(window)
            out, a, b, c = gumbel_max_logits(
                scores, key, window, temperature, top_p,
                skip_watermark=repeat, rng_state=tuple(rng_state))
            rng_state[0], rng_state[1], rng_state[2] = a, b, c
            return out.to(scores.dtype)

    return _Proc()


def text_detection_scores(answer_ids: Sequence[int], prompt_len: int, key: int,
                          k: int = 2):
    """Metin tespiti için token başına (pozisyon, pencere) üretir.

    Bölüm 3.4 kuralları (a),(b),(c) burada; (d) global küme detect.py'da.
    Dönüş: list[(pos, window_tuple)] — puanlanacak pozisyonlar.
    """
    ids = list(answer_ids)
    out = []
    seen_here: set[tuple[int, ...]] = set()
    for pos in range(len(ids)):
        if pos < k:                       # (a) cevabın ilk k tokenı
            continue
        window = tuple(ids[pos - k:pos])
        if window in seen_here:           # (c) aynı cevapta daha önce
            continue
        seen_here.add(window)
        out.append((pos, window))
    return out