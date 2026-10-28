"""vLLM V1 entegrasyonu — watermark LogitsProcessor sınıfı.

vLLM V1 motoru istek başına `SamplingParams.logits_processors` callable'larını
KABUL ETMEZ ("does not support per request user provided logits processors").
Resmi yol: motor açılırken sınıf kaydedilir, istek başına ayarlar
`SamplingParams.extra_args` ile taşınır. AdapterLogitsProcessor, istek
düzeyindeki bir callable'ı (prompt_ids, output_ids, logits) bu yapıya sarar.

  llm = LLM(model, logits_processors=["wm.vllm_proc:WatermarkLogitsProcessor"])
  SamplingParams(temperature=0, extra_args={"wm_key": "<hex>", "wm_T": 0.8,
                                            "wm_top_p": 0.95})

"wm_ban": [id, ...] verilirse o token'lar yasaklanır (thinking <think>/</think>);
watermark'sız isteklerde yalnız yasaklama uygulanır. extra_args'ı olmayan
istekler işlemciden etkilenmez.
Matematik wm.sampler'dakiyle AYNIDIR (make_vllm_request_processor onu sarar);
burada yalnızca motor bağlantısı vardır.
"""

from __future__ import annotations

from typing import Any

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (AdapterLogitsProcessor,
                                              RequestLogitsProcessor)

from wm.sampler import make_ban_processor, make_vllm_request_processor

FQCN = "wm.vllm_proc:WatermarkLogitsProcessor"


def watermark_extra_args(key: int, temperature: float, top_p: float,
                         k: int = 2) -> dict[str, Any]:
    """İstek başına ayarlar. Anahtar hex metin: 64-bit değer IPC'de taşma/
    imza sorunu yaşamasın."""
    return {"wm_key": f"{int(key):016x}", "wm_T": float(temperature),
            "wm_top_p": float(top_p), "wm_k": int(k)}


class WatermarkLogitsProcessor(AdapterLogitsProcessor):
    """Gumbel-max watermark'ı (T/top-p dahil) istek düzeyinde uygular."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        ea = params.extra_args or {}
        if "wm_ban" in ea:
            if not all(isinstance(i, int) for i in ea["wm_ban"]):
                raise ValueError(f"wm_ban tam sayı listesi olmalı: {ea['wm_ban']!r}")
        if "wm_key" not in ea:
            return
        try:
            int(ea["wm_key"], 16)
            float(ea["wm_T"])
            float(ea["wm_top_p"])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"geçersiz watermark extra_args: {ea!r} ({e})")
        if params.temperature != 0:
            raise ValueError("watermark'lı istek temperature=0 (greedy) olmalı; "
                             "T ve top-p işlemcinin içinde uygulanır")

    def is_argmax_invariant(self) -> bool:
        # Logits'i yeniden şekillendirir → greedy argmax'ı DEĞİŞTİRİR.
        return False

    def new_req_logits_processor(self, params: SamplingParams
                                 ) -> RequestLogitsProcessor | None:
        ea = params.extra_args or {}
        ban = ea.get("wm_ban") or ()
        if "wm_key" in ea:
            return make_vllm_request_processor(
                key=int(ea["wm_key"], 16), temperature=float(ea["wm_T"]),
                top_p=float(ea["wm_top_p"]), k=int(ea.get("wm_k", 2)),
                ban_ids=ban)
        if ban:
            return make_ban_processor(ban)
        return None
