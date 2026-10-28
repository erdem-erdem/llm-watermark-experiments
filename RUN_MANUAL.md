# RUN_MANUAL — Watermark Deney Pipeline'ı Çalıştırma Kılavuzu

Bu klasör, blog yazısının Deneyler (Bölüm 5) bölümü için E0–E4 + mini
denemelerin tamamını üretir. Hedef makine: 4×H100, `nvidia-smi` görünür.
Her şey `results/` altına yazılır; her aşama idempotenttir (çıktısı olan
aşama atlanır; `--force` ile zorlanır).

```
llm-watermark-experiments/
├── wm/            # watermark çekirdeği (PRF, sampler, tespit, okuma modu)
├── data/          # InstrucTurca hazırlama + anahtar kalibrasyonu
├── gen/           # üretim (vLLM + custom logits processor)
├── train/         # SFT (HF Trainer) + koşu yapılandırmaları
├── eval/          # AIME25-TR, GPQA-TR, eval_loss, 8-gram örtüşme
├── analyze/       # tespit orkestrasyonu, figürler, tablolar, özet
├── configs/       # YAML yapılandırmaları
├── scripts/       # ortam, koşucu, koşum betikleri
├── results/       # çıktılar (CSV/JSON; şekiller figures/ altında)
├── figures/       # tr/ ve en/ şekiller
├── SPEC.md              # teknik şartname
└── RUN_MANUAL.md        # bu dosya
```

---

## 1. Ortam Hazırlığı

> **Önerilen yol: Docker.** Conda/venv/sistem-kütüphanesi çakışmalarını (ör. `libstdc++`
> `CXXABI_1.3.15`) tamamen ortadan kaldırır. Bkz. **`DOCKER_GUIDE.md`**. Aşağıdaki elle kurulum
> yalnızca Docker kullanılamadığında içindir.

```bash
cd llm-watermark-experiments
python -m venv .venv && source .venv/bin/activate
bash scripts/setup_env.sh
```

- `setup_env.sh`: torch, vllm, transformers, datasets, sentence-transformers,
  scipy, matplotlib kurar ve CUDA'yı doğrular.
- Hugging Face cache'i `HF_HOME` ile önceden tanımlayın (4 GB'lık
  InstrucTurca + 5 model ~60 GB disk):
  ```bash
  export HF_HOME=/disk/hf_cache
  ```
- İndirilecek modeller: `Qwen/Qwen3.5-{0.8B,2B,4B,9B}`, `Qwen/Qwen3.5-2B-Base`,
  `intfloat/multilingual-e5-large`; veri: `turkish-nlp-suite/InstrucTurca`
  (streaming — dosya tam yüklenmez), `ytu-ce-cosmos/tubitak-science-olympiad-tr` (kodda `aime_*` adları miras isimdir),
  `ytu-ce-cosmos/gpqa-extended_tr`.
- Windows dışı ortamda UTF-8 genelde hazır; her koşum betiği
  `PYTHONIOENCODING=utf-8` zaten ayarlar.

### HF önbelleği ve çevrimdışı mod (429 önleme)

`run_all.sh` başında `scripts/predownload.py` çalışır: gereken modelleri/veri setlerini
configlerden türetir, **önce ağsız doğrular**, yalnızca eksikleri indirir. Önbellek
tamamsa `results/.hf_offline_ok` yazılır ve veri hazırlığından sonrası
`HF_HUB_OFFLINE=1` ile koşar (Hub'a istek yok). Bir kez elle de çalıştırabilirsiniz:
`export HF_TOKEN=hf_...; python scripts/predownload.py`. Kapatmak: `HF_OFFLINE_MODE=off`.
Kapılı veri seti (GPQA) alınamazsa çevrimdışı mod AÇILMAZ, uyarı verilir.

## 2. Duman Koşusu (önce bunu çalıştırın)

```bash
bash scripts/run_smoke_all.sh
```

Sıra ve neye bakılacağı:

| # | Aşama | Kontrol |
|---|-------|---------|
| 1 | PRF/sampler testleri | "PASS" satırları; PRF referans≠vektör olursa DUR |
| 2 | Veri hazırlama (N≈200) | `data/train.jsonl`, `DATA_CARD.md` oluştu; alanlar dolu |
| 3 | Anahtar seçimi | `results/E0/selected_keys.json` yazıldı |
| 4 | Tokenizer karşılaştırma | 5 model aynı token dizisi; "UYUMSUZ" ise §4.1 kuralı: DUR |
| 5 | E1 sweep (20×1×1) | `results/E1/sweep.csv` satırları dolu, neglog10p makul |
| 6 | E2 üretim (50×2) | `results/E2/D_gen.jsonl` alan şeması tamam |
| 7 | E3 tek eğitim (1 epoch, 50 satır) | `results/E3/models/A-gen-10` klasörü |
| 8 | Tespit (E0+E2 mini) | CSV'ler dolu |
| 9 | Figürler | `figures/` altında en az 1 PNG oluştu |

Duman koşusu temizse tam koşuya geçin.

**Küçük GPU (≈6 GB, Ampere+) ile duman:** `bash scripts/run_smoke_6gb.sh` — aynı kod
yollarını Qwen3.5-0.8B ile sınar (watermark LogitsProcessor, eğitim, okuma modu, tespit).
Boyut davranışını (9B/2B), çoklu GPU'yu, AIME/GPQA'yı ve E4 B-eğitimlerini sınamaz.
Ayarlar: `WM_GPU_MEM_UTIL`, `WM_MAX_MODEL_LEN`, `WM_EAGER`, `WM_TRAIN_LOWMEM`,
`WM_MODEL_OVERRIDES` (ayarlı değilse üretim koşusu etkilenmez).

## 3. Tam Koşum (elle inceleme YOK)

```bash
bash scripts/run_all.sh 4
```

Betik sırası: testler → veri → anahtarlar → **E1 tarama** → `analyze.e1_report`
→ seçim otomatik yazılır → `run_e1_full.sh` **kendiliğinden** sürer. Durma yok.

**E1 seçimi (otomatik, belirlenimli):**
1. KARAR-8: benzerlik >= 0.85 ve uzunluk geçişi >= %90 koşullarını sağlayanlar
   içinden en yüksek medyan -log10p.
2. Hiçbiri sağlamıyorsa **FALLBACK "verim"**: satırların en büyük payının aynı
   anda benzer (>=0.85), uzunluk bandında (0.75-1.25) ve tespit edilebilir
   (-log10p >= 6) olduğu ayar. Bu bir **SAPMA**dır: `SWEEP_REPORT.md`,
   `selected.json` (`deviation_from_karar8: true`) ve SUMMARY'de açıkça yazılır.
3. Seçim `results/E1/selected.json`'a `approved: true, auto: true` olarak yazılır.
   Elle yazılmış bir `selected.json` varsa **ezilmez**.

Eski davranış (elle onay için dur): `E1_GATE=manual bash scripts/run_all.sh`.
Bir model taramasını yeniden üretmek: ilgili `results/E1/sweep_parts/sweep_*.csv`
dosyasını silin ya da `--force` verin (var olan parçalar atlanır).

`run_e1_full.sh` sırası: E2 üretimi → E1 full (D_post, D_rew) → E3
eğitimleri (4 GPU'da 2 dalga) → E0 kalibrasyon → E2 tespit → E3 eğrileri →
E4 ajan zinciri → AIME/GPQA/eval_loss → mini denemeler → figürler +
tablolar + SUMMARY.

## 4. GPU Dağılımı

Her koşum betiği GPU sayısını ilk argümandan ya da `GPUS` ortam değişkeninden
alır (varsayılan 4); bitişik olmayan seçim için `GPU_IDS`:

```bash
bash scripts/run_all.sh 4          # ya da: GPUS=4 bash scripts/run_all.sh
bash scripts/run_e1_full.sh 2      # 2 GPU ile devam
GPU_IDS=1,3 bash scripts/run_e1_full.sh
```

Ağır aşamaların GPU'ya bölünmesi (GPUS=N'e göre):

| Aşama | Bölünme |
|---|---|
| E1 tarama (12 üretim partisi) | 4 model → model başına 1 iş, N GPU'da paralel |
| E2 üretimi | 3 bağımsız görev (D_gen / D_clean / çeşitlilik) → min(N,3) GPU |
| E3 eğitimleri | 8 koşul, dalga başına N paralel eğitim |
| E4 üretimi | 2 görev (ajan / A₀ kontrol) → min(N,2) GPU |
| E4 öğrenci eğitimleri | 3 eğitim (B-greedy / B-sample / B-control; adım 10b) → min(N,3) GPU |
| GPQA + eval_loss | 12 model, her ölçüt N parçaya bölünür (`--only`), parçalar otomatik birleşir |
| E1 full (D_post/D_rew) | tek sıralı koşum (9B öğretmen; yavaşsa config'te `tp_size: 4`) |
| Olimpiyat (aime_tr) | 8 koşul `--only` ile GPU'lara bölünür; parçalar (`results/E2/aime_parts/`) her koşumda birleşir. Ölçek: `aime_max_problems` (eval.yaml; 0 = tüm sayı-cevaplı satırlar) |
| E3 tespit eğrileri | 8 koşul `--only` ile GPU'lara bölünür; parçalar (`results/E3/detect_curves_parts/`) birleşir. Ölçek: `e3_detect_n_prompts` (detect.yaml; 0 = tüm D_gen) |
| Tespit/figürler/miniler | küçük; tek GPU |

Notlar:
- `N > bağımsız iş sayısı` olan aşamalarda fazladan GPU'lar o aşamada boş
  kalır (ör. E2'de 3 görev var; GPUS=4 ise 1 GPU bekler) — kuyruk modeli
  iş bitince sıradaki işe atlar.
- Duman koşusu küçük ölçeklidir; tek GPU yeter, GPUS onu değiştirmez.
- Üretimler varsayılan TP=1; 9B tek GPU'ya sığmazsa ilgili YAML'da
  `tp_size: 4` yapın (o iş 4 GPU'yu tek vLLM'e verir, paralellik düşer).
- 100-prompt hız testi duman koşusunun E1 adımında otomatik ölçülür.

## 5. Çıktı Yerleri

Şekiller `figures/<tr|en>/main/` altına yazılır (`python -m analyze.make_figures`); tablolar ve ham çıktılar `results/` altındadır.

| Konu | Çıktı |
|---|---|
| Şekiller (TR + EN) | `figures/<lang>/main/fig_*.png/.svg` |
| E0 kalibrasyon, anahtar ayrımı | `results/E0/` (`E0.json`, `T_E0.csv`, `pvals_*.csv`) |
| E1 tarama + seçim | `results/E1/` (`sweep.csv`, `T_E1.csv`, `SWEEP_REPORT.md`, `selected.json`) |
| E2 eğri/dağılım/konu | `results/E2/` (`curves*.csv`, `T_E2.csv`, `per_record.csv`) |
| Matematik olimpiyatı kalite (`aime_*`) | `results/E2/aime_summary.json` |
| E3 okuma modu, kayıp, GPQA | `results/E3/` (`reading_*.csv`, `T_E3.csv`, `loss/*.csv`, `gpqa_tr.json`) |
| E4 zincir | `results/E4/` (`chain.csv`, `T_E4.csv`) |
| Mini denemeler | `results/mini/` |
| Günlük ve manifest | `results/LOG.md`, `results/run_manifest.json` (`results/SUMMARY.md`: `python -m analyze.summary`) |

## 6. Yeniden Koşum / Devam

- Bir aşamayı tek başına koşturun (örnek):
  ```bash
  python -m analyze.detect_all --config configs/detect.yaml --stage E2,E3
  python -m eval.gpqa_tr --config configs/eval.yaml --force
  ```
- Çıktısı olan her adım atlanır; zorlamak için `--force`.
- Eğitim çıktısı varsa `train.sft` atlar; `--force` ile ezilir.
- Okuma modu tahminleri `results/E{0,4}/preds/*.npz` önbelleğidir; anahtar
  değişikliğinde yeniden puanlama bedava (npz'yi silmeyin, sadece
  `selected_keys.json` güncellenirse `--force` gerekmez).

## 7. Sorun Giderme

- **`libstdc++.so.6: version CXXABI_1.3.15 not found` (vLLM ilk açılışında)**:
  venv pythonunuz conda env'indendir; conda'nın `libicui18n.so.78`'i yeni
  libstdc++ ister, sisteminki eskidir. Ortamda çözülür (betikler müdahale
  etmez): `conda install -c conda-forge 'libstdcxx-ng>=13'` ya da dağıtım
  depodan yeni `libstdc++6`. Doğrulama:
  `python -c "import sqlite3, vllm"` hatasız geçmeli.
- **Veri havuzu boş ("missing_fields" her şeyde)**: `data/field_peek.json`
  dosyasına gerçek alan adları yazılır; `data/prep_data.py` içindeki
  `FIELD_CANDIDATES` sözlüğüne veri setinizin adlarını ekleyip yeniden
  koşturun.
- **Tokenizer uyumsuzluğu**: `scripts/check_tokenizers.py` çıkışı 1 verirse
  §4.1 kuralı gereği DURUN; model setini blog notlarıyla karşılaştırın.
- **vLLM logits-processor**: duman koşusundaki E1 adımında üretilen
  metinlerde `check_no_thinking` hata verirse (düşünce tokenları sızması)
  chat template `enable_thinking=False` yolunu kontrol edin
  (`gen/generate.py`).
- **OOM**: `max_new_tokens` veya batch azaltın; 9B üretimde `tp_size: 4`.
- **E0 kabul kapısı**: `pvals_a0.csv` ortalaması 0.4–0.6 aralığında değilse
  SUMMARY "KONTROL ET" der; veri/anahtar adımlarını `--force` ile tekrarlayın.
- **scipy logsf -inf**: kod, analitik Γ fallback'e otomatik geçer; "underflow"
  uyarıları normaldir (§7.3 — kırpma yok).
- **GPQA metni**: hiçbir çıktı dosyasına yazılmaz; yalnız sayılar.

## 8. Koşum Sonrası Kontrol Listesi (SPEC.md §12)

- [ ] `wm/tests` iki süite de PASS
- [ ] Tokenizer karşılaştırma UYUMLU
- [ ] E0 kabul: parça p ortalaması 0.4–0.6, KS p > 0.01
- [ ] E1: KARAR-8 seçimi raporda, `selected.json` onaylı
- [ ] E2: D_gen tespit oranı (n=400) yüksek, D_clean düşük; AIME farkı küçük
- [ ] E3: 8 koşul eğrileri `detect_curves.csv` içinde; eval_loss/GPQA tabloda
- [ ] E4: 4 halka `chain.csv`; yanlış anahtar medyanları düşük
- [ ] Miniler: benchmark sayı korunması (≤3 deneme) raporu


## Düzeltme koşumu (ana koşu bittikten sonra): `scripts/rerun_reading.sh`

Ana koşunun sonuçları incelenirken üç sorun bulundu; bu betik hepsini giderir:

| Sorun | Düzeltme |
|---|---|
| A-gen/post-10 ve -50'de ρ iki kez uygulanmıştı (gerçek oran %1 ve %25) | `train/sft.py` düzeltildi; 4 model yeniden eğitilir (`MIX_VERSION="subset-v2"` işareti), GPQA/eval_loss bu 4 için tekrarlanır |
| E3 eğrileri serbest üretimden çıkarılmıştı (spesifikasyon: okuma modu) | `analyze/reading_rerun.py --stage model`: her model eğitimde GÖRDÜĞÜ watermark'lı cevaplar üzerinde, ana + 5 yanlış anahtar, 20 noktalı eğri, 50 parça p-değeri |
| E4 halka 4 yanlış veri üzerinde okunmuştu; halka 3 kontrolleri eksikti | Halka 4: B-greedy/B-sample/B-control → D_gen üzerinde okuma; halka 3: agent greedy/sampled + A₀ kontrolü (metin modu). Eski `chain.csv` → `chain_v1_hatali.csv` |

```bash
bash scripts/rerun_reading.sh 4          # tüm adımlar; ~40-60 dk (4×H100)
```
Sonra `results/` klasörünü aktarın. CPU tamamlamaları (`analyze.e0_complete`, `e1_complete`,
`e2_complete`) betiğin sonunda da koşar; yerelde `data/*.jsonl` ile tekrar çalıştırılabilir.

Not: `wm.reading_mode.rescore_predictions` kural (c)'yi tüm küme boyunca uygular
(spesifikasyondan sıkı); yeni sonuçlar `score_reading` (cevap başına (c), küresel (d)) ile hesaplanır.
