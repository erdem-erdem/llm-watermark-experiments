# llm-watermark-experiments

[English](README.md) · **Türkçe**

**LLM watermark'ı** ve **watermark radyoaktivitesi** üzerine Türkçe bir çalışmanın kodu, veri hattı,
sonuçları ve şekilleri: metne yerleştirilen bir watermark, o metinle eğitilmiş bir modelde hâlâ
tespit edilebilir mi?

Yöntemin, deneylerin, sonuçların ve yorumlarının tamamı blog serisindedir:

- 🇹🇷 Türkçe: <https://kureansiklopedi.com/tr/detay/llm-watermarking-12-iz-birakmak-calisma-prensipl-2>
- 🇬🇧 İngilizce: <https://kureansiklopedi.com/en/detay/llm-watermarking-12-watermarking-working-principle>

Bu depo bulguları **tartışmaz**; sonuçları yeniden üretecek kodu, ham sonuç tablolarını ve tüm şekilleri
Türkçe ve İngilizce olarak verir.

[![CI](https://github.com/erdem-erdem/llm-watermark-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/erdem-erdem/llm-watermark-experiments/actions)
[![Lisans: MIT](https://img.shields.io/badge/lisans-MIT-blue.svg)](LICENSE)

---

## İçindekiler

1. [Beş dakikada kavramlar](#beş-dakikada-kavramlar)
2. [Depo yapısı](#depo-yapısı)
3. [Kendiniz deneyin (GPU gerekmez)](#kendiniz-deneyin-gpu-gerekmez)
4. [Şekilleri yeniden üretin](#şekilleri-yeniden-üretin)
5. [Tam hattı çalıştırın (GPU)](#tam-hattı-çalıştırın-gpu)
6. [`results/` içinde ne var](#results-içinde-ne-var)
7. [Veri, modeller ve lisanslar](#veri-modeller-ve-lisanslar)
8. [Notlar ve bilinen tuhaflıklar](#notlar-ve-bilinen-tuhaflıklar)

---

## Beş dakikada kavramlar

**Üretim sırasında watermark.** Bir dil modeli bir sonraki tokenı `p` olasılık dağılımından seçer. Rastgele
çekimi *anahtarlı* bir çekimle değiştiririz. Gizli 64 bitlik bir `key` ve son `k = 2` token kimliği
(`wm/prf.py`) her sözlük tokenı `v` için `R_v ∈ (0, 1)` sözde rastgele sayısına dönüştürülür. Model
`argmax_v R_v^(1/p_v)` çıktısını verir (Gumbel-max / üstel yarış hilesi). Anahtar olmadan bu, `p`'den
birebir örneklemedir — metin her zamanki dağılımdadır — ama anahtarla bakıldığında seçilen tokenların
`R` değerleri büyük olma eğilimindedir.

**Tespit.** Bir metin için her tokenın `R` değeri anahtarla yeniden hesaplanır ve `s = −ln(1 − R)` skoru
alınır. Watermark'sız metinde `s ~ Exp(1)` olduğundan `N` puanlanan tokenın toplamı `Gamma(N, 1)`
izler ve p-değeri verir. Her zaman `−log₁₀ p` ile `N` birlikte raporlanır. Tekrar eden pencereler
atlanır (`wm/detect.py` içindeki tekilleştirme kuralları); aksi halde tekrar eden n-gramlar kanıtı şişirir.

**Post-hoc ve üretim sırasında.** Aynı şema bir model cevap *yazarken* ya da var olan bir cevabı *yeniden
yazarken* uygulanabilir (`gen/gen_e1.py`).

**Radyoaktivite ve "okuma modu".** Bir model watermark'lı metinle ince ayar gördüyse, kendi tahminleri
watermark'ın tercih ettiği tokenlara kayabilir. Bunu hiç metin üretmeden sınamak için metni şüpheli
modele veririz (tek ileri geçiş, teacher forcing), her konumda modelin tahmin ettiği tokenı alırız ve
metindeki token yerine *bu tahmini* anahtara göre puanlarız (`wm/reading_mode.py`).

Her parametre ve kuralın bulunduğu teknik şartname [`SPEC.md`](SPEC.md) dosyasındadır. Sezgi, şekiller
ve sonuçlar için blogu okuyun.

---

## Depo yapısı

```
wm/            watermark çekirdeği: PRF, Gumbel-max örnekleyici, skor, tespit, okuma modu, vLLM adaptörü
  tests/       birim testleri (CPU; PRF bit-düzeyi doğruluk, örnekleyici↔dedektör tutarlılığı, …)
data/          veri hazırlama (prep_data.py) ve anahtar seçimi (prep_keys.py); DATA_CARD.md
gen/           üretim betikleri (vLLM + özel logits processor)
train/         SFT (HF Trainer) + model başına bir YAML (train/configs/)
eval/          GPQA-TR, matematik olimpiyatı doğruluğu, ayrılmış küme kaybı, n-gram örtüşme kontrolü
analyze/       tespit orkestrasyonu, tablolar ve tüm şekil kodu (iki dilli)
configs/       her aşamanın YAML yapılandırması
scripts/       ortam kurulumu, GPU iş koşucusu, uçtan uca koşum betikleri
docker/        Dockerfile, öz-denetim, yedek requirements
examples/      toy_demo.py — oyuncak modele watermark + tespit, CPU'da saniyeler içinde
results/       işlenmiş sonuç tabloları ve kayıtlar (CSV / JSON / JSONL / MD), aşamaya göre
figures/       tr/ ve en/ — main/ (ana şekiller) ve extra/ (aşama tanılama şekilleri), PNG + SVG
SPEC.md        teknik şartname (Türkçe)
RUN_MANUAL.md  hattın çalıştırılması (Türkçe)      DOCKER_GUIDE.md  Docker anlatımı (İngilizce)
```

Kodda ve `results/` içinde geçen aşama adları:

| Aşama | Kod ne yapar |
|---|---|
| `E0` | dedektörü temiz modellerde ve yanlış anahtarlarla sınar; anahtarı seçer |
| `E1` | var olan cevapları watermark'lı (post-hoc) ve watermark'sız (kontrol) yeniden yazar |
| `E2` | üretim sırasında watermark'la cevap üretir; kalite ve çeşitlilik kontrolleri |
| `E3` | yukarıdakilerin karışımlarıyla öğrenci modelleri eğitir, okuma modunda tespiti ölçer |
| `E4` | bir öğrencinin çıktılarından yeni bir öğrenci damıtır, zincir boyunca tespiti ölçer |
| `mini` | blogun giriş bölümlerindeki küçük gösterimler |

---

## Kendiniz deneyin (GPU gerekmez)

Bu bölümdeki her şey dizüstü CPU'da çalışır.

```bash
git clone https://github.com/erdem-erdem/llm-watermark-experiments.git
cd llm-watermark-experiments
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu-tests.txt
```

**1. Oyuncak bir "dil modeline" watermark koyun ve tespit edin** (≈10 sn):

```bash
python -m examples.toy_demo
```

Rastgele bir bigram modeli LLM'in yerini tutar. Betik anahtarla metin örnekler, `wm/` içindeki gerçek
dedektörü doğru ve yanlış anahtarla çalıştırır, sıradan örneklemeyle de aynısını yapar. Üç satırda
`−log₁₀ p` değerinin nasıl davrandığına bakın. (Oyuncak model gerçek bir LLM'den çok daha "sivri"dir;
mutlak sayılar blogdakilerle kıyaslanamaz.)

**2. Parçaları doğrudan kullanın:**

```python
from wm.detect import detect_text

res = detect_text(prompt_ids=[...], answer_ids=[...], key=0x1234ABCD5678EF01, k=2)
print(res.N, res.neglog10p)            # puanlanan token sayısı, −log10(p)
```

`answer_ids` token kimlikleridir. Tespit metinle değil kimliklerle çalışır; çünkü metni yeniden
tokenlaştırmak pencereleri sessizce değiştirebilir (bkz. `SPEC.md`, kural 2).

**3. Birim testlerini çalıştırın:**

```bash
make test        # ya da: python -m wm.tests.test_prf vb.
```

---

## Şekilleri yeniden üretin

Tüm şekiller, `results/` içindeki işlenmiş tablolardan iki dilde üretilir:

```bash
pip install -r requirements-figures.txt
python -m analyze.make_figures              # tr + en, bir dakika kadar
python -m analyze.make_figures --lang tr    # tek dil
python -m analyze.blog_figures --lang tr E0 E1_1     # yalnız seçili şekiller
```

Çıktı `figures/<dil>/main/` (ana şekiller) ve `figures/<dil>/extra/` (aşama tanılama şekilleri)
klasörlerine PNG ve SVG olarak yazılır. Bazı şekillerin içindeki Türkçe model çıktıları (örnek metinler,
tokenlar) veridir ve İngilizce sürümlerde de Türkçe kalır. Şekiller gerçek veri ister: bir sonuç dosyası
yoksa betik durur, dosyayı ve onu üreten aşamayı söyler — hiçbir şey yer tutucu değerle doldurulmaz.

---

## Tam hattı çalıştırın (GPU)

Çalışma **4 × H100** (80 GB) üzerinde yapıldı. Tek GPU çalışır ama yavaştır; betikler ağır işleri verdiğiniz
GPU sayısına böler. Yaklaşık 45 GB Hugging Face önbelleği, 20–25 GB Docker imajı ve birkaç GB eğitilmiş
kontrol noktası gerekir.

**Önerilen: Docker** (vLLM/torch/CUDA uyumsuzluklarını önler). Ayrıntılı anlatım
[`DOCKER_GUIDE.md`](DOCKER_GUIDE.md) dosyasındadır (İngilizce); kısa hali:

```bash
docker build -f docker/Dockerfile -t llm-watermark-experiments:latest .
docker run --rm --gpus all --ipc=host llm-watermark-experiments:latest python /opt/selfcheck.py   # önce bu geçmeli

export HF_TOKEN=hf_xxx                       # bazı veri kümeleri kapılıdır; bkz. "Veri, modeller ve lisanslar"
alias wm='docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN -v "$PWD":/workspace -v "$HOME/hf_cache":/hf llm-watermark-experiments:latest'

wm bash scripts/run_smoke_all.sh             # küçük uçtan uca kontrol (indirmeler dahil ~1 saat)
wm bash scripts/run_all.sh 4                 # veri, anahtarlar, E1 taraması + otomatik ayar seçimi, sonra devam eder
# run_all.sh, scripts/run_e1_full.sh'ı kendisi çağırır (E2–E4, değerlendirmeler, tablolar, şekiller);
# elle yalnızca `E1_GATE=manual` sonrası ya da devam etmek için: wm bash scripts/run_e1_full.sh 4
```

**Docker olmadan:** `bash scripts/setup_env.sh` ([`RUN_MANUAL.md`](RUN_MANUAL.md) §1). Sürüm kilit dosyası
`docker/requirements.lock.txt` depoda **yoktur**; çalışan bir ortamdan `DOCKER_GUIDE.md` §4'teki komutla
üretin ya da `--build-arg REQ_FILE=requirements.txt` ile gevşek listeden kurun.

Her aşama idempotenttir: çıktısı olan aşama atlanır (`--force` yeniden yapar). Her komut `results/LOG.md`
dosyasına eklenir; kütüphane sürümleri, tohumlar ve süreler `results/run_manifest.json` dosyasına yazılır.
Aşama aşama ayrıntı, GPU dağılımı ve sorun giderme `RUN_MANUAL.md` içindedir.

---

## `results/` içinde ne var

Aşama başına CSV / JSON / JSONL / Markdown çıktıları: tespit tabloları ve eğrileri, p-değeri dağılımları,
değerlendirme skorları, üretim örnekleri, anahtar seçim kaydı, koşum günlüğü ve manifest.

Depoda **yok** (büyük ya da yeniden üretilebilir): model kontrol noktaları, `data/*.jsonl` üretilmiş veri
kümeleri (`python -m data.prep_data` ile yeniden kurulur), `.npz` tahmin dizileri, iş başına ham günlükler
ve post-hoc ham üretim parçaları (`results/E1/full_parts/`). Bunlar hat tarafından yeniden üretilir;
şekiller depodaki tablolardan çizilir.

---

## Veri, modeller ve lisanslar

| Kaynak | Kullanım | Lisans / not |
|---|---|---|
| [`turkish-nlp-suite/InstrucTurca`](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca) | talimat verisi (akışla okunan, süzülmüş altküme) | CC BY-SA 4.0 — atıf zorunlu; bkz. `data/DATA_CARD.md` |
| [`ytu-ce-cosmos/tubitak-science-olympiad-tr`](https://huggingface.co/datasets/ytu-ce-cosmos/tubitak-science-olympiad-tr) | matematik olimpiyatı kalite kontrolü | veri kartına bakın |
| [`ytu-ce-cosmos/gpqa-extended_tr`](https://huggingface.co/datasets/ytu-ce-cosmos/gpqa-extended_tr) | aşağı akış değerlendirmesi | kapılı olabilir: koşullarını kabul edip `HF_TOKEN` verin |
| Qwen3.5 (0.8B, 2B, 2B-Base, 4B, 9B) | üreticiler / öğrenciler | Hugging Face model kartlarına bakın |
| `intfloat/multilingual-e5-large` | anlamsal benzerlik | model kartına bakın |

Burada model ağırlığı ya da veri dökümü yeniden dağıtılmaz. Kod [MIT Lisansı](LICENSE) ile yayımlanmıştır.

---

## Notlar ve bilinen tuhaflıklar

- **`aime_*` adları mirastır.** Şartname başta `aime25-tr` adlı bir matematik kümesi öngörmüştü. Gerçek
  koşumlar `ytu-ce-cosmos/tubitak-science-olympiad-tr` (`ortaokul_matematik` bölümü, yalnız sayısal
  cevaplı satırlar) kullanır. `aime_*` / `AIME25-TR` adlı dosya ve fonksiyonlar (ör. `eval/aime_tr.py`,
  `results/E2/aime_summary.json`) AIME'yi değil, bu olimpiyat kümesini değerlendirir.
- **Yorum satırları ve docstring'ler çoğunlukla Türkçedir**; tanımlayıcılar ve README'ler değildir.
  `SPEC.md` ve `RUN_MANUAL.md` Türkçe, `DOCKER_GUIDE.md` İngilizcedir.
- Betikler UTF-8 bekler; Windows'ta `PYTHONIOENCODING=utf-8` ayarlayın (`Makefile` bunu yapar).
- Şekiller kuruluysa Arial, değilse DejaVu Sans kullanır; SVG'lerde metin metin olarak kalır.
- `gen/gen_opening.py` kendi `data/opening_draft.txt` dosyanızı bekler (ilk paragraf = söz, boş satır,
  sonra paragraf); dosya depoda yoktur, hazır çıktısı `results/mini/` altındadır.
- `python -m analyze.summary` `results/SUMMARY.md` yazar, ancak E3 tablosu nihai okuma modu çıktılarına
  göre güncellenmediğinden dosya depoya konmamıştır.
