# Teknik Şartname — Watermark & Radyoaktivite Deneyleri

> Bu dosya projenin teknik şartnamesidir (kodun docstring'leri bölüm numaralarına atıf yapar). Orijinal metin Türkçedir.
>
> **Not (AIME → TÜBİTAK):** Şartname başta `ytu-ce-cosmos/aime25-tr` adlı bir matematik kümesi öngörür. Gerçek koşumlarda bunun yerine
> `ytu-ce-cosmos/tubitak-science-olympiad-tr` (`ortaokul_matematik` bölümü, yalnız sayısal cevaplı satırlar) kullanıldı. Kodda ve
> çıktılarda `aime_*` / `AIME25-TR` adları **miras isimdir**; değerlendirme aslında bu olimpiyat kümesidir.

## 0. Bağlam

Bu proje, Türkçe bir blog yazısının (“Watermark Bir Modele Ne Bırakır?”) Bölüm 5'indeki deneyleri ve Bölüm 1–2'deki küçük gösterimleri (mini denemeler) üretir. Deneylerin anlatımı ve yorumu blog yazısındadır; bu depo yalnızca üretim kodunu ve çıktıları içerir.

Senin işin: deneyleri çalıştırmak, sayıları ve figürleri üretmek, sonuçları `results/` altına makine-okunur biçimde yazmak ve yazar için `results/SUMMARY.md` hazırlamak. **Yorum paragraflarını sen yazmıyorsun.** Taslakta her deney için “sonuç şöyle çıkarsa…” senaryoları zaten var; sen yalnızca hangi senaryonun gerçekleştiğini sayılarla raporlarsın.

Deney zinciri:

```
InstrucTurca'dan 5.000 Türkçe soru-cevap (D_orig: özgün cevaplar)
 ├─ E1: post-hoc watermark (D_orig cevaplarını yeniden yaz)   → D_post, D_rew (watermark'sız yeniden yazım)
 ├─ E2: üretim sırasında watermark (öğretmen aynı soruları cevaplar) → D_gen, D_clean
 ├─ E3: öğrenci A'yı karışımlarla eğit, radyoaktivite ölç     → A₀, A-gen-*, A-post-*, A-rew
 └─ E4: A-gen-100'ün watermark'sız çıktılarıyla öğrenci B     → B-greedy, B-sample, B'
E0 (ön kontrol): testin temiz modelde ve yanlış anahtarlarda şans düzeyinde kaldığını doğrula.
```

## 1. Pazarlıksız kurallar

1. **Tek PRF, tek kod yolu.** Üretimdeki watermark ile tespitteki watermark aynı `wm/prf.py` fonksiyonunu çağırır. Kütüphanelerin hazır watermark uygulamalarını (vLLM'in yerleşik watermark'ı dahil) ancak PRF'lerini bizim dedektörümüzden birebir çağırabiliyorsan kullan; aksi halde kendi örnekleyicimizi kullan.
2. **Token ID'leri sakla, yeniden tokenize etme.** Üretilen her cevabın token ID'leri JSONL'de saklanır. Tespit ve eğitim bu ID'leri kullanır. Metni yeniden tokenize etmek pencereleri sessizce bozar.
3. **Watermark yalnızca cevaplarda.** Eğitimde kayıp yalnızca cevap (assistant) tokenlarında hesaplanır.
4. **Tekilleştirme her zaman açık** (Bölüm 3.4). Tek istisna E0'daki “tekilleştirme kapalı” gösterim koşusudur.
5. **Negatif kontrol geçmeden sonuç yok.** E0 başarısızsa (Bölüm 7.1 eşikleri) E3–E4 sonuçlarını raporlama; sorunu bul.
6. **Thinking modu kapalı.** Qwen3.5 sohbet şablonunda düşünme modunu kapat (`enable_thinking=False` ya da modelin güncel karşılığı). Çıktılarda `<think>` görürsen dur ve düzelt.
7. **Her şey tekrar üretilebilir.** Tohumlar, anahtarlar, model revizyonları (HF commit hash), kütüphane sürümleri `results/run_manifest.json`'a yazılır.
8. **Sonuç uydurma, sonuç eğme yok.** Beklenmeyen bir sonuç beklenmeyen olarak raporlanır. Bir koşu başarısızsa yeniden çalıştırmadan önce nedenini `results/LOG.md`'ye yaz.

## 2. Kararlar

| Kod | Konu | Karar |
|---|---|---|
| KARAR-1 | Veri ve E1 kaynağı | `turkish-nlp-suite/InstrucTurca`. E1'de yeniden yazılan metin bu kümenin özgün cevaplarıdır (D_orig). E3'te watermark'lı olmayan kısım da D_orig'den gelir. |
| KARAR-2 | Veri boyutu | N = 5.000 soru-cevap. Radyoaktivite testi için ayrılmış küme yok; yalnızca downstream kayıp ölçümü için 500 örneklik `eval_loss` kümesi var. |
| KARAR-3 | Öğrenci modelleri | Qwen3.5-2B-Base (hem A hem B) |
| KARAR-4 | Eğitim | Tam fine-tuning, 3 epoch, lr 2e-5 |
| KARAR-5 | Örnekleme | T = 0.8, top-p = 0.95, top-k yok, max_new_tokens = 512 |
| KARAR-6 | Pencere boyutu | k = 2 |
| KARAR-7 | Değerlendirme kümeleri | E2 kalite kontrolü: `ytu-ce-cosmos/aime25-tr`. E3–E4 downstream: `ytu-ce-cosmos/gpqa-extended_tr` + 500 ayrılmış InstrucTurca örneğinde kayıp (Bölüm 4.3). |
| KARAR-8 | E1 ayar seçimi | Agent ayarı kendisi seçmez. Taramadan sonra rapor hazırlar ve yazar onayını bekler (E1 adım 4). Öneri olarak: medyan benzerlik ≥ 0.85 ve uzunluk filtresinden geçme ≥ %90 olan ayarlar arasında medyan −log₁₀(p) en yüksek olan. |
| KARAR-9 | E2 çeşitlilik ölçütü | Aynı prompt+anahtarla iki üretimin birebir aynı olma oranı ve korpus düzeyinde distinct-2 |

## 3. Watermark modülü (`wm/`)

### 3.1 PRF

Anahtar `key` 64-bit tamsayıdır. Pencere, üretilecek tokendan önceki son k token ID'sidir (prompt ve sohbet şablonu tokenları dahil, tam dizinin son k tokenı).

Referans tanım (saf Python, uint64 aritmetiği; bu tanım **doğruluk kaynağıdır**):

```python
M = (1 << 64) - 1
def mix64(z):
    z = (z + 0x9E3779B97F4A7C15) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)

def ctx_hash(key, window):          # window: k adet token id
    h = mix64(key)
    for t in window:
        h = mix64(h ^ t)
    return h

def R(key, window, v):              # (0,1) aralığında
    x = mix64(ctx_hash(key, window) ^ v) >> 11
    return (x + 0.5) / 2**53
```

GPU için vektörleştirilmiş bir sürüm yaz (tüm sözlük için R vektörü). Torch'ta uint64 çarpım desteği sınırlıysa int64 ile taşmalı çarpım (bitler aynıdır) ve maskeli mantıksal sağa kaydırma kullan. **Birim test zorunlu:** rastgele 10.000 (key, window, v) üçlüsünde vektörleştirilmiş sürüm referansla birebir aynı sonucu vermeli; tek-token yolu ile tüm-sözlük yolu da birebir aynı olmalı.

### 3.2 Örnekleyici (Gumbel-max)

Her adımda:

1. Modelin logit'lerinden sıcaklık ve top-p uygulanmış dağılım p hesaplanır (maskelenen tokenlar p = 0).
2. Pencere bu cevapta **daha önce görülmediyse**: `R` vektörü hesaplanır ve `argmax_v R_v^(1/p_v)` seçilir.
3. Pencere bu cevapta daha önce görüldüyse (tekrar): watermark atlanır, p'den normal (anahtardan bağımsız, tohumlu RNG ile) örnekleme yapılır.

Uygulama ipucu: `argmax R^(1/p) = argmax [log p − log(−log R)]`. Bu ifadeyi logit olarak döndüren bir logits processor yazıp açgözlü (greedy, temperature = 0) çözümleme ile kullanabilirsin; vLLM'de özel logits processor, HF'de `LogitsProcessor` + `do_sample=False`. Tekrar eden pencerede aynı formülü anahtardan bağımsız bir tekdüze U ile kullan; bu normal örneklemeye denktir.

Watermark'sız üretimler aynı örnekleme ayarlarıyla (T, top-p) standart rastgele örneklemeyle yapılır.

### 3.3 Skor ve p-değeri

Puanlanan her token için skor `s = −ln(1 − R)`. N token için toplam S; H₀ altında S ~ Γ(N, 1).

```python
from scipy.stats import gamma
log10_p = gamma.logsf(S, a=N) / np.log(10)    # çok küçük p'lerde taşmayı önler
```

Her zaman `−log₁₀(p)` ve N (puanlanan token sayısı) birlikte raporlanır.

### 3.4 Tekilleştirme kuralları

Bir pozisyon ancak şu koşulların hepsini sağlıyorsa puanlanır:

- (a) Pozisyon cevabın içindedir ve cevabın ilk k tokenı değildir.
- (b) Pencere, bu örneğin prompt'unda geçmez.
- (c) Pencere, aynı cevapta daha önce geçmemiştir.
- (d) (pencere, puanlanan token) ikilisi, aynı test içinde (tüm veri kümesi boyunca) daha önce puanlanmamıştır.

Metin tespitinde (E1, E2, E4 halka 3) “puanlanan token” metindeki tokendır. Okuma modunda (E3, E4 halka 2 ve 4) “puanlanan token” modelin tahminidir.

### 3.5 Okuma modu (radyoaktivite testi)

Watermark'lı her örnek için: sohbet şablonuyla `prompt + cevap` ID'leri modele verilir (tek ileri geçiş, teacher forcing). Cevaptaki her t pozisyonu için modelin tahmini `ŷ_t = argmax logits_t` alınır. Pencere, **metindeki** gerçek önekin son k tokenıdır. Skor `R(key, pencere, ŷ_t)` ile hesaplanır. Bölüm 3.4 kuralları uygulanır; kural (d)'deki ikili (pencere, ŷ_t)'dir.

Modelin tahminleri (ŷ ve pencere) bir kez hesaplanıp `preds/*.npz` olarak saklanır; farklı anahtarlarla yeniden puanlama bu dosya üzerinden yapılır.

### 3.6 Anahtar seçimi

20 aday anahtar (sabit tohumdan türetilmiş) D_orig ve D_clean cevaplarında metin tespitinden geçirilir (örnek başına p-değeri). Örnek p-değerlerinin tekdüze dağılıma uyumu KS testiyle ölçülür. KS p-değeri en yüksek anahtar **ana anahtar** olur. Kalanlardan 5'i **yanlış anahtar** olarak saklanır. Seçim ve tüm KS sonuçları `results/E0/key_selection.csv`'ye yazılır.

## 4. Modeller ve veri

### 4.1 Modeller

| Rol | Model |
|---|---|
| Öğretmen, ana watermark'layıcı | Qwen3.5-9B |
| E1 yeniden yazma taraması | Qwen3.5-0.8B / 2B / 4B / 9B |
| Öğrenci A ve B | Qwen3.5-2B-Base |
| Anlam benzerliği | Çok dilli bir cümle gömme modeli (ör. `intfloat/multilingual-e5-large` ya da `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`; seçtiğini manifeste yaz) |

Notlar:
- Qwen3.5 modelleri yerel olarak çok kipli (görüntü-metin) modellerdir. Yalnızca metin yolunu kullan. Doğru model sınıfını ve gerekli `transformers` sürümünü (kaynaklara göre ≥ 5.2) doğrula.
- Mimari, hızlı çalışma için ek çekirdekler (doğrusal dikkat katmanları için) gerektirebilir; kurulu değilse üretim çok yavaşlar. İlk iş olarak 9B ile 100 prompt üretip token/saniye ölç ve manifeste yaz.
- Bütün modeller aynı tokenizer'ı kullanmalı. Başlangıçta sözlük boyutlarını ve birkaç Türkçe cümlenin tokenizasyonunu dört model + Base arasında karşılaştır; fark varsa dur ve raporla.
- Base model için sohbet şablonu olarak post-trained modelin şablonunu kullan; eğitim ve tespit aynı şablonla yapılır.

### 4.2 Veri: InstrucTurca

Kaynak: `turkish-nlp-suite/InstrucTurca` (Hugging Face, CC BY-SA 4.0). İngilizce talimat kümelerinin makine çevirisiyle oluşturulmuş, kod, matematik, deneme, şiir, tıp gibi alanları kapsayan sentetik bir Türkçe talimat kümesidir; tek bir ~4 GB JSONL dosyasıdır. Bütününü belleğe yükleme; akışla (streaming) oku.

Adımlar:

1. Alan adlarını (talimat, varsa ek girdi, cevap) dosyadan doğrula. Ek girdi alanı varsa prompt = talimat + boş satır + girdi.
2. Filtreler: cevap 120–500 token (Qwen3.5 tokenizer'ı ile); prompt 10–300 token; Türkçe karakter ve kelime oranı kontrolü (çeviri bozukluklarını ele); cevabın büyük kısmı kod bloğu olan örnekleri `matematik_kod` kotasıyla sınırla; tam ve yakın tekrarları (normalize edilmiş metin üzerinde) ele; dış bağlam gerektiren (“yukarıdaki metne göre” gibi, ama girdisi olmayan) promptları ele.
3. Filtreden geçenlerden sabit tohumla rastgele bir aday havuzu (~20.000) çek.
4. **Konu etiketi:** Havuzdaki promptlar Qwen3.5-9B ile (watermark'sız) üç gruptan birine etiketlenir: `yaratici`, `olgusal`, `matematik_kod`. 100 örneklik rastgele bir kontrol listesini `data/label_check.csv` olarak bırak.
5. Havuzdan tabakalı olarak 5.000 eğitim + 500 `eval_loss` örneği seç; hedef dağılım ~%40 `yaratici`, ~%40 `olgusal`, ~%20 `matematik_kod`. Bir grupta yeterli örnek yoksa mevcut dağılımı kullan ve raporla.
6. Bu 5.000 örneğin özgün cevapları **D_orig**'dir. Seçimi `data/train.jsonl` olarak sabitle; bir daha değiştirme.

`data/DATA_CARD.md`'ye yaz: kaynak ve revizyon, lisans ve atıf, filtrelerin her birinde elenen sayı, son konu dağılımı, ortalama ve medyan cevap uzunluğu (token), toplam token sayısı, 5 örnek.

Bölmeler (sabit tohumla, bir kez):
- `train` (5.000), `sweep` (`train` içinden 300, konu dağılımına göre tabakalı), `eval_loss` (`train` dışından 500, aynı dağılımda).
- E3 için iç içe alt kümeler: `S10 ⊂ S50 ⊂ S100 = train` (ρ = %10, %50, %100).

### 4.3 Değerlendirme kümeleri

**AIME 2025 Türkçe — `ytu-ce-cosmos/aime25-tr` (E2 kalite kontrolü, ÖRNEK 4.1)**

- Alanlar: `Problem` (metin), `Answer` (sayı). Alan adlarını ve soru sayısını doğrula (AIME 2025'te 30 soru beklenir).
- Talimat (Türkçe, sabit): “Aşağıdaki problemi adım adım çöz. Son cevabını yalnızca bir tam sayı olarak \boxed{} içinde ver.” + problem.
- Puanlama: çıktıdaki **son** `\boxed{...}` içeriği alınır, boşluk ve noktalama temizlenir, tam sayıya çevrilir, `Answer` ile karşılaştırılır. `\boxed` yoksa yanlış sayılır; bu oranı ayrıca raporla.
- Ayar: Qwen3.5-9B, thinking kapalı (çözüm adımları cevabın içinde yazılabilir), T = 0.8, top-p = 0.95, **max_new_tokens = 4096** (genel 512 sınırı burada geçerli değil).
- Koşullar: watermark'lı 4 farklı anahtar (ana anahtar + 3 yanlış anahtar) ve watermark'sız 4 farklı tohum → 30 × 8 = 240 üretim.
- Rapor: koşul başına doğruluk; watermark'lı ve watermark'sız için ortalama ± std (4 tekrar üzerinden); soru başına doğru sayısı tablosu `E2/aime_per_problem.csv`.
- Dikkat: 30 soru küçük ve zor bir küme; doğruluk düşük ve gürültülü çıkabilir. Farkı yorumlamak yazara aittir; sen yalnızca ortalama, std ve iki koşulun dağılımlarının örtüşüp örtüşmediğini raporla.

**GPQA Extended Türkçe — `ytu-ce-cosmos/gpqa-extended_tr` (E3–E4 downstream)**

- Alanlar: `Question`, `Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, `Incorrect Answer 3`. Doğrula.
- Biçim: dört seçenek sabit tohumla karıştırılıp A–D olarak sıralanır (karıştırma bir kez yapılır, bütün modellerde aynı). 0-shot şablon: soru + seçenekler + “Cevap:”. Puanlama: “ A”, “ B”, “ C”, “ D” tokenlarının log-olasılıkları arasında en yükseği. Watermark kullanılmaz.
- Modeller: Qwen3.5-2B-Base (taban), tüm A modelleri, B modelleri. 1.000 bootstrap ile %95 GA.
- Dikkat: Küme çok zordur; 2B modeller şans düzeyine (%25) yakın kalabilir. Taban model %25 ± 3 puan içindeyse bunu `SUMMARY.md`'de “bu ölçüt ayırt edici değil” diye açıkça belirt.
- **Lisans/yayın notu:** GPQA yazarları soruların internette açıkça yayımlanmamasını ister. GPQA sorularını hiçbir çıktı dosyasına, örnek listesine ya da figüre koyma; yalnızca sayısal sonuçları raporla.

**Ayrılmış kayıp — `eval_loss` (E3–E4 downstream, ikinci ölçüt)**

- `train` dışından, aynı filtrelerden geçmiş ve aynı konu dağılımında 500 InstrucTurca örneği (watermark'sız, özgün cevaplar). Bu küme yalnızca kalite ölçümü içindir; radyoaktivite testlerinde kullanılmaz.
- Ölçüm: cevap tokenları üzerinde ortalama kayıp (ve perplexity), bütün A ve B modelleri ile taban model için; örnek düzeyinde bootstrap %95 GA.

Her iki eğitim kümesiyle (InstrucTurca `train`) değerlendirme kümeleri arasında 8-gram örtüşme kontrolü yap; örtüşen eğitim örneklerini ele ve sayısını raporla.

### 4.4 Dosya şeması

Her üretim `data/<küme>.jsonl`, satır başına:

```json
{"id": "...", "src_id": "<InstrucTurca satır no>", "split": "train", "topic": "olgusal", "prompt": "...", "prompt_ids": [...],
 "answer": "...", "answer_ids": [...], "wm": true, "key_id": "main", "k": 2,
 "gen_model": "Qwen/Qwen3.5-9B@<rev>", "T": 0.8, "top_p": 0.95, "skipped_positions": [...],
 "source_answer": "... (yalnızca post-hoc)"}
```

## 5. Deneyler

Genel üretim ayarları: `[KARAR-5]`. Genel eğitim ayarları: tam fine-tuning, bf16, AdamW, lr 2e-5, cosine, %3 warmup, 3 epoch, efektif batch 32, max uzunluk 2048, kayıp yalnızca cevap tokenlarında, paketleme (packing) kapalı, eğitim tohumu ve veri sırası her koşuda aynı. Girdi ID'leri saklanan `prompt_ids` ve `answer_ids`'ten kurulur (kural 2).

### E0 — Ön kontrol (testin kendisi)

1. Anahtar seçimi (Bölüm 3.6).
2. E3 bittikten sonra: A₀ üzerinde D_gen ve D_post (5.000'er örnek) ile okuma modu, **ana anahtar**. Bütün A modellerinde aynı tahminlerle **5 yanlış anahtar**.
3. Gösterim: A₀ üzerinde aynı test, tekilleştirme kapalı (yalnızca kural (a)).
4. Örnek düzeyinde p-değerleri için: D_gen'i 50 eşit parçaya böl, her parçada ayrı p-değeri hesapla → histogram ve KS testi.

**Kabul ölçütü:** A₀/ana anahtar ve tüm yanlış anahtarlar için parça p-değerlerinin ortalaması 0.4–0.6, KS p > 0.01. Sağlanmazsa dur; tokenizer, pencere tanımı ve tekilleştirmeyi kontrol et.

Çıktılar: `results/E0/pvals.csv`, `fig_E0.png/.svg`, `E0.json` (ortalama p'ler, KS p'leri, tekilleştirme kapalı p-değeri).

### E1 — Post-hoc watermark

1. **Tarama:** `sweep` (300) D_orig cevapları × {0.8B, 2B, 4B, 9B} × T ∈ {0.7, 1.0, 1.2}, watermark açık, ana anahtar.
2. Yeniden yazma talimatı (Türkçe, sabit): “Aşağıdaki metni anlamını, içerdiği bilgileri ve biçimini koruyarak yeniden yaz. Bilgi ekleme ya da çıkarma. Yalnızca yeniden yazılmış metni ver.” Kaynak > 400 token ise paragraf sınırlarından parçalara böl, her parçaya bir önceki parçanın yeniden yazılmış hâlini bağlam olarak ver. Parça sınırlarını `chunk_bounds` alanına kaydet; tespitte her parçanın ilk k tokenı kural (a) gibi atlanır.
3. Ölçümler (her çıktı için): −log₁₀(p) ve N (metin tespiti); anlam benzerliği (kaynak ile çıktı gömmeleri arası kosinüs); uzunluk oranı (karakter). Uzunluk oranı [0.75, 1.25] dışındaysa “filtrede elendi” işaretle.
4. **Dur ve danış:** `E1/SWEEP_REPORT.md` hazırla: `T_E1.csv`, `fig_E1_1`, her ayar için 2 örnek çift, KARAR-8'deki öneri kuralının seçeceği ayar ve buna en yakın iki alternatif. **Yazar onayı gelmeden 5. adıma geçme.** Bu sırada E2 üretimleriyle devam edebilirsin. Onaylanan ayarı `E1/selected.json`'a yaz.
5. **Tam veri:** `train`'deki 5.000 D_orig cevabı seçilen ayarla yeniden yazılır → `D_post`. Aynı model ve T ile watermark'sız yeniden yazım → `D_rew`. Elenen çıktılar en fazla 2 kez farklı RNG tohumuyla yeniden üretilir (watermark'lı olanlarda tekrar üretim için talimata görünmez bir değişiklik yapma; bunun yerine elenenleri işaretle ve oranını raporla).

Çıktılar: `E1/sweep.csv` (her satır bir çıktı), `T_E1.csv` (model × T: medyan −log₁₀p, IQR, medyan benzerlik, geçme oranı), `fig_E1_1` (saçılım: x = −log₁₀p, y = benzerlik, renk = model boyutu, işaret = T, seçilen ayar yıldız), `fig_E1_2_examples.md` (3 örnek çift: yüksek benzerlik, orta, düşük; her biri için benzerlik, uzunluk oranı, p), token başına entropi dağılımı (model başına) `E1/entropy.csv`.

### E2 — Üretim sırasında watermark

1. Qwen3.5-9B, `train` promptlarına bir kez watermark'lı (`D_gen`), bir kez watermark'sız (`D_clean`) cevap verir.
2. **Tespit gücü:** Her D_gen ve D_clean cevabı için, ilk n puanlanan token üzerinden −log₁₀(p), n ∈ {25, 50, 100, 200, 300, 400}. Konuya göre medyan eğrileri. Ayrıca n = 400'de p < 0.01 oranı (D_gen: tespit oranı, D_clean: yanlış alarm oranı; ikincisi ~%1 olmalı).
3. **Post-hoc ile karşılaştırma:** Aynı promptlar için D_gen ve D_post'ta puanlanan token başına ortalama skor (= S/N) dağılımları.
4. **Çeşitlilik `[KARAR-9]`:** 200 prompt × 2 üretim, aynı anahtarla (watermark'lı) ve farklı tohumlarla (watermark'sız): birebir aynı çıktı oranı. Korpus düzeyinde distinct-2 (D_gen vs D_clean).
5. **Kalite kontrolü:** AIME 2025 Türkçe (Bölüm 4.3): watermark'lı 4 anahtar ve watermark'sız 4 tohum, doğruluk ortalama ± std.

Çıktılar: `E2/curves.csv`, `fig_E2_1` (konu başına üç eğri + p = 0.01 çizgisi), `fig_E2_2` (D_gen vs D_post token başına skor, keman grafiği), `T_E2.csv` (satırlar: D_gen, D_post, D_clean; sütunlar: medyan −log₁₀p, n=400 tespit oranı, aynı-çıktı oranı, distinct-2, ortalama uzunluk, AIME doğruluğu ± std).

### E3 — Watermark'lı veriyle eğitim

Her koşuda eğitim kümesi `train`'in 5.000 promptudur; ρ alt kümesindeki promptlar watermark'lı cevapla, kalanlar özgün cevapla (D_orig) eğitilir. Böylece her koşu “bir Türkçe talimat kümesinin bir kısmı watermark'lı” durumunu temsil eder.

| Koşu | ρ alt kümesindeki cevaplar | Kalan cevaplar |
|---|---|---|
| A₀ | — (ρ = 0) | D_orig |
| A-gen-10 / 50 / 100 | D_gen | D_orig |
| A-post-10 / 50 / 100 | D_post | D_orig |
| A-rew | D_rew (ρ = %100, watermark'sız yeniden yazım) | — |

Tespit (okuma modu, ana anahtar):
- A-gen-ρ: modelin eğitimde gördüğü D_gen örnekleri. A-post-ρ: gördüğü D_post örnekleri.
- A₀: tüm D_gen ve tüm D_post (iki ayrı test). A-rew: tüm D_post.
- Her koşu için eğri: puanlanan token sayısı arttıkça −log₁₀(p) (örnekler rastgele sırayla eklenerek, 20 nokta, logaritmik aralık).

Downstream: tüm A modelleri ve taban model için GPQA-TR doğruluğu ve `eval_loss` kaybı (Bölüm 4.3), 1.000 bootstrap ile %95 güven aralığı.

Sağlık kontrolleri: eğitim kaybı eğrileri `E3/loss/*.csv`; koşu başına eğitim süresi.

Çıktılar: `E3/curves.csv`, `fig_E3_1` (x: puanlanan benzersiz token, log ölçek; y: −log₁₀p; renk = ρ, düz = gen, kesikli = post; A₀ ve A-rew gri; p = 0.01 ve 10⁻⁶ çizgileri), `T_E3.csv` (koşu, ρ, kaynak, N, son −log₁₀p, yanlış anahtarlarda ortalama p, GPQA-TR doğruluğu ± GA, eval_loss ± GA).

### E4 — Radyoaktif modelden distillation

1. Aracı = A-gen-100. `train` promptlarına **watermark'sız** iki üretim: greedy ve T = 0.7 (top-p 0.95). Kontrol: A₀'ın greedy üretimi.
2. Yeni öğrenciler (aynı Base modelden, E3 ile aynı eğitim ayarları): B-greedy, B-sample, B'.
3. Zincir ölçümleri (ana anahtar):
   - Halka 1: D_gen'in kendi metin tespiti.
   - Halka 2: A-gen-100 okuma modu (E3'ten).
   - Halka 3: aracının üretimlerinin **doğrudan metin tespiti** (greedy ve örnekleme ayrı; A₀'ın üretimleri kontrol).
   - Halka 4: B'lerin D_gen üzerinde okuma modu.
   - Her halka için yanlış anahtar ortalaması.
4. Downstream: B'ler için E3 ile aynı iki ölçüm.

Çıktılar: `E4/chain.csv` (halka × koşul: N, −log₁₀p, yanlış anahtar ortalama p), `fig_E4_1` (dört halka, log eksen çubuklar; greedy ve örnekleme iki renk, kontrol gri), `T_E4.csv`.

## 6. Mini denemeler (Bölüm 1–4 için)

| Yer tutucu | Ne yapılacak | Çıktı |
|---|---|---|
| Açılış sözü + ilk paragraf | Yazarın `docs/opening_draft.txt` metnini Qwen3.5-9B ile ana anahtarla post-hoc yeniden yaz (E1 talimatı). 5 farklı talimat varyantıyla 5 aday üret (talimat varyantları dosyada). Her aday için söz ve paragraf ayrı ayrı: N, −log₁₀p, token renkleri. | `mini/opening_candidates.md`, `mini/opening_tokens.csv` |
| GÖRSEL 1.3 | D_gen'den 150–250 tokenlık bir `yaratici` örnek ve D_clean'den eşdeğeri: token başına skor ve birikimli −log₁₀p serisi | `mini/fig13_tokens.csv`, `mini/fig13_curve.csv`, ön izleme `fig13.png` |
| ÖRNEK 1.1 | E1 tam verisinden 2 kısa örnek çift (kaynak, watermark'lı yeniden yazım, benzerlik, p) | `mini/example_1_1.md` |
| MİNİ DENEME 2.1 | 3 prompt; watermark'lı iki üretim (aynı anahtar) ve watermark'sız iki üretim (farklı tohum) | `mini/same_prompt_twice.md` |
| MİNİ DENEME 2.2 | 3 prompt; watermark'lı ve watermark'sız cevap çifti, p-değerleriyle; hangisinin hangisi olduğu ayrı satırda | `mini/guess_pairs.md` |
| ÖRNEK 4.1 | AIME 2025 Türkçe'nin 30 problemi, E1 talimatıyla Qwen3.5-9B tarafından ana anahtarla yeniden yazılır. **Sayılar, formüller ve problemin istediği şey birebir korunmalı:** her yeniden yazımda problemdeki bütün sayıların korunduğunu otomatik kontrol et, bozulanları en fazla 3 kez yeniden üret (talimata ek “Sayıları ve matematiksel ifadeleri değiştirme.” cümlesiyle), yine bozuk kalanları raporla. Rapor: tek tek her problemin N ve −log₁₀p'si, 30 problemin birlikte (tekilleştirilmiş) N ve −log₁₀p'si, 2 örnek çift. | `mini/benchmark_marking.md`, `mini/benchmark_marking.csv` |

## 7. İstatistik ve raporlama kuralları

1. **Eşikler:** E0 kabul ölçütü Bölüm 5'te. Anlamlılık için p < 0.01 ve “güçlü” için p < 10⁻⁶ çizgileri figürlerde gösterilir; metinde p-değeri her zaman N ile birlikte verilir.
2. Medyanlar IQR ile, ortalamalar std ya da %95 güven aralığıyla raporlanır. Tek bir koşudan gelen downstream skorlar için bootstrap (1.000).
3. −log₁₀(p) > 300 olduğunda bile `logsf` ile gerçek değeri yaz; “< 10⁻³⁰⁰” gibi kırpma yapma.
4. Figürler: matplotlib, Türkçe eksen etiketleri, 300 dpi PNG + SVG, her figürün verisi aynı adla CSV olarak. Renk paleti tüm figürlerde aynı: watermark'lı = koyu mavi tonları, watermark'sız/kontrol = gri, post-hoc = turuncu tonları.
5. Sayısal yazım: ondalık ayırıcı olarak yazının kullandığı nokta (0.85), yüzdeler “%10” biçiminde.

## 8. Zaman bütçesi ve çalıştırma sırası (1–2 H100, tahmini)

| Adım | Tahmini süre |
|---|---|
| Ortam, tokenizer kontrolü, hız ölçümü | ~30 dk |
| Veri: akışla okuma, filtreler, etiketleme, seçim | ~45 dk |
| Anahtar seçimi + PRF birim testleri | ~15 dk |
| E2 üretimleri (5.000 × 2) | ~20–30 dk |
| E1 tarama → **yazar onayı** → tam yeniden yazım (watermark'lı ve watermark'sız) | ~45 dk + onay süresi |
| E3: 8 koşu (2B, 3 epoch) | ~1–1.5 saat |
| E4: 3 üretim + 3 koşu | ~45 dk |
| Tespit, figürler, mini denemeler | ~1 saat |
| Değerlendirmeler: AIME-TR (240 üretim), GPQA-TR ve eval_loss (11 model + taban) | ~45 dk |
| **Toplam** | **~5.5–6.5 saat** (onay beklemesi hariç) |

Sıra: kurulum → PRF testleri → veri → anahtar seçimi → E1 tarama (rapor, onay beklenir) → E2 → E1 tam veri → E3 → E0 (A₀ ve yanlış anahtarlar) → E4 → mini denemeler → değerlendirmeler (AIME-TR, GPQA-TR, eval_loss) → SUMMARY. İkinci GPU varsa E3 koşularını paralel çalıştır.

## 9. Klasör yapısı

```
wm/            prf.py, sampler.py, detect.py, reading_mode.py, tests/
data/          *.jsonl, DATA_CARD.md, label_check.csv
gen/           üretim betikleri (E1, E2, E4, mini)
train/         sft.py, configs/*.yaml
eval/          gpqa_tr.py, aime_tr.py, eval_loss.py
results/       E0/ E1/ E2/ E3/ E4/ mini/  run_manifest.json  LOG.md  SUMMARY.md
```

Her betik tek bir YAML yapılandırmasından çalışır; komutlar `results/LOG.md`'ye eklenir.

## 10–11. (Blog taslağına özgü bölümler bu kopyada çıkarıldı)

## 12. Bilinen tuzaklar (kontrol listesi)

- [ ] Thinking modu kapalı mı, çıktılarda `<think>` yok mu?
- [ ] Bütün modellerde tokenizer aynı mı?
- [ ] Eğitim girdileri saklanan ID'lerden mi kuruldu (yeniden tokenizasyon yok)?
- [ ] Kayıp yalnızca cevap tokenlarında mı?
- [ ] Pencere tanımı üretim ve tespitte aynı mı (tam dizinin son k tokenı, şablon tokenları dahil)?
- [ ] Tekrar eden pencerelerde watermark atlanıyor ve tespitte de puanlanmıyor mu?
- [ ] Watermark'sız üretimler (D_clean, D_rew, aracı üretimleri) gerçekten anahtardan bağımsız mı?
- [ ] E1 tam verisine geçmeden yazar onayı alındı mı?
- [ ] InstrucTurca atfı (CC BY-SA 4.0) DATA_CARD'da var mı?
- [ ] E3'te iç içe alt kümeler ve aynı veri sırası kullanıldı mı?
- [ ] E0 kabul ölçütü sağlandı mı?
- [ ] Değerlendirme soruları eğitim promptlarıyla örtüşmüyor mu (8-gram kontrolü)?
- [ ] Hiçbir çıktı dosyasında GPQA sorusu yok mu?
- [ ] AIME yeniden yazımlarında sayılar korunmuş mu?
