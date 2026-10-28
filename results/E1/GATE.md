# E1 SEÇİMİ (otomatik)

**SAPMA — KARAR-8 sağlanamadı** (benzerlik >= 0.85 VE uzunluk geçişi >= %90 koşulunu hiçbir ayar karşılamıyor; en iyi geçiş %87). Otomatik FALLBACK uygulandı: satırların en büyük payının aynı anda benzer (>=0.85), uzunluk bandında (0.75-1.25) ve tespit edilebilir (-log10p >= 6) olduğu ayar.

**Seçilen:** model `Qwen/Qwen3.5-4B`, T=1.2 — verim %53, medyan -log10p = 106.06, benzerlik 0.933, uzunluk geçişi %56. Yazı bu sapmayı belirtmelidir.

Pipeline durmaz: `results/E1/selected.json` otomatik yazıldı ve `run_e1_full.sh` devam eder. Elle değiştirmek için `selected.json`'u kendiniz yazın (varsa ezilmez) ve `bash scripts/run_e1_full.sh` çalıştırın.
