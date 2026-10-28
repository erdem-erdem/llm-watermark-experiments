# E1 Tarama Raporu

**SAPMA — KARAR-8 sağlanamadı** (benzerlik >= 0.85 VE uzunluk geçişi >= %90 koşulunu hiçbir ayar karşılamıyor; en iyi geçiş %87). Otomatik FALLBACK uygulandı: satırların en büyük payının aynı anda benzer (>=0.85), uzunluk bandında (0.75-1.25) ve tespit edilebilir (-log10p >= 6) olduğu ayar.

**Seçilen:** model `Qwen/Qwen3.5-4B`, T=1.2 — verim %53, medyan -log10p = 106.06, benzerlik 0.933, uzunluk geçişi %56. Yazı bu sapmayı belirtmelidir.

| model | T | n | medyan -log10p | benzerlik | uzunluk geçiş | verim | uzunluk oranı q10/medyan/q90 | kısa% / uzun% | uygun |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B | 0.7 | 300 | 0.76 | 0.979 | 72% | 4% | 0.81 / 1.00 / 1.75 | 8% / 20% | — |
| Qwen/Qwen3.5-0.8B | 1.0 | 300 | 25.40 | 0.948 | 51% | 29% | 0.66 / 1.07 / 2.06 | 13% / 37% | — |
| Qwen/Qwen3.5-0.8B | 1.2 | 300 | 251.22 | 0.874 | 27% | 18% | 0.37 / 1.26 / 2.42 | 23% / 50% | — |
| Qwen/Qwen3.5-2B | 0.7 | 300 | 0.88 | 0.981 | 74% | 7% | 0.81 / 1.00 / 1.68 | 7% / 19% | — |
| Qwen/Qwen3.5-2B | 1.0 | 300 | 14.35 | 0.962 | 53% | 26% | 0.77 / 1.10 / 1.93 | 9% / 38% | — |
| Qwen/Qwen3.5-2B | 1.2 | 300 | 177.71 | 0.904 | 35% | 26% | 0.50 / 1.22 / 2.32 | 18% / 46% | — |
| Qwen/Qwen3.5-4B | 0.7 | 300 | 2.20 | 0.979 | 87% | 6% | 0.91 / 1.04 / 1.24 | 3% / 10% | — |
| Qwen/Qwen3.5-4B | 1.0 | 300 | 9.06 | 0.971 | 76% | 50% | 0.91 / 1.10 / 1.43 | 4% / 20% | — |
| Qwen/Qwen3.5-4B | 1.2 | 300 | 106.06 | 0.933 | 56% | 53% | 0.82 / 1.16 / 1.82 | 8% / 36% | — |
| Qwen/Qwen3.5-9B | 0.7 | 300 | 2.61 | 0.970 | 75% | 9% | 0.86 / 1.10 / 1.41 | 5% / 20% | — |
| Qwen/Qwen3.5-9B | 1.0 | 300 | 7.79 | 0.965 | 65% | 41% | 0.87 / 1.16 / 1.55 | 4% / 30% | — |
| Qwen/Qwen3.5-9B | 1.2 | 300 | 65.99 | 0.937 | 47% | 46% | 0.80 / 1.20 / 1.88 | 8% / 45% | — |

Notlar:
- uzunluk oranı = çıktı karakter sayısı / kaynak karakter sayısı; band 0.75-1.25. 'kısa%' = bandın altı, 'uzun%' = bandın üstü.
- verim = satırın aynı anda benzer (>=0.85) + bantta + -log10p >= 6 olma payı.
- filtreli (uzunluk dışı / think_leak) satırlar İŞARETLENİR, eğitimden ÇIKARILMAZ; oranı raporlanır (Bölüm 5-E1.5).
- Seçim otomatiktir (`E1/selected.json`, auto=true). Elle bir `selected.json` varsa üzerine yazılmaz.
