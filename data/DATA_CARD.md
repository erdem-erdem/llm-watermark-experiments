# DATA_CARD — InstrucTurca seçkisi

- Kaynak: `turkish-nlp-suite/InstrucTurca` (CC BY-SA 4.0; atıf zorunlu: Turkish NLP Suite,
  *InstrucTurca: Diverse Instructional Content Dataset for Turkish*, Hugging Face)
- Kaynak revizyon: akışla okundu; tohum 20260814. Kod: `data/prep_data.py`
- Filtre sayımları: {
 "total": 62468,
 "f_answer_len": 28889,
 "kept": 20000,
 "f_prompt_len": 11930,
 "f_code_quota_hit": 1242,
 "f_context": 171,
 "f_turkish": 948,
 "f_duplicate": 288
}
- Konu dağılımı (train): {'yaratici': 2000, 'olgusal': 2000, 'matematik_kod': 1000}
- Cevap uzunluğu (token): ort 282.3, medyan 272
- Toplam token (train, prompt+cevap): 1,883,680
- Bölmeler: train=5000, eval_loss=500, sweep=300
- İç içe: S10=500, S50=2500, S100=5000

Örnekler:
```json
{"id": "train-00000", "src_id": 16191, "split": "train", "topic": "yaratici", "prompt": "Aşırı Kulak Hijyenini Hesaplayan bir JavaScript komut dosyası pasajı oluşturun: Profesyoneller İçin Karar Verme Amaçlı Kulak Temizleme İpuçlarından Kaçınmak. Güvenliğe dayalı olarak farklı eylemleri koşullu olarak gerçekleştirmek için if/else veya switch/case ifadelerini kullanın. Prova yapın ve ardından kontrol akışının ana hatlarını çizen ve farklı senaryoları nasıl ele aldığınıza dair yorumları ekleyin.", "prompt_ids": [32, 6996, 191231, 81591, 570, 65695, 73544, 6383, 242402, 385, 157715, 14804, 12544,
```
```json
{"id": "train-00001", "src_id": 58528, "split": "train", "topic": "yaratici", "prompt": "Ekstrem Çevrimiçi Alışverişi Denetleyen bir C# modülü pasajı oluşturun: Uzmanlar için Analiz için Ürün Kullanılabilirliği. Sorumluluğa bağlı olarak farklı durumları ele almak için if/else veya switch/case ifadelerini ekleyin. Yaklaşımınızı açıklayan yorumlarla kontrol akışınızın iyi bir şekilde belgelendiğinden emin olun.", "prompt_ids": [40535, 267, 1763, 237888, 235581, 177498, 1615, 31400, 203710, 76200, 9482, 294, 206938, 14804, 351, 2, 1410, 202874, 167080, 3649, 8069, 163012, 160422, 25, 533, 78125, 
```
