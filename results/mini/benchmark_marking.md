# Benchmark işaretleme — örnek çiftler

100 problem birlikte: N = 11949, −log₁₀(p) = 427.7; sayıları bozulan: 9


## prob-15 — N=237, −log₁₀(p)=19.8
**Orijinal**

Bir *ABC* üçgeninde \([AB]\) kenarı üstündeki *D* noktası ve \([AC]\) kenarı üstündeki *E* noktası için, \(s(\overline{ABD}) = s(\overline{ABC})\), \(|AE| = 2\), \(|AD| = 5\) ve \(|BD| = 3\) ise, \(|CE|\) nedir?

**Yeniden yazım**

ABC üçgeninin içinden, AB kenarına D, AC kenarına da E noktaları düşülmüştür. Verilen bilgilere göre; ABD üçgeninin alanı, tüm ABC üçgeninin alanına eşit olup, bu durumun açıları kullanmadan sadece uzunluklarla çelişmesi nedeniyle klasik bir alan oranı çözümüyle |CE| uzunluğunu hesaplamak mümkün değildir. Mevcut koşullarda (\(s(\overline{ABD}) = s(\overline{ABC})\)), D noktası AB üzerinde olduğu için ABD alanı, ABC alanından kesinlikle küçük olmalıdır; alanlar ancak iki üçgen özdeş olursa eşit olabilir ki bu da D noktası ile A, B noktalarının üst üste binmesi gerektiğini (yani D'nin A veya B ile aynı olması) gerektirir. Ayrıca verilen diğer uzunluklar (AE = 2, AD = 5, BD = 3) ile bu doğrusal bağımlılık ilişkisi kurularak |CE| değerine varılabilen tutarlı bir geometrik yapı sağlanamamaktadır. Sorunun koşulları kendi içinde çelişkili olduğu için |CE| uzunluğu tanımlanamaz.


## prob-84 — N=331, −log₁₀(p)=18.7
**Orijinal**

Bir kavanozda bulunan portakal suyuna iki kez havuç suyu ekleniyor. Her iki eklemede kavanozdaki sıvının hacmi eşit yüzdeyle artmıştır. Son durumda kavanozdaki sıvının hacmi 600 birim ve ikinci eklemede kavanoza eklenen havuç suyu miktarı 120 birim ise, başlangıçta kavanozda bulunan portakal suyunun hacmi kaç birimdir?

**Yeniden yazım**

İlk olarak kavanozun içindeki sıvının hacminin her iki adımda da aynı oranda artması verildiği için, bu artış oranını tespit etmek gerekir. Bilinenlere göre ikinci ekleme yapıldığında kavanozdaki toplam hacim 600 birim ve bu ekleme miktarı 120 birimdir. Hacmin eşit yüzdeyle artması ifadesi, bu artışın ilk eklemeden önceki hacmin %5'ini oluşturduğunu gösterir (yani orani). Bu durumda ikinci eklemeden önceki hacim, toplam hacimden ikinci ekleme miktarı çıkarılarak bulunabilir; 600 birimden 120 birim çıkarıldığında 480 birim elde edilir. Daha sonra, ilk eklemeden önceki başlangıç hacmini hesaplamak adına, 480 birim hacmin ikinci ekleme ile %5 oranında artarak 600 birime ulaşması sonucuna varıldığı anlaşılır. Bu nedenle, ikinci eklemeden önceki hacmin (480 birim) %5'i yine 120 birim olacak şekilde doğrulanır ve bu da mevcut verilerle tutarlıdır. Şimdi geriye dönerek ilk eklemeden önceki durumu bulalım: İlk eklemede de aynı şekilde hacim artışı meydana gelmiş ve bu artış miktarı da 120 birim olmalıdır (zira artış oranı ve ikinci ekleme sonrası sonucuna ulaşmak için mantıksal akışta her adımda artış miktarının sabitlendiği varsayımına dayanarak, ancak daha kesin yol şudur: İlk ekleme sonrası hacim X ise, ikinci ekleme ile X + (X/100 * r) = 600 ve ikinci ekleme miktarı (X/100 * r) = 120 olmalı. X = 480 bulunur. İlk eklemeden önceki hacim Y ise, ilk ekleme sonrası hacim (Y + Y * oran) = 480 ve bu artış miktarı da aynı oranda 120 birim olmalıdır. Dolayısıyla Y + 120 = 480 şeklinde denklem kurularak Y = 360 birim elde edilir. Başlangıçta kavanozda bulunan portakal suyunun hacmi 360 birimdir.
