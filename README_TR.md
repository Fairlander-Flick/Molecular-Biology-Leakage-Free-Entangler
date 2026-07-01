# Türkçe Anlatım — Her Şey, İncik Cincik

> **Bu dosya iki katmanlı:**
> - 🔬 **TEKNİK** kısımlar = senin *anlaman* için (slaytta anlatmayacaksın).
> - 🍏 **BASİT (elma-armut)** kutuları = slaytta *böyle* anlatacaksın.
>
> Sonda ayrı bir "Slaytta ne diyeceğim" bölümü var: sadece skorlar + basit fikir.

---

## BÖLÜM 0 — Tek cümlede olay
İki proteinin birbirine yapışıp yapışmadığını tahmin eden bir yapay zekâ yaptık;
ama asıl mesele **dürüst** bir skor almak — çünkü literatürdeki yüksek skorlar
**kopya (data leakage)** yüzünden şişmiş. Biz kopyayı ölçtük, temizledik ve dürüst
skorun aslında **iyileştirilebilir** olduğunu gösterdik.

---

## BÖLÜM 1 — Problem: Protein ne, PPI ne, neden umurumuzda?

🔬 **Teknik:** Proteinler hücrenin iş yapan makineleridir. Çoğu iş, iki (veya daha
çok) proteinin fiziksel olarak **birbirine bağlanmasıyla** (protein–protein
interaction = PPI) olur. Hangi protein çiftinin etkileştiğini bilmek; hastalık
mekanizmalarını, ilaç hedeflerini, hücre yollarını anlamak için kritik. Deneyle
tek tek ölçmek pahalı ve yavaş → **bilgisayarla tahmin** istiyoruz.

🍏 **Basit:** Proteinleri **lego parçaları** gibi düşün. Bazı parçalar birbirine
tıpatıp oturur (etkileşir), bazıları oturmaz. Biz, "bu iki parça birbirine oturur
mu?" sorusuna evet/hayır diyen bir makine yaptık. Neden önemli? Çünkü hangi
parçaların oturduğunu bilirsek, hücrenin nasıl çalıştığını ve ilaçların nereye
takılacağını anlarız.

---

## BÖLÜM 2 — Paperın anlattığı: Bernett ve arkadaşları (2024)

Bu, kullandığımız veri setinin **kendi makalesi**. Söyledikleri şu:

### 2.1 İddia: yayınlardaki skorlar sahte-yüksek
🔬 **Teknik:** Sekans-tabanlı PPI modelleri literatürde ~%80–90 doğruluk
raporluyor. Bernett gösterdi ki bu sayılar **data leakage** (veri sızıntısı)
yüzünden şişkin. Sızıntıyı temizleyince aynı yöntemler **~%50–65'e**, yani
rastgeleye yakına düşüyor.

🍏 **Basit:** Sınavdan önce cevap anahtarını görmüş öğrenci gibi. %90 alıyor ama
öğrendiğinden değil, **ezberden**. Cevap anahtarını elinden alınca notu çöküyor.

### 2.2 Leakage (kopya) nedir — 3 çeşidi
🔬 **Teknik:**
1. **Kimlik (identity) sızıntısı:** *Aynı* protein hem eğitimde hem testte var →
   model proteini ezberler, etkileşimi öğrenmez.
2. **Homoloji sızıntısı (en sinsi):** Testteki proteinin **çok benzer bir akrabası
   (homolog)** eğitimde var. Model o proteini "görmemiş" olsa da neredeyse-ikizini
   görmüş → yine kopya. Bernett'in asıl vurguladığı sızıntı budur.
3. **Derece/hub sızıntısı:** Bazı proteinler çok popülerdir (çok partneri var). Model
   "popüler protein her şeyle etkileşir" kestirmesini öğrenip biyolojiyi öğrenmeden
   skor kasabilir.

🍏 **Basit:** 
- Kimlik: testte **birebir aynı** meyveyi daha önce görmüşsün.
- Homoloji: testte gördüğün elma yeni ama eğitimde **neredeyse aynısı** olan başka
  bir elma vardı → "yeni" demek zor.
- Hub: "Çok satan meyve popülerdir, herkes sever" gibi ucuz kestirme.

### 2.3 C1 / C2 / C3 = sınav zorlukları
🔬 **Teknik:** Test çiftlerinin, eğitimdeki proteinlerle ne kadar örtüştüğüne göre
üç rejim (Park–Marcotte çerçevesi):
- **C1:** her iki protein de eğitimde görülmüş (kolay, sızıntılı).
- **C2:** biri görülmüş, biri yepyeni (gerçekçi).
- **C3:** ikisi de yepyeni (en zor, en dürüst).

🍏 **Basit (sınav):** C1 = açık kitap sınavı (çalıştığın konular çıktı). C2 = yarısı
tanıdık yarısı yeni. C3 = tamamen yeni konudan kapalı kitap. **Aynı öğrenci, üç
farklı zorluk** → not, öğrencinin zekâsından çok **hangi sınav** olduğuna bağlı.

### 2.4 Paperın sonucu
🔬 **Teknik:** Leakage-free (C3) rejimde sekans yöntemleri neredeyse rastgeleye
düşüyor; makale "sekans-tabanlı derin öğrenme aslında genelleme yapmıyor, ezberliyor"
mesajını veriyor. **Not:** bu **ampirik bir gözlem**, matematiksel bir kanıt değil.

🍏 **Basit:** "Bu modeller aslında öğrenmiyor, ezberliyor; kopyayı kesince çuvallıyor."

---

## BÖLÜM 3 — Bizim modelimiz ve araçlar

### 3.1 BMSE modeli
🔬 **Teknik:** Her proteini iki **protein dil modeli** ile sayılara çeviriyoruz:
**ESM2** (dizilim/işlev) + **ProstT5** (yapı-farkında). Sonra bir **cross-attention**
ağı iki proteini kalıntı-kalıntı karşılaştırıp bir etkileşim olasılığı üretiyor.
Elle özellik yok. Her şey FAU **HPC kümesinde**, **SLURM** iş kuyruğuyla, RTX 3080
(10 GB) kartlarda çalışıyor.

🍏 **Basit:** İki proteini önce **sayı listesine** çeviriyoruz (parmak izi gibi),
sonra bu iki parmak izini karşılaştıran bir hakem "yapışır/yapışmaz" diyor.

### 3.2 Protein dil modeli & "embedding" nedir?
🔬 **Teknik:** ESM2, Meta'nın modeli; ChatGPT metinden öğrenirken, ESM2 **milyonlarca
protein diziliminden** öğrenmiş. Protein = amino asit harf dizisi (`M K T A Y...`).
Model bu diziyi okuyup her proteini bir **vektöre (embedding)** çeviriyor; bu vektör
proteinin yapısı/işlevi/evrimi hakkında bilgi taşıyor.

🍏 **Basit:** Milyonlarca yemek tarifi okumuş bir aşçı gibi — malzemeyi görünce
"bu şuna yakışır" sezgisi var. Embedding = bir meyveyi sayılarla tarif etmek
(tatlılık, boyut, renk, sertlik...). Model her proteini böyle bir "özellik listesine"
çeviriyor.

### 3.3 650M vs 3B
🔬 **Teknik:** İkisi de ESM2 ama farklı boyutta. **650M = 650 milyon parametre**
(embedding boyutu 1280). **3B = 3 milyar parametre** (~4.6× büyük, embedding boyutu
2560). Parametre = modelin öğrenerek ayarladığı iç "düğme" sayısı; çok düğme = çok
kapasite ama aynı zamanda **daha çabuk ezber (overfit)** riski.

🍏 **Basit:** 650M = küçük beyin, 3B = ~5 kat büyük beyin. Büyük beyin daha zengin
tarif çıkarır ama bazen **fazla ezberler**.

---

## BÖLÜM 4 — Ne yaptık, adım adım

### Adım 1 — Boru hattını kurduk
🔬 Model + embedding cache (HDF5) + SLURM. İlk sonuç: **0.642 doğruluk**.
🍏 Makineyi kurduk, ilk denemede %64.

### Adım 2 — Kritik bug (verinin yarısını kaybediyorduk!)
🔬 **Teknik:** Embedding çıkarımını SLURM'de parçalara (shard) bölüyorduk. Parçalama
bir Python `set`'ine göre yapılıyordu; `set`'in sırası process'ler arası
**deterministik değil** → parçalar çakışıp boşluk bırakıyor, proteinlerin ~yarısı hiç
gömülmüyordu. Çözüm: sıralamayı **id ile sabitlemek**.
```python
# ÖNCE — eşitlikler rastgele sırada → proteinlerin ~yarısı hiç gömülmedi
ordered = sorted(used, key=lambda p: len(seqs[p]))
# SONRA — id ile deterministik → her SLURM görevi aynı bölünmeyi yapıyor
ordered = sorted(used, key=lambda p: (len(seqs[p]), p))
```
Sonuç: kapsama **11.018/11.018**, doğruluk **0.642 → 0.660**.

🍏 **Basit:** Meğer verinin yarısını yanlışlıkla çöpe atıyormuşuz. Tek satırlık ayarla
düzelttik; skor 0.642 → 0.660 oldu, 0.65 barajını geçtik.

### Adım 3 — "0.660 gerçek mi, kopya mı?" — Sızıntı dedektörü
🔬 **Teknik:** Modelin "popüler proteini her şeyle eşleştirme" kestirmesi yapıp
yapmadığını ölçüyoruz: tahmin ile proteinin **derecesi** (kaç çiftte geçtiği)
arasındaki korelasyon. ~0 = temiz.
```python
def build_degree(dataset):
    c = Counter()
    for a, b, _ in dataset.pairs: c[a]+=1; c[b]+=1
    return np.array([c[a]+c[b] for a,b,_ in dataset.pairs], float)
m["degree_corr"] = float(np.corrcoef(p, degree)[0,1])   # her temiz koşuda ~0
```
C3 sonucu: acc **0.660** / AUROC **0.722**, degree_corr **−0.01** → hub kopyası yok.

🍏 **Basit:** "Model sadece popüler proteinlere mi oynuyor?" diye bir yalan dedektörü
koyduk. Sonuç ~0 → hile yok, skor gerçek.

### Adım 4 — C1/C2/C3'ü **kendi verimizden** kurduk
🔬 **Teknik:** Yeni veri seti indirmeden, aynı cache'i üçe böldük (`resplit.py`) +
esnek bir bölme kancası (`PPI_SPLIT_DIR`).
```python
# C2 = proteinlerin %15'ini "yeni" ilan et
novel = set(proteins[:int(0.15*len(proteins))])
for a,b,y in edges:
    na, nb = a in novel, b in novel
    if not na and not nb: c2_train.append((a,b,y))   # ikisi de tanıdık -> eğitim
    elif na ^ nb:         c2_eval.append((a,b,y))     # tam biri yeni -> C2 test
    # ikisi de yeni -> at (o zaten C3 rejimi)
```
Doğrulama: C1 testin %99.6'sı "ikisi de tanıdık"; C2'de yeni proteinler eğitime
hiç girmiyor.

🍏 **Basit:** Yeni bir sınav satın almadık; kendi soru havuzumuzdan üç zorlukta
sınav hazırladık ve her birinin kopyaya kapalı olduğunu kontrol ettik.

### Adım 5 — Regime curve (leakage eğrisi)
🔬 650M sonuçları: **C1 AUROC 0.814 → C2 0.747 → C3 0.721.** Sızıntı azaldıkça skor
düşüyor — Bernett'in bahsettiği etki, tek veri setinde **sayısal olarak** kanıtlanmış.

🍏 Kolay sınavda 0.81, gerçekçide 0.75, zorda 0.72. Sadece kolay sınavı raporlayan
biri ~9 puan **fazla** iyi görünür.

### Adım 6 — Daha büyük model (ESM2-3B)
🔬 **Teknik:** Kodu env ile model-değiştirebilir yaptık; ESM2-3B'yi bir kez çıkardık
(indir → `.bin`'i safetensors'a çevir → 64 parçalı SLURM extraction, 10 GB'de).
```python
ESM2_MODEL = os.environ.get("PPI_ESM2_MODEL", "facebook/esm2_t33_650M_UR50D")
ESM2_DIM   = int(os.environ.get("PPI_ESM2_DIM", "1280"))  # 1280(650M) | 2560(3B)
```
41.9 GB'lik 3B cache, tam kapsama, OOM yok.

🍏 Daha büyük beyni (3B) getirdik, tüm proteinlerin zengin parmak izini çıkardık.

### Adım 7 — Dürüst 3B sonucu (NEGATİF — önemli)
🔬 3B, C2'de yardım etti (0.747→0.757) ama **C3'te daha KÖTÜ** (0.721→0.714) ve
epoch 0'da ezberledi. Yani **tek başına büyütmek çözüm değil; darboğaz overfit.**

🍏 Büyük beyin tek başına gerçekçi sınavda biraz yardım etti ama en zor sınavda
**daha çok ezberleyip** kötüleşti. "Daha büyük = daha iyi" değilmiş.

### Adım 8 — Asıl kaldıraç: Ensemble + regularization
🔬 **Teknik:** Aynı modeli **3 farklı rastgele tohumla (seed)** eğitip **olasılıkları
ortaladık** (deep ensemble); ayrıca anti-overfit koşusu (dropout 0.3, weight decay 0.05).
```python
ps = [np.load(f"{r}/test_preds.npz")["p"] for r in seed_runs]  # her modelin olasılığı
p  = np.mean(ps, axis=0)                                        # ortalama = ensemble
auroc = roc_auc_score(y, p)
```
Sonuç: **C3 0.714 → 0.736** (650M'e göre +0.015), **C2 0.757 → 0.767** (+0.020);
anti-overfit (0.725) de tek 3B'yi geçti.

🍏 **Basit:** Aynı işi 3 arkadaşa yaptırıp cevaplarını **ortaladık**. Herkesin
kendine has hatası birbirini götürdü, ortak doğru kaldı → tek modelden daha iyi.
"Kazanç kaslardan değil, **takım oyunundan** geldi."

---

## BÖLÜM 5 — Sonuçlar ve metrikler

### 5.1 Master tablo (AUROC)
| Rejim | 650M | 3B (tek) | 3B ensemble |
|---|---|---|---|
| C1 (kolay/sızıntılı) | 0.814 | 0.817 | — |
| C2 (gerçekçi) | 0.747 | 0.757 | **0.767** |
| C3 (en zor/dürüst) | 0.721 | 0.714 | **0.736** |

### 5.2 Metrikler ne demek (rastgele = 0.5 / 0.5 / 0.5 / 0)
- **acc (Doğruluk):** doğru bilme oranı (0.5 eşikte). 0.66 = %66 doğru.
- **AUROC:** **sıralama** kalitesi — gerçek çifti sahte çiftin üstüne koyabiliyor mu?
  Eşikten bağımsız, **ana metriğimiz.** 0.72 = %72 ihtimalle doğru sıralıyor.
- **AUPRC:** aynı ama **pozitif (etkileşen) sınıfa** odaklı.
- **MCC:** −1…+1 arası **dengeli tek sayı**; tek sınıfa oynayarak kandırılamaz. 0 = yazı-tura.

🍏 **Basit:** Doğruluk = kaçını bildi. AUROC/AUPRC = gerçekleri sahtelerin üstüne
doğru sıralayabildi mi (asıl baktığımız). MCC = kandırılamayan dürüst tek not.

### 5.3 Neden C2 ile C3 birbirine yakın?
🔬 C1→C2 düşüşü büyük (−0.067), C2→C3 küçük (−0.026). Çünkü PPI **çiftli** bir problem:
karar, **daha az tanınan tarafla** sınırlı. C1→C2'de ezber tamamen ölüyor (büyük
düşüş); C2→C3'te zaten yeni-protein darboğazı vardı, ikinci proteini de kaybetmek az
ekliyor.

🍏 **Basit:** Bir bilinmeyen protein bile işi neredeyse "iki bilinmeyen" kadar
zorlaştırıyor. Asıl uçurum kolaydan-gerçekçiye geçişte; ondan sonrası zaten zor.

### 5.4 DÜRÜST uyarı: C2 tam temiz DEĞİL
🔬 C2'de **kimlik** ve **hub** sızıntısı yok (doğrulandı), ama **homoloji kontrol
edilmedi** — yeni bir proteinin çok benzer akrabası eğitimde olabilir. Yani C2'nin
0.767'si biraz iyimser olabilir. **Tam temiz olan tek rejim C3** (Bernett'in
homoloji-farkında bölmesini kullandığı için). Bu yüzden asıl dürüst manşetimiz
**C3 ensemble = 0.736**.

🍏 **Basit:** C2 "bir tarafı yeni" ama tamamen kopyasız değil; en dürüst sayımız
**C3**.

---

## BÖLÜM 6 — Ana mesaj (rebuttal)
🔬 Leakage-free PPI **rastgele değil, optimize edilebilir.** Ensemble + ölçek, en zor
(C3) rejimde AUROC'u **+0.015**, gerçekçi (C2) rejimde **+0.020** yükseltti — hepsi
sızıntı-kontrollü. Bernett'in "neredeyse rastgele" tablosu, **zayıf tek modelleri**
yansıtıyor; problemin doğal sınırı değil. **Her zaman rejimi raporla.**

**Dürüst negatifler (skoru oynatmadı):** koevolüsyon (0.537), yapı (≈rastgele),
fusion (≈0). Bunları saklamıyoruz — dürüst bilim.

🍏 **Basit:** "Kopyayı kesince skor sıfıra çakılmıyor; doğru yöntemle (takım + büyük
beyin) yukarı çıkıyor. Yani literatürdeki karamsarlık, zayıf modeller yüzünden."

---

## BÖLÜM 7 — SLAYTTA NE DİYECEĞİM (teknik yok, skor + fikir + analoji)

Slaytta teknik detaya girme; şu akışı anlat:

1. **Problem:** Proteinler lego gibi; hangileri yapışır tahmini. (Neden önemli: ilaç/hastalık.)
2. **Kopya sorunu:** Cevap anahtarını görmüş öğrenci → %90'lar sahte. Kopya kesilince ~%50–65.
3. **Üç sınav:** C1 açık kitap / C2 yarı tanıdık / C3 tamamen yeni. Aynı model, üç zorluk.
4. **Biz ne yaptık:** Dürüst üç sınavı kendi verimizden kurduk + yalan dedektörü koyduk.
5. **Skor eğrisi:** 0.81 → 0.75 → 0.72. Kopya azaldıkça düşüyor (etki gerçek).
6. **Büyük beyin denedik:** Tek başına en zor sınavda yardım etmedi (fazla ezber).
7. **Takım oyunu (ensemble):** 3 modeli ortaladık → en zor sınavda skor **yukarı** (0.72 → 0.74).
8. **Mesaj:** Dürüst PPI rastgele değil, **iyileştirilebilir**; her zaman zorluğu (rejimi) söyle.

**Kullanışlı analoji cümleleri:**
- Etkileşim = lego/anahtar-kilit uyumu.
- Kopya = sınavdan önce cevabı görmek.
- C1/C2/C3 = açık kitap / yarı tanıdık / kapalı kitap sınav.
- Protein dil modeli = milyonlarca tarif okumuş aşçının sezgisi.
- Embedding = meyveyi sayılarla tarif etmek (tatlılık, boyut, renk).
- 650M vs 3B = küçük beyin vs büyük beyin.
- Ensemble = 3 arkadaşa sorup cevabı ortalamak.

---
*Teknik ayrıntı: `FINDINGS.md` · Kod haritası: `README.md` · Grafik: `regime_curve.png`.*
