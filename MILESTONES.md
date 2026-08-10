# hifusim — Milestone Merdiveni ve Başarı Kriterleri

> Kural: **Bir milestone'un TÜM başarı kriterleri sağlanmadan bir sonrakine geçilmez.**
> Kriterler ölçülebilir yazılır; "geçti" kararı test çıktısı/rapor ile belgelenir (docs/devlog.md).
> Durumlar: `[ ]` başlanmadı · `[~]` devam ediyor · `[x]` TAMAMLANDI (kriter kanıtıyla) · `[!]` bloklu
> GUI bilinçli olarak kapsam dışı bırakıldı (kullanıcı kararı, 2026-08-10); eski plandaki GUI fazı iptal, planner/CLI çıktıları metin+figür tabanlı.
> Kaynaklar: PLAN.md (mimari), gemini1/2.md (araştırma — doğrulanmadan güvenilmez; şüpheli noktalar milestone içinde "VERIFY" olarak işaretli).

---

## Faz Grubu A — Temel (yerel, CPU)

### M0 — Repo iskeleti ve araç zinciri `[x]` (2026-08-10)
Paket kurulabilir, test/lint altyapısı çalışır durumda.
- [x] `pyproject.toml` (src-layout, `pip install -e .` çalışıyor; extras: `dev`, `gpu`)
- [x] `ruff check src tests` → 0 hata
- [x] `pytest` → smoke testi yeşil (paket import + sürüm)
- [x] `.gitignore`, `git init`, README taslağı
- [x] `.venv` ile tekrarlanabilir yerel ortam (Python 3.12; numpy 2.5.2, scipy 1.18, pydantic 2.13)
- Not: paket adı "hifusim" ÇALIŞMA ADI'dır; PyPI'a ilk publish ÖNCESİ isim kararı + kontrol yapılacak (M11 civarı).

### M1 — Core temeller: backend + Grid + PML + config `[x]` (2026-08-10)
Her şeyin üstüne oturacağı katman. API-first kararının ilk somut yüzeyi.
- [x] `core.backend`: `get_backend("auto"|"numpy"|"cupy")`; cupy yoksa numpy'a düşer; birim testli
- [x] `core.grid.Grid`: ndim∈{1,2,3}, izotropik `dx`, k-vektörleri (rfft/fft uyumlu), ppw hesabı, mm↔voxel yardımcıları
- [x] `core.pml.PMLSpec`: kalınlık mm→voxel türetimi; Gaussian sponge profili (backend-bağımsız üretim)
- [x] `config`: pydantic v2 taban modeli (`extra="forbid"`), `GridConfig` JSON round-trip; mm-tabanlı alanlardan türetilmiş voxel değerleri TEK yönlü (mm → voxel; voxel el ile yazılamaz)
- Başarı kriterleri (hepsi test olarak kodlandı):
  - k-vektörleri `2π·fftfreq` analitik değerleriyle birebir (rel err < 1e-12)
  - `Grid.ppw(f0, c_min)` bilinen örneği doğrular (dx=0.30 mm, f0=1.1 MHz, c=1450 → 4.39 ppw)
  - Config JSON round-trip: `cfg == GridConfig.model_validate_json(cfg.model_dump_json())`
  - Bilinmeyen alan → ValidationError (sessiz yutma yok)
  - Sponge profili: kenarda min, içeride tam 1.0; genişlik-0 durumunda tamamen 1.0

### M2 — Materials + Medium `[x]` (2026-08-10)
- [x] `materials.Material` (alpha_np_m, rho, c, beta; opsiyonel termal alanlar — M18 kancası)
- [x] `materials.MaterialDB`: id→Material; notebook `TISSUE_PROPS` birebir port (`breast_default()`), `water()` preseti
- [x] `medium.Medium`: `homogeneous(...)` ve `from_id_map(...)`; float32 contiguous property hacimleri (alpha/rho/c/beta); c_min/c_max
- Başarı kriterleri:
  - `breast_default()` değerleri notebook TISSUE_PROPS ile birebir aynı (test sabitlerle karşılaştırır)
  - `from_id_map`: her doku id'sinin voxel'leri doğru özellik değerini alır; bilinmeyen id → hata
  - Homojen su ortamı: tüm hacimler sabit; c_min==c_max==c0
  - dtype/bellek: tüm hacimler float32 C-contiguous

### M3 — Analitik referans paketi `[x]` (2026-08-10)
Çözücüden ÖNCE gelir: çözücünün doğrulanacağı zemin. GUI olmasa da ArrayDesigner'ın
gelecekteki "anlık beam önizleme"si de bu modüldür.
- [x] `analytic.rayleigh`: vektörize + parça-parça (chunked) Rayleigh–Sommerfeld integrali (kavisli kaynak nokta bulutu → keyfi hedef noktalar)
- [x] `analytic.oneill`: O'Neil (1949) odaklı çanak eksenel basınç (kapalı form) + odak kazancı
- [x] `analytic.planewave`: üstel zayıflama yasası; Fubini harmonik serisi B_n(σ)=2 J_n(nσ)/(nσ); shock mesafesi x̄=1/(βεk)
- Başarı kriterleri (test olarak kodlandı):
  - O'Neil ekseni vs Rayleigh sayısal (aynı çanak, süreklilik→nokta bulutu): odak bölgesi korelasyonu r > 0.999, tepe konum farkı < örnekleme adımı
  - O'Neil odak basıncı ~ Rayleigh odak basıncı (rel fark < %2)
  - Fubini: σ→0 limitinde B1→1, B2→σ/2 (küçük-σ açılımı, rel < %1); σ=1'de seri yakınsak ve B1 monoton azalan
  - Zayıflama: p(x)=p0·e^(−αx) birebir
  - Tüm analitik testler CPU'da < 60 s

---

## Faz Grubu B — Çözücüler (CPU referans → GPU)

### M4 — Lineer k-space PSTD çözücüsü (numpy; 1D/2D/3D) `[ ]`
Boyut-agnostik ilk tam dalga çözücü; CW + steady-state fazor çıkarımı.
- [ ] `solvers.base.SolverBase` + yetenek deklarasyonu + `solvers.registry`
- [ ] 1. mertebe kuple denklemler (p, u), k-space gradyan/diverjans, kappa sinc düzeltmesi, exact-period dt, Gaussian sponge PML, CW kaynak enjeksiyonu (ramp'li), tek-bin DFT fazor + p_max
- Başarı kriterleri:
  - Düzlem dalga (1D/2D/3D, periyodik yönde): faz hızı hatası < %0.1 @ 4 ppw, 50 periyot
  - Üstel absorpsiyon: ölçülen α, konfigüre α'dan < %1 sapar
  - PML: normal geliş yansıma genliği < giriş genliğinin %3'ü (≈ −30 dB)
  - Küçük 3D odaklı çanak (su, lineer) vs O'Neil: normalize eksenel profil r > 0.99; −6dB eksenel/lateral genişlik farkı < %5; odak konumu < 1 voxel
  - Fazor çıkarımı: saf CW sinüs girişinde genlik hatası < %0.5 (sızıntısızlık)
  - Kararlılık: 200 periyot koşuda enerji patlaması yok (peak drift < %1)

### M5 — Westervelt nonlineerlik + p_max/2f0 `[ ]`
- [ ] β terimi, p_max takibi, opsiyonel 2f0 fazor, `westervelt` çözücüsü registry'de
- Başarı kriterleri:
  - β=0 ⇒ `westervelt` ≡ `linear` (aynı grid'de rel fark < 1e-6)
  - Fubini ön-şok rejimi (σ ≤ 0.3): A2/A1 oranı analitikten < %5 sapar
  - amp ≤ p_max · (1/cos(π/spp)) diskre tavan değişmezi her voxel'de
  - Notebook'un bilinen bandı yeniden üretilir: küçük senaryoda amp/p_max ∈ [0.85, 1.0]

### M6 — Kaynak modeli + transducer arrays `[ ]`
- [ ] `arrays.ArchimedeanSpiral` (notebook portu) + genel `CustomArray`; DAS fazlama; eleman→voxel kabuk projeksiyonu; faz haritası (sin/cos) temsili
- Başarı kriterleri:
  - Eleman sayısı/aktif alan/eleman yarıçapı notebook değerleriyle birebir (128, r≈… testte sabitlenir)
  - Kaynak voxel'leri: 128/128 eleman temsil edilir; tekrarlı voxel yok
  - DAS fazlarıyla Rayleigh önizleme: hedef odak, geometrik odaktan kaydırılmış hedefe < λ/2 mesafede odaklanır
  - Entegrasyon: M4 çözücü + spiral array su içinde koşar; odak konumu geometrik odaktan < 1 voxel sapar

### M7 — CuPy backend (CUDA) `[ ]` — Colab oturumu gerektirir
- [ ] ElementwiseKernel'ların portu; aynı çözücü kodu iki backend'de; fp32 yolu
- Başarı kriterleri:
  - numpy↔cupy parite: mini 3D senaryoda fazor/p_max rel fark < 1e-5 (fp32 toleransı belgelenir)
  - Colab T4 VE A100'de tam boy (dx=0.30, 512³ FFT sınıfı) koşu OOM'suz tamamlanır
  - Adım süresi ölçülür ve `benchmarks/`e damgalanır (baseline; M19 bunu referans alır)
  - GPU yokken testler otomatik SKIP (CI kırılmaz)

### M8 — Planner v1 (süre + VRAM tahmini) `[ ]`
- [ ] Statik VRAM modeli (tampon dökümü + cuFFT workspace payı + %15 marj); süre modeli a·N·logN + b·N; `gpu_db.json` (T4/L4/V100/A100/H100); cihazda kalibrasyon (~20 adım) → `~/.hifusim/calibration.json`; `sim.estimate(gpu=...)` + `planner.compare(...)`
- Başarı kriterleri:
  - VRAM tahmini, Colab'da ölçülen mempool tepe değerinin ±%10'u içinde (≥2 farklı grid boyutunda)
  - Kalibrasyon SONRASI süre tahmini gerçekleşenin ±%25'i içinde (aynı cihaz, ≥2 senaryo)
  - Tahmin kaynağı raporda etiketli: `db` | `calibrated` | `measured`
  - OOM öngörüsünde eyleme geçirilebilir öneri metni (dx büyüt / AOI küçült / linear'a geç)

### M9 — KZK çözücüsü `[ ]`
- [ ] Operator splitting: difraksiyon = angular spectrum (VERIFY: gemini2 önerisi; CN alternatifi karşılaştırılacak), absorpsiyon, nonlineerlik = zaman-uzayı Burgers (şok emniyeti); Rayleigh ile başlangıç düzlemi projeksiyonu; registry'de `kzk`
- Başarı kriterleri:
  - Lineer odaklı piston (paraksiyel, F-number ≥ 2): eksenel profil O'Neil'e r > 0.99
  - Fubini düzlem-dalga limiti: A2/A1 < %5 sapma (σ ≤ 0.3)
  - Çapraz doğrulama: zayıf odaklı su senaryosunda kzk vs westervelt fokal basınç farkı < %5, odak konumu < 1 voxel (paraksiyel geçerlilik bölgesinde)
  - Süre: aynı senaryoda full-wave'e karşı ≥ 50× hız (beklenti ~100×; ölçülüp belgelenir)

---

## Faz Grubu C — Veri, rapor, doğrulama (v1 çizgisi)

### M10 — IO: HDF5 kontratı + atomik yazım + resume `[ ]`
- [ ] Notebook kontratının portu (input/, output/, attrs; float16 dinamik kuantizasyon); `tmp→os.replace` atomik yazım; `DriveResilientStore` (mkdir-doğrulama, write-probe); resume skip-guard
- Başarı kriterleri:
  - Round-trip: float16 kuantizasyon max norm hata ≤ 1e-3 kontratı testte doğrulanır; kontrat aşılırsa float32'ye düşer
  - Kesinti simülasyonu: yazım ortasında kill → görünür bozuk dosya YOK (tmp temizlenir, ana dosya ya eski ya tam)
  - Resume: 10 örneklik mini sette ortadaki 1 dosya silinince yalnızca o id yeniden üretilir
  - Faz konvansiyonu/absorpsiyon modeli attr'ları her dosyada mevcut (downstream sözleşmesi)

### M11 — Study/Report + doğrulama harness'i `[ ]`
- [ ] `study.Study`: config + koşu + sonuç + figürler; `report()` → Markdown (+JSON); ortam/GPU/git-hash damgası; `Study.sweep(...)`; analitik doğrulama süiti tek komutla rapor üretir
- Başarı kriterleri:
  - Tek komut: `python -m hifusim.validation run-analytic` → damgalı JSON+MD rapor `benchmarks/reports/` altına düşer
  - Rapor, planner tahmini vs gerçekleşen tablosunu içerir
  - Sweep: 3-noktalı p0 taraması uçtan uca koşar, birleşik rapor üretir
  - **v1 ön-etiketi**: M4–M11 kriterlerinin tamamı yeşil → isim kararı + GitHub public + `v0.1` tag

### M12 — k-Wave karşılaştırma harness'i `[ ]`
- [ ] `kwave-python` (CPU/OMP) entegrasyonu; T0 sanity kapısı (homojen su — GPU binary'yi her ortamda önce bundan geçir; kalırsa "environment-broken" damgası); karşılaştırma metrikleri (relL2, Pearson, odak konum/amp, −6dB genişlikler, sidelobe)
- Başarı kriterleri:
  - Küçük grid lineer su çanağı: hifusim vs k-Wave CPU → relL2 < %3, r > 0.99, odak konumu < 1 voxel
  - Heterojen küçük fantom (2 doku): relL2 < %5, odak basınç farkı < %10 (ITRUSST koridoru)
  - Nonlineer küçük senaryo: 2f0/f0 oranı farkı < %10
  - Rapor damgalı; k-Wave sürümü/binary tipi (CPU/GPU) kayıtlı
  - VERIFY: k-Wave GPU binary'nin Colab durumu yeniden test edilir (notebook v11 bulgusu + gemini "eski mimari desteği çekildi" iddiası)

### M13 — Dataset pipeline (Katman C) `[ ]`
- [ ] `pipelines.DatasetGenerator`: dondurulmuş LHS (seed→checksum), üretim döngüsü, background save, ETA (planner'dan), metadata/timing CSV, disk kontrolü
- Başarı kriterleri:
  - Mini dataset (N=3, küçük grid): iki ayrı koşuda AYNI checksum'lar (LHS dondurma değişmezi)
  - Kesinti+resume testi: 2. örnekte kill → yeniden başlatma kaldığı yerden, çift üretim yok
  - metadata.csv satır bütünlüğü (append-only, kolon şeması sabit)
  - ETA log'ları planner modelini kullanır (start→start cadence ölçümü korunur)

### M14 — Colab üretim doğrulaması `[ ]` — Colab oturumu gerektirir
- [ ] Tam boy dx=0.30 senaryosu Colab A100'de kütüphaneden koşar; notebook v12 ile karşılaştırma
- Başarı kriterleri:
  - amp/p_max bandı [0.85, 0.95] içinde; focus_ratio ve fokal alan deseni notebook koşusuyla tutarlı (normalize r > 0.99 — bire-bir sayısal eşitlik ARANMAZ, karar #2)
  - t_end > 110 µs kontratı korunur
  - Cadence ≤ notebook v12 cadence'i (~65 s/sample A100) — kütüphaneleşme yavaşlatmadı kanıtı
  - `examples/02_dataset_generation.ipynb` ince sürücüsü uçtan uca çalışır

---

## Faz Grubu D — Fizik genişlemesi

### M15 — Eksenel simetri (AS) çözücüsü `[ ]`
- [ ] VERIFY ÖNCE: güncel CuPy'de `cupyx.scipy.fft.dct/dst` durumu (gemini2 "yok, ayna-FFT gerekir" diyor — sürüm notlarıyla doğrula). Yoksa ayna-genişletme (mirror+FFT) DTT katmanı; WSWA/WSWS şemaları (Treeby yaklaşımı)
- Başarı kriterleri:
  - Eksenel simetrik çanak: AS çözücü vs 3D full-wave → eksenel profil r > 0.995, odak basıncı < %3 fark
  - Aynı fiziksel problemde ≥ 50× hızlanma (beklenti 100–300×; ölçülür)
  - DTT katmanının birim testleri: DCT/DST kimlikleri (bilinen analitik çiftler) < 1e-10

### M16 — Power-law absorpsiyon (fractional Laplacian) `[ ]`
- [ ] α(f)=α0·f^y (y∈[1,1.5]) + Kramers-Kronig dispersiyonu; k-uzayında |k|^y çarpanı; çözücülerde feature flag
- Başarı kriterleri:
  - Çok-frekanslı test: ölçülen α(f) eğrisi hedef güç yasasından < %2 sapar (f0, 2f0, 3f0 noktalarında)
  - Dispersiyon: faz hızı farkı Kramers-Kronig öngörüsüyle < %1
  - k-Wave power-law senaryosuyla çapraz test: relL2 < %5
  - Kapalıyken (flag off) mevcut üstel model bit-değişmez

### M17 — Broadband zaman alanı `[ ]`
- [ ] Keyfi kaynak sinyali (puls, tone-burst), `sensors.TimeSeries` genel kaydı, spektral çıkarım genellemesi
- Başarı kriterleri:
  - Gaussian puls su içinde: varış zamanı TOF analitiğinden < dt; spektrum şekli korunur (r > 0.999)
  - Tone-burst → CW limitine yakınsama: uzun burst fazoru CW fazoruyla < %1 fark
  - Bellek: kayıt pencereli/decimated çalışır (tam alan × tüm adımlar tutulMAZ)

### M18 — Termal modül: Pennes + CEM43 `[ ]`
- [ ] `sensors.HeatingSource` (Q = 2·α·I; harmonik katkılarıyla) → `thermal.PennesSolver` (GPU difüzyon+perfüzyon) → CEM43 doz haritası
- Başarı kriterleri:
  - Analitik nokta-kaynak/Gaussian ısı difüzyonu karşılaştırması: rel err < %2
  - Perfüzyon terimi: bilinen kararlı-durum çözümüyle < %2
  - Uçtan uca örnek: sonication → T(r,t) → CEM43 haritası; k-Wave `kWaveDiffusion` senaryosuyla çapraz test < %5
  - Tıbbi sorumluluk notu dokümanda (araştırma amaçlı, klinik karar aracı değil)

---

## Faz Grubu E — Performans ve ölçek

### M19 — GPU performans turu `[ ]`
- [ ] cuFFT plan cache doğrulaması, RawKernel füzyonu (PML+absorpsiyon+nonlineerlik tek kernel), CUDA Graphs (VERIFY: CuPy'de graph capture API olgunluğu), TF32 güvenlik çalışması
- Başarı kriterleri:
  - M7 baseline'a göre adım süresi ≥ 1.5× iyileşme (A100'de; her teknik ayrı ölçülür ve raporlanır)
  - Parite süiti değişmez: fp32 sonuçlarında rel fark < 1e-5 (TF32 ayrı raporlanır, default OFF)
  - Planner katsayıları yeni backend'e göre yeniden kalibre edilir

### M20 — Çoklu GPU / çok büyük problemler `[ ]`
- [ ] VERIFY: cuFFT-Mp/NCCL slab decomposition fizibilitesi (spektral global FFT maliyeti); alternatif: domain-bölgeli hibrit veya out-of-core
- Başarı kriterleri:
  - 2×GPU'da weak-scaling verimi ≥ %60 VEYA fizibilite raporu "yapılmaz çünkü…" kararıyla kapatılır (negatif sonuç da geçerli çıktı)
  - ≥ 1024³ problem tek H100'de (out-of-core/chunked) VEYA 2×A100'de çözülür

### M21 — ITRUSST benchmark süiti `[ ]`
- [ ] PH1-BM1…BM9 alt kümesi (düz piston + odaklı çanak; su, tek katman kemik, çok katman, tam kafatası ilerledikçe); sonuçlar yayınlanmış kod medyanlarıyla karşılaştırılır
- Başarı kriterleri (ITRUSST koridorları):
  - Odak basıncı: yayınlanan kodlar medyanından < %10 sapma
  - Odak konumu: < 1 mm
  - −6dB boyutları: literatür bandı içinde (0.2–0.6 mm toleranslar problem başına)
  - Sonuçlar `benchmarks/reports/itrusst/` altında yayınlanır (repo'nun vitrin raporu)

---

## Faz Grubu F — "Piyasanın üstü" vizyonu

### M22 — Viskoelastik/kemik (kayma dalgaları) `[ ]`
- Kafatası/kemik için elastik PSTD (Kelvin-Voigt); ITRUSST kafatası senaryolarında akustik-yalnız çözümden ölçülebilir iyileşme; k-Wave `pstdElastic` ile çapraz test.

### M23 — Tedavi planlama araçları `[ ]`
- Faz optimizasyonu (time-reversal + kısıtlı optimizasyon), aberasyon düzeltme, hedef-fonksiyonlu planlama (odak basıncı maksimize / yan lob minimize). Kriter: heterojen fantomda düzeltme sonrası fokal basınç ≥ %20 artış (senaryo bazlı, raporlanır).

### M24 — İleri entegrasyonlar `[ ]`
- ML surrogate köprüsü (dataset pipeline'ın doğal devamı: eğitilmiş modelin çözücü yanında "tahminci" olarak servis edilmesi), belirsizlik miktarlama (doku özellik dağılımlarıyla MC), adjoint/türevlenebilirlik fizibilite raporu (j-Wave'e karşı konum), bulut orkestrasyonu (çoklu Colab/VM koşu yöneticisi).
- Kriter: her biri ayrı fizibilite + prototip raporu; hangisinin v2 olacağına kullanıcı karar verir.

---

## Sıradaki iş (canlı)

- **Şimdi**: M4 — lineer k-space PSTD çözücüsü (numpy, boyut-agnostik).
- M0–M3 kriter kanıtları: `pytest` çıktısı (devlog 2026-08-10 girişi) + bu dosyadaki işaretler.
