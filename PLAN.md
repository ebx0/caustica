# HIFU Simülasyon Kütüphanesi — Mimari Plan (v2)

> Tarih: 2026-08-10 · Hazırlayan: Claude (Fable) · Sonraki oturumlar: Opus
> Kaynak: `hifu_pred_dx300_t128.ipynb` (v12.3, tek hücre, ~3750 satır) + `mtype.txt` (0.5 mm meme fantomu, 310×355×253, Fortran order)
> Notebook'un düz metin dökümü: `_code_cells.py` (bu klasörde; refactor sırasında referans olarak kullan)
> v2 değişikliği: çoklu çözücü formülasyonu (Westervelt/Lineer/KZK), 2D/3D/eksenel-simetri boyut stratejisi, maliyet-tahmin (planner) modülü, termal modül arayüzü, COMSOL-vari Study/Report katmanı eklendi.

## 0. Kullanıcı kararları (kesinleşti — yeniden tartışma)

2026-08-10, tur 1:
1. **Önce genel amaçlı API** (Grid/Medium/Source/Sensor soyutlaması); dataset pipeline'ı bu API'nin üstünde bir "uygulama".
2. **Eski dx300_t128 dataseti ile bire-bir sayısal uyumluluk ŞART DEĞİL** — fizik doğruluğu yeterli. (Kütüphane kendi içinde golden-regression testleri tutar.)
3. **GUI notebook-native**: ipywidgets + PyVista/Plotly — aynı kod yerelde (VS Code/Jupyter) ve Colab'da çalışır.
4. **GitHub public repo**; Colab kurulumu `pip install git+https://...`.

2026-08-10, tur 2:
5. **Üç çözücü de v1'de**: Westervelt (full-wave PSTD), Lineer (optimize edilmiş beta=0 yolu), KZK (parabolik, z-marş). Kullanıcı v1'in gecikme maliyetini bilerek kabul etti — API'nin en baştan çok-çözücüye oturması öncelikli.
6. **3D + 2D birlikte** (çekirdek boyut-agnostik; 2D aynı zamanda ucuz test/CI ortamı). **Eksenel simetri (AS) sonraki faz** — FFT yerine DTT/WSWA-WSWS transformları gerektirir (k-Wave'in axisymmetric çözücüsündeki gibi), ayrı grid+transform katmanı olarak eklenir.
7. **Termal modül (Pennes bioheat + CEM43 doz) kapsamda, GEÇ fazda** — ama mimari ilk günden ısı-kaynağı (Q) arayüzünü tanımlar.
8. **v1 = CW + steady-state** (mevcut güçlü yan: adaptif settling, fazor/p_max/probe). Source/Sensor API'si keyfi zaman sinyaline izin verecek şekilde TASARLANIR, broadband implementasyonu sonraki faza kalır.

## 1. Vizyon

"Colab'da çalışan gerçek bir k-Wave rakibi, COMSOL gibi duran" bir sistem. Üç katman:

- **Katman A — Simülasyon çekirdeği**: çok-formülasyonlu (Westervelt/Lineer/KZK, ileride AS + broadband), çok-boyutlu (2D/3D), backend'li (numpy/CuPy-CUDA), her parametresi config'ten ayarlanabilir çözücü ailesi.
- **Katman B — Mühendislik kabuğu (COMSOL hissi)**: Study/Report katmanı, koşu ÖNCESİ kaynak tahmini (planner: GPU'ya göre süre + VRAM), doğrulama/benchmark raporları, GUI.
- **Katman C — Uygulamalar**: mevcut "learned Green's functions" dataset üretimi, Colab sürücü notebook'ları.

Notebook'taki fizik motoru **korunacak değer**: k-space PSTD + kappa sinc düzeltmesi, exact-period dt (sızıntısız tek-bin DFT), Westervelt nonlineerlik, üstel absorpsiyon, Gaussian sponge PML, adaptif settling, fazor + p_max çıkarımı, atomik HDF5 yazımı, resume mantığı, float16 dinamik kuantizasyon, O'Neil/Rayleigh analitik doğrulaması. Bunlar modüllere dağıtılır, yeniden yazılmaz.

## 2. Paket yapısı

Çalışma adı: **`hifusim`** (M0'da PyPI kontrolüyle kesinleştirilecek; öneriler: `sonoforge`, `pywestervelt`, `openhifu`).

```
hifusim/                      (repo kökü)
├─ pyproject.toml             # extras: [gpu]=cupy, [ui], [kwave], [dev]
├─ src/hifusim/
│  ├─ core/                   # Grid (ndim=2|3), PML tanımı, birimler, backend dispatch (xp),
│  │                          # logging, ResourceProfiler
│  ├─ config/                 # pydantic v2 modelleri; JSON/YAML load-save; türetilmiş değerler
│  │                          # (mm → voxel HER ZAMAN türetilir; recompute_grid_derived mantığı)
│  ├─ materials/              # MaterialDB (alpha, rho, c, beta, + termal: k, cp, perfüzyon — geç faz)
│  ├─ geometry/               # Phantom importerları (mtype.txt, ileride NIfTI/NRRD), NN-upsample,
│  │                          # padding+PML yerleşimi, focal takibi, dx-damgalı geometri cache
│  ├─ arrays/                 # ArchimedeanSpiral (mevcut 128'lik), annular/ring/custom;
│  │                          # DAS fazlama; faz-haritası (sin/cos) temsili
│  ├─ sources/                # Source soyutlaması (CW sürüş v1; API keyfi sinyale açık tasarlanır)
│  ├─ sensors/                # SteadyStatePhasor (f0, 2f0), PeakPressure, TimeSeriesProbe, FullField,
│  │                          # HeatingSource (Q çıkarımı — termal arayüzün akustik ucu)
│  ├─ solvers/                # ÇOK-ÇÖZÜCÜ MİMARİSİ (aşağıda §3)
│  │  ├─ base.py              # SolverBase + yetenek deklarasyonu
│  │  ├─ registry.py          # isimle seçim + plugin (entry-points)
│  │  ├─ westervelt/          # full-wave k-space PSTD (2D/3D)
│  │  ├─ linear/              # lineer k-space (beta=0 optimize yolu)
│  │  ├─ kzk/                 # parabolik, retarded-time, z-marş
│  │  └─ backends/            # numpy.py (CPU referans), cupy.py (ElementwiseKernel/CUDA)
│  ├─ planner/                # KOŞU ÖNCESİ KAYNAK TAHMİNİ (aşağıda §4)
│  ├─ analytic/               # Rayleigh integral, O'Neil, düzlem dalga; Fubini (nonlineer referans)
│  ├─ io/                     # HDF5 kontratı, atomik yazım (tmp→replace), float16 kuantizasyon,
│  │                          # DriveResilientStore (v12.3 dersleri)
│  ├─ pipelines/              # DatasetGenerator: LHS (dondurulmuş seed), resume/skip-guard,
│  │                          # background save, ETA/disk takibi, metadata/timing CSV
│  ├─ study/                  # COMSOL-vari katman: Study = config + koşu + sonuç + rapor (§5)
│  ├─ validation/             # analitik metrikler + k-Wave harness + JSON/MD rapor
│  ├─ thermal/                # (GEÇ FAZ) Pennes bioheat + CEM43; v1'de sadece Q arayüzü tanımlı
│  ├─ viz/                    # fig01–23 + figV/figD süitinin modülerleştirilmiş hali
│  └─ ui/                     # notebook-native widget'lar (PhantomViewer/Editor, ArrayDesigner
│                             # + Rayleigh anlık beam önizleme, SimulationBuilder → Config JSON)
├─ tests/                     # pytest; 2D/CPU mini-grid'ler; GPU parite testleri (varsa koşar)
├─ benchmarks/                # k-Wave/analitik senaryolar + damgalı raporlar
├─ examples/                  # ince Colab sürücü notebook'ları
└─ docs/
```

### Hedef API (taslak — Opus ilk iş bunu netleştirmeli)

```python
import hifusim as hs

grid   = hs.Grid(ndim=3, dx=0.30e-3, pml=hs.PML(thickness_mm=5.0))
medium = hs.Medium.from_phantom("mtype.txt", materials=hs.materials.breast_default(), grid=grid)
array  = hs.arrays.ArchimedeanSpiral(n=128, d_outer=0.100, d_inner=0.044, roc=0.100)
source = array.source(f0=1.1e6, p0=1.0e6, phases=phases)

solver = hs.solvers.get("westervelt")(backend="auto", precision="fp32")   # "linear" | "kzk"
sim    = hs.Simulation(grid, medium, source,
                       sensors=[hs.sensors.SteadyStatePhasor(harmonics=(1, 2)),
                                hs.sensors.PeakPressure()],
                       solver=solver)

est = sim.estimate(gpu="A100-80GB")     # KOŞMADAN: süre, VRAM dökümü, adım sayısı, uyarılar
print(est.table())

res = sim.run(until=hs.SteadyState(tol=0.01, min_periods=44, t_end_min_us=110))
res.p_amp, res.p_phase, res.p_max
res.save("sample_00000.h5")
```

Her şey **Config**'e serileşir: `sim.config.to_json()` ↔ `hs.Simulation.from_config("run.json")`. GUI aynı JSON'u üretir; Colab'daki main kod sadece yükleyip koşturur.

## 3. Çok-çözücü mimarisi (karar #5)

**Registry + yetenek deklarasyonu.** Her çözücü `SolverBase`'i uygular ve yeteneklerini bildirir:

```python
class WesterveltPSTD(SolverBase):
    name = "westervelt"
    capabilities = Caps(ndim={2, 3}, nonlinear=True, absorption={"exponential"},
                        drive={"cw"}, geometry="full-wave")
```

- `Simulation` kurulurken config, seçilen çözücünün yetenekleriyle DOĞRULANIR (ör. KZK'ya sponge-PML parametresi verilirse anlaşılır hata). COMSOL hissinin yarısı budur: yanlış kombinasyon sessizce koşmaz, kurulum anında açıklanır.
- Üç v1 çözücüsü:
  - **`westervelt`**: mevcut motorun taşınması (2D/3D). Referans çözücü.
  - **`linear`**: beta=0 + nonlineer kernel'ların atlandığı ayrı optimize yol. Aynı fazor çıkarımı; hızlı parametre taramaları ve doğrulama koşuları için.
  - **`kzk`**: parabolik yaklaşım, retarded-time çerçevesi, z-yönünde marş; operator splitting (difraksiyon [açısal spektrum veya Crank–Nicolson] + absorpsiyon + nonlineerlik [Burgers adımı]). Kavisli/fazlı array için başlangıç düzlemi Rayleigh ile projekte edilir (analytic/ modülü burada da kullanılır). Full-wave'e göre ~2 mertebe ucuz → tasarım iterasyonu ve planner'ın "hızlı önizleme" modu.
- Ortak altyapı paylaşılır: Grid/Medium/Source/Sensor, backend, IO, viz. Çözücüler yalnızca zaman/uzay ilerletme şemasını getirir.
- Plugin: registry entry-points okur → üçüncü taraf pip paketi kendi çözücüsünü `hs.solvers.get(...)` ile sunabilir (ölçeklenebilirlik gereksinimi).
- **Çözücüler arası çapraz doğrulama testi**: aynı lineer senaryoda `linear` ≈ `westervelt(beta=0)` ≈ `kzk` (paraksiyel bölgede) ≈ Rayleigh — dört yollu tutarlılık, validation/ süitinin parçası.

## 4. Planner — koşu öncesi kaynak tahmini (karar: kullanıcı için birinci sınıf özellik)

`sim.estimate(gpu=...)` koşmadan şunları verir:

- **VRAM dökümü**: kalıcı tamponların (p, U×3, grad_k×3 complex, div_k, pmax, fazor, 4 özellik haritası, sponge'lar) grid boyutu + çözücü + hassasiyete göre bayt-bayt tablosu + FFT çalışma alanı payı + emniyet marjı. Seçilen GPU'nun VRAM'ini aşıyorsa net uyarı ve öneri (dx büyüt / AOI küçült / linear çözücüye geç).
- **Süre tahmini**: adım maliyeti ≈ a·(N log N)·n_fft + b·N model; adım sayısı settling modelinden (TOF + min_settle + record) × spp. Katsayılar iki kaynaktan:
  - **GPU veritabanı**: T4, L4, V100, A100-40/80, H100 için ölçülmüş katsayılar (`planner/gpu_db.json`, sürümlenir; Colab koşularıyla doldurulur).
  - **Kalibrasyon modu**: mevcut cihazda ~20 adım koşup katsayıyı fit eder, cache'ler (`~/.hifusim/calibration.json`). Tahmin > kalibrasyonlu tahmin > ölçüm hiyerarşisi raporda açıkça etiketlenir.
- **Karşılaştırma tablosu**: `hs.planner.compare(config, gpus=["T4", "A100-80GB", "H100"])` → hangi GPU'da kaç s/örnek, 1000 örnek kaç saat, VRAM sığar mı.
- UI'da SimulationBuilder her parametre değişiminde tahmini canlı gösterir; pipelines/ ETA'sı da aynı modeli kullanır (mevcut cadence ölçümü kalibrasyonu besler).

## 5. Study/Report katmanı (COMSOL hissi)

- `hs.Study`: config + koşu(lar) + sonuçlar + figürler tek pakette; `study.report()` → Markdown/HTML rapor (koşu parametreleri, ortam/GPU, planner tahmini vs gerçekleşen, QA istatistikleri, figürler). Notebook'taki "numeric report + figür süiti" buraya taşınır.
- Parametre taraması: `hs.Study.sweep(config, over={"source.p0": [...], ...})` — her koşu damgalı, raporlar birleştirilebilir. (Dataset pipeline'ı da teknik olarak bir sweep'tir; pipelines/ bunu LHS + resume ile özelleştirir.)
- Her koşunun `run_config.json` + ortam bilgisi + git sürümü sonuçların yanına yazılır (tekrarlanabilirlik — COMSOL'un "model dosyası" karşılığı).

## 6. Backend stratejisi ve C/C++ kararı

- **numpy backend**: CPU referansı; 2D mini-grid'lerle CI'da her şey koşar. **cupy backend**: CUDA — mevcut ElementwiseKernel'lar zaten derlenmiş CUDA, çözücü cuFFT-domine. **Şimdilik C/C++ YOK** (kazanç ~%10–30, bakım maliyeti yüksek); backend arayüzü ileride native/mükemmelleştirilmiş backend'e açık.
- İleriki hız işleri: RawKernel füzyonu, planlı FFT'ler, CUDA Graphs, stream overlap, TF32 denemesi, çok-GPU domain decomposition (spektral yöntemde zor — Gemini sorgusunda).

## 7. Test stratejisi

1. **Birim testler** (hızlı, CPU, çoğu 2D): geometri dönüşümleri, array üretimi, faz haritası yerleşimi, LHS değişmezliği (seed→checksum), atomik IO + kesinti simülasyonu, float16 kontratı, config türetme/doğrulama (yetenek kontrolü dahil), resume/skip-guard, planner VRAM modeli (bilinen küçük case'e karşı).
2. **Fizik testleri** (küçük grid, yavaş işaretli): düzlem dalga faz hızı + dispersiyon, üstel absorpsiyon yasası, lineer odaklı çanak vs **O'Neil**, fokal düzlem vs **Rayleigh** (normalize şekil), nonlineer harmonik büyüme vs **Fubini**; **çözücüler arası çapraz tutarlılık** (§3).
3. **Golden regression**: dondurulmuş küçük senaryoların alanları repo'da; her PR'da toleranslı karşılaştırma (kendi geçmişimize karşı).
4. **GPU parite**: numpy vs cupy aynı mini senaryo (GPU varsa koşar; CI'da atlanır).
5. CI: GitHub Actions (ruff + pytest CPU). GPU testleri Colab'da elle veya self-hosted runner.

## 8. k-Wave karşılaştırması (validation/)

Kritik bulgu (notebook v11): **k-wave-python'ın GPU binary'si Colab'da bozuk** (T0: homojen su küpü 2.09e12 Pa'ya patladı — ortam hatası). Bu yüzden:
- Karşılaştırmalar **k-Wave CPU (OMP) binary'siyle küçük/orta grid'lerde**; her ortamda önce T0 sanity, geçemezse rapora "environment-broken" damgası.
- Metrikler: relL2, Pearson r, fokal konum/amp farkı, -6dB genişlikler, sidelobe; + performans (s / 1e6 voxel-adım, VRAM — planner verisiyle aynı formatta).
- Her koşu `benchmarks/reports/` altına **JSON + Markdown, tarih/ortam/GPU/sürüm damgalı** ("raporlarını tutmalı" gereksinimi).
- Analitik süit birincil doğrulama; k-Wave ikincil çapraz kontrol. KZK için ayrıca literatür karşılaştırması (ör. FDA HITU simulator senaryoları — Gemini sorgusunda).

## 9. Colab çalışma akışı

- `examples/01_quickstart.ipynb`: pip install → JSON config → estimate → run → görselleştir.
- `examples/02_dataset_generation.ipynb`: üretim döngüsünün ince sürücüsü (Drive mount, resume, ETA). Notebook'ta iş mantığı OLMAZ.
- `examples/03_array_designer.ipynb`: ui.ArrayDesigner (Colab'da da çalışır).
- Drive dayanıklılık dersleri (v12.3: mkdir doğrulama, write-probe, yazım anında re-assert) `io.DriveResilientStore`'da.
- **VS Code ↔ Colab**: birincil akış "yerelde geliştir → push → Colab'da pip install + ince notebook". `colab-ssh`/cloudflared tüneli mümkün ama kırılgan/gri — birincil yapılmaz.

## 10. GUI (ui/, notebook-native)

- **PhantomViewer/Editor**: ortho-slice + PyVista 3B; doku ID/özellik düzenleme, MaterialDB editörü.
- **ArrayDesigner**: eleman yerleşimi 3B, faz ayarı; **Rayleigh ile anlık beam önizleme** (GPU'suz, saniyeler). KZK gelince "hızlı fizik önizleme" seçeneği de eklenir.
- **SimulationBuilder**: form → Config JSON → `Simulation.from_config`; **planner tahmini canlı gösterilir** (süre/VRAM/GPU seçimi — COMSOL hissinin görünen yüzü).
- PyVista'nın Colab'da trame/panel backend durumu Gemini sorgusunda.

## 11. Yol haritası

| Faz | İçerik | Çıktı |
|-----|--------|-------|
| M0 | Repo iskeleti: pyproject, src-layout, ruff, pytest, CI, README; isim kararı | pip kurulabilir boş paket |
| M1 | Core API: Grid(2D/3D)/Medium/Source/Sensor/Config + solver registry + numpy backend + Westervelt taşıması (2D önce, 3D hemen ardından) + analytic/ + planner v0 (statik VRAM modeli) | CPU'da 2D/3D mini senaryo; O'Neil/Rayleigh testleri geçer |
| M2 | cupy backend + parite testleri + tam boy 3D Colab koşusu + `linear` çözücü + planner v1 (GPU DB + kalibrasyon) | Notebook ile aynı problemde eşdeğer sonuç + estimate() çalışır |
| M3 | `kzk` çözücü + çözücüler-arası çapraz doğrulama + Study/Report katmanı | Üç çözücü tek API'de; ilk otomatik rapor |
| M4 | io/ + pipelines/ (LHS, resume, atomik yazım) + viz/ süiti | Dataset üretimi kütüphaneden |
| M5 | validation/: analitik süit + k-Wave harness + damgalı raporlar → **v1 sürümü** | İlk benchmark raporu; v1 tag |
| M6 | ui/: PhantomViewer, ArrayDesigner, SimulationBuilder (planner entegre) | GUI'li örnek notebook'lar |
| M7 | Eksenel simetri (DTT/WSWA grid+transform katmanı) + broadband kaynak/sensör | AS + puls desteği |
| M8 | thermal/: Pennes bioheat + CEM43 (Q arayüzü M1'den beri hazır) | Tedavi-planlama örneği |
| M9 | Performans: kernel füzyonu, FFT planları, CUDA Graphs, (gerekirse) çok-GPU | Hız raporu |

## 12. Bilinen teknik gerçekler (yeniden tartışmaya gerek yok — notebook v12 başlığından)

- dx=0.30 mm: 2f0 Nyquist analiziyle kilitli (0.35 reddedildi, 1.88 ppw).
- Mutlak tepe değerlerinde ~%5 dt-bağımlı bias; dataset kendi içinde tutarlı.
- amp/p_max ~0.89–0.92 bandı ve spp=10 discrete ceiling (1.0515) beklenen davranış.
- p_phase kaynak-referanslı DEĞİL; sabit global offset HDF5 attr'da kayıtlı.
- Absorpsiyon frekans-bağımsız üstel; power-law absorpsiyon (fractional Laplacian) k-Wave paritesi için **gelecek özellik** (muhtemelen M7 civarı — AS ile aynı transform altyapısına dokunur).

## 13. Açık sorular / Opus'un netleştireceği konular

- Kütüphane adı (PyPI kontrolü).
- Lisans (öneri: MIT veya LGPL — k-Wave'in kısıtlı lisansına karşı rekabet avantajı).
- KZK'nın difraksiyon adımı: açısal spektrum mu Crank–Nicolson mu (Gemini sonuçlarına göre).
- CuPy'de DTT (DCT/DST) durumu — AS çözücüsünün ön koşulu (cuFFT native DCT vermez; FFT-tabanlı DTT gerekebilir).
- `mtype.txt` (128 MB text) → ilk import'ta sıkıştırılmış cache; raw dosya repo'ya GİRMEZ.
- Gemini Deep Research sonuçları `research/` klasörüne gelince M7/M9 detayları güncellenmeli.

## Ek: Gemini Deep Research sorgusu (v2 — güncellenmiş; bunu ver)

```
I am building an open-source, GPU-accelerated acoustic simulation library in Python
for HIFU/therapeutic ultrasound, designed to run locally and on Google Colab
(CuPy/CUDA on NVIDIA T4/L4/A100/H100). It offers MULTIPLE solver formulations behind
one API — full-wave nonlinear Westervelt k-space PSTD, an optimized linear k-space
solver, and a parabolic KZK solver — in 2D and 3D, with 2D-axisymmetric planned.
It intends to compete with k-Wave and feel like COMSOL (pre-run cost estimation,
study/report system). Research and report on:

1. LANDSCAPE: Actively maintained open-source full-wave ultrasound/acoustics
   simulators as of 2026 — k-Wave and k-wave-python, j-Wave (JAX), Stride, mSOUND,
   FOCUS, fullwave, BabelBrain, OptimUS, the FDA HITU Simulator, and newer entrants.
   For each: numerical method(s), dimensionality (2D/3D/axisymmetric), GPU story,
   license, maintenance activity, API design, known weaknesses. What API-design
   lessons should a new multi-solver competitor learn?

2. VALIDATION STANDARDS: The ITRUSST transcranial benchmark suite (Aubry et al.,
   JASA 2022) and successors — problems, metrics, participating codes. Standard
   analytic references for HIFU solvers (O'Neil focused bowl, Rayleigh integral,
   Fubini/Blackstock nonlinear plane wave) and standard KZK validation cases
   (e.g., FDA HITU Simulator scenarios). What does a credible "we match / beat
   k-Wave" report look like? Also: known issues with k-Wave's precompiled CUDA
   binary (kspaceFirstOrder3D-CUDA) on Google Colab GPUs (we observed catastrophic
   blow-up on a trivial homogeneous-water test on Colab).

3. SOLVER NUMERICS:
   a) KZK on GPU: best-practice operator splitting (diffraction via angular spectrum
      vs Crank-Nicolson; nonlinearity via frequency-domain vs time-domain Burgers),
      handling strongly focused/phased sources (Rayleigh projection to the initial
      plane), published GPU implementations.
   b) Axisymmetric k-space methods (Treeby et al.'s axisymmetric k-Wave): the
      WSWA/WSWS symmetry + discrete trigonometric transform formulation, and the
      practical availability of fast DTTs (DCT/DST) in CuPy/cuFFT or FFT-based
      workarounds on NVIDIA GPUs.
   c) Power-law absorption via fractional Laplacian in k-space solvers: cost and
      implementation notes, since we currently use frequency-independent
      exponential absorption.

4. PERFORMANCE + COST MODELING: State of the art for PSTD solvers on A100/H100
   (CuPy vs JAX vs custom CUDA; cuFFT plan reuse, R2C tricks, kernel fusion, CUDA
   Graphs, TF32/mixed-precision safety, multi-GPU domain decomposition for
   global-FFT methods) with realistic per-technique speedups over a plain CuPy
   implementation. ALSO: approaches for PREDICTING runtime and VRAM of FFT-dominated
   GPU workloads before running (analytic cost models, per-GPU calibration
   microbenchmarks) — prior art in other scientific codes.

5. NOTEBOOK-NATIVE 3D GUI: In 2026, what actually works INSIDE Google Colab for
   interactive 3D voxel visualization/editing widgets — PyVista (trame), K3D-Jupyter,
   ipyvolume, Plotly, niivue, itkwidgets? Compatibility matrix and the most robust
   stack for (a) orthoslice + 3D tissue-map viewing/editing and (b) interactive
   transducer-array design with live Rayleigh-integral beam preview, sharing one
   codebase between local Jupyter/VS Code and Colab.

6. WORKFLOW: Officially supported and community ways (2026) to develop for Colab
   from VS Code — pip install from GitHub, colab-ssh/cloudflared tunnels
   (reliability + ToS status), Colab local runtimes, any new official integrations.
   Recommended GitHub Actions CI for a scientific Python library whose GPU tests
   cannot run on free CI runners.

Prioritize primary sources (docs, papers, repos, issue trackers). Output: comparison
table for (1), metrics checklist for (2), implementation-recipe summaries for (3),
ranked technique list with expected gains for (4), compatibility matrix for (5),
concrete recommended workflow for (6).
```
