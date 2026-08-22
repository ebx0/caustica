# caustica — Mimari Plan (v3)

> **Canlı belge.** Mimarinin ne olduğunu ve NEDEN öyle olduğunu anlatır.
> Yol haritası ve başarı kriterleri: **MILESTONES.md** · oturum kayıtları: **docs/devlog.md** ·
> kütüphane-önce dönüşümünün iş planı: **docs/library_first_plan.md** (İngilizce, uygulama detayı).
>
> **v3.1 (2026-08-22).** M10h/M10i/M10k kapandı: kütüphane UWCEM'siz, ortam kapıları içeride.
> Yeni kararlar K15–K17 (tam plugin mimarisi, UWCEM tek dosyada EN SON — `docs/uwcem.md`,
> operatör modeli). Yalınlaştırma turu #1: kök veri/notebook dosyaları silindi.
>
> **v2 → v3 (2026-08-21).** Kütüphane-önce dönüşüm kararlarıyla yeniden yazıldı. Değişenler:
> notebook-native GUI kararı **iptal**; UWCEM fantom katmanı kütüphaneden **çıkarıldı**; public
> API **üç katmana** ayrıldı; Colab'da Drive kullanımı **kaldırıldı**; M0–M9 yol haritası tablosu
> MILESTONES.md'ye devredildi; Gemini araştırma sorgusu eki kaldırıldı (sonuçlar `research/`).

---

## 0. Kesinleşmiş kararlar

Bunlar yeniden tartışılmaz. Değişmesi gerekiyorsa önce burası güncellenir, sonra kod.

### 0.1 Fizik ve API (2026-08-10, tur 1–2 — hâlâ geçerli)

1. **Önce genel amaçlı API.** Grid/Medium/Source/Solver soyutlaması birinci sınıf; dataset üretimi
   bu API'nin üstünde bir *uygulama*.
2. **Eski `dx300_t128` dataseti ile bire-bir sayısal uyum ŞART DEĞİL** — fizik doğruluğu yeterli.
   Kütüphane kendi golden-regression testlerini tutar. (Nedeni artık somut: KEŞİF 1, §15.)
3. **Üç çözücü de v1 hedefinde**: `westervelt` (full-wave PSTD), `linear` (β=0 optimize yol),
   `kzk` (parabolik, z-marş — M9).
4. **3D + 2D birlikte**; çekirdek boyut-agnostik, 2D ucuz CI ortamı. Eksenel simetri (AS) sonraki
   faz (M15, DTT/WSWA transform katmanı gerektirir).
5. **v1 = CW + steady-state.** Source/Sensor API'si keyfi zaman sinyaline izin verecek şekilde
   TASARLANIR; broadband implementasyonu M17.
6. **Termal modül kapsamda, geç fazda** (M18); mimari ilk günden ısı-kaynağı (Q) arayüzünü tanır.
7. **Lisans MIT** (k-Wave'in kısıtlı lisansına karşı rekabet avantajı).
8. **İsim: `caustica`** (2026-08-21, M10e; "hifusim" çalışma adıydı, PyPI/GitHub çakışma
   kontrolüyle değiştirildi).

### 0.2 Kütüphane-önce kararlar (2026-08-21)

| # | Karar | Gerekçe / sonuç |
|---|---|---|
| K1 | **Gerçek public kütüphane** hedefi | Dışarıdan gelen bir araştırmacı `pip install` deyip koşabilmeli; hiçbir şey senin Drive'ına, yollarına veya checkout'una bağlı olamaz |
| K2 | Kurulum: **`git+https` şimdi → PyPI v0.1'de** | Sürüm makinesi henüz kurulmaz; ama wheel doğruluğu temiz ortamda test edilir |
| K3 | **Üretilmiş fantom dataseti hiç dağıtılmaz** | 4.5 GB yerelde kalır; kütüphane hiçbir anatomik veri taşımaz |
| K4 | Public API **üç katman**: facade → nesneler → job JSON | Üçü de public; facade İKİNCİ bir kod yolu DEĞİL (§4) |
| K5 | GPU yoksa: **`auto` + gürültülü uyarı + büyük işte red** | Planner tahmini > **5 dk** ise reddet; `--allow-slow-cpu` kaçış kapısı |
| K6 | **cupy asla otomatik kurulmaz** | Aksiyonlu hata: Colab'da "runtime GPU değil", yereldeyse `pip install cupy-cuda12x` |
| K7 | **Kütüphane tamamen UWCEM'siz** | `phantom_dataset`, `stored_setup`, `_require_uwcem` silinir (§9) |
| K8 | Yerine **genel `medium_volume` kind'ı** | caustica'nın sahibi olduğu format; her fantom kaynağı aynı kapıdan girer |
| K9 | UWCEM işleri **ayrı repo** (`uwcem-phantom`) | Lisans/şart riski kütüphaneden çıkar; repo caustica'ya bağımlıdır, tersi değil |
| K10 | Literatür akustik doku değerleri **`caustica.materials`e** | UWCEM media-numarası eşlemesi ayrı repoda kalır |
| K11 | **Koşu-içi ilerleme + önizleme varsayılan AÇIK** | Periyot başına callback (adım başına ASLA); 8 periyotta bir odak kesiti |
| K12 | **Colab çıktısı `/content`** — kütüphane Drive mount ETMEZ | Kalıcılık isteyen kullanıcı Drive'ı kendi mount edip `--out` verir |
| K13 | **GUI ayrı repo** (`caustica-gui`), teknoloji seçilmedi | Şimdi yalnızca sözleşmeler dondurulur (§11) |
| K14 | v1.0'a kadar **API stabilite garantisi yok** | Kırıcı değişiklik serbest; ama `__all__` dürüst kalır ve kırılma devlog'a yazılır |
| K15 | **Tam plugin mimarisi** (2026-08-22) — beş eksen entry-point'li: solver ✅, medium kind, array kind, backend, report renderer | Erken-soyutlama riski söylendi, kullanıcı teyit etti. Panzehir: çekirdek kendi plugin API'sinin birinci müşterisidir (özel yol yok). M10m + M10n |
| K16 | **UWCEM kalanları EN SON; her şey TEK dosyada** (2026-08-22) | Ayrışım (M10k) zaten kapandı; kalan push/bakım işleri `docs/uwcem.md`'de, yol haritasının sonunda |
| K17 | **Yönetim modeli** (2026-08-22): operatör = Fable 5, kod = Opus 5 alt-ajanları; PM sistemi = MILESTONES.md | Önemli kararlar kullanıcıya sorulur; gerisi operatörde. Milestone kanıt disiplini değişmez |
| K18 | **Çok-motor çatı kimliği** (2026-08-22): "bir API, N motor" — native aile (westervelt/linear/AS, ileride KZK) + sarılmış motorlar (kwave ✅, jwave M25, stride M26) aynı harness'te çapraz doğrulanır | Landscape: k-wave-python v0.6 saf-CuPy çözücü çıkardı, "saf Python" farkı kapandı; savunulabilir konum SÖZLEŞME (job→planner→damgalı koşu) + çok-motor. Rakiple fizik yarışı değil, rakibi motor olarak sarma |
| K19 | **v0.1 kapısı = ITRUSST PH1 dokuzunun TÜMÜ, akustik-yalnız** + JOSS + PyPI (2026-08-22) | Self-serve ekosistem (Zenodo/GitHub donmuş, bekçisiz); 500 kHz LİNEER olduğundan CW motoru yeter, elastik gerekmez (katılımcı normu). Koridorlar: fokal basınç <%10, konum <1 mm |
| K20 | **HIFU-önce, doğruluk-önce** (2026-08-22): termal öne (M18), AS KZK'nın önüne (M15), kafatası/elastik v0.2 (M22), broadband ikinci faz (M17) | "En doğru çözücü" iddiası kazanılamaz (hidrofon karşısında alan %20-31 hatada) — iddia: doğrulanabilirlik + tekrarlanabilirlik + çok-motor mutabakat |
| K21 | **Görüntü köprüsü entegrasyon-önce** (2026-08-22): `caustica[imaging]` extra'sı hazır segmentasyon/pseudo-CT çıktılarını alır + adlandırılmış literatür haritalamaları; kendi AR-GE'si yok | Üretim araçları (k-Plan vs BabelBrain) tam bu haritalamada güvenlik-kritik ayrışıyor — çoğul-strateji KARŞILAŞTIRMASI ucuz fark alanı |

### 0.3 Vazgeçilenler (kayıt — yeniden gündeme gelirse buraya bakılır)

| Vazgeçilen | Ne zaman / neden |
|---|---|
| **Notebook-native GUI** (ipywidgets + PyVista/Plotly, `ui/` paketi) | 2026-08-10'da kapsam dışı, 2026-08-21'de tamamen iptal: GUI ayrı repo, teknoloji seçilmedi (K13). `ui/` hiç yazılmadı, planlanan hâli de silindi |
| **`hs.Medium.from_phantom("mtype.txt", ...)`** | UWCEM'e özgü import kütüphaneden çıkıyor (K7); yerine `medium_volume` |
| **`phantom_dataset` medium kind + `stored_setup` job kind** | K7/K9 — `caustica-job/1` bu iki kind'ı kaybeder (kırıcı şema değişikliği, bilinçli) |
| **Fantom datasetinin Drive'dan birincil beslenmesi** (eski M10f) | K3/K12 — kütüphanede anatomik veri yolu kalmıyor, Drive'a hiç dokunulmuyor |
| **Colab'da Drive mount + Drive'a atomik yazma** | K12 — riski (R10) kabul edildi: kullanıcı isterse kendi mount eder |
| **PLAN.md'deki M0–M9 yol haritası tablosu** | MILESTONES.md M0–M24 ile çakışıyordu; tek kaynak MILESTONES.md |
| **Gemini Deep Research sorgusu eki** | Sorgu koşuldu; sonuçlar `research/` altında, milestone'lara "VERIFY" notlarıyla işlendi |
| **C/C++ backend** | §12 — kazanç %10–30, bakım maliyeti yüksek; backend arayüzü açık bırakıldı |
| **Ayrı M12/M13/M14/M10g milestone'ları** | 2026-08-22 yeniden plan: M12→M11'e, M13+M14+M10g→M29'a birleşti |
| **"Saf Python, binary'siz" ana farkı** | k-wave-python v0.6 (2026-03) aynısını yaptı; kimlik K18'e taşındı (sözleşmeli çok-motor çatı). Teknik özellik duruyor, PAZARLAMA iddiası olmaktan çıktı |

---

## 1. Vizyon

"Bir API, N motor" (K18): planner-kapılı, damgalı, tekrarlanabilir koşu sözleşmesi altında
native çözücü ailesi + sarılmış motorlar (kwave, jwave, stride) — fokuslu ultrason için
**doğrulanabilirliği kanıtlanmış** çok-motor çatı. COMSOL hissi korunur; rakiple fizik
yarışı değil, rakibi de motor olarak sarma. Ve **önce kütüphane**.
Üç katman:

- **Katman A — Simülasyon çekirdeği**: çok-formülasyonlu (Westervelt / lineer / KZK, ileride AS +
  broadband), çok-boyutlu (1D/2D/3D), backend'li (numpy / CuPy-CUDA), her parametresi config'ten
  ayarlanabilir çözücü ailesi.
- **Katman B — Mühendislik kabuğu**: koşu ÖNCESİ kaynak tahmini (planner), job şeması + runner,
  rapor/önizleme, doğrulama ve benchmark raporları.
- **Katman C — Uygulamalar**: dataset üretimi (M13), Colab sürücü notebook'ları, ayrı repolardaki
  fantom araçları ve GUI.

**Korunan değer** (notebook v12.3'ten taşındı, yeniden yazılmadı): k-space PSTD + kappa sinc
düzeltmesi, exact-period dt (sızıntısız tek-bin DFT), Westervelt nonlineerliği, üstel absorpsiyon,
Gaussian sponge PML, adaptif settling, fazor + p_max çıkarımı, atomik HDF5 yazımı, resume mantığı,
float16 dinamik kuantizasyon, O'Neil/Rayleigh analitik doğrulaması.

**Yapısal fark (rakiplere karşı konum):** derlenmiş ikili yok. Çekirdek saf Python + numpy/scipy;
GPU yolu cupy. k-Wave'in Colab'daki kırılma noktası (yanlış CUDA/glibc'ye derlenmiş binary) burada
yapısal olarak imkânsız; j-Wave/JAX'ın Windows zayıflığı da yok.

---

## 2. Katmanlar ve bağımlılık yönü

```
L5  caustica-gui  (ayrı repo)     yalnızca L3 çıktılarıyla konuşur
L4  caustica.colab                ortam kontrolü + /content altında koşu
L3  job JSON + CLI                caustica-job/1, runner, status.json, preview, çıkış kodları
L2  facade                        caustica.simulate(...)
L1  nesneler                      Grid / Medium / Source / Solver / CWRunSpec
L0  çekirdek                      Backend, spectral, PML, materials, io, medium_volume

    uwcem-phantom (ayrı repo) ──bağımlı──▶ caustica
    üretir: medium_volume dosyaları + explicit job JSON
```

**Kurallar (öncelik sırasıyla):**

1. **Oklar yalnızca aşağı bakar.** `caustica.*` asla `apps`, `uwcem_phantoms` veya bir GUI paketini
   import etmez. AST testiyle zorlanır. *(İhlal M10k/W0c ile kapandı — `tests/test_import_direction.py` yeşil, 2026-08-22.)*
2. **L2 ikinci bir kod yolu değildir.** Facade girdisini job'a çevirip AYNI `build_job`'dan geçer.
   Facade'ın ifade edebildiği her şey job dosyasıyla da ifade edilebilmeli, tersi de.
3. **L4 fikirli olabilir, L0–L3 olamaz.** Colab varsayımları (`/content`, runtime restart) yalnızca
   `caustica.colab` içinde yaşar.
4. **L5'e özel API yoktur.** GUI bir şeye ihtiyaç duyuyorsa önce belgelenmiş bir L3 sözleşmesi olur.
5. **Hacim ortamları için tek kapı.** Her fantom kaynağı `medium_volume`'den girer. Bir kaynak
   kütüphanede özel durum gerektiriyorsa, özel durum yanlıştır.
6. **Her eksen genişletilebilir (K15).** Solver, medium kind, array kind, backend ve report
   renderer registry + entry-point seam'i taşır; üçüncü taraf paket çekirdeğe dokunmadan uzanır.
   Çekirdek implementasyonlar da AYNI kapıdan kaydolur — seam'in çalıştığının sürekli kanıtı budur.

---

## 3. Paket yapısı

`✅` mevcut · `🔄` bu dönüşümde geliyor · `⬜` planlı (milestone numarasıyla)

```
caustica/                        (repo kökü)
├─ pyproject.toml                extras: [gpu]=cupy, [kwave], [report], [dev]
├─ src/caustica/
│  ├─ core/          ✅ backend (numpy/cupy dispatch, dtype-koruyan fft), grid, pml
│  ├─ config/        ✅ models.py (pydantic taban) + job.py (caustica-job/1 şeması)
│  ├─ materials.py   ✅ Material / MaterialDB / water() / breast_default()  (+🔄 literatür doku tablosu, K10)
│  ├─ medium.py      ✅ Medium: homogeneous / from_id_map; float32 özellik hacimleri
│  ├─ geometry/      ✅ CSG şekiller, sahne, hacim import, dx yeniden örnekleme (M6b)
│  ├─ arrays/        ✅ spiral/bowl/annular transducer üretimi, DAS fazlama, faz haritaları
│  ├─ sources.py     ✅ CWSource + plane/bowl builder'ları (kütle-kaynak normalizasyonu 2c·dt/dx)
│  ├─ spectral.py    ✅ tek fazor implementasyonu (çözücü + adaptör + testler aynı kodu kullanır)
│  ├─ solvers/       ✅ base (yetenek deklarasyonu), registry (entry-point plugin),
│  │                    kspace/{engine,linear,westervelt,operators}, kwave_adapter
│  ├─ planner/       ✅ VRAM envanteri + süre modeli + gpu_db.json + cihaz kalibrasyonu
│  ├─ io/            ✅ atomic, quantize (float16 kontratı), store (caustica-result/1), checkpoint,
│  │                    medium_volume (K8 — okuyucu+yazıcı, bit-aynılık kanıtlı; M10k/W0a)
│  ├─ report/        ✅ metrics (tek kaynak), preview (≤10 MB), figures, html, run_report
│  ├─ runner.py      ✅ tek job koşumu: plan-önce, ayrık çıkış kodları, kalp atışı, damga
│  ├─ __main__.py    ✅ validate / run / report / example + `caustica` konsol komutu (M10h)
│  ├─ facade.py      🔄 simulate(...) — L2 (K4)
│  ├─ env.py         ✅ env_report(), require_gpu(), CausticaWarning, CPU kapısı (M10i)
│  ├─ progress.py    🔄 ilerleme payload'u + tqdm/metin sunumu (K11)
│  ├─ examples/      ✅ water_bowl_mini — dış veri gerektirmeyen paketli örnek (M10h)
│  ├─ colab.py       ⬜ M10f — ortam kontrolü + /content altında koşu (K12)
│  ├─ analytic/      ✅ rayleigh, oneill, planewave (Fubini)
│  ├─ study/         ⬜ M11 — Study + sweep + damgalı rapor
│  ├─ validation/    ⬜ M11 — analitik süit tek komutla rapor
│  ├─ pipelines/     ⬜ M13 — DatasetGenerator (LHS, resume, ETA)
│  └─ thermal/       ⬜ M18 — Pennes + CEM43
├─ tests/            ✅ 279 test (M10k sonrası; taşınan süit ../uwcem-phantom'da 165+ yeşil)
├─ benchmarks/       ✅ damgalı doğrulama raporları
├─ notebooks/        ⬜ M10f — colab_run.ipynb (değişmeyen dosya)
├─ apps/             ✅ focus_study (kütüphane tüketicisi; wheel'e girmez)
└─ docs/             ✅ devlog.md, library_first_plan.md   ⬜ gui_contract.md
```

**`sensors/` neden yok:** v1'de kayıt, çözücü `run()` argümanları (`record_region`, `harmonics`,
`reference_point`) ve `SolverResult` ile yapılıyor. Ayrı bir sensör modülü ancak broadband zaman
alanı (M17, `sensors.TimeSeries`) ve termal (M18, `HeatingSource`) gelince gerekecek — o zaman
eklenir, şimdi değil.

**`viz/` neden yok:** `report/` bu işi görüyor (figürler + HTML + metrikler tek kaynaktan).

---

## 4. Public API — üç katman (K4)

Üçü de public, üçü de aynı `build_job` çekirdeğinden geçer. **Facade ikinci bir kod yolu değildir.**

**L2 — facade (notebook için tek satır):** — M10j'de indi (`caustica/facade.py`). Girdi listesi
KAPALIDIR: çıplak `Grid`/`Medium`/`CWSource` KABUL EDİLMEZ — voxelize edilmiş bir kaynak job'a
geri çevrilemez, kabul etmek ikinci bir kurulum yolu demek olurdu. Onlar için L1 kullanılır.

```python
import caustica

res = caustica.simulate(
    setup="job.json",       # job yolu | dict | ExplicitJobConfig | BuiltJob
    solver="westervelt",
    backend="auto",         # GPU varsa cupy; yoksa uyarı + 5 dk üstü işte red (K5)
    harmonics=(1, 2),
    out=None,               # None = bellekte; yol verilirse tam runner çıktı klasörü
    progress="auto",        # ilerleme + odak önizlemesi, varsayılan açık (K11)
)
res.metrics       # caustica.report.metrics — focus_study ile aynı tanımlar
res.preview()     # ≤10 MB önizleme paketi, bellekte
res.save(path)    # caustica-result/1
```

**L1 — nesneler (bugün çalışan yol):**

```python
import caustica as hs
import caustica.solvers as solvers
from caustica.arrays import archimedean_spiral
from caustica.materials import water
from caustica.solvers import CWRunSpec

grid   = hs.Grid(shape=(96, 96, 96), dx=0.5e-3, pml=hs.PMLSpec(thickness=5e-3))
medium = hs.Medium.homogeneous(grid.shape, water())
array  = archimedean_spiral(n_elements=32, d_outer=0.030, d_inner=0.010, roc=0.030)
src    = array.voxelize(grid, apex_vox=(48, 48, 12), f0=1.0e6, amplitude=1e5).source

res = solvers.get("westervelt")().run(grid, medium, src, CWRunSpec(), harmonics=(1, 2))
```

**Tasarım kuralı — basit varsayılan, detaylı erişim.** İki uç da birinci sınıf; biri diğerine
feda edilmez. Uygulamada bu dört madde demek:

1. **Minimal job kısa olmalı.** Su içinde bir çanak ~15 satır JSON ya da tek `simulate(...)`
   çağrısıdır. Her zorunlu alan gerçekten zorunlu olduğu için oradadır.
2. **Her düğme erişilebilir kalır.** Varsayılanla gizlenen hiçbir fizik yoktur; `CWRunSpec`'ten
   PML profiline, harmonik listesinden kayıt bölgesine kadar her şey job'dan ve Python'dan
   ayarlanabilir. "Basit" demek "kısıtlı" demek DEĞİLDİR.
3. **Varsayılanlar belgelenir ve gerekçelendirilir.** `docs/job_reference.md` her varsayılanın
   değerini ve NEDEN o değer olduğunu yazar. Sessiz sihir yok.
4. **Hatalar öğretir.** Yanlış kombinasyon sessizce koşmaz; mesaj neyin yanlış olduğunu ve nasıl
   düzeltileceğini söyler (`SolverCapabilityError`, planner OOM önerileri, `require_gpu`
   mesajları bu kalıbın mevcut örnekleridir). Yeni kod bu çıtayı düşürmez.

**L3 — job JSON + CLI (kuyruğun ve GUI'nin sözleşmesi):**

```bash
caustica validate job.json     # çözmeden: şema, dosyalar, geometri, PML, odak, ppw
caustica run job.json --out runs/x   # plan-önce; çıkış kodları 0/2/3/4/5 AYRIK
caustica report runs/x               # REPORT.md + index.html + figürler
```

Çıktı klasörü sabittir ve resume buna bağlıdır: `job.json`, `plan.json/.txt`, `status.json`
(kalp atışı), `checkpoint.npz`, `result.h5`, `preview.npz`, `metrics.json`, `run_meta.json`.

---

## 5. Çok-çözücü mimarisi

**Registry + yetenek deklarasyonu.** Her çözücü `SolverBase`'i uygular ve yeteneklerini bildirir
(`SolverCaps`: ndim, nonlinear, absorption, drive, backends). Kurulum anında doğrulanır — yanlış
kombinasyon sessizce koşmaz, açıklamalı hata verir (`SolverCapabilityError`). Fizik sessizce
düşürülmez.

| çözücü | fizik | durum |
|---|---|---|
| `linear` | lineer k-space PSTD, 1/2/3-D | ✅ doğrulandı (M4) |
| `westervelt` | nonlineer Westervelt, çok-harmonikli yakalama | ✅ doğrulandı (M5) |
| `kwave` | k-Wave `kspaceFirstOrder` (CPU/OMP binary), harici | ✅ sarıldı + çapraz doğrulandı (M4b) |
| `kzk` | parabolik KZK, z-marş | ⬜ M9 |
| AS (eksenel simetri) | DTT/WSWA-WSWS | ⬜ M15 |

- **Ortak altyapı paylaşılır**: Grid/Medium/Source, backend, io, report. Çözücüler yalnızca
  zaman/uzay ilerletme şemasını getirir.
- **Plugin**: registry entry-point okur → üçüncü taraf paketi kendi çözücüsünü sunabilir.
- **Çözücüler arası çapraz doğrulama**: aynı lineer senaryoda `linear` ≈ `westervelt(β=0)` ≈
  `kwave` ≈ Rayleigh. `westervelt(β=0) ≡ linear` kapısı < 1e-6.
- **kwave adaptörü kwarg sözleşmesi**: bilinmeyen `run()` argümanlarını REDDEDER. Native'e özgü
  argümanlar (`backend=`, `checkpoint=`, gelecekte `progress=`) ona ASLA geçirilmez.

---

## 6. Planner — koşu öncesi kaynak tahmini

`planner.estimate(...)` koşmadan şunları verir:

- **VRAM dökümü**: kalıcı tamponların (p, u×nd, k-uzayı geçicileri, pmax, fazorlar, 4 özellik
  hacmi, sponge'lar) bayt-bayt tablosu + FFT çalışma alanı + emniyet marjı. Envanter motora
  birebir bağlıdır; motora kalıcı tampon ekleyen `test_memory_inventory_matches_hand_count`
  testini **bilerek** kırar.
- **Süre tahmini**: adım maliyeti ≈ `a·P·log2(P) + b·P`; adım sayısı settling modelinden
  (TOF + min_settle + record) × spp.
- **Kaynak etiketi**: `db` (gpu_db.json, 7 cihaz) < `calibrated` (~20 adım fit,
  `~/.caustica/calibration.json`) < `measured` (bu koşuda ölçüldü). Rapor hangisi olduğunu yazar —
  CPU kapısının (K5) güvenilirliği buna dayanır.
- **Karşılaştırma**: `planner.compare(..., gpus=[...])` → hangi GPU'da ne kadar, sığar mı.
- OOM durumunda öneri üretir (dx büyüt / kayıt bölgesini küçült / linear'a geç) ve runner
  **koşmadan** reddeder (çıkış kodu 3).

---

## 7. Rapor, Study ve doğrulama

- ✅ **`caustica.report`** — metriklerin TEK kaynağı (`apps/focus_study` buraya delege eder),
  her koşuda yazılan ≤10 MB önizleme paketi, `caustica report` ile tam figür seti veya
  yalnızca önizlemeden hızlı görünüm. matplotlib/h5py **lazy** (PEP 562) — ikisi de kurulu
  olmayan bir makinede önizleme yazımı çalışmaya devam eder.
- ⬜ **`study/` (M11)**: config + koşu(lar) + sonuç + figürler tek pakette; `report()` →
  Markdown/JSON; ortam/GPU/git-hash damgası; `Study.sweep(...)`.
- ⬜ **`validation/` (M11)**: analitik süit tek komutla damgalı rapor üretir; planner tahmini vs
  gerçekleşen tablosu raporun parçası.
- Her koşunun `job.json` + ortam + git damgası sonuçların yanına yazılır (tekrarlanabilirlik).

---

## 8. Ortam ve güvenlik politikası

Ortak tema: **kullanıcı sessizce yanmasın.** Yanlış backend, yanlış çözünürlük, görünmeyen uyarı,
tek çekirdekte sürünen CPU — hepsi "çalışıyor gibi görünüp yanlış ya da yavaş sonuç veren"
sınıfından ve bir kütüphanenin en pahalı hata türü budur.

- **Backend dispatch**: `get_backend("auto"|"numpy"|"cupy")`. cupy importu ve CUDA yoklaması
  **lazy** ve süreç boyunca cache'li; GPU'suz makinede `import caustica` cupy'ye hiç dokunmaz.
  `backend.fft` dtype korur (numpy.fft float32'yi complex128'e terfi ettirir — fp32 paritesi için
  kritik). Çözücüler backend'i **parametre olarak** alır; global durum yoktur.
- **`env_report()`**: sürümler, CUDA driver/runtime, GPU adı, boş/toplam VRAM, caustica sürümü,
  git commit. Runner damgası ve notebook AYNI fonksiyonu çağırır — ikisi çelişemez. Asla exception
  atmaz.
- **`require_gpu()`**: cupy backend'ini döndürür ya da **aksiyonlu** hata verir. Colab'da mesaj
  "Runtime → Change runtime type → GPU" der (oradaki gerçek arıza neredeyse her zaman budur ve
  hiçbir `pip install` onu çözmez); yereldeyse `pip install cupy-cuda12x` der. **Kütüphane asla
  pip çağırmaz.**
- **CPU kapısı**: backend numpy'a düştüyse ve planner beklenen süreyi **5 dk**'nın üstünde
  görüyorsa koşu reddedilir (çıkış kodu 2 — yeni kod EKLENMEZ, kod kümesi kuyruğun API'sidir).
  Mesaj tahmini ve `est.source` etiketini alıntılar, iki kaçışı adıyla söyler: GPU backend'i ya da
  `--allow-slow-cpu`. Eşiğin altında koşar ama **tek** bir uyarı basar.
- **CPU çok çekirdekli FFT**: `scipy.fft` çağrıları `workers=-1` ile koşar. **Sıra bağlayıcıdır**:
  önce çok çekirdek, sonra planner CPU kalibrasyonu, EN SON 5 dk eşiği — tek çekirdeğe göre
  kalibre edilmiş bir eşik hiçbir şey ölçmez.
- **Görünürlük**: kütüphane `import` anında hiçbir logging handler'ı kurmaz (doğru kütüphane
  davranışı), ama kritik olaylar — backend'in numpy'a düşmesi, düşük ppw — `warnings.warn` ile
  yayılır: notebook'ta da CI'da da görünür, gerekirse filtrelenebilir. `caustica run` ve facade
  girişte loglamayı açar.
- **Düşük ppw**: uyarıdır, engel DEĞİLDİR (üretim ayarı 2f0'da 1.88 ppw ve bu bilinçli bir seçim;
  sert eşik kendi datasetimizi bloke ederdi). Ama uyarı plan çıktısında, `status.json`'da,
  `run_meta.json`'da ve raporun başında tekrar eder — görmezden gelinebilir, kaydırılıp geçilemez.
- **İlerleme kancası**: periyot sınırında (adım başına ASLA — her adımda device→host senkronu GPU
  verimini yok eder) tek bir payload yayılır:
  `{period, periods_expected, step, steps_expected, peak, converge_delta, elapsed_s, eta_s, stage}`.
  Bu DOKUZ anahtar serileştirilebilir sözleşmedir. Payload ayrıca **onuncu** bir anahtar taşır:
  `snapshot` — sıfır argümanlı, odaktan geçen 2-B kesiti döndüren bir CALLABLE (M10j). Tembel
  olması bilinçli: kopya yalnız tüketici isterse yapılır, yani 8 periyotta bir önizleme çizen
  tüketici o periyotlarda TEK device→host kopya öder, diğerlerinde sıfır. **Serileştiren tüketici
  (`status.json`, GUI soketi) bu anahtarı DÜŞÜRMEK ZORUNDADIR** — `json.dumps` onu kaldıramaz.
  `status.json` kalp atışı bu payload'un TÜKETİCİSİDİR, ikinci bir implementasyon değil. Sunum
  çözücünün dışındadır (tqdm varsa ve izleyen bir şey varsa tqdm, yoksa düz satır; tqdm bir
  runtime bağımlılığı DEĞİLDİR ve `extra` olarak da tanımlı değildir). Callback hatası koşuyu
  düşürmez.

---

## 9. Veri ve fantom stratejisi

**Kural: kütüphane hiçbir anatomik veri ve hiçbir kaynağa-özgü içe aktarıcı taşımaz.**

- **`medium_volume` (K8)** — caustica'nın sahibi olduğu genel format. Ya etiket haritası + malzeme
  veritabanı, ya **voxel-başına özellik hacimleri** (c, ρ, α, β) taşır; **grid'i dosyadan alır**
  (shape + dx dosyanındır, explicit `grid` bölümü reddedilir — bu kural bir job'ın dataset'in
  yeniden örneklenmiş bir hayaletini sessizce koşmasını engeller). Mevcut 4.5 GB dosyaları
  **olduğu gibi** okur; format değişikliği yeniden üretim gerektiriyorsa yanlıştır.
- **`uwcem-phantom` (K9/K16)** — ayrı repo, **taşındı ve çalışıyor** (2026-08-22; yerel git,
  push yok). caustica'ya **bağımlıdır**; ürettiği şey `medium_volume` dosyaları ve **explicit
  job JSON**'dur (`setup_to_job`). Lisans/atıf yükümlülüğü orada; caustica'nın MIT lisansı temiz.
  Güncel durum + kalan işler: **docs/uwcem.md** (tek dosya, EN SON).
- **Veri yerelde kalır (K3)**: üretilmiş 4.5 GB ve dokuz setup JSON'u senin diskinde, git'in
  dışında. caustica repo'sundaki `data/` boşalır. Dışarıdan gelen kullanıcı UWCEM dosyalarını
  **kendi** indirir (repo'nun katalog + checksum yolu) ve kendi fantomunu üretir.
- **Format hem okunur hem YAZILIR**: `write_medium_volume(...)` public. Kendi CT/NIfTI/numpy
  verisinden gelen bir kullanıcı kendi ortam dosyasını üretebilir; `uwcem-phantom` repo'su da aynı
  fonksiyonu çağırır. Yalnız okuyucu olsaydı "kendi hacmini getir" vaadi çalışmazdı — tek yazıcı
  kullanıcıda olmayan bir repoda kalırdı.
- **Genel geometri yolu kütüphanede kalır**: `geometry/` (CSG, sahne, hacim import, dx yeniden
  örnekleme) hiçbir kaynağa özgü değildir ve `volume_import` medium kind'ı olarak zaten public.
  `load_labels_txt` genel kaldı (kaynağa özgü kısım `mapping` callable'ında); `load_breast_phantom`
  UWCEM'e özgü olduğu için yeni repoya taşındı (`legacy_import.py`).
- **Kendi kurulumunu getir**: job şeması keyfi eleman geometrisini (`elements` kind'ı) tanır,
  `caustica schema` şemayı JSON Schema olarak basar, `docs/job_reference.md` ve
  `docs/conventions.md` dışarıdan gelenin kaynak koda inmeden job yazmasını sağlar. Bu bir
  "sonra yaparız" kalemi değil, kütüphane olmanın kabul kriteridir.

---

## 10. Colab çalışma akışı

- **Kurulum**: `pip install git+https://github.com/ebx0/caustica` (token gerekmez, repo public).
  Colab'da cupy zaten kurulu gelir; `[gpu]` extra'sını Colab'da kurmaya gerek yok.
- **`caustica.colab` (M10f)**: ortam kontrolü (GPU var mı, cupy import ediliyor mu, boş VRAM
  planner tahminine yetiyor mu — **hiçbir şey hazırlanmadan önce** reddeder), sonra `run_job_file`,
  çıktı `/content` altında.
- **Drive yok (K12)**: kütüphane `drive.mount()` çağırmaz, Drive yollarını bilmez, Drive'a özgü
  yeniden deneme mantığı taşımaz. Kalıcılık isteyen kullanıcı Drive'ı kendi mount eder ve `--out`
  ile oraya yazdırır — runner bunu zaten destekler. **Kabul edilen risk**: oturum çökerse
  `/content` gider; checkpoint oturum içi restart'ı kurtarır, VM teardown'ı kurtarmaz.
- **Notebook sözleşmesi**: 4–5 hücre, tek düzenlenen satır CONFIG. Mantık notebook'ta DEĞİL
  repoda yaşar; güncelleme `pip install -U` ile gelir, `.ipynb` diff'i SIFIR.
- **Kuyruk (M10g)**: `jobs/pending → running → done|failed`, atomik rename ile claim, ölü oturum
  devri. Protokol **paylaşılan bir klasör yolu** alır; Drive onun bir örneğidir ve kütüphane bunu
  bilmez. Runner çıkış kodları kuyruğun API'sidir; değiştirilmez.
- **VS Code ↔ Colab**: birincil akış "yerelde geliştir → push → Colab'da pip install + ince
  notebook". Tünel çözümleri (colab-ssh/cloudflared) kırılgan; birincil yapılmaz.

---

## 11. GUI — kapsam dışı, sözleşme dondurulur

GUI bu planın kapsamında **değil** (K13). Teknoloji seçilmedi, bağımlılık eklenmedi, `gui` extra'sı
yok. Yapılan tek şey **sözleşmeleri dondurmak** — GUI ne zaman gelirse gelsin bunların üstüne
oturur:

`caustica-job/1` şeması + `validate` + `caustica schema` (JSON Schema → otomatik form) · runner
çıkış kodları (0/2/3/4/5) · `--dry-run` → `plan.json` (koşmadan VRAM/süre/öneri) · `status.json`
alanları · `caustica-preview/1` önizleme paketi · `caustica-result/1` sonuç formatı ·
`env_report()` · ilerleme payload'u (§8) · **`cancel` dosyası** (periyot sınırında temiz durur,
checkpoint bırakır, çıkış 5) · **`error.json`** (`stage`, `exit_code`, `error_class`, `message`,
`advice[]`).

Son iki kalem 2026-08-21'de eksik olduğu tespit edilip M10l'e eklendi: iptal sinyali hiç yoktu ve
koşu başlamadan oluşan hatalar `status.json` üretmiyordu — GUI'ye stderr ayrıştırmak kalıyordu.

Girdi tarafında sözleşme **tek dosyadır**: `job.json` koşuyu tümüyle tarif eder (voksel ortam
kullanılıyorsa artı yolla referans verilen hacim dosyası — 4.5 GB JSON'a gömülmez). Çıktı tarafında
sözleşme tanımlı bir **klasördür**; tek dosya olmaması bilinçlidir, resume ve canlı ilerleme tek
dosyayla çalışmaz. GUI'nin göstermek için ihtiyaç duyduğu şey yine de tek pakettedir:
`preview.npz` + `metrics.json`, ≤10 MB.

Listelenmeyen hiçbir şey sözleşme değildir. GUI bir şeye ihtiyaç duyarsa önce buraya eklenir.
`docs/gui_contract.md` bunu yazıya döker; AST testi de import yönünü zorlar.

---

## 12. Backend stratejisi ve C/C++ kararı

- **numpy backend**: CPU referansı; 2D mini-grid'lerle CI'da her şey koşar.
- **cupy backend**: CUDA; ElementwiseKernel'lar zaten derlenmiş CUDA, çözücü cuFFT-domine.
- **C/C++ YOK**: beklenen kazanç %10–30, bakım maliyeti yüksek. Backend arayüzü ileride native bir
  backend'e açık bırakıldı.
- İleriki hız işleri (M19): RawKernel füzyonu (PML+absorpsiyon+nonlineerlik tek kernel), cuFFT plan
  cache doğrulaması, CUDA Graphs (CuPy'de graph capture olgunluğu VERIFY), TF32 güvenlik çalışması
  (default OFF, ayrı raporlanır). Çok-GPU (M20) spektral yöntemde zor — fizibilite raporu da
  geçerli çıktıdır.

---

## 13. Test stratejisi

**Kural: testler milestone kriterlerini kodlar.** Bir milestone çalışan kodla değil, kriter
gerilerse KIRILACAK bir testle kapanır.

1. **Birim** (hızlı, CPU): geometri dönüşümleri, array üretimi, faz haritası yerleşimi, atomik IO +
   kesinti simülasyonu, float16 kontratı, config türetme/doğrulama, resume/skip-guard, planner VRAM
   envanteri.
2. **Fizik** (küçük grid, `slow` işaretli): düzlem dalga faz hızı + dispersiyon, üstel absorpsiyon
   yasası, odaklı çanak vs **O'Neil**, fokal düzlem vs **Rayleigh**, harmonik büyüme vs **Fubini**
   (ppw=16 ister), çözücüler arası tutarlılık.
3. **Golden regression**: dondurulmuş küçük senaryolar; toleranslı karşılaştırma.
4. **GPU parite** (`gpu` işaretli): numpy vs cupy aynı mini senaryo — GPU varsa koşar, yoksa skip.
   **Bugün hiç koşmadı**: cupy bu makinede kurulu değil ve CI'da GPU yok. GPU iddiaları ilk Colab
   oturumuna kadar "doğrulanmadı" olarak işaretli kalır.
5. **Mimari testleri**: import yönü (AST taraması), wheel içeriği, temiz-venv kurulumu.
6. **CI**: Linux + Windows × py3.12, ayrıca py3.10 bacağı; `ruff check` + `ruff format --check` +
   `pytest -m "not kwave"`. GPU testleri Colab'da elle.
7. **Yerel koşum**: `./.venv/Scripts/python.exe -m pytest -q` — sistem Python'unda caustica kurulu
   değil.

---

## 14. k-Wave karşılaştırması ve doğrulama

- **Kritik bulgu (notebook v11)**: k-wave-python'ın **GPU binary'si Colab'da bozuk** (homojen su
  küpü 2.09e12 Pa'ya patladı). Bu yüzden karşılaştırmalar **CPU/OMP binary'siyle**; her ortamda
  önce T0 sanity, geçemezse rapora "environment-broken" damgası.
- **Gerçekleşen çapraz doğrulama (normalize alan)**: 2D lineer relL2 %1.14 (r=0.9998), heterojen
  %1.29, 3D çanak %1.57. Birim dönüşümleri (Np/m ↔ dB/cm, β ↔ B/A) testli.
- **Metrikler**: relL2, Pearson r, fokal konum/genlik farkı, −6 dB genişlikler, sidelobe +
  performans (s / 1e6 voxel-adım, VRAM — planner ile aynı formatta).
- **Adaptör sözleşmesi**: `pml_size = grid.pml_vox`; PML bandına düşen kaynak REDDEDİLİR.
- **Analitik süit birincil**, k-Wave ikincil çapraz kontrol. ITRUSST benchmark süiti M21'de
  repo'nun vitrin raporu olur.

---

## 15. Bilinen teknik gerçekler

Yeniden tartışmaya gerek yok; ama fizik yorumu yapılırken bilinmeli.

- **KEŞİF 1 — notebook'un absorpsiyonu etiketin YARISI.** Üstel sönüm yalnız basınca uygulanırsa
  dispersiyon bağıntısı `k = ω/c + i·α/2` verir (kaybın yarısı u'da kalır, o sönümsüz). Kaynak
  notebook (v6–v12) böyle yapıyordu → `dx300_t128` dataseti doku absorpsiyonunun **yarısını**
  gördü (deri 15→7.5, yağ 6→3, kas 10→5 Np/m). **Kütüphanede düzeltildi**: sönüm p VE u'ya
  simetrik; ölçülen α artık < %1 doğrulukta.
- **KEŞİF 2 — PML'siz grid = periyodik sınır tuzağı.** FFT periyodik sarma ±%40 duran dalga üretir.
  Çözücü artık PMLSpec yoksa gürültülü uyarı basar.
- **KEŞİF 3 — dataset faz haritaları 64×64'tü** (32 değil).
- **Fazor konvansiyonu (kütüphane çapında)**: `p(t) = Re{P·e^{-iωt}}`, giden dalga `e^{+ikx}` —
  analitikle aynı. Eski çözücü fazorları eşlenikti; 2026-08-10 öncesi raporlardaki mutlak fazlar
  eski konvansiyondadır.
- **Kaynak genliği**: kütle-kaynak normalizasyonu `2c·dt/dx` → gerçekleşen genlik ≈ `amplitude`,
  ortam-değişmez. (Eskiden `c_max`/dt üzerinden %27 kayıyordu.)
- **dx = 0.30 mm**: 2f0 Nyquist analiziyle kilitli (0.35 reddedildi, 1.88 ppw).
- Mutlak tepe değerlerinde ~%5 dt-bağımlı bias; dataset kendi içinde tutarlı.
- `amp/p_max` ~0.89–0.92 bandı ve spp=10 discrete ceiling (1.0515) beklenen davranış.
- `p_phase` kaynak-referanslı DEĞİL; sabit global offset HDF5 attr'ında kayıtlı.
- **Absorpsiyon frekans-bağımsız üstel**; power-law (fractional Laplacian) M16'da, k-Wave paritesi
  için gerekli.
- **Settling**: `settle_capped` dürüsttür ve yerleşme rampayı bekler
  (`eff_min ≥ tof + ceil(ramp) + 1`); planner aynı formülü kullanır.
- **Bilerek değiştirilmeyenler** (notebook paritesi): eğik eleman ayak izi (M12 adayı), faz haritası
  yarım-piksel maske merkezi (dataset kodlama paritesi).

---

## 16. Açık sorular

1. **`uwcem-phantom` ne zaman GitHub'a çıkar** — repo yerelde var ve çalışıyor
   (2026-08-22; şart okuması yapıldı, kod-only). Kullanıcı kararı: şimdilik yerel. Tek kopya
   riski kabul edildi; kalanlar `docs/uwcem.md`.
2. **Genel hacim araçlarının (crop/resample/simplify, sentetik heterojenlik) ileride caustica'ya
   dönüp dönmeyeceği.** Bilinçli olarak ertelendi: önce ayır, ikinci bir fantom kaynağı çıkarsa
   genelleştir. Şimdiden inşa edilmez.
3. **KZK'nın difraksiyon adımı**: açısal spektrum mu Crank–Nicolson mu (M9'da karara bağlanacak).
4. **CuPy'de DTT (DCT/DST) durumu** — AS çözücüsünün (M15) ön koşulu; yoksa ayna-genişletme +
   FFT tabanlı DTT katmanı gerekir. VERIFY işaretli.
5. **Çok-GPU fizibilitesi** (M20): cuFFT-Mp/NCCL slab decomposition mı, out-of-core mu, yoksa
   "yapılmaz çünkü…" raporu mu.
