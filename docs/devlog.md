# hifusim — Geliştirme Günlüğü (devlog)

> Amaç: her oturumun kararları, kanıtları ve açık uçları burada. Milestone "geçti" kararlarının
> kanıtı bu dosyaya işlenir (MILESTONES.md kuralı). Ters kronolojik değil, kronolojik.

---

## 2026-08-10 — Oturum 3 (Fable): M0–M3 tamamlandı

### Yapılanlar
- `.venv` kuruldu (Python 3.12.10; numpy 2.5.2, scipy 1.18.0, pydantic 2.13.4, h5py 3.16.0, pytest 8.x, ruff).
- **MILESTONES.md** yazıldı: M0→M24, her birinde ölçülebilir başarı kriterleri. Kural: kriterler
  sağlanmadan sonraki milestone'a geçilmez. GUI kullanıcı kararıyla kapsam dışı.
- **M0** — repo iskeleti: `pyproject.toml` (src-layout, extras: gpu/dev), `.gitignore`, `README.md`,
  `git init`. Editable kurulum çalışıyor.
- **M1** — core: `core/backend.py` (lazy-cupy dispatch; auto→numpy fallback), `core/grid.py`
  (1/2/3B izotropik grid, fft/rfft k-vektörleri, ppw, mm→voxel yardımcıları), `core/pml.py`
  (PMLSpec + notebook'un `_make_sponge_1d` portu), `config/models.py` (pydantic v2,
  `extra="forbid"`, GridConfig: mm→voxel TEK yönlü türetme, JSON round-trip).
- **M2** — `materials.py` (Material + MaterialDB + notebook TISSUE_PROPS birebir portu
  `breast_default()`; termal alanlar opsiyonel olarak şimdiden şemada), `medium.py`
  (homogeneous / from_id_map; float32 C-contiguous; bilinmeyen id → id listesiyle hata).
- **M3** — `analytic/`: `geometry.py` (Fibonacci eş-alan küresel kapak örnekleme),
  `rayleigh.py` (chunked vektörize Rayleigh–Sommerfeld, fiziksel prefaktörle),
  `oneill.py` (O'Neil 1949 eksenel kapalı form + odak limiti −ikhρcu0·e^{ikF}),
  `planewave.py` (üstel zayıflama, Fubini B_n = 2J_n(nσ)/(nσ), shock mesafesi).

### Kanıt (milestone geçiş kriterleri)
- `pytest`: **43 passed** (0.6 s). Kapsananlar: k-vektör analitik eşitliği (rtol 1e-12),
  ppw=4.394 dataset değeri, config round-trip + forbid, TISSUE_PROPS birebir eşitlik,
  **O'Neil vs Rayleigh çapraz doğrulama: fokal bölge korelasyonu r > 0.999, odak genliği
  farkı < %2, tepe konumu ≤ 0.2 mm** (iki bağımsız implementasyon birbirini doğruladı),
  Fubini limitleri (B1→1, B2→σ/2, enerji ≤ 1).
- `ruff check src tests`: temiz. `ruff format` uygulandı (stil bundan sonra formatter'a emanet).

### Kararlar ve gerekçeler
- **Kod + docstring İngilizce, devlog/plan Türkçe** — public k-Wave rakibi hedefi.
- `water()` preset'i **beta=0 lineer/kayıpsız** (doğrulama ortamı); fiziksel su isteyen beta'yı
  açıkça verir — test zincirinin varsayımı docstring'de.
- Grid **medium'suz**: dt/CFL/kappa çözücüde (c_max malzemeye bağlı). Grid yalnız geometri + k-uzayı.
- Rayleigh prefaktörü fiziksel (−iρck/2π): O'Neil ile MUTLAK genlik karşılaştırması yapılabiliyor
  (odak limitinde ikisi analitik olarak aynı k·h kazancına iner — test bunu doğruluyor).
- Lisans şimdilik MIT (pyproject'te); ilk public release öncesi teyit edilecek.
- Paket adı "hifusim" çalışma adı; PyPI kontrolü M11 (v1 öncesi) yapılacak.

### Gemini raporlarından alınanlar (şüpheyle işaretli olanlar MILESTONES'ta "VERIFY")
- ITRUSST koridorları milestone kriterlerine işlendi (odak basıncı <%10, konum <1 mm) — M21.
- KZK difraksiyon adımı için angular spectrum önerisi → M9'da CN ile karşılaştırılarak seçilecek.
- CuPy'de native DCT/DST yok iddiası → M15 öncesi sürüm notlarından DOĞRULANACAK
  (yanlışsa ayna-FFT workaround'a gerek kalmaz).
- k-Wave GPU binary'nin Colab'da bozukluğu bizim v11 T0 bulgumuzla tutarlı → M12'de sanity kapısı.

### Açık uçlar / sonraki adım
- **Sıradaki: M4** — lineer k-space PSTD çözücüsü (numpy, boyut-agnostik):
  `solvers/base.py` (SolverBase + yetenek deklarasyonu), `solvers/registry.py`,
  1. mertebe p–u şeması + kappa + sponge + CW kaynak + exact-period dt + fazor çıkarımı.
  Kriterler MILESTONES M4'te; düzlem dalga dispersiyon testi ilk yazılacak test.
- git deposu init edildi ama **commit atılmadı** (kullanıcı isteğiyle atılacak).
- `_code_cells.py` (notebook dökümü) referans olarak duruyor; .gitignore'da.

---

## 2026-08-10 — Oturum 4 (Fable): M4 + M4b tamamlandı; İKİ FİZİK KEŞFİ

### Yapılanlar
- Kullanıcı izniyle git commit akışı başladı: `bc715cd` (M0–M3), `71d9a5a` (M4+M4b).
- Kullanıcı kararı işlendi: **k-Wave registry'de doğrudan çözücü** (`get("kwave")`) ve
  doğrulamanın merkezinde (MILESTONES M4b eklendi, M12 onun üstüne oturuyor).
- `solvers/` paketi: `base.py` (SolverCaps + CWRunSpec + SolverResult + kurulum-anında
  yetenek doğrulama), `registry.py` (entry-point plugin desteği), `kspace/operators.py`
  (2-3-5-smooth pad, k-vektörler, kappa, sponge), `kspace/linear.py` (boyut-agnostik
  lineer çözücü), `kwave_adapter.py` (k-wave-python 0.6.2, CPU/OMP binary).
- `sources.py` (CWSource + plane/bowl builder'ları; duplicate-voxel reddi),
  `spectral.py` (tek fazor implementasyonu — çözücü, adaptör ve testler aynı kodu kullanır).
- `Backend.fft` eklendi: numpy yolunda scipy.fft (dtype-koruyan), cupy yolunda cupyx.scipy.fft.
  (numpy.fft float32'yi complex128'e terfi ettiriyor — fp32 GPU paritesi için kritik.)

### KEŞİF 1 — Notebook'un absorpsiyonu etiketin YARISI
M4 absorpsiyon testi (ölçülen α ≈ konfigüre α) İLK KOŞUDA %50 sapma yakaladı.
Analiz: üstel sönüm `exp(-α·c·dt)` YALNIZ basınca uygulanırsa dispersiyon bağıntısı
k = ω/c + i·α/2 verir (enerji eşbölüşümü: kaybın yarısı u'da, o sönümsüz) → uzamsal
sönüm α/2. Kaynak notebook (v6–v12) tam olarak böyle yapıyor → **dx300_t128 dataseti
doku absorpsiyonunun etikettekinin yarısını gördü** (deri 15→7.5, yağ 6→3, kas 10→5 Np/m
efektif). Dataset kendi içinde tutarlı; ama fiziksel yorum/karşılaştırma yapılırken bu
bilinmeli. KÜTÜPHANEDE DÜZELTİLDİ: sönüm p VE u'ya simetrik (ω→ω+iα·c ikamesi, tam üstel,
faz hızı değişmez); ölçülen α artık <%1 doğrulukta.

### KEŞİF 2 — PML'siz grid = periyodik sınır tuzağı
k-Wave karşılaştırma testinde grid'i PML'siz kurunca alan ±%40 duran-dalga desenine
boğuldu (FFT periyodik sarma). Çözücüye loud warning eklendi ("attach a PMLSpec unless
periodic boundaries are intended"). Test PML'le düzeltildi.

### Kanıt (milestone geçişleri)
- **77 test yeşil (11.6 s), ruff temiz.** M4 kapıları: dispersiyon <%0.1 (k-space+kappa
  ile ~kesin), absorpsiyon <%1 (fix sonrası), PML ripple <%3, saf CW fazor 1e-9 doğruluk,
  200-periyot kararlılık (drift <%1), 3D çanak vs O'Neil: odak ≤1 voxel, eksenel r>0.99,
  −6dB eksenel+lateral genişlikler <%5 (lateral referansı Rayleigh).
- **M4b canlı çapraz doğrulama: `linear` vs `kwave` (gerçek OMP binary, Windows yerel),
  2D su, normalize alan korelasyonu r > 0.99, tepe konumu ≤1 voxel.** Birim dönüşümleri
  (Np/m↔dB/cm, β↔B/A) testli. kwave yokken süit skip'lerle yeşil kalıyor.
- Test-tasarım dersleri: ölçüm bölgesi ASLA sponge içine taşmamalı (O'Neil eksenel
  penceresi PML'e değince r 0.974'e düşüyordu; domain z 96→120 + pencere sınırı);
  −6dB genişlik ölçümü distal kesişimi tam içermeli.

### Sonraki adım
- **M5**: `westervelt` çözücüsü (β terimi + p_max + 2f0; linear'dan türetilmiş tek
  fark nonlineer basınç güncellemesi). Kapılar: β=0 ≡ linear (<1e-6), Fubini A2/A1 <%5
  (σ≤0.3), amp/p_max tavan değişmezi. Ardından M6 arrays (spiral port).
- GitHub'a çıkış: kullanıcı `gh auth login` yapınca repo oluşturulup push edilecek.

---

## 2026-08-10 — Oturum 5 (Fable): M5 + M6 + görsel doğrulama raporu + GitHub

### Yapılanlar
- Kullanıcı kararları (tur 5): rapor = repo(MD+PNG) + Artifact web sayfası; kapsam = M5+M6+rapor;
  k-Wave seti = 2D×3 + 3D çanak; GitHub'a şimdi çıkılıyor.
- Profesyonelleşme: `.gitattributes` (LF normalize), `LICENSE` (MIT — kullanıcı telif satırını
  kendi adına güncelledi), GitHub Actions CI (ubuntu+windows, kwave testleri CI'da deselect),
  kwave adapter uyarı hijyeni (bilinen zararsız FutureWarning/UserWarning'ler koşu çevresinde
  filtreli; 8 uyarı → 1).
- **M5**: `solvers/kspace/engine.py` — linear+westervelt TEK numerik yüzeyde; westervelt
  `dp_nl = −2·β·dt·p·divu` (notebook formu); çok-harmonik fazor (`harmonics=(1,2,3)`) tek
  kayıt geçişinde; `SolverResult.phasors` + `harmonic_amp(n)`. kwave adapter de aynı
  harmonics API'sini aldı (kayıttan n·f0 tek-bin DFT).
- **M5 kalibrasyonu**: ppw=8'de Fubini A2/A1 ~%10 sapıyor (3f0 @2.67ppw aliası); ppw=16'da
  %0.85–3.2 → kapı ppw=16'da tanımlandı, çözünürlük kuralı belgelendi.
- **M6**: `arrays/` paketi — `archimedean_spiral` (parametre-generik notebook portu; üretim
  128'lik birebir: r_elem=3.205mm), `TransducerArray` (DAS fazlama, `rayleigh_preview`,
  `voxelize` eleman-sahiplikli kabuk projeksiyonu), `phasemaps` (sin/cos + boyut seçici).
- **GitHub**: repo canlı — https://github.com/ebx0/hifusim (public, master push'landı,
  gh hesabı ebx0 zaten girişliydi; winget ile gh CLI kuruldu). README rozetli/profesyonel.
- Görsel rapor altyapısı: `scripts/gen_validation_report.py` (8 senaryo, paralel koşulabilir,
  metrics fragment + PNG üretir; dataviz kurallarına uygun: tek-renk sequential rampa,
  diverging fark haritaları, sabit kategorik sıra). Senaryolar workflow ile paralel koşuldu.

### KEŞİF 3 — Dataset'in faz haritası 64×64'müş (32 değil)
M6 testi: üretim 128-spirali 32×32 faz haritası yerleşiminden GEÇEMİYOR (95 eleman,
0.25-piksel merkezleme toleransını aşıyor; matematik notebook'la birebir aynı). Notebook
runtime'da sessizce 64×64 fallback'ine düşüyordu → **dataset'in gerçek phase_map_size'ı 64**.
(HDF5 attr'ı doğru yazıyor; ama "default 32" zihinsel modeli yanlıştı.) Test bunu regresyon
olarak sabitliyor.

### Test-kapısı fizik dersleri (yanlış kapıyı düzeltmek de iş)
- DAS kapısı "tepe==hedef" OLAMAZ: sonlu açıklık odak kayması tepe noktasını faz hedefinin
  proksimaline çeker. Doğru kapı: uniform-vs-DAS tepe YER DEĞİŞTİRMESİ = komut edilen
  kayma (yanal <λ/2, eksenel <λ; sistematik bias farkta iptal) + hedefte ≥3× genlik artışı.
- Entegrasyon kapısında eksenel pencere O'Neil-öngörülü tepeyle sınırlandı (geometrik odak değil).
- Rapor senaryosunda dispersiyon ölçümü kayıpsız koşuya ayrıldı (kayıplı koşuda decay ripple
  faz-gradyan kestirimini kirletiyor: %0.28 görünüyordu; kayıpsızda %0.004).

### Kanıt
- **90 test yeşil** (M5: β=0 birebir eşitlik [array_equal], Fubini A2/A1 <%5 @ σ∈[0.06,0.61],
  A3/A1 <%10, tavan değişmezi; M6: geometri regresyonu, DAS, voxelizasyon, faz haritası,
  uçtan uca spiral+çözücü odaklanması). Commit: 4b81f8c.
- Rapor metrikleri (bu oturum): O'Neil 3D axial r=0.9916 / lateral r=0.9989; absorpsiyon
  hatası %0.33; dispersiyon %0.004; spiral DAS tepe (8.0, 84.5)mm vs hedef (8.0, 85.0)mm.
  k-Wave karşılaştırma sayıları workflow bitince metrics.json'da.

---

## 2026-08-11 — Oturum 5 devamı: rapor yayınlandı, review sertleştirmesi

### Rapor
- 8 senaryo koşuldu (workflow ile paralel; k-wave-python'ın SANİYE-damgalı temp .h5 adı
  yüzünden aynı saniyede başlayan paralel kwave koşuları çakışıyor — nonlineer senaryo seri
  yeniden koşuldu, kısıt scripte not edildi).
- Sonuçlar: k-Wave vs hifusim — 2D lineer relL2 %1.14 (r=0.99981), heterojen %1.29
  (r=0.99977), nonlineer f0 %1.14, 3D çanak %1.57 (r=0.99981, odak ≤1 voxel). 2f0 alanı
  zayıf-σ'da relL2 %17.3 ama A2/A1 seviyeleri %9.8 farkla koridor içinde.
- Çıktılar: benchmarks/reports/2026-08-10/ (REPORT.md + metrics.json + 9 PNG) +
  Artifact web sayfası: https://claude.ai/code/artifact/af2a6222-6ffb-4efe-b811-ee06f1f1479f

### Review turu (kısmi) ve sertleştirme
- Adversarial review workflow'u oturum limitine takıldı (17/19 ajan düştü) — "0 bulgu"
  İNCELENMEDİ demek. İki finder'ın 19 bulgusu journal'dan kurtarıldı, elle doğrulandı.
- GERÇEK bulgular düzeltildi:
  1. record_region dilimleri PADDED FFT dizisine uygulanıyordu — slice(None)/negatif stop
     pad voxellerini içeriyordu → normalize_record_region() (aktif şekle karşı çözümleme).
  2. spp/2 üstü harmonikler sessizce aliaslanıyordu → temporal-Nyquist kontrolü (2h < spp).
  3. kwave adaptörünün sabit programı ramp'i beklemiyordu → settle ≥ ramp+2 periyot.
  4. reference_point'e metre verilirse sessizce yanlış TOF → voxel-tamsayı doğrulaması.
  5. kwave record_region doğrulanmıyordu → aynı normalize yolu; kare sensör-verisi
     yönelim belirsizliğine uyarı.
  6. TransducerArray caller dizilerini aliaslıyordu → kopya.
  7. build_phase_maps çakışan elemanları sessizce eziyordu → ValueError.
- Test sertleştirmesi (tests/test_review_hardening.py, 9 test): yukarıdakilerin kapıları +
  FAZLI kaynakla canlı k-Wave testi (Fortran-order LUT'u gerçekten zorlar; r>0.99 &
  relL2<%5 GEÇTİ) + heterojen+soğurmalı ortam canlı testi (GEÇTİ) + hızlı nonlineer smoke.
- Kabul edilen sınırlamalar (düzeltilmedi, belgelendi): Fubini σ-ekseni self-kalibrasyonu
  (enjekte kaynakta mutlak genlik tanımsız — yapısal), rapor scriptinin assert'süzlüğü
  (kapılar pytest'te, script raporlama aracı), r-kapısının ~%6 λ-hatası toleransı (relL2
  kapısı eklendi), DAS-çözücü-içi yönlendirme testi (M12'ye not).
- **99 test yeşil.** Sonraki oturum: M7 CuPy (Colab) veya M8 planner; review'un physics
  finder'ı hiç koşamadı — bir sonraki oturumda tam tur tekrarlanmalı.

---

## 2026-08-11 — Oturum 6 (Fable): M6b geometri sistemi (araya alınan iş)

### Yapılanlar
- Kullanıcı talebi: COMSOL-vari geometri. `src/hifusim/geometry/` paketi (materyallerden AYRI):
  - `shapes.py`: Ball/Box/Cylinder/Ellipsoid/HalfSpace + `| & - ~` CSG cebiri +
    translated/rotated/scaled (AffineShape, ters-dönüşüm noktalarla; tek kontrat:
    `contains((n,ndim) noktalar)`).
  - `scene.py`: Scene(ndim, background, axisymmetric) — boyama sıralı etiket ataması,
    `rasterize(grid, supersample)` (s^ndim alt-örnek + çoğunluk oyu, satır-chunk'lı),
    `add_volume` (import edilen fantom sahnede konumlandırılır, ignore etiketleri şeffaf),
    `to_medium`.
  - `volumes.py`: LabelVolume (dx+origin'li heterojen etiket hacmi), mtype-tarzı text import
    (genel `mapping` callable + meme fantomu preseti; .labels.npz otomatik önbellek),
    `resample(dx, "nearest"|"smooth")` (smooth = one-hot lineer + argmax, etiket icat etmez).
  - `configs.py`: pydantic tagged-union CSG ağacı JSON'da; import DOSYA REFERANSI
    (yol+format+konum+resample ayarı) JSON'da.
- Refactor/temizlik: PLAN.md'ye "tarihsel belge" bandı (canlı doküman MILESTONES),
  README'ye geometri bölümü. Gereksiz dosya taraması: silinecek bir şey bulunamadı
  (_code_cells.py gitignore'da referans olarak duruyor — bilinçli).

### Test kapısı dersleri
- Süperörnekleme kapısı "hacim hatası küçülür" DEĞİL: büyük pürüzsüz şekilde kenar hataları
  istatistiksel dengelenir (s=1 hacim hatası ~%0.08!). Ölçülebilir doğru kapı: s=3
  rasterizasyonu s=5 (yakınsak) referansına s=1'den daha yakın (7 vs 9 sınır voxeli).
- Axisymmetric r≥0 kuralı voxel MERKEZLERİ için (r=0 merkezli eksen voxeli meşru).

### Kanıt
- **121 test yeşil** (22 yeni geometri testi: primitif analitiği, CSG ≡ numpy boolean birebir,
  dönüşümler, boyama sırası, axisym, mtype-format round-trip + NaN + Fortran + önbellek,
  0.5→0.3 mm resample (gerçek oran) arayüz ≤1 voxel, config JSON round-trip + build ≡ elle
  kurulum, Scene→Medium→çözücü smoke, gerçek mtype.txt yükleme+resample [yerel]).

### Sonraki
- M7 CuPy (Colab) veya M8 planner. M6b'nin axisym sahneleri M15 çözücüsünü bekliyor.
- Bir sonraki oturumda tam adversarial review turu (bu turda da atlandı — önceki oturumda
  limit yüzünden kesilmişti; geometri paketi henüz bağımsız review görmedi).

---

## 2026-08-11 — Oturum 7: M8 planner (yerel yarı) + tam adversarial review turu

### M8 — Planner v1 (`hifusim.planner`)
- VRAM modeli: `run_cw_kspace_pstd`'nin tampon envanterinin birebir dökümü (durum p+u,
  özellik haritaları, sünger, spektral çarpanlar, kayıt tamponları, adım geçicileri,
  FFT workspace payı) + %15 ayırıcı marjı. `test_memory_inventory_matches_hand_count`
  envanteri elle sayımla sabitler — motora kalıcı tampon ekleyen bu testi kırar (bilerek).
- Süre modeli `t_step = a·P·log2(P) + b·P`; üç kaynak, sonuçta ETİKETLİ:
  `db` (gpu_db.json datasheet, kaba ~2x), `calibrated` (cihazda ~20 gerçek adım →
  `~/.hifusim/calibration.json`, ±%25 Colab kapısı), `measured` (şimdi bu makinede ölç).
- `planner.estimate(...)` / `planner.compare(...)` (fits/sure tablosu); OOM'da eyleme
  geçirilebilir öneriler: dx ×m (hesaplı), AOI küçült, linear'a geç, daha büyük cihaz.
- Motor refaktörü: dt/spp ve tof türetimi `cw_discretization`/`cw_tof_periods` olarak
  motordan çıkarıldı — planner ve motor AYNI fonksiyonu çağırır (test: planner==engine).
- Colab'a kalan iki kapı MILESTONES M8'de açık işaretli (VRAM ±%10, kalibre süre ±%25).

### Adversarial review turu (2 bağımsız ajan: geometri + fizik motoru)

**Geometri (9 bulgu; 5 MED düzeltildi, hepsi regresyon testli):**
1. `resample`: scipy `zoom(grid_mode=False)` uç-hizalama → içerik %1.5'e varan gerilme,
   0.5→0.3 mm fantomda arayüz 1 voxel kayıyordu. Çözüm: eksen-ayrık TAM fiziksel pozisyon
   örnekleme (`j·dx_new/dx`); arayüz testte tam 18.0 mm'de sabitlendi.
2. `.labels.npz` önbelleği yalnız mtime'a bakıyordu — farklı argümanlarla (transpose/dx/
   mapping) çağrı bayat önbelleği sessizce döndürüyordu. Çözüm: argüman parmak izi npz'de.
3. `add_volume` `volume.origin`'i yok sayıyordu (rasterize→add_volume round-trip konum
   kaybediyordu). Çözüm: position verilmezse origin geçerli.
4. Axisymmetric + supersample: eksen voxelinin r<0 alt-örnekleri aynalanmadan
   değerlendiriliyordu (odak tam orada!). Çözüm: `pts[:,0]=|r|`.
5. Chunk böleni s çarpanı eksikti (bellek s kat büyüyordu; sonuç doğruydu). Ayrıca:
   majority tie artık gerçekten "son boyanan kazanır"; `LabelVolume.__eq__` düzgün;
   HalfSpace + affine Transform config'leri eklendi (JSON kapsamı tamamlandı);
   SceneConfig boyama sırası (import → obje) belgelendi.

**Fizik motoru (fizik çekirdeği TEMİZ çıktı: dispersiyon 1500.000 m/s, absorpsiyon
5.0017/5.0 Np/m, Westervelt/Fubini, birim dönüşümleri, Fortran sıralaması, analitik
önfaktörler — hepsi doğrulandı; bulgular ÇÖZÜCÜ SINIRLARINDA):**
1. **Kaynak genliği kalibre edildi**: ham additive enjeksiyon ~amp/(2·CFL_local)
   gerçekleşiyordu ve dt(c_max) üzerinden UZAKTAKİ ortam içeriğine bağlıydı (uzak hızlı
   inklüzyon sürücüyü %27 değiştirdi!). Çözüm: k-Wave-eşdeğeri kütle-kaynak ölçeği
   `2·c·dt/dx` — gerçekleşen düzlem genliği ≈ amplitude, grid/ortam-değişmez (test:
   sünger içine gömülü c=1800 blok spp'yi değiştirir ama genlik <%2 oynar).
2. **Fazor konvansiyonu kütüphane çapında sabitlendi**: çözücü analitik referansların
   KOMPLEKS EŞLENİĞİNİ üretiyordu. Artık `p(t)=Re{P·e^{-iωt}}`, giden dalga `e^{+ikx}`
   (analitikle aynı). `single_bin_phasor` + motor demodülasyonu çevrildi; dispersiyon
   ve spektral testler yeni konvansiyonu sabitler.
3. **kwave adapteri PML**: `pml_size=grid.pml_vox` geçirilir (önce k-Wave default'u
   sessizce farklı banttı); kaynak k-Wave'in iç PML bandına girerse ValueError.
4. `settle_capped` dürüstlüğü (açık `converged` bayrağı); yerleşme penceresi artık
   kaynak rampasını bekler (`eff_min ≥ tof + ceil(ramp)+1`; planner aynı formülü kullanır).
5. Belgelendi/bilinçli bırakıldı: eğik eleman ayak izi ~πr²/cosγ (birebir notebook portu,
   M12 adayı); faz haritası maskesinin yarım-piksel merkezi (dataset kodlama paritesi).

### Kanıt
- **139 + 4 kwave = 143 test yeşil** (11 planner + 8 geometri-regresyon + 3 fizik-regresyon
  yeni). Canlı k-Wave çaprazları adapter değişikliği sonrası yeniden koşuldu.
- NOT: 2026-08-10 tarihli görsel rapor ESKİ mutlak genlik/faz konvansiyonuyla üretildi;
  normalize karşılaştırmalar geçerliliğini korur, mutlak değerler o günün anlık görüntüsüdür.
  Bir sonraki rapor üretimi yeni konvansiyonla damgalanır.

### Sonraki
- M7 CuPy (Colab oturumu; M8'in iki Colab kapısı aynı oturumda ölçülür) veya M10 IO.

---

## 2026-08-18 — Oturum 6 (Opus): M6c — UWCEM fantom alt modülü + Phantom Studio GUI

Araya alınan iş (kullanıcı talebi): "UWCEM fantomlarını indir, simülasyona import edilebilir
TEK dosya üreten API-benzeri bir alt modül yaz, bir de bunu kullanan GUI yap."

### Yapılanlar

**Veri.** https://uwcem.ece.wisc.edu/phantomRepository.html — dokuz MRI-türevi meme
fantomu (ACR yoğunluk sınıfı 1–4, 0.5 mm izotropik), `breastInfo.txt` + `mtype.zip` +
`pval.zip`. Hepsi indirildi (179 MB, `src/hifusim/phantoms/_data/uwcem/`, .gitignore'da).
InstructionManual.pdf çözümlendi: `mtype` = Tablo-1 medya numarası/voxel (−1 daldırma
ortamı, −2 deri, −4 kas, 1.1–1.3 fibroglandüler, 2 geçiş, 3.1–3.3 yağ), `pval` = sınıf içi
[0,1] konum, ikisi de Fortran sırasında tek sütun ASCII.

**ZIP'ler AÇILMADI, bilinçli.** Dokuz `mtype.txt` + `pval.txt` açıldığında ~3.5 GB; zipli
179 MB. Okuyucu doğrudan zip'in içinden çözüyor, gerçekten yeniden kullandığımız açılmış
biçim ise int8 `.npz` önbelleği (fantom başına ~600 kB, ~1000:1 sıkışıyor).

**`src/hifusim/phantoms/` (10 modül).** `catalog` (katalog + atomik indirici + CRC
doğrulaması), `reader` (hızlı çözücüler), `orientation`, `tissue`, `processing`,
`heterogeneity`, `spec` (pydantic tarif), `builder` (boru hattı + `plan()`), `asset`
(dışa aktarım), `cli`.

**Hızlı çözücüler (18× ve 4×).** `np.loadtxt` en büyük fantomda dakikalar sürüyor; GUI her
parametre değişiminde bunu ödeyemez.
- `mtype`: satırlar tam on token; (byte0,byte1,byte2) üçlüsü bu on token içinde TEKİL →
  tek gather + `searchsorted`, hiç float parse etmeden int8 sınıf kodu. 18.0 s → 0.96 s.
- `pval`: `%1.5f` yazılmış, yani kaydı 8 bayt SABİT ADIM → baytları (n,8) reshape edip
  ondalık aritmetiği numpy'de. 4.36 s → 1.07 s.
- İkisi de varsayımını gerçek baytlarda DOĞRULUYOR (token kümesi / adım + ayraç konumları)
  ve tutmazsa yavaş referansa düşüyor. Test: her iki çözücü aynı baytlarda birebir eşit
  (CRLF ve eksik son satırsonu dahil).
- `flat.reshape((s3,s2,s1))` + transpose(2,1,0) = SIFIR KOPYA (dosya zaten column-major).
  Kopya tek sefer, yönelim adımında. Toplam: 10.1 s → 2.8 s (071904, pval dahil).

**Yönelim — tahmin edilmedi, ÖLÇÜLDÜ.** Manuel hangi eksenin ne olduğunu söylemiyor.
Kılavuzun vaat ettiği 0.5 cm kas göğüs duvarı, dokuz fantomun HEPSİNDE s1'in SON on
diliminde, dilim alanının %100'ünü kaplayan tek slab olarak çıkıyor; s1=0 saf daldırma
ortamı. Yani **s1 = meme ucu → göğüs duvarı ekseni**, indeks 0 transdüser tarafı.
Kanonik dönüşüm `flip(transpose(2,1,0), axis=0)`: tek eksen takası TEK permütasyon
(det −1, yani ayna — sol meme sessizce sağ meme olur), ek flip elle-yönlülüğü geri
getiriyor. Mevcut `load_breast_phantom` ile aynı dönüşüm; test dokuz fantomu da tarıyor.

**Akustik tablo.** Depo ELEKTROMANYETİK; akustik sayı içermiyor. On sınıf için c, rho,
alpha (güç yasası `alpha0·f^b`, dB/(cm·MHz^b)) ve B/A → beta tablosu yazıldı. Uç noktalar
literatür (Duck 1990; IT'IS; Mast 2000; UST meme ölçümleri), fibroglandüler/yağ alt
grupları UWCEM'in kendi su-içeriği sıralamasında İNTERPOLE — her satır `interpolated`
bayrağı + kaynak dizesi taşıyor, CLI tablosunda `~` ile işaretli. Alpha güç yasası olarak
saklanıp `f0`'da değerlendiriliyor: Np/m saklamak fantomu sessizce tek frekansa çivilerdi.

**pval'in tuzağı (yakalandı).** Kılavuz: "diğer tüm voxel'lerde p sıfıra ayarlanmıştır" —
yani deri/kas/banyo için p=0 "VERİ YOK" demek, "en düşük değer" değil. İlk sürüm bunu
alt sınır sanıp fantomdaki HER deri voxel'ini literatür minimumuna çekiyordu
(alpha 32.8 → 25.3 Np/m). Düzeltme: `TissueTable.pval_ids` — yalnız pval'i olan
sınıflar interpole edilir, kalanlar orta noktada kalır.

**Diğer düzeltilen gerçek hatalar.**
- `crop mode="tissue"` yanal olarak HİÇ kırpamıyordu: kas slab'ı VE memenin arkasındaki
  deri altı yağ katmanı dilimin tamamını kaplıyor. `mode="breast"` eklendi — yanal kutuyu
  yalnız "çıkıntı yapan" dilimlerde (doku kapsaması < %90) ölçüyor, ilerleme ekseni tüm
  göğüs duvarını koruyor. 212×328 → 180×288 (%29 daha az voxel).
- Gürültü alanı yalnız std'ye normalize ediliyordu; korelasyon uzunluğu büyüdükçe
  bağımsız örnek sayısı düşüyor ve kalan DC ofseti tüm dokuya SİSTEMATİK sapma olarak
  biniyordu (40³'te 4 voxel korelasyonda genliğin %10'u — "%3 gürültü" aynı zamanda
  memeyi %1 yoğunlaştırıyordu). Artık önce demean, sonra normalize.
- `np.savez_compressed(path)` adı `.npz` ile bitmiyorsa `.npz` EKLİYOR → atomik yazımın
  `.part` geçici dosyası kayboluyordu. Dosya NESNESİ ile yazılıyor.
- `tissue_only` gürültü maskesi id 0'ı banyo sanıyordu; `simple` modelinde banyo id 4
  (breast_default uyumu için) — scatterer'lar suya gidiyordu. `TissueTable.coupling_id`.

**Koşarak bulunan hata (en değerlisi).** İlk uçtan uca denemede dx=1.2 mm / f0=1 MHz
seçildi; build başarılı, medium geçerli, çözücü sonuna kadar koştu — ve tepe basınç
TRANSDÜSERİN ÜSTÜNDE çıktı, odak hiç oluşmadı. Sebep: 1.2 mm'de en yavaş dokuda dalga
başına 1.3 nokta; k-space PSTD ~2 ppw altında dalgayı taşıyamıyor. Hiçbir katman şikâyet
etmiyordu. Artık `builder` f0 + dx + en yavaş dokunun c'sinden ppw hesaplayıp <2'de
"UNUSABLE", <4'te "marginal" uyarısı veriyor ve HANGİ dx / hangi f0 gerektiğini yazıyor.

**Dışa aktarım — tek dosya, iki okuyucu.** `.npz` anahtarları: `labels`, `dx`, `origin`
(bunlar TAM OLARAK `LabelVolume.load_npz`'in okuduğu anahtarlar → eski kod hiç
değişmeden aynı dosyayı açıyor), + `materials` (MaterialDB JSON), `meta` (spec,
uyarılar, log, atıf), `format`, ve heterojenlik varsa `alpha/rho/c/beta` float32
hacimleri. Heterojenlik kapalıyken yoğun hacimler YAZILMAZ: etiket+DB'den birebir
üretilebilir, yazmak 2 MB'lık dosyayı 500 MB yapardı.

**`plan()`.** Pahalı adımları yapmadan boyut/bellek tahmini (kırpma kutusu gerçekten
ölçülüyor, gerisi aritmetik). Test: `plan(spec).shape == build(spec).shape`, üç farklı
dx/standoff için birebir. GUI önizleme çözünürlüğünü bununla seçiyor.

**GUI — `apps/phantom_studio/`, sıfır bağımlılık.** Stdlib `http.server` + elle yazılmış
WebGL2 ray caster. Sol: `PhantomSpec`'in her düğmesi. Sağ: 3-B hacim + üç kesit (slider'lı,
hover okumalı). Kesit/hacim tel üzerinde ham `uint8` + değer aralığı olarak gidiyor,
renklendirme tarayıcıda → alan/renk rampası değiştirmek anında, yeniden build yok.
Kesme (cutaway) köşe ısırığı olarak yapılıyor ve açığa çıkan üç yüz TAM OLARAK alttaki üç
kesit düzlemi. Önizleme build'i voxel bütçesine sığacak kadar kabalaştırıyor, bunu SÖYLÜYOR
ve önizlemenin dışa aktarılmasını REDDEDİYOR.

**Palet — göz kararı değil, doğrulanmış.** On sınıf aslında BEŞ aile, ikisi kendi içinde
sıralı (fibroglandüler 1→3, yağ 1→3). Beş aile rengi bu koyu yüzeyde (#12181f) tüm-çiftler
renk körlüğü kapılarını geçiyor (en kötü çift CVD ΔE 9.0 ≥ 8 hedefi, normal görüş 15.4 ≥ 15
tabanı, hepsi ≥ 3:1 kontrast); iki alt rampa ordinal testi geçiyor. Banyo rengi seri değil
YÜZEY (arka plan; 3-B'de şeffaf), o yüzden kapıların dışında. Sürekli alanlar için tek
hue'lu sequential rampa (asla gökkuşağı).

### Kanıt
- **258 test yeşil** (115'i yeni; 22'si inceleme turunun regresyon testleri),
  `ruff check src tests apps` temiz.
- **Uçtan uca fizik koşusu** (asıl iddianın kanıtı): 012304 @ 0.6 mm, f0=0.5 MHz, 26 mm su
  standoff, bowl (a=28 mm, ROC=55 mm) → `build → save → load_phantom → grid + to_medium →
  westervelt`. Tepe 6.00 MPa, konum z=54.0 mm; geometrik odak suda 51.8 mm olurdu —
  **2.2 mm derine kayma**, yağ dokusundan (c=1475 < su 1522) geçmenin beklenen etkisi.
  Tepe "Fatty-1" voxel'inde, yani memenin içinde. `to_medium(linear=True)` ile lineer
  çözücü de koştu: 6.09 MPa (nonlineer doygunluk temel harmoniği hafifçe düşürüyor).
  Alan her yerde sonlu.
- Hızlı↔yavaş çözücü birebir eşitliği; dokuz fantomda göğüs duvarı geometrisi;
  `plan()==build()`; export'un HEM `LabelVolume.load_npz` HEM `load_phantom` ile
  açılması; uçtan uca `build → save → load_phantom → grid + to_medium` (Medium
  invariyantları: float32, C-contiguous, sonlu, 1000 < c < 2000 m/s).
- GUI headless Edge'de (SwiftShader) render edildi ve ekran görüntüsüyle doğrulandı
  (etiket haritası + sürekli alan, 3-B + üç kesit, path-traversal denemeleri 404).

### Düşman gözle inceleme turu (4 mercek x N doğrulayıcı, 40 ajan)
36 aday bulgu, 26'sı çürütmeye dayandı ve DÜZELTİLDİ. Kayda değer olanlar:

**Geometri / fizik**
1. **FFT dolgusu standoff'u şişiriyordu.** `pad_to_fft_friendly(anchor="center")` büyümeyi
   HER eksende ikiye bölüyordu — yani transdüser yüzünün ÖNÜNE de su ekliyordu. dx=0.55 mm'de
   istenen 20 mm standoff sessizce 23.65 mm oluyordu (1 MHz'de ~2.6 dalgaboyu) ve dx ile
   0–3.85 mm arasında düzensizce değişiyordu. Artık z büyümesi tamamen ARKAYA gidiyor;
   ölçüldü: her dx'te standoff bir voxel toleransında tam.
2. **pval birleştirilmiş bandın üstüne uygulanıyordu.** `p` MEDYA NUMARASI içinde tanımlı;
   `grouped` modelinde yağ-1..yağ-3 birleşince blend 2× geniş banda yayılıyor ve saf lipid
   voxel'e yağ-1 hızı veriliyordu. `lookup_by_code` eklendi.
3. **pval yeniden örneklemede sentinel karışıyordu.** Deri/kas/banyodaki p=0 "veri yok"
   demek; lineer interpolasyon bunu gerçek p ile harmanlayıp memenin TÜM yüzeyini saran bir
   voxel'lik şeritte dokuyu sınıf minimumuna çekiyordu (oran 1.4'te 0.600 → 0.360). Artık
   maskeli (p·valid ve valid ayrı örneklenip bölünüyor) — sentetik testte tam 0.600.
4. **`drop_skin` memeyi küçültüyordu.** En yakın deri-olmayan komşu alınıyordu; deri ~3
   voxel'lik bir kabuk ve dışı su olduğu için %57.5'i suya dönüşüyor, doku yüzeyi 1.22 mm
   geri çekiliyordu — yani ışının girdiği arayüz kayıyordu. Artık en yakın DOKU ile
   dolduruluyor: ölçülen kayma 0.00 mm.
5. **Su 20 °C absorpsiyonu taşıyordu** ama satır "37 °C" diyordu (c ve rho 37 °C idi).
   alpha0 0.0022 → 0.0015 dB/(cm·MHz²).
6. **`realized_cv` gürültüyü değil doku kontrastını ölçüyordu** (%3 istenince %6.76 rapor
   ediyordu — kullanıcı düğmeyi yarıya indirirdi). Artık çarpanın std'si.
7. **coupled/uncoupled empedans kontrastı docstring'de TERSTİ.** Ortak alanla z=ρc
   (1+sn)² gibi ölçekleniyor → coupled √2 kat DAHA FAZLA kontrast veriyor, daha az değil.
8. **`remove_islands` bitirmiyordu**: A adasını B'ye çevirmek B'de yeni küçük bileşen
   yaratıyor, döngü ise yazımdan önce alınmış anlık görüntü üzerinde. 63.502 voxel "silindi"
   raporlanırken 7.669 voxel hâlâ eşik altı bileşendeydi. Artık sabit noktaya kadar; kalan 0.

**Sağlamlık / kaynak**
9. Paylaşılan sabit `.part` geçici adı: iş parçacıklı sunucuda `/api/plan` ve `/api/build`
   aynı fantomu aynı anda çözüyor → Windows'ta PermissionError, POSIX'te kayıp rename.
   Artık pid+uuid'li ad ve önbellek yayınlama "best effort" (başarısızlık yüklemeyi düşürmez).
10. Bozuk önbellek `zipfile.BadZipFile` fırlatıyordu — `except (OSError, ValueError, KeyError)`
    bunu yakalamıyor. Artık her okunamayan önbellek yeniden çözülüyor. Ayrıca önbellek kaynak
    arşivin boyut+mtime parmak iziyle geçersizleştiriliyor.
11. `decode_mtype` yalnız 3 baytlık ÖNEKİ eşleştiriyordu: "1.15" sessizce sınıf 1.1 oluyordu.
    Artık token uzunluğu da doğrulanıyor.
12. `decode_pval` 111 MB sonuç için ~1.4 GB scratch kullanıyordu (5 sütunluk int32 dizisi +
    dört tam boy geçici). Tek akümülatör: 445 MB, aynı hız, aynı değerler.
13. `plan().peak_bytes` yeniden örnekleme aşamasını HİÇ saymıyordu — dx=0.35 mm'de 267 MB
    tahmin, 1180 MB gerçek (4.4×). Artık üç aşamanın maksimumu, katsayılar ölçümden;
    iki fantom × yedi senaryoda en kötü oran 1.10 (asla düşük raporlamıyor).
14. `max_voxels` bariyeri pahalı adımdan SONRA tetikleniyordu; artık öncesinde (0.16 s).

**GUI**
15. Görünüm isteği her seferinde TÜM hacmi materyalize ediyordu: 132 kB'lık bir kesit
    199 MB / 131 ms. Artık düzlem doğrudan alınıyor: 1.1 MB / 1.6 ms. Hacim isteği
    348 → 67 MB (parçalı blok indirgeme + parçalı bincount).
16. `couplingId()` banyoyu İSME göre buluyordu; `simple` modelinde id 0 sıfır-voxel'lik bir
    yer tutucu ve aynı ismi taşıyor → hacmin %52'si opak su olarak çiziliyordu. Artık
    sınıf koduna göre; voxel'i olmayan materyal hiç boyanmıyor.
17. Kesit/hacim istekleri sıraya sokulmuyordu — slider sürüklerken EN YAVAŞ yanıt kazanıyordu.
    Artık istek jetonu. Ayrıca baytlar üretildikleri alanla etiketli, böylece geç gelen bir
    yanıt yeni alanın renk skalasıyla boyanıp uydurma bir değer okutamıyor.
18. İptal edilen istek (sekme kapandı) iki aşamalı traceback döküyordu ve yarı yazılmış
    yanıtın üstüne ikinci bir yanıt yazmaya çalışıyordu. Artık sessiz.
19. Dışa aktarım adı sanitize edilmiyordu: `"../../x"` dizinden çıkıyor, mutlak yol tamamen
    değiştiriyordu. `export_path()` düz dosya adı dayatıyor.

Kalan 10 aday çürütüldü (doğrulayıcı ajan gösteremedi).

### Açık uçlar / sonraki adım
- Akustik tablonun interpole satırları literatürle bağımsız olarak bir kez daha
  karşılaştırılmalı (uç noktalar sağlam, ara basamaklar bizim modelleme kararımız).
- Fantom + çözücü uçtan uca fizik koşusu (aberasyon/odak kayması) M12 karşılaştırma
  harness'ine aday.
- Sıradaki: M7 CuPy (Colab oturumu) veya M10 IO.

## 2026-08-19 — Oturum 8 (Fable): M6d — fantomlar bağımsız pakete taşındı + standart hizalı 0.25 mm dataset

Kullanıcı talebi: fantom modülü `src`'nin dışına, yan paket olarak `uwcem_phantoms` adıyla
taşınsın ("uwcam" yazımı soruyla düzeltildi — depo adı UWCEM); hifusim için gerekli veriler
otomatik üretilsin: 0.25 mm, hepsi AYNI boyda, merkezleri hizalı, `data/phantoms/` içinde,
pval'lar düzgün. Kurulum soruları soruldu; kullanıcı dört öneriyi de seçti: 9 fantomun hepsi,
meme kırpması + birleşim kutusu, ön yüz + transvers merkez hizalama.

### Taşıma

`src/hifusim/phantoms/` → `uwcem_phantoms/` (repo kökü, `apps/` gibi; wheel'e GİRMEZ —
hifusim'i tüketir, hifusim onu asla import etmez). Tüm importlar, `phantoms.bat`/`.sh`,
launcher, studio GUI, testler, README, `.gitignore` yeniden yazıldı; `pyproject`'e
`pythonpath = ["."]` (pytest) ve ruff src listesine `uwcem_phantoms` eklendi. Taşıma sonrası
124 mevcut fantom testi değişiklik olmadan yeşil.

### `uwcem_phantoms/dataset.py` — dokuz meme, TEK grid

İki fazlı: (1) **survey** — dokuz fantomun kırpma kutusu native 0.5 mm'de (önbellekli mtype,
saniyeler) ölçülüp `0.5/dx` ile ölçeklenir, birleşim + 3 voxel emniyet + FFT-dostu boyut =
ortak kutu; builder'ın fitted peak-RAM modeli de satır başına taşınır ve build başlamadan
boş RAM'e karşı kapı var. (2) **build** — fantom başına normal boru hattı (breast kırpma,
smooth resample, pval), sonra ortak kutuya su dolgusuyla hizalama: deri ön yüzü her dosyada
TAM z=front_gap (20 voxel = 5 mm); x/y'de ÇIKINTI YAPAN memenin bbox merkezi kutu merkezinde.
Dolgu fizik taşır: etiketler coupling id, özellik hacimleri suyun KENDİ mid değerleri.

Üretim (bu makinede, arka planda): **540×700×625 @ 0.25 mm = 135×175×156.25 mm, 236 Mvox**;
9 dosya toplam **5.66 GB**, ~55 dk, sıfır builder uyarısı, her dosyada ön yüz z=20, trim 0.
`manifest.json` grid'i, fantom başına gerçekleşen hizalamayı, sınıf-başına özellik
istatistiklerini ve atfı kaydediyor.

### Düşman gözle inceleme turu (5 mercek + bulgu başına şüpheci, 27 ajan)

22 aday bulgu → 17'si doğrulamaya dayandı → HEPSİ düzeltildi (5'i çürütüldü). Kayda değer:

1. **x/y "doku bbox merkezi" kırpma-penceresi merkezine dejenereydi.** Göğüs duvarı yağ
   slab'ı her transvers dilimi kapladığından doku bbox'u = pencere; merkezlenen şey meme
   değil pencereydi ve verify aynı dejenere niceliği ölçtüğü için HİÇ başarısız olamazdı.
   Düzeltme: `_breast_transverse_bbox` — dilim "göğüs duvarı" sayılır eğer su-dışı sayısı
   hacim-geneli transvers bbox alanının %90'ını aşarsa; sayılar da bbox da su dolgusundan
   etkilenmediği için tanım hizalama ÖNCESİ ve SONRASI aynı dilimleri seçer (verify gerçekten
   ölçebilir). Dokuz gerçek fantomda yeni survey birebir AYNI grid'i veriyor ve hiçbirinde
   kırpma clamp'i yok → eski üretim yeni tanıma göre de doğru merkezli (doğrulandı).
2. **NaN-körü doğrulama.** Her sayısal kapı "bant dışıysa raise" kıyasıydı ve NaN için
   her kıyas False → NaN'li hacim temiz geçerdi. `isfinite` kapısı eklendi (build + verify).
3. **RAM rayı yoktu.** Fitted peak fantom başına 6.1–15.0 GB (24 GB makinede); plan artık
   en kötü peak'i basıyor, build boş RAM'den büyükse `--force`'suz reddediyor.
4. **`savez_compressed` seviye 6'ya çivili (~3 MB/s).** Elle zip (aynı npz düzeni,
   `compresslevel=1`, ~20 MB/s, aynı boyut) — `np.load`/`LabelVolume.load_npz`/`load_phantom`
   değişmeden okuyor.
5. **236 Mvox'ta ölçek maliyetleri.** `PhantomAsset.__post_init__`'teki `np.unique`
   (945 MB kopya + 5.7 s, fantom başına 5 kez) → min/max + chunk'lı histogram;
   `_nonwater_bbox`'taki `np.isin` (11.7 s) → `!= 0` (0.2 s); sınıf-başına bant kontrolü
   40 maske+extract geçişi (72 s) → tek chunk'lı geçiş: `bincount` (count/sum/sumsq) +
   `ufunc.at` (min/max), ~8 s.
6. **Manifest alt-küme rebuild'de ezilip dosyalar öksüz kalıyordu.** Artık aynı tarifli
   rebuild MEVCUT grid'i benimseyip manifest'e BİRLEŞİR; farklı tarif (ör. `--f0 3`)
   dosya adı sadece dx kodladığından sessiz ezme demekti → reddedilir (`--force` bilinçli
   ezme). `--verify` manifest format tag'ini, dosya başına dataset tag'ini VE manifest'in
   listelemediği `uwcem-*.npz` dosyalarını (öksüzleri) da kontrol ediyor.
7. **CLI/launcher kabloları.** `--verify` yanındaki id/build bayraklarını sessizce yutuyordu
   → kullanım hatası (rc 2); `--verify --json` çıktısına insan satırı ekleniyordu → saf JSON
   (progress stderr'e); launcher tek-atış modda `DatasetError`'da ham traceback basıyordu →
   menüyle aynı dostane hata; launcher'ın yükleme snippet'i dx'e bakmadan `dx0p25` yazıyordu
   → `dataset_filename(id, dx)`.

Çürütülenler (örnek): survey'in z-yüz haritalama konvansiyonu resampler'ınkinden farklı ama
hata SAFETY_VOX içinde kalıyor (ulaşılabilir kırılım yok); `detailed` sabitken etiket==kod
varsayımı kırılamaz; CLI literal defaultları sabitlerle bugün eşit.

### Kanıt
- **Tam suite yeşil** (taşıma sonrası, düzeltmeler sonrası iki ayrı tam koşu; 32'si yeni
  dataset testi — hizalama sentetikleri, NaN, RAM rayı, CLI kablo/routing testleri,
  manifest birleştirme + farklı-tarif reddi, format/öksüz/ön-yüz sabotaj yakalama,
  dx=1 mm'de iki fantomla gerçek uçtan uca). `ruff check` temiz.
- **`python -m uwcem_phantoms dataset --verify`: 9/9 dosya her kontrolü geçti** —
  format tag'leri, ortak şekil/dx, ön yüz z=20, meme merkezi ±1 voxel, su dolgusu
  değerleri, sınıf-başına özellik bantları (pval kontratı: her voxel kendi medya
  numarasının [lo, hi] bandında, NaN yok), öksüz dosya yok.
- pval gerçekten dokuyu değiştiriyor: doğrulayıcı pval'li sınıflarda sınıf-içi std > 0
  şartını her dosyada sağladı.

### Açık uçlar
- Dosyalar eski (yavaş) kayıt yoluyla yazıldı; içerik doğrulandı, yeniden üretim gerekmiyor —
  ama bir sonraki tam rebuild ~2× hızlı olur (deflate seviye 1).
  *(Sonraki oturumda M6e rebuild'i bunu doğruladı: fantom başına 492 s → 50–92 s.)*
- Manifest'teki fantom girdileri eski hizalama sözlüğünü taşıyor (breast_center_vox alanı
  sonraki rebuild'de eklenir); verify diskten yeniden ölçtüğü için kontrat etkilenmiyor.
- Colab'a taşımada 5.66 GB'lık `data/phantoms/` Drive üzerinden taşınabilir ya da Colab'da
  `python -m uwcem_phantoms dataset` ile (~1 saat) yeniden üretilebilir.
  *(M6e ile grid 100 mm'ye tavanlandı: 4.2 GB, ~13 dk — aşağıdaki oturum.)*

---

## 2026-08-19 — Oturum 9 (Opus): dataset denetim raporu + M6e derinlik tavanı (100 mm)

### Bölüm 1 — üretilen datasetlerin bağımsız denetimi (kullanıcı talebi, geçici)

"Ürettiğin datasetleri kontrol et, rapor oluştur, 3D ve kesitlerle" — ve **rapora ait hiçbir
kod repoda kalmayacak**. Denetim `uwcem_phantoms`'tan HİÇBİR ŞEY import etmeden yapıldı: düz
`numpy.load` ve her beklenti dosyanın KENDİ `meta.materials_table` kaydından yeniden türetildi,
böylece bir hata onu üreten mantığın aynısıyla değil dışarıdan yakalanabilecekti. Rapor
artifact olarak yayımlandı; bütün script'ler scratchpad'de kaldı (`git status` temiz).

En güçlü sonuç — **ortak-p kontratı**: p, c/ρ/α/β'nın her birinden AYRI AYRI geri hesaplandı ve
dördü birbirini 4·10⁻⁶ içinde tutuyor (denetim ajanı aynı testi iki dosyanın 236 milyon
voxel'inin tamamında koşturdu: 9.6·10⁻⁸). Özellik başına ayrı p ya da birleştirilmiş band
üzerinden blend olsaydı bu mertebelerce patlardı. "pval'lar düzgün hesaplandı" iddiasının
kanıtı budur; bant-içi-aralık istatistiği değil.

Adversarial tur (31 ajan, 26 aday bulgu, 14'ü ayakta kaldı). Kritik iddiaları kendim yeniden
ölçüp İKİSİNİ daralttım: (a) "merkez 16 mm kayık" abartılıydı — bbox ±0.25 mm, kütle merkezi
≤8.07 mm; (b) "5 mm ön pay PML'i odağı bozar" — payı belirleyen kesiti kaplayan slab kenarı,
çıkıntı yapan meme hiçbir fantomda 5.75 mm'nin altında değil. (c) p=1'de birikmenin (doku
voxel'lerinin %22–37'si) boru hattımızdan mı geldiğini HAM arşivden kontrol ettim: 062204'ün
işlenmemiş pval dosyasında zaten %36.9 — kaynağın karakteri, sadakatle taşınmış.

### Bölüm 2 — M6e: derinlik tavanı (önce 100, sonra 120 mm)

Kullanıcı: "göğüs duvarının ilerisi boş; o boşluğu almak ya da saklamak istemiyorum — hepsi
yine aynı ızgarada kalsın, en küçük memede bilgi kaybı olmasın; derinlik sınırını 100 mm yap."

Önce ölçtüm (dokuz dosyanın z-başına sınıf histogramı):

| | z |
|---|---|
| ön su payı | 0 → 5.00 mm (her dosyada) |
| çıkıntı yapan meme biter (kesit kaplaması < %90) | 83.0 – 132.0 mm |
| kesiti TAMAMEN kaplayan göğüs duvarı slab'ı | 21.5 mm (65 voxel yağ + 21 voxel kas) |
| doku biter | 104.5 – 153.5 mm |
| arkada saf su | 2.75 – 51.75 mm |

Yani 156.25 mm'lik eksenin arka yarısı su + düz slab. 100 mm tavan bunu atıyor ama **yıkıcı**:
en sığ fantom (012304, doku 104.5 mm'de biter) SIFIR meme dokusu kaybediyor — kullanıcının
"en küçük memede bilgi kaybı olmayacak" beklentisi birebir tutuyor — buna karşılık derin
fantomlar kas dışı dokularının %27–49'unu kaybediyor. Kaybın büyük kısmı göğüs duvarı yağ
slab'ı (Fatty-1/2), meme değil. Bu bilinçli bir karar olduğu için kod bunu SESSİZ yapmıyor.

- `depth_limit_mm` (varsayılan 100 mm, `--depth`, `0` = tavansız) plan → build → CLI → launcher.
  Tavan TAVAN olduğu için z `prev_fft_friendly` ile aşağı yuvarlanır (yeni yardımcı; diğer iki
  eksen hâlâ yukarı) — istenen mm asla aşılmaz. 100/0.25 = 400 zaten 2/3/5/7-düzgün.
- `_align_into_common` arkadan kesiyor ve kesileni sınıf sınıf sayıyor:
  `back_trim_vox/_mm`, `truncated_tissue_vox`, `truncated_by_class`, `tissue_at_back_face`
  → manifest + `asset.meta["warnings"]` + build günlüğünde fantom başına WARNING satırı.
  `--dry-run` maliyeti build'den ÖNCE fantom fantom yazıyor.
- **Tavan yokken** aynı taşma survey hatası sayılıp reddediliyor — sessiz kırpma yolu yok.
- `DATASET_FORMAT` /1 → /2 ve `depth_limit_mm` tarif anahtarlarına eklendi: eski (156 mm) ve
  yeni (100 mm) dosyalar aynı dizinde karışamaz, dosya adı sadece dx kodladığı için.
- `verify` iki yeni kapı: grid ilan ettiği tavanı gerçekten sağlıyor mu; ve kesim kaydı dosyayla
  çelişiyor mu — "kesildi" diyen fantomun SON z düzleminde doku olmak ZORUNDA (gövde z boyunca
  bitişik). Kesilen doku geri getirilemez ama iddia tek yönde yanlışlanabilir.

**Ara bulgu (kendi hatam, dry-run ile yakalandı):** transvers merkez survey'de TÜM hacimden
ölçülüyordu; meme tabana doğru genişlediği için atılacak dilimler hem merkezi hem gerekli
yarı-genişliği yanlış veriyordu. Survey ve build artık ikisi de KALAN dilim üzerinden ölçüyor;
düzeltme ortak kutuyu x'te 540 → 560 büyüttü. İlk (yanlış) rebuild bu yüzden iptal edilip
baştan koşuldu.

### Üretim ve kesim maliyeti (manifest'ten, fantom başına)

**560×700×400 @ 0.25 mm = 140×175×100 mm, 156.8 Mvox**; 9 dosya toplam **4.2 GB**
(önce 5.3 GB), ~13 dk (fantom başına 50–92 s).

| fantom | ACR | kesilen | kesilen doku | kas dışı | kas dışının %'si |
|---|---|---|---|---|---|
| 071904 | 1 | 53.50 mm | 45.5 M | 39.9 M | %47.95 |
| 012804 | 1 | 29.50 mm | 35.3 M | 28.3 M | %33.82 |
| 012204 | 2 | 46.00 mm | 48.6 M | 41.6 M | %48.65 |
| 070604PA1 | 2 | 47.00 mm | 47.2 M | 40.5 M | %48.38 |
| 010204 | 2 | 22.50 mm | 16.8 M | 12.8 M | %26.95 |
| 080304 | 3 | 28.00 mm | 19.9 M | 15.8 M | %31.21 |
| 070604PA2 | 3 | 25.00 mm | 25.7 M | 20.0 M | %32.90 |
| 062204 | 3 | 6.50 mm | 4.8 M | 0.9 M | %2.69 |
| 012304 | 4 | 4.50 mm | 3.5 M | 0 | **%0.00** |

En sığ fantom (012304) yalnız kas slab'ını kaybediyor — meme dokusundan tek voxel gitmiyor.
Kesilenin ağırlığı Fatty-1/2, yani göğüs duvarı yağ slab'ı.

**Çapraz doğrulama:** kesim sayılarını build'den ÖNCE, ESKİ 625 derinlikli dosyalardan z>400
histogramıyla bağımsız hesaplamıştım; build'in kendi `bincount` sayımı dokuzunda da BİREBİR
aynı çıktı (45,544,884 / 35,317,288 / …). İki ayrı kod yolu, aynı voxel sayısı.

Doğrulama: `dataset --verify` 9/9; verify'ın diskten kendi ölçümü dokuzunda da şekil
560×700×400, ön yüz z=20, arka yüz z=400 (doku SON düzleme değiyor), meme merkezi kutu
merkezinden en fazla 0.5 voxel sapma.

### Bölüm 3 — "simülasyon hazır mı?" ve ön payın 20 mm'ye çıkması

Zinciri konuşmak yerine koşturdum: 012304'ü 0.8 mm'de kurup (1 MHz'te builder 1.8 ppw diye
HAKLI OLARAK reddetti; f0'ı 0.4 MHz'e indirdim → 4.45 ppw), 6.4 mm PML, F/1.5 ROC 54.6 mm
çanak, apex 8.8 mm, kabuk 11.9 mm'de bitiyor (deri 38.4 mm) → **264 adım / 76 s, period 22'de
yakınsadı, tepe 0.701 MPa, 2. harmonik 0.025 MPa.** Odak geometrik konumdan **10.4 mm öne**
kaydı: yağ (1440–1475 m/s) sudan yavaş, yakınsayan hüzme arayüzde kırılıyor — heterojenlik
gerçekten iş görüyor.

Planner, gerçek 0.25 mm gridi için: A100'e **sığıyor** (17.0 / 38.9 GiB, 2534 adım, datasheet
tahmini ~3.5 dk; H100'de ~77 s).

**Ama transducer sığmıyordu.** Ön pay 5 mm ve tipik PML de 5 mm (0.25 mm'de 20 voxel), sünger
grid'in İÇİNDE → serbest su tam 0 mm. Odaklı bir çanak ayrıca kendi kabuğu için yer istiyor:

| dizilim | kabuk derinliği | gereken ön su |
|---|---|---|
| üretim spirali (128 el, ROC 100) | 11.6 mm | ≥ 18.6 mm |
| çanak F/1.0 ROC 60 | 8.0 mm | ≥ 15.0 mm |
| çanak F/1.5 ROC 60 | 3.4 mm | ≥ 10.4 mm |

Kullanıcı kararı: **ön pay 20 mm, tavan 120 mm** (doku kapsamı 100 mm'de sabit).
`FRONT_GAP_MM` 5 → 20, `DEPTH_LIMIT_MM` 100 → 120, grid **560×700×480 = 140×175×120 mm,
188.2 Mvox**, 9 dosya 4.5 GB, ~13 dk. Tavan artık 9/9 değil 8/9 fantomu kesiyor:

| fantom | kesilen | kesilen doku | kas dışı |
|---|---|---|---|
| 071904 | 48.50 mm | 42.1 M | 36.4 M |
| 070604PA1 | 42.00 mm | 43.4 M | 36.7 M |
| 012204 | 41.00 mm | 44.8 M | 37.9 M |
| 012804 | 24.50 mm | 31.2 M | 24.2 M |
| 080304 | 23.00 mm | 17.6 M | 13.5 M |
| 070604PA2 | 20.00 mm | 21.6 M | 16.0 M |
| 010204 | 17.50 mm | 13.4 M | 9.3 M |
| 062204 | 1.50 mm | 1.1 M | **0** |
| 012304 | 0.00 mm | **0** | **0** |

**Yerleştirme doğrulaması (gerçek dosyalar üzerinde):** 5 mm PML → 15 mm serbest su; deri ön
düzlemi z=80 (20.00 mm), eksende 23.00 mm. Üretim 128-elemanlı spirali apex z=22 (5.50 mm),
kabuk 17.13 mm'de bitiyor → deriyi 2.87 mm boşlukla geçiyor; 65,826 voxel, **128/128 eleman**
temsil ediliyor, `validate()` geçiyor, DAS ile 60 mm'ye yönlendirilen odak Transitional dokuda.
F/1.5 ROC 60 çanak da sığıyor (odak 65.5 mm, Fibroglandüler-1). Planner: A100'de **20.36 /
38.88 GiB, 2016 adım, ~3.4 dk**.

**Yolda bulunan sessiz hata:** yerel k-space çözücüsü, tamamı süngerin içinde kalan bir kaynağı
reddetmiyordu — koşu hata vermeden yakınsıyor ve sessizce yanlış alan döndürüyordu (k-Wave
adaptöründe kontrol vardı, yerel yolda yoktu; 5 mm su + 5 mm PML tam da bu tuzağı kuruyordu).
`SolverBase.validate` → `check_source_clears_pml`. İlk versiyonum FAZLA sıkıydı ve dört testi
kırdı: tam genişlikli bir düzlem kaynağın yanal uçları zaten yanal PML'e girer, o normal
düzlem-dalga kurulumu. Doğru kural: **hiçbir voxeli süngerin dışında değilse** reddet. Ayrıca
sünger tüm domaini kaplayacak kadar küçük gridlerde kontrol devre dışı (engine'in kendi
convergence-region mantığıyla aynı gerekçe).

### Açık uçlar / bilinmesi gerekenler
- **Arka sınır dokunun İÇİNDE** (8/9 fantomda). Buradaki PML yarı-sonsuz gövdeyi kesmenin
  standart yolu; eskiden orada duran şey (kusursuz düz kas/su düzlemi) zaten kaynak verinin
  modeliydi, fizik değil. Ama geri yansıyan alanla çalışıyorsanız göğüs duvarı yansıtıcısı YOK.
- `apps/focus_study`'de **fantom senaryosu yok** (water_bowl / spiral_array / layered_tissue).
  Fantom koşusunun glue'u şu an elde yazılıyor; senaryo haline getirilmeli.
- **cupy bu makinede kurulu değil.** Çözücü döngüsü tamamen backend-generic (`xp` +
  `backend.fft`), yani Colab'da cupy kurulur kurulmaz koşmalı; M7 kernel füzyonu (hız) —
  ilk GPU koşusu planner'ın datasheet tahmininden yavaş olacaktır.
- alpha hâlâ 1 MHz'e gömülü; başka f0 için `--f0` verip yeniden üretin.
- Tam suite yeşil (296 test); `ruff check src uwcem_phantoms apps tests` temiz.

### Bölüm 4 — geometri raporu (4 dizilim × 9 fantom) ve M6f: depolanmış kurulumlar

Kullanıcı transducer + PML geometrisi için bol resimli bir rapor istedi: 64 elemanlı dört
Arşimet spirali, boyut ve eğrilik farklarıyla, dokuz fantomun hepsinde. Tasarım matrisi
2×2 kuruldu — açıklık (60/100 mm) × eğrilik (ROC 60–150) — böylece her fark iki eksene
atfedilebilir. 36 yerleşimin **hiçbirinde çakışma yok**, en dar geçiş 9.25 mm.

Sezgiye aykırı bulgu: **geniş dizilimler daha rahat sığıyor.** Meme dışbükey olduğu için
büyük yarıçapta doku çok daha derinde; dar bir çanağın kenarı memenin en öne çıkan bölgesine
denk geliyor.

Asıl mühendislik sonucu: odağı deri + 25 mm'ye çekmek için gereken yönlendirme, ROC oranı
olarak S1'de %17–32, S3'te %50–59, S2'de %58–66, S4'te %67–73. Yani **bu domain için doğru
eğrilik ROC ≈ 60 mm**; uzun ROC'lu tasarımların doğal odağı domainin arkasında kalıyor.

Görselleştirmede iki karar kayda değer. (1) **Kabuk her yerde tek bir "alet rengi"nde (altın)**,
dizilim başına renkle değil: doku paleti zaten kırmızı-turuncuyu (deri), macentayı (kas),
camgöbeğini (fibroglandüler) ve moru (transitional) kullanıyor, dizilim rengi verseydim kabuk
bir doku sınıfı gibi okunurdu. Dizilim renkleri (Okabe–Ito türevi, iki yüzey için ayrı adım,
renk körlüğü doğrulayıcısından geçti) sayfada ve tissue içermeyen figürlerde yaşıyor.
(2) **Yan görünümde analitik ark çiziliyor**, sadece düzlemi kesen voxeller değil — bir spiral
herhangi bir düzlemi bir avuç yerde keser, ham kesit iki kopuk parça gösteriyordu.

Ayrıca kodda belgeli bir voxelizasyon artefaktı sayısallaştırıldı: `voxelize` disk testini xy
düzleminde yapıp z'yi sonradan kaydırdığı için eğik elemanın yaması ~1/cos(eğim) fazla alan
kaplıyor — 30° yarı-açılı S1/S3'te **+15.5%**. Yerleşimi değil, kenar elemanların göreli
kaynak gücünü etkiliyor.

### M6f — `uwcem_phantoms/setup.py`

Kullanıcı S1'i seçti ve dokuz geometriyi "simülasyona yüklemeye hazır" depolamayı istedi.
Sorulan dört karar: **JSON tarif + yükleyici** (npz pişirme değil), **apex dokuzunda sabit
z = 5.50 mm**, **doğal odak 65.50 mm — yönlendirme yok**, **9 kurulum**.

Tasarımın özü: dosya tarif, gerçek değil. Eleman konumları ve 23.283 kaynak voxeli yükleme
anında türetiliyor; dosyanın kaydettiği türetilmiş değerler (eleman yarıçapı, kabuk derinliği,
r_max, voxel sayısı, PML payı) bunlara karşı sınanıyor. Dizilim kurulumu kütüphanede değişirse
koşu sessizce başka bir transducer'la devam etmiyor — yükleme patlıyor ve nedenini söylüyor.

`build_setups` yazmayı REDDETTİĞİ durumlar: kabuk dokuya giriyorsa, odak suya düşüyorsa,
dataset'in pişirdiğinden farklı bir f0 isteniyorsa (alpha yanlış olurdu), ya da tek bir kaynak
voxeli süngerin içindeyse. Bu son kural çözücüdekinden **daha sıkı** ve bilinçli: genel kapı
kısmi örtüşmeye izin vermek zorunda (tam genişlikli düzlem kaynağın yanal uçları), depolanmış
bir kurulumun ise böyle bir mazereti yok. Bunu testte apex'i z=2'ye taşıyarak buldum — genel
kapı geçirdi, çünkü kabuğun bir kısmı bandın dışındaydı.

Kayıt bölgesi de kurulumda: hüzme kutusu (±35 mm yanal, sünger dışı eksen) 34.5 Mvox — tam
gridi kaydetmek harmonik başına 1.5 GB döndürürdü.

### Kanıt
- Dokuz kurulum yazıldı ve `setup --verify` 9/9 geçti; su yolu 16.00–24.75 mm, kabuk–deri
  geçişi 9.25–14.50 mm, odak dokuz fantomda da dokuda (sınıf 3/5/6/7), odak deri altı
  35.25–44.00 mm. Sayılar rapor için yapılan bağımsız ölçümle birebir aynı.
- `load_setup("s1-012304")` → `westervelt.validate()` OK; planner A100'de 20.36/38.88 GiB,
  1890 adım, ~3.1 dk.
- Tam suite yeşil (307 test, 11'i yeni setup testi); `ruff check` temiz.
- Scratchpad 591 MB → 7.8 MB (eski denetim turlarının ~200 tek kullanımlık script'i,
  `pref.npy` 309 MB, `ds04/` 160 MB, `cref.npy` 77 MB, eski ekran görüntüleri silindi).


---

## 2026-08-19 — Oturum 10 (Fable): Colab entegrasyon planı (a-yolu) MILESTONES'a işlendi

### Bağlam
Kullanıcının hedefi: lokalde ayarları yap → `job.json` Drive'a düşür → Colab'daki DEĞİŞMEYEN
notebook koşsun → çıktı Drive'a insin → lokalde sonuç görüntülensin. İki varyant soruldu
(a: GUI'siz, b: GUI ile); önce a-yolu, GUI hariç tam entegrasyona kadar planlandı. Kod
değişikliği YOK — yalnız MILESTONES.md yeniden yapılandırıldı.

### Kullanıcı kararları (soruldu, cevaplandı)
1. **Repo erişimi:** public'leşme M11'den öne çekildi → yeni **M10e** (Colab clone token'sız;
   `v0.1` tag M11'de kalıyor).
2. **Dataset staging:** kullanıcı ölçütü "H100'de < 5 dk ise yeniden üret". Ölçüm: 9 fantomun
   yerel üretimi ~8 dk (data/phantoms dosya damgaları 18:04→18:12) + indirme/decode; iş CPU'da
   (scipy resample), H100 GPU'su katkısız → eşik aşılıyor. Karar: **Drive birincil + checksum;
   fallback yerinde üretim** (kullanıcı isteğiyle fallback her koşulda duruyor).
3. **Kuyruk:** ayrı milestone (**M10g** — jobs/pending→running→done protokolü); ileride GUI'nin
   "Run in Colab" düğmesinin altyapısı.
4. **Job şeması:** TAM genişletme (**M10b**): sadece dokuz stored setup değil; scene (CSG),
   volume import ve serbest array reçetesi de job'dan tarif edilebilecek.

### MILESTONES.md değişiklikleri
- Faz Grubu C yeniden adlandı: "Veri, IO, Colab entegrasyonu (v1 çizgisi)" + omurga girişi:
  M10 → M10b → M10c → M10d (hepsi CPU) → M10e public → M10f Colab köprüsü → ilk Colab oturumu
  (M7+M8 kapılarıyla birleşik) → M10g kuyruk.
- **M10** genişledi: koşu-içi checkpoint eklendi (her N periyotta atomik alan durumu;
  checkpoint-resume kriteri: kesintisiz koşuya fazor rel < 1e-6).
- Yeni milestone'lar: **M10b** JobConfig (`hifusim-job/1`, medium/source tagged union'ları,
  override katmanı, `hifusim validate`), **M10c** runner (`hifusim run`: plan-önce, OOM reddi,
  status.json kalp atışı, `--dry-run/--resume/--max-hours`, ortam+planner damgası), **M10d**
  rapor+önizleme (`hifusim report`, ≤10 MB önizleme paketi, focus_study figür kodu ortak modüle),
  **M10e** public'leşme, **M10f** `hifusim.colab` + değişmeyen notebook (mantık repoda, notebook
  4–5 hücre; kontrat testi), **M10g** kuyruk.
- M11 v1 ön-etiketi güncellendi (public M10e'ye taşındı); M13'e runner+kuyruk bağı, M14'e
  "job akışıyla koşar" notu; üst nottaki GUI satırı tazelendi (GUI ayrıca planlanacak; M10b/M10d/
  M10g onun kontratları).
- "Sıradaki iş" güncellendi: sıra M10'dan başlıyor; ilk Colab oturumu tek seferde üç kapıyı
  kapatıyor (M7 parite/tam-boy + M8 VRAM ±%10 ve kalibre süre ±%25 + M10f E2E).

### Açık uçlar
- M10e isim kararı kullanıcıda (PyPI/GitHub çakışma kontrolüyle).
- M10 checkpoint'in N periyot varsayılanı ve VRAM/disk maliyeti M10 sırasında ölçülecek.
- Colab'ın önceden kurulu cupy sürümüyle `cupy-cuda12x>=13` uyumu M10f ortam kontrolünde ele
  alınacak (yeniden kurulum gerektirmeden kullanmak tercih).


---

## 2026-08-19 — Oturum 10 devamı (Fable): M10 kapandı — IO kontratı + atomik yazım + resume + koşu-içi checkpoint

### Yapılanlar
- **`src/hifusim/io/` paketi** (4 modül, ayrık kaygılar):
  - `atomic.py` — `atomic_write()` context manager (tmp → `os.replace`; istisnada tmp silinir,
    hedef ya eski ya tam) + `sweep_temp_debris()` (v12.2 dersi: .tmp cesetleri açılışta süpürülür).
  - `quantize.py` — `try_float16()`: alan kendi tepe değerine bölünüp float16'ya atılır, round-trip
    hatası ÖLÇÜLÜR; kontrat (vars. 1e-3·|peak|) aşılırsa float32'de kalır. Ölçülen hata dosyayla taşınır.
  - `store.py` — **`hifusim-result/1`** HDF5 kontratı: `input/` (kaynak voxelleri+fazlar+drive),
    `output/p_real_h{n}`, `p_imag_h{n}`, `p_max` (kuantizeli, her dataset'te scale/stored_dtype/
    quant_norm_err/reload tarifi), `convergence/history`; kökte format/sürüm/çözücü/backend/dt/spp/
    bölge/grid + **faz konvansiyonu ve absorpsiyon modeli attr'ları HER dosyada** (downstream
    sözleşmesi). amp/faz asla saklanmaz, hep türetilir. `ResultStore` = DriveResilientStore portu:
    doğrulanmış mkdir (FUSE mkdir yalanına karşı isdir+backoff), write-probe (yazılamayan klasör
    ŞİMDİ patlar), yazım anında yeniden mkdir doğrulaması (v12.3), resume skip-guard
    (`missing()`: liste bir kez okunur, yalnız su-çizgisi yapısal doğrulanır; `deep=True` hepsi).
  - `checkpoint.py` — koşu-içi durum: `CheckpointSpec(path, every_periods, stop_when)`,
    `RunInterrupted`, `CheckpointMismatch`; sıkıştırmasız atomik `.npz` (np.savez'in ".npy" uzantı
    oyununa karşı açık dosya tanıtıcısıyla).
- **Motor entegrasyonu** (`engine.py`): settle for-döngüsü while'a çevrildi (davranış birebir —
  yakınsamada prev_peak güncellenmez, aynı sıra); periyot sınırında `_period_boundary()` kancası
  (kadans yazımı + `stop_when` yoklaması); kayıt penceresi ÖNCESİ zorunlu "record" anlık görüntüsü
  (kayıt sırasında ölüm = pencereyi aynı durumdan birebir yeniden koşmak); başarıda checkpoint
  silinir. **Parmak izi**: scheme etiketi + çözücü/backend/grid/dx/pml/spp/f0/genlik/ramp/
  harmonikler/nonlineer/kayıt bölgesi/referans noktası/spec + kaynak-SHA1 + medium-SHA1 (c,rho,
  alpha,beta) — tutmayan checkpoint FARKLI ANAHTARLARI İSİMLEYEREK reddedilir. `linear`/
  `westervelt` `checkpoint=` alır; kwave adaptörü açıkça reddeder (harici binary durduramayız).
- h5py **lazy kaldı**: `hifusim/io/__init__` PEP 562 `__getattr__` ile store'u geç yükler;
  motor yalnız `io.checkpoint`'e (numpy-only) bağımlı → `import hifusim.solvers` h5py çekmez.

### Kanıt (M10 kriterleri)
- **Tam suite 330 test yeşil** (önceki 307 + 23 yeni: `test_io.py` 16, `test_checkpoint.py` 7);
  2:55 dk; `ruff check` + `ruff format` temiz.
- float16 round-trip ≤ 1e-3 ölçülüp doğrulandı; 1e-5 kontratı float32'ye düşürüyor (birebir eşit).
- Kill kapısı GERÇEK: subprocess yazımın ortasında `proc.kill()` (TerminateProcess) —
  hedef dosya YOK, .tmp cesedi var, süpürme temizliyor. Python cleanup'ı koşmadan test edildi.
- Resume kapısı: 10 dosyalık mini set, ortadaki silindi → `missing()` = yalnız o; sayaçla tek
  yeniden üretim kanıtlandı. Su-çizgisi bozulması (çöp bayt) yakalanıyor; derin tarama her yerde.
- **Checkpoint-resume: BİREBİR AYNI** (beklenen rel < 1e-6'dan güçlü): float32 durum sıkıştırmasız
  npz'den bit-kayıpsız döner, adım döngüsü deterministik → kesilen+devam eden koşunun fazor,
  p_max, TÜM harmonikler ve yakınsama geçmişi kesintisiz koşuyla `assert_array_equal` düzeyinde
  eşit. Kapsam: 1D linear, 2D westervelt (beta yolu + u-listesi), zincirleme iki kesinti,
  record-aşaması anlık görüntüsünden devam, parmak izi reddi (genlik ve medium değişimi).

### Notlar / açık uçlar
- `every_periods` varsayılanı 8; tam gridde bir checkpoint ≈ 4 × padded float32 hacim (~3 GB
  sınıfı) host'a iner — GPU'da maliyet M10f oturumunda ölçülüp kadans ona göre belgelenecek.
- `stop_when` kancası M10c `--max-hours`ın hazır temeli; kayıt penceresi içinde yoklanmıyor
  (2 periyot, bilinçli).
- Backend adı parmak izinde: numpy checkpoint'i cupy'de devam ETTİRİLMEZ (bit-eşitlik ancak aynı
  backend'de anlamlı; çapraz devam istenirse bilinçli bir karar olarak açılır).


---

## 2026-08-19 — Oturum 10 devamı (Fable): M10b kapandı — `hifusim-job/1` + `python -m hifusim validate`

### Yapılanlar
- **`src/hifusim/config/job.py`**: tek JSON = tam koşu. İki job kind'ı (`kind` ile ayrışık):
  `stored_setup` (data/setups referansı + override katmanı) ve `explicit` (tam ağaç).
- **TASARIM SAPMASI (bilinçli, MILESTONES'a not düşüldü)**: plan `stored_setup`'ı source
  union'ına koymuştu; JOB seviyesine alındı — depolanmış kurulum medium+grid+yerleşim+run'ı
  birlikte sabitler, onu source yapıp başka medium'la eşleştirmek M6f garantilerini kırardı.
- `medium` union: `phantom_dataset` (grid dosyadan; job'da grid bölümü YASAK — pml_mm tek seçim;
  odak sınıf 0/su'ya düşerse JobError) | `scene` (malzeme tablosu eksik etiketi kurulumda
  yakalar) | `volume_import` (tek-import'lu SceneConfig üzerinden aynı yerleştirme yolu) |
  `homogeneous`. `source`: `array` reçetesi — spiral (n_el/d_outer/d_inner/roc/af) | bowl
  (d_outer/roc; yarıküre üstü reddedilir; steer/faz isteği "single focused element" hatası).
  Odak: natural (apex+roc) | steered (DAS fazları, c0=1500 su varsayımı belgelendi) | açık faz
  listesi. Kullanıcı birimleri mm/MHz/kPa; her model extra=forbid + JSON round-trip.
- Override katmanı: amplitude/harmonics/run-policy/steer; **f0 override eşitse no-op, farklıysa
  alpha gerekçesiyle RED** (M6f override katmanından sağ çıkıyor). Steering voxel kümesine
  dokunmuyor (parite testi indeksleri birebir doğruluyor), yalnız fazlar + odak voxeli değişiyor;
  kaynak-PML kontrolü yeniden koşuyor.
- `check_derived()`: M6f "hiçbir şey pişirilmiyor" deseninin genellemesi — job çıktısına
  damgalanan türetilmiş geometri (elem_radius/shell_depth/r_max/f_number/half_angle) yeniden
  türetilip karşılaştırılıyor; oynanan değer İSİMLE reddediliyor. M10c runner damgası bunu kullanacak.
- **`python -m hifusim validate job.json [--fast]`** (`src/hifusim/__main__.py`): şema (typo),
  dosya varlığı, kaynak-PML, odak-dokuda (dataset işleri; steered stored-setup'ta etiketler
  yüklenip sınanıyor), çözücü yetenekleri (medium yüklüyse; yüklü değilse ERTELENDİĞİNİ söyleyen
  uyarı), harmonik başına ppw (medium'suz yollarda "approx. c_min" etiketiyle). Exit 0/2.
- Lazy sınırlar korundu: `hifusim.config.__init__` job'ı PEP 562 ile geç yüklüyor (geometry↔config
  döngüsü kırık); `uwcem_phantoms` yalnız iki kind'da ve import anında değil kullanım anında.

### Kanıt (M10b kriterleri)
- `tests/test_job.py` 35 test; **tam suite 365 yeşil**; ruff temiz.
- Round-trip: 12 düğüm parametrize + union + dump/load; typo üst VE iç içe seviyede hata.
- Parite: stored job == load_setup — grid/indeksler/fazlar/genlik/f0/ramp/spec/bölge/odak birebir.
- Scene smoke: bowl + fat-top sahnesi gerçek mini çözüm; odakta alan > 0.5·genlik; medium c
  min/max doğru (1450/1500).
- validate yakalıyor: typo, bozuk setup referansı, PML'e gömülü çanak, suya çekilen odak
  (dataset), bilinmeyen çözücü (kzk), lineer çözücüde beta≠0, --fast ertelme uyarısı.
- CLI gerçek çağrı: stored-setup job → exit 0 + özet; olmayan dosya → exit 2.

### Notlar / açık uçlar
- Stored-setup'ın ucuz yolunda c_min bilinmiyor → ppw "approx. c_min (1450)" etiketiyle; h2
  2.90 ppw uyarısı bu yaklaşıklıktan (gerçek dataset c_min'iyle ~3.0 sınırında). M10c koşu
  damgası gerçek medium'dan kesin ppw yazacak.
- Steering DAS fazları su yolu (c0=1500) varsayıyor — doku aberasyonu M23 planlama işi, job
  knob'u değil (docstring'de).
- `output.folder/quantize/max_norm_err` şemada hazır; M10d önizleme alanı gerektiğinde eklenecek
  (şema genişlemesi geriye uyumlu: yeni alan default'lu gelir).


---

## 2026-08-19 — Oturum 10 devamı (Fable): M10c runner kapandı + adversarial review turu (M10/M10b/M10c)

### M10c — `python -m hifusim run job.json`
- `src/hifusim/runner.py` (+ `__main__.py run` alt komutu): focus_study'nin kuralı korunur —
  planner konuşmadan pahalı hiçbir şey olmaz. Akış: yükle→kur→plan yazdır/kaydet→VRAM kapısı
  (cupy'de cihaz VRAM'i, testlerde `--vram-limit-gib`)→çöz→M10 store→damga. Çıktı düzeni
  deterministik ve JOB DOSYASINA göreli (CWD kayması resume'u bozamaz): job.json, plan.json/.txt,
  status.json, checkpoint.npz, result.h5, run_meta.json.
- Ayrık exit kodları: 0 (başarı/zaten-tam skip-guard) · 2 config · 3 OOM reddi · 4 çözücü/store ·
  5 kesildi-resumable. Kuyruk (M10g) metin ayrıştırmadan tepki verebilecek.
- Kalp atışı: `CheckpointSpec.stop_when` yoklaması periyot sayacı olarak kullanılır (motor
  değişikliği/adım başı maliyet YOK); ETA planner değil ölçülen kadanstan. `--max-hours` 0 dahi
  anlamlı (ilk periyot sınırında checkpoint + zarif duruş). Resume AÇIK istek: checkpoint varken
  `--resume`suz koşu exit 2; `--resume` var ama checkpoint yoksa yüksek sesli not.
- Damga: git commit + ortam (GPU adı/driver/cupy; numpy'da boş) + planner-vs-gerçekleşen
  (M8'in iki Colab kapısı bu dosyadan ölçülecek) + türetilmiş geometri (check_derived kontratı).
- Testler: `tests/test_runner.py` 14 test — dry-run hiçbir şey çözmez; exit kodları; uçtan uca
  damga; skip-guard mtime'ı bile değişmez; kesinti+resume BİREBİR; status alanları; CLI.
- Yakalanan gerçek bug (test yazarken): `--max-hours 0` truthiness yüzünden limiti kapatıyordu →
  `is not None`.

### Adversarial review turu — 5 boyut, 14 bulgu → 12 düzeltme + 1 belgelendi + 1 çürütüldü
Workflow'un doğrulama ayağı oturum limitine takıldı; 3 boyutun 14 bulgusu elle doğrulandı,
eksik 2 boyut (motor-checkpoint, job-config/steering) elle tarandı.
- **[HIGH] kwave job'ları hiç koşamıyordu**: runner her çözücüye `backend=` geçiyordu, kwave
  adaptörü bilinmeyen kwarg'ı reddeder → TypeError. `backend`/`checkpoint` artık yalnız native
  çözücülere gider; sahte-harici-çözücü testi eklendi (plan.json yok + run_meta.planner null).
- **[HIGH] explicit phantom_dataset yolu M6f f0-alpha korumasını ATLIYORDU**: stored yol reddeder,
  explicit yol 1.5 MHz'i sessizce 1.0 MHz alpha'sıyla koşardı. `_check_dataset_f0` asset
  meta'sındaki pişirilmiş f0'ı sürücüyle karşılaştırır; testli (JobError "alpha").
- **[HIGH] save_result çökmesi bitmiş çözümü yutuyordu**: motor checkpoint'i başarıda siliyordu,
  store sonra patlarsa geriye hiçbir şey kalmıyordu. `CheckpointSpec.keep_on_success` eklendi:
  runner checkpoint'i result GÜVENLE yazıldıktan sonra siler; store çökmesi exit 4 + "çözüm
  kayıp değil, --resume yalnız kayıt penceresini yineler" mesajı; testli (sahte OSError →
  resume → baseline'la birebir). Yazım öncesi ensure_dir_verified (v12.3) eklendi.
- **[MED] pre-solve delikler exit 1 sızdırıyordu** (bilinmeyen `--gpu`, plan yazım hatası) →
  config-try genişletildi, exit 2; testli.
- **[MED] outdir CWD'ye görelidi** → job dosyasının klasörüne göreli (Colab CWD kayması resume'u
  sıfırdan başlatamaz); `output.folder` göreli testi eklendi.
- **[MED] steered stored su-odağı reddi yalnız validate'teydi** (run kabul ederdi) → build'e
  taşındı (npz'den YALNIZ labels üyesi okunur, ucuz); run ve validate artık aynı fikirde; testli.
- **[MED] atomik tmp adı deterministikti**: iki oturak aynı adı paylaşıp torn dosyayı final ada
  terfi ettirebilirdi → yazar-benzersiz ad (pid+token). **[MED] süpürme canlı komşunun tmp'sini
  silebilirdi** → yalnız bayat (mtime>1 saat; testler 0 ile zorlar). **[LOW] os.replace Windows
  kilidi tam yazılmış tmp'yi sildiriyordu** → PermissionError'da retry, son çarede tmp KORUNUR
  ("preserved" mesajıyla); ikisi de testli. **[LOW] probe adı id() türeviydi** → pid+entropi.
- **[LOW] dataset job'da etiket kontrolü GB'lik to_medium'dan SONRAYDI** → yeniden sıralandı
  (tüm geometri redleri medium kurulumundan önce). **[LOW] tam-grid kayıt sessizdi** → validate
  özet satırı + >10 Mvox uyarısı; testli.
- **Belgelendi (kozmetik)**: heartbeat pre-record yoklamada periyodu +1 sayar — status telemetri,
  kesin sayaçlar checkpoint meta/run_meta'da (docstring'e yazıldı).
- **Çürütüldü**: "steering apex-çerçeve dönüşümü yanlış olabilir" — voxelize okundu: eleman
  voxeli = apex_vox + round(pos/dx), yarım-voxel kayması yok; `target_m - apex_vox*dx` doğru.

### Kanıt
- Tam suite 383 test yeşil; ruff temiz. Yeni bulgu-regresyon testleri: test_io +2 (replace-retry,
  bayat süpürme), test_job +3 (f0 guard, su-steer build reddi, tam-grid uyarısı), test_runner +4
  (kwave kwarg, bilinmeyen gpu, store-çökmesi kurtarma, job-göreli outdir).

## 2026-08-21 — Oturum 11 (Fable): M10d kapandı — `hifusim.report` + önizleme paketi + `hifusim report`

### Ne yapıldı
- **`hifusim.report` paketi** (yeni): figür/metrik/rapor kodu `apps/focus_study`'den kütüphaneye
  çıkarıldı. İçe aktarma disiplini `hifusim.io` ile aynı: `metrics` + `preview` numpy-only ve
  eager; `figures` (matplotlib) ve `run_report` (h5py) lazy — matplotlib'siz bir Colab/başsız
  ortamda runner önizleme yazabilir, yalnız `hifusim report` figür çizerken matplotlib ister
  (`pip install hifusim[report]`; dev extra'ya da eklendi).
- **Metrik tek doğruluk kaynağı**: `focus_metrics(result, dx, grid_shape, pml_vox, apex_vox,
  focus_vox, source_amplitude, medium, solver)` — peak/target/focal_spot/run/harmonics
  alt-ağaçları focus_study `analyze()` çıktısıyla ANAHTAR SIRASI dahil aynı; `analyze()` artık
  delege edip yalnız O'Neil çapraz kontrolünü ekliyor (senaryo bilgisi — aperture/ROC — result
  dosyasında yok, focus_study'de kalması doğru). Eksik girdiler dürüst düşer: medium yoksa
  isppa=None, amplitude yoksa gain=None, focus yoksa target bölümü yok.
- **Önizleme paketi** (`hifusim-preview/1`, `preview.npz` + `metrics.json`, atomik): tepe
  voxel'inden geçen 3 eksen dilimi (her harmonik + p_max), blok-ortalama kabalaştırılmış
  fundamental amp hacmi (dinamik float16 + scale, M10 kontratı), mm eksenleri, yakınsama
  geçmişi, meta_json. Kabalaştırma adımı bütçeden (varsayılan 10 MB'ın %60'ı hacme) hesaplanır
  ve paket YAYIMLANMADAN önce bellekte ölçülür — sığmazsa adım büyütülür (kriter tahmin değil
  ölçüm). Bilinçli sapma: milestone metni "orta dilimler" diyordu; gerçekleşen TEPE voxel'inden
  geçen dilimler seçildi (odak kaçtıysa orta dilim boş su gösterir) ve meta_json'a not düşüldü.
- **Runner entegrasyonu**: başarılı store'dan sonra `focus_metrics` (medium elde — isppa dolu)
  + `write_preview`; önizleme çökmesi koşuyu düşürmez (result zaten güvende, yalnız warning).
  result.h5 attrs'ına `apex_vox`/`focus_vox` damgası eklendi — rapor, job/medium olmadan
  apex-çerçeveli mm konumları üretebilsin.
- **`python -m hifusim report <out-dir>`**: result.h5 varsa tam figür seti (alan haritaları,
  profiller, harmonikler, yakınsama; kaynak noktaları h5'teki input grubundan) + REPORT.md +
  index.html; `--preview` ile (ya da result yokken) yalnız paketten quicklook figürü. Boş
  klasör exit 2. metrics.json varsa ona güvenir (runner medium'la hesapladı); yoksa h5'ten
  yeniden hesaplar ve isppa'nın neden olmadığını Caveats'e yazar.
- **focus_study ince adaptör**: `analysis` delege + O'Neil; `figures` Setup→FigureContext
  çevirisi; `report` ortak render (CSS/markdown/html iskeleti + Focus/Focal spot/Run/Harmonics
  satır üreticileri `hifusim.report.html`'de) üstüne Setup/Analytic/Planner satırları.

### Kriter kanıtları
- **Regresyon (bayt)**: water_bowl + layered_tissue (dx=0.6, min/max settle 2/6, --no-measure)
  refactor ÖNCESİ ve SONRASI koşuldu; REPORT.md ve index.html bayt-aynı (wall-time satırı
  normalize edilerek karşılaştırıldı — süre koşudan koşuya değişir), metrics.json'da tek fark
  `--out` yolunu içeren `command` alanı. Diff scripti scratchpad'de.
- **≤10 MB tam gridde**: 256³ sentetik alan (float16 ham hacim 32 MB olurdu) → paket ölçüldü,
  ≤ 10 MB, coarse_step>1; dilimler f16 kontratı içinde (atol 1.1e-3·peak) — testli.
- **Tek kaynak**: `test_focus_study_and_library_compute_identical_metrics` bölüm bölüm dict
  eşitliği; `test_metrics_from_result_file_match_the_runner_metrics` h5-roundtrip yakınlık
  (quantize=False job'da; isppa istisnası açıkça assert edilir). Anahtar-sırası koruması ayrı test.
- Uçtan uca duman: mini job (dx=0.5, westervelt h1+h2) → run → preview.npz 135 KB + metrics.json
  → `hifusim report` 4 figür + REPORT.md + index.html; `--preview` yalnız paketten render etti.

### Kanıt
- Tam suite **393 test yeşil** (383 + 10 yeni `tests/test_report.py`), ruff temiz.

## 2026-08-21 — Oturum 11 devamı (Fable): İSİM DEĞİŞTİ — hifusim → caustica (M10e ilerleme)

### Karar
- Kullanıcı önce "Kymata" istedi; PyPI'da `kymata` DOLU çıktı (Cambridge Kymata Atlas, v1.0.6,
  aktif; ayrıca ünlü `kymatio` wavelet kütüphanesiyle marka yakınlığı). İkinci turda kullanıcı
  **caustica**'yı seçti: PyPI'da BOŞ (simple index 404); GitHub'da aynı adlı en görünür proje bir
  Minecraft ray-tracer (Java, farklı ekosistem) — bizim ad alanımız `ebx0/caustica`.
- Ad fizikte de oturuyor: odaklanmış dalga alanlarının kostik (caustic) yüzeyleri.

### Ne yapıldı
- `git mv src/hifusim src/caustica` + kod ve YAŞAYAN dokümanlarda (src/tests/apps/
  uwcem_phantoms/scripts, pyproject, README, PLAN, MILESTONES, phantoms.bat/sh, .gitignore)
  hifusim→caustica / Hifusim→Caustica / HIFUSIM→CAUSTICA. Devlog ve benchmarks/reports
  TARİHÇE olarak bırakıldı (eski girdiler eski adla doğru).
- Format etiketleri de yeni ada geçti (`caustica-job/1`, `caustica-result/1`,
  `caustica-checkpoint/1`, `caustica-preview/1`, `caustica-setup/1`, `caustica-phantom/1`,
  `caustica-phantom-dataset/2`). **5.66 GB dataset REBUILD EDİLMEDİ**: okuyucular eski
  etiketleri belgeli legacy alias olarak kabul eder (`LEGACY_FORMAT_TAGS` asset.py,
  `LEGACY_DATASET_FORMATS` dataset.py+setup.py) — yeni yazımlar hep yeni etiketi yazar.
- İnce tuzak: setup üretimi `dataset_format`'ı SABİTTEN kaydediyordu; diskteki npz'ler legacy
  etiket taşıdığı için load_setup'ın "aynı build mi" kontrolü kırılırdı → setup artık dataset
  manifest'inin GERÇEK etiketini kaydeder. 9 setup yeniden üretildi (`caustica-setup/1`),
  `setup --verify` 9/9.
- Kalibrasyon dizini `~/.caustica/` oldu (eski `~/.hifusim` kalibrasyonu yetim kalır —
  yeniden kalibre etmek saniyeler). Env var `CAUSTICA_PHANTOM_DATA`. Editable kurulum
  yenilendi (`pip install -e .`), ruff format tüm pakete uygulandı (CI format kapısı).
- GitHub: repo ZATEN public'ti (baştan beri) — `gh repo rename` ile **ebx0/caustica** oldu
  (GitHub eski URL'den yönlendirir), origin güncellendi. README "working name" notu
  "hifusim'den yeniden adlandırıldı" notuna çevrildi; PyPI kontrolü: `caustica` boş.
- Yerel klasör adı `Desktop\hifusim` BİLEREK kaldı: .venv mutlak yolları ve oturum
  hafıza bağlaması klasör adına bağlı; istenirse ayrı bir adımda taşınır (venv yeniden kurulur).

### Kanıt
- Tam suite **393 test yeşil** (rename sonrası; aynı günkü janitor turu 402'ye çıkardı), ruff check + format temiz.
- M10e'nin kalan kriterleri commit+push bekliyor: public repoda CI yeşili + temiz ortamda
  `pip install git+https://github.com/ebx0/caustica` (push kullanıcı onayına bağlı — commit
  kuralı: kullanıcı istemeden commit atılmaz).

## 2026-08-21 — Oturum 11 devamı (Fable): Janitor turu #1 — 44 bulgu, 22 düzeltme, `janitor/` defteri

### Süreç
- Kullanıcı isteği: bakım işleri lokal bir klasörde ticket olarak tutulacak (yaz/sil),
  gelecek planları güncellenecek, eski hatalar test edilecek, gereken refactor'lar yapılacak.
- 7 boyutlu çok-ajanlı tarama başlatıldı (rename-kalıntıları, TODO/ölü-kod, doc-drift,
  paketleme/CI, test boşlukları, eski-hata→test haritası, refactor adayları) + boyut başına
  adversarial doğrulayıcı. Oturum limiti 3 tarayıcı + doğrulayıcıları düşürdü (M10c turundaki
  gibi) → eksik boyutlar elle tarandı, kullanılan bulgular elle doğrulandı. 4 tamamlanan
  boyuttan 44 bulgu geldi.

### En kritik düzeltme
- **Wheel `gpu_db.json` içermiyordu**: `pip install` (Colab'ın yapacağı şey) sonrası
  `planner.estimate` FileNotFoundError'la çökerdi. pyproject'e `[tool.setuptools.package-data]`
  eklendi; `pip wheel` ile paket kuruldu, wheel'den import edilip `planner.estimate` CANLI
  koşturularak doğrulandı (M10e "pip install çalışır" kriterinin ön şartıydı).

### Diğer düzeltmeler (hepsi testli ya da davranış-nötr refactor)
- `caustica report` bozuk/yarım-sync preview.npz'de raw traceback kusuyordu (Drive'ın NORMAL
  arıza modu) → geniş yakalama + temiz exit 2; `--preview` paket yokken yönlendiren mesaj;
  render_html boş rows'ta sahte `</table>`; run_report metrics.json atomik yazım.
- Tek-kaynaklaştırma: FIG_CAPTIONS (iki kopya → `report.html`), float16 geri-yükleme tarifi
  (5 site → `quantize.restore()`), figures._mm_axes (metrics.mm_axes'e delege — delegasyonda
  yakalanan ×1e3 çifte-ölçek hatası dahil), legacy alias koşulları (4 site → `ACCEPTED_*`).
- Runner: 35 satırlık inline önizleme bloğu `_write_preview_package` yardımcısına; apex_vox
  tek kez türetiliyor.
- focus_study: analysis/figures ölü re-export'ları kırpıldı (tüketici analizi: cli yalnız
  region_origin/analyze/profiles + make_all kullanıyor), report.py çift import bloğu birleşti.
- Dokümanlar: README'ye CLI bölümü (`validate|run|report`) + Layout'a io//report//runner +
  extras listesi; apps/README.md sed'den kaçan 5 hifusim düzeltildi; MILESTONES M0 notu ve
  PLAN.md'nin sed'le YANLIŞLANAN tarihsel ad satırları gerçek tarihçeye döndürüldü (editör
  notuyla); uwcem `plan` üst-seviye export (README'nin vaat ettiği import çalışmıyordu).
- Paketleme/CI: SPDX lisans (`license = "MIT"` + license-files, setuptools>=77), ruff pini
  (>=0.16,<0.17 — format kapısı sürüm atlamasında kırılmasın), CI'ya ubuntu-3.10 taban ayağı
  (requires-python >=3.10 hiç test edilmiyordu; 3.11+ sözdizimi taraması temiz), `runs/`
  .gitignore'a, bayat `src/hifusim.egg-info` silindi.

### Eski hatalar
- `janitor/eski-hatalar-haritasi.md`: 2026-08-11 fizik+geometri, 2026-08-19 dataset + M10c,
  2026-08-21 turlarının HER bulgusu → bugünkü koruyucu test eşlemesi. Korumasız üç kalem
  açıkça işaretli (probe entropisi, etiket-kontrol sıralaması, heartbeat ±1 belgeli).
  Bu turda üç tarihsel bulguya İLK koruma testi eklendi: periyodik-sınır uyarısı
  (2026-08-10 footgun), tmp yazar-benzersiz adlar (M10c), önizleme-çökmesi sözleşmesi (M10d).
- Yürütülen kanıt: tam suite 402 test yeşil (393 + 9 yeni: rapor 6, runner 1, io 1, pml 1),
  ruff check + format temiz.

### Janitor defteri
- `janitor/` (gitignore'lu, lokal): 00-durum özeti, 7 açık ticket (en önemlisi 06 —
  push sonrası M10e kapanış kontrolleri), eski-hatalar haritası. Kural: bir dosya = bir iş,
  bitince sil.

## 2026-08-21 — Oturum 12 (Fable): M10h kapandı — paketleme + temiz ortam kapısı (devir paketi 1/4)

### Bağlam
- Kütüphane-önce devir paketi başladı (docs/library_first_plan.md; sıra W1→W2→W0→W5 =
  M10h→M10i→M10k→M10m). Dal: `library-first`; ilk commit çalışma ağacındaki commit'lenmemiş
  M6c–M10e durumunu (hifusim→caustica rename dahil) olduğu gibi sabitledi (`ed9c7a4`) —
  master ref'ine dokunulmadı, push yok.

### Yapılanlar (W1)
- `[project.scripts] caustica`, `py.typed` + package-data, `caustica.examples` paketi
  (`available/path/copy`) + `water_bowl_mini.json`, CLI `example` alt komutu (kopyalar,
  üzerine yazmayı reddeder, adsız listeler), README Quickstart (`[report]` extra satırıyla),
  CI'ya `wheel` temiz-venv işi + `network` markası. `prog` "caustica" oldu.
- Örnek şablondan tek bilinçli sapma: dx 0.75 → 0.5 mm. Şablon dx'i ppw 2.00'da
  "under-resolved" uyarısı üretiyordu; quickstart'ın ilk çıktısı uyarı olamaz.
  dx=0.5 → ppw 3.00, uyarısız; çözüm 0.2 sn (≤30 sn kapısının çok altında).
- `setuptools>=77` dev extra'ya girdi (runtime DEĞİL): test_packaging wheel'i
  `--no-build-isolation` ile kurup içeriği ağsız sabitleyebilsin diye.

### Kapı kanıtları
- Wheel içeriği: `tests/test_packaging.py` (8 test) — py.typed, gpu_db.json, örnek job,
  console-script entry point, yan-paket (uwcem_phantoms/apps/tests) sızıntısı yok.
- Yerinde-koşturmama (T4): kurulum dizini içerik anlık-görüntüsü koşu öncesi/sonrası birebir;
  çıktı kopyanın yanındaki `runs/`e düşüyor; süre < 30 sn assert'lü.
- Temiz ortam provası (yerel, CI wheel ayağının birebir adımları): `pip wheel` →
  scratchpad'de taze venv → repo DIŞINDAN `import caustica`, `caustica --version`,
  `example --to`, `validate` (OK, uyarısız), `run --dry-run`, ardından TAM koşu — 1.5 sn'de
  sekiz dosyalık çıktı kontratı (matplotlib'siz venv'de preview.npz dahil — numpy-only yol).
  CI yeşili push'ta görülecek (push kullanıcı onayı bekliyor, janitor/06).

### Bilerek yapılmayanlar
- CI matrisi genişletilmedi (D14); yeni runtime bağımlılığı yok; `data/setups` örnek olarak
  kullanılmadı (fantom referanslı, D13 gereği sentetik örnek).

### Mutasyon denetimi sonrası düzeltme (aynı oturum, kullanıcı denetimi üzerine)
- Kullanıcı "package-data satırını sil, test kırmızı olmalı" provasını istedi. İLK sonuç:
  üç mutasyon da YEŞİL kaldı — test ısırmıyordu. Kök neden iki katmanlı:
  (1) `py.typed` için setuptools ≥69 dosyayı package-data'sız da otomatik paketliyor
  (v69 "Include type information by default") — o satır emniyet kemeri, mutasyonu
  görünmez kılan bu; (2) `gpu_db.json`/örnek JSON içinse repodaki BAYAT `build/lib`
  kalıntısı: setuptools önceki build'in kopyalarını wheel'e taşımaya devam ediyor,
  pyproject mutasyonu wheel içeriğini değiştirmiyordu. Yani test "bayat kopyalı"
  wheel'i doğruluyordu — tam da planın uyardığı maskeleme sınıfı, yeni kostümle.
- Düzeltmeler: wheel fixture'ı artık PRISTINE geçici kaynak kopyasından build ediyor
  (pyproject+README+LICENSE+src, `__pycache__`/egg-info hariç); bayat `build/` silindi;
  yeni test `test_wheel_ships_no_file_absent_from_src` — wheel'de src'de olmayan dosya
  = hayalet (M10k'daki silmelerde hortlama bununla yakalanır); `_dir_snapshot`
  (yol, boyut, mtime_ns) üçlüsüne güçlendirildi (yerinde üzerine yazma da yakalanır).
- Düzeltme SONRASI mutasyon protokolü: gpu_db satırı sil → KIRMIZI; examples satırı
  sil → KIRMIZI; örnek dosyayı kaldır → KIRMIZI; geri al → 9/9 yeşil. Eski prova
  wheel'i hayalet içermiyordu (tehlike gizildi, gerçekleşmemişti); prova temiz
  wheel'le tazelendi (validate ppw 3.00 uyarısız, koşu 0.2 sn).

### Adversarial review turu (M10h, 14 ajan: 3 mercek + bulgu başına şüpheci)
- 11 aday bulgu → 7 doğrulandı, 4 çürütüldü. Doğrulananlardan CRITICAL olan (bayat
  `build/lib` → wheel-içerik testi gerilemiş wheel'de yeşil kalır; ajan `STALE_MARKER.txt`
  ekiyle deneysel kanıtladı) kullanıcı mutasyon denetimimin bağımsız teyidi — yukarıdaki
  fixture düzeltmesiyle kapanmıştı. Kalan doğrulanmışlar da düzeltildi:
  - CLI `example` hata yolu: `KeyError` repr tırnakları temizlendi (`exc.args[0]`);
    `except` OSError'a genişledi (POSIX'te `--to mevcut-dosya/alt` → NotADirectoryError
    raw traceback veriyordu; WSL'de repro edildi, artık temiz exit 2)
  - Wheel fixture'ı build hatasında pip/setuptools stderr'ini yutuyordu → returncode
    kontrolü + stderr'li RuntimeError
  - "Kurulum dizinine yazma yok" kapısı: snapshot yalnız examples/ adlarını topluyordu →
    TÜM paket kökü, (yol, boyut, mtime_ns), `__pycache__` hariç; docstring sınırı
    (create-then-delete geçici yazım snapshot'la yakalanamaz) açıkça yazıldı
  - Örnek tam ppw=3.0 eşiğinde ve hiçbir test uyarısızlığı sabitlemiyordu →
    `test_packaged_example_validates_clean`: `report.warnings == []` assert'i (dx bozulursa
    quickstart'ın uyarıyla açılması artık test kırar)
- Çürütülenler (kayıt): CI wheel ayağının yeşili push öncesi gözlenemez (×2 — milestone
  metni yerel provayı kanıt olarak açıkça kabul ediyor, push kullanıcı onayında);
  `network` markası şu an hiçbir teste takılı değil (plan metni tam olarak bunu istiyor —
  marka + CI dışlaması; ağa çıkan testler W0'da uwcem repo'suna gidiyor); README quickstart
  satırları test-pinli değil (W1 kabulü bunu kapsam dışı bırakıyor; [report] satırı var).

## 2026-08-22 — Oturum 13 (Fable): M10h borcu kapandı (CI yeşil) + M10i başladı

### M10h borcu
- Kullanıcı raporu kabul etti ama milestone'u `[~]`'ye çevirdi: "CI'da yeşil" ölçütü CI'da
  hiç koşmadan işaretlenemez. Ders kayda geçti: doğrulanmamış ölçüt = `[~]`, istisna açık yazılır.
- Push (kullanıcı onayı) + draft PR #1 (library-first → master, merge YOK — pull_request
  tetikleyicisi CI'ı koşturuyor). İlk tur: wheel + windows YEŞİL, iki ubuntu ayağı KIRMIZI —
  taşınan M6c kodunda platform hatası: `x\y` export adı POSIX'te meşru dosya adı, "plain
  filename" korkuluğu tetiklenmiyor (kod ilk kez Linux görüyor). Korkuluk her platformda `\`
  reddedecek şekilde düzeltildi (`efc86dd`). İkinci tur **4/4 YEŞİL**
  (run 32529033382; wheel 21 sn). Örnek eşikten çekildi: f0 1.0→0.8 MHz, ppw 3.75 (`60b73c1`).

### M10i adım 1 — workers (D32): ölçüm karara üstün geldi, varsayılan 1
- Tesisat: `Backend.fft` numpy yolunda `_ScipyFFTWithWorkers` (tek nokta; çözücü sıcak
  döngüsünde dallanma yok); `cupyx.scipy.fft`'de `workers` parametresi YOK (CuPy stable
  docs'tan doğrulandı: `rfftn(x, s, axes, norm, overwrite_x, *, plan)`) — cupy yolu ham modül.
  Çözünürlük: `set_cpu_fft_workers()` > `CAUSTICA_CPU_WORKERS` > 1.
- Ölçüm ortamı (rapor şartı): tam boşta DEĞİL — kullanıcının Edge/dwm süreçlerinden %14–22
  arka plan yükü (16 mantıksal CPU'nun ~2-3'ü); benim süreçlerimden koşan yoktu (CI izleyicisi
  yalnız ağ+bekleme). i5-13450HX, 10 çekirdek (6P+4E), 16 mantıksal; scipy 1.15.3.
- Tarama (medyan adım ms; motor adım karışımı `measure_step_time` ile):
  | şekil | w=1 | w=2 | w=4 | w=8 | w=16 | w=-1 |
  |---|---|---|---|---|---|---|
  | 96³ | 51.2 | 50.3 | 70.2 | 68.6 | 70.7 | 70.6 |
  | 64×80×100 | 39.0 | 39.1 | 40.2 | 38.7 | 37.3 | 38.6 |
  | 240×300×360 | 2369 | 2410 | 1600 | 1861 | 2396 | 2388 |
- Teyit turu iki "sinyali" de ÇÜRÜTTÜ: 96³ w=-1 → 51.8 ms (regresyon yok; ilk turdaki 70 ms
  gürültüymüş), 240×300×360 w=4 → 2322 ms (1.48× kazanç tekrarlanmadı; w=1 aynı anda 2250 ms).
  İki tur birlikte: hiçbir worker sayısı KARARLI kazanç vermiyor (bellek bantgenişliği sınırlı
  + hibrit çekirdek + arka plan gürültüsü). Kullanıcının kendi ölçümü de aynı yönde.
- KARAR (kullanıcı yetkisi "ölçüm karara üstün gelir"): varsayılan **workers=1**; isteyen
  `CAUSTICA_CPU_WORKERS` / `set_cpu_fft_workers` ile açar. D32'nin "-1 varsayılan" kararı
  MILESTONES'ta ölçüm referansıyla düzeltildi. Hızlanma sayısı: **1.00× (tekrarlanabilir
  kazanç yok)** — beklenen değil, ölçülen.
- Bit-aynılık kapısı (kullanıcı ölçümüyle uyumlu, SIKI): w=1 vs w=-1 fazor alanları
  `assert_array_equal` ile BİREBİR — linear 3D + westervelt 3D mini çözümlerde (testli;
  pocketfft 1-D dilimleri thread'lere dağıtır, toplama sırası değişmez).

### M10i adım 2 — planner CPU kalibrasyonu
- Önceki durum: `~/.caustica/calibration.json` HİÇ YOKTU (bu makine hiç kalibre edilmemiş).
- Kalibrasyon yükü: %13.7–18.9 (öncesinde ölçüldü); workers=1 (yeni varsayılan) ile koştu.
- Yeni cpu girdisi: **a = 2.831e-09, b = 0.0** (48³: 4.12 ms, 72³: 19.86 ms). Sağlama:
  model 240×300×360 için ~1.8 s öngörüyor, ölçülen 2.25-2.37 s (~%20 altında — iki küçük
  şekilli fit için makul; eşik 5 dk ölçeğinde bu sapma karar değiştirmez).

### M10i adım 3 — 5 dk CPU kapısı (`d2f3c75`)
- VRAM reddinin hemen ardında, yalnız native+numpy: eşik `CAUSTICA_CPU_LIMIT_MIN` (vars. 5 dk),
  aşımda EXIT_CONFIG(2) — yeni çıkış kodu yok; mesaj tahmin + `est.source` + iki kaçış
  (`--backend cupy`, `--allow-slow-cpu`); kaçış aynı commit'te. Eşik altı: koşu başına TAM BİR
  `CausticaWarning`. `--dry-run` da kapılı (VRAM reddiyle aynı semantik).
- Tasarım düzeltmesi (planın harfinden sapma, gerekçeli): `--no-measure` yolunda `est` GPU
  datasheet sayısıdır — kapı ona kurulsaydı 10 saatlik CPU işi sessizce geçerdi. Kapı bu yolda
  kalibre cpu girdisinden yeniden ölçekliyor; ikisi de yoksa "yargılayamıyorum" uyarısı.
- KANIT KOŞUSU (tam boy 560×700×480 @ 0.25 mm homojen, numpy, --no-measure): exit 2,
  "estimated wall time ... ~9509 s (~2.6 h, estimate source: calibrated), over the 5 min CPU
  limit". Aynı işte A100 db tahmini 58.7 sn — GPU sayısına kurulan kapının neden anlamsız
  olduğunun canlı kanıtı.

### M10i kalan kalemler (`dc0747f`, `25bfc65`, `c60945b`, `74212bc`)
- `caustica.env`: `env_report()` (damga anahtarları korunarak GENİŞLETİLDİ: + scipy/pydantic/
  h5py/resolved_backend; asla raise etmez) + `require_gpu()` (Colab: Runtime menüsü, pip'siz;
  yerel: `pip install cupy-cuda12x`; pip çağrısı YOK). Runner damgası aynı fonksiyondan geçiyor.
- `CausticaWarning(UserWarning)` public kategori; backend auto→numpy düşüşü süreç başına BİR
  uyarı (eski INFO handler'sızdı, kimse görmüyordu); CLI `run` girişte logging açıyor,
  kütüphane import'ta handler kurmuyor.
- Taze CLI sürecinde CPU koşusu İKİ uyarı gösterir: 1 backend düşüşü (süreç başına) + 1 kapı
  bildirimi (koşu başına) — iki ayrı ölçütün bileşimi, testte belgeli.
- VRAM reddi boş VRAM'e bakıyor (`vram_free_gib`, yoksa toplam + etiketli); mesaj hangi sınırı
  kullandığını söylüyor. Sahte-GPU monkeypatch testleri: toplam 40 GiB "sığar" derken boş
  0.001 GiB reddettiriyor.
- Düşük ppw dört yerde: `low_ppw_warnings()` tek kaynak; plan.txt/json + status.json (her kalp
  atışı) + run_meta.json + rapor BAŞI (⚠ banner, full+preview). Dört test. mini_job (ppw 2.0)
  artık koşuda görünür uyarı üretiyor — bilinçli.
- Plan'a "expected result.h5 size: ~X MB" satırı (quantize-farkında); `--preview-only`
  (varsayılan değişmedi; preview-only modda preview yazımı FATAL — çıktının kendisi o).

### M10i hafif adversarial tur (tek mercek: "mevcut davranış sessizce bozuldu mu?")
- Bulucu ajan 5 bulgu üretti; şüpheci doğrulayıcılar oturum limitine takıldı → beşi de
  ELLE doğrulandı, beşi de GERÇEKti, beşi de düzeltildi (`57b9eed`):
  1. **CRITICAL — boş VRAM prob SONRASI okunuyordu:** measure probu cupy havuzunu koşunun
     kendi ayak izi kadar doldurur ve havuz blokları cihaza dönmez → boş-VRAM kapısı ~yarım
     VRAM üstü işleri (40 GiB A100'de 25 GiB'lik M10 sınıfı!) yanlış reddederdi. GPU ortamı
     artık plan'dan ÖNCE anlık görüntüleniyor; çağrı sırası testle sabit. Kendi boş-VRAM
     düzeltmemin yan etkisiydi — mercek tam bunun için kurulmuştu.
  2. **MAJOR (repro'lu) — kapı resume'u reddediyordu:** kapı-öncesi başlatılmış ya da %95
     bitmiş bir CPU koşusu `--resume` ile tamamlanamaz, checkpoint yetim kalırdı. Mevcut
     checkpoint'e açık `--resume` kapıyı uyarıyla atlar (batık emek + açık niyet); uçtan uca
     test (kesinti → bayraksız resume → bit-devam).
  3. **MAJOR — prob "auto" backend'de ölçüyordu:** GPU'lu makinede `--backend numpy` işinde
     kapı cuFFT süresine "measured" etiketiyle güvenirdi. `estimate(measure_backend=...)`
     eklendi, runner çözümlenen backend'i geçiriyor (testli). Yerelde repro edilemez (GPU yok)
     ama kod yolu net — Colab oturumunda canlı doğrulanacak.
  4. **MINOR — dry-run sözleşmesi:** kapının dry-run'ı reddetmesi (benim kararımdı) exit-0'a
     bağlı betikleri ve "Colab işini CPU'da planla" akışını kırıyordu. Geri alındı: dry-run'da
     kapı "would be refused" NOTU basar, exit 0 kalır. VRAM reddi dry-run'da reddetmeye devam
     ediyor (M10c'den beri mevcut sözleşme — tutarsızlık bilinçli ve belgeli).
  5. **MINOR — damga/loglama:** `environment["numpy"]` yine `np.__version__` (importlib.metadata
     egzotik kurulumda None dönebilirdi); CLI loglaması root yerine `caustica` logger'ına
     kapsandı (main() çağıran notebook üçüncü-parti INFO seline maruz kalmasın).
- Ders: 1. bulgu bu milestone'da eklenen boş-VRAM düzeltmesinin etkileşim hatası — "düzeltme
  de bir değişikliktir" merceği kendini kanıtladı.

## 2026-08-22 — Oturum 13 devamı (Fable): M10k başladı — W0a + W0b + W0c indi

### W0a — medium_volume (`e16deb2`)
- `caustica.io.medium_volume`: format artık kütüphanenin malı. Okuyucu mevcut dosya
  etiketlerini (`caustica-phantom/1` + `hifusim-phantom/1`) kabul ediyor — plan metnindeki
  `*-dataset/2` adları manifest etiketiydi, dosya etiketi ground-truth'tan düzeltildi (satır
  kayması değil, ad düzeltmesi; sapma raporu gerektirmeyen türden — davranış: mevcut dosyalar
  OLDUĞU GİBİ okunuyor, T7 hedefi bu). Yazıcı `write_medium_volume` public, yeni etiket
  `caustica-medium-volume/1`.
- BİT-AYNILIK KANITI (R11): gerçek 560×700×480 dataset dosyası → `medium_volume` Medium'u,
  PhantomAsset yolunun Medium'uyla sha256 düzeyinde birebir (alpha/rho/c/beta + id_map;
  25 sn). Round-trip (yaz→oku) iki modda da bit-aynı. 13 yeni test.
- M6f f0-alpha koruması ve su-odak reddi medium_volume'a genelleşti; su-odak `water_label`
  alanıyla (varsayılan 0) kapatılabilir — genel bir hacimde etiket 0 su olmayabilir.

### W0b — literatür doku değerleri (`d2bac95`)
- `TISSUE_LIBRARY` (5 giriş) + `AcousticTissue` + dB/cm↔Np/m caustica.materials'a HARFİYEN
  taşındı (isimler dahil — isim, MaterialDB JSON'una gömülü). uwcem tissue.py uçları artık
  kütüphaneden `is`-aynı nesne olarak alıyor; ramp + eşleme + pval kuralları yerinde.
  Rakamlar donmuş literalle test-pinli; 93 fantom testi değişmeden yeşil.

### W0c — bağ kesildi (kırıcı şema değişikliği, K14)
- SİLİNDİ: `_require_uwcem`, `PhantomDatasetMediumConfig`, `StoredSetupJobConfig`,
  `StoredSetupOverrides`, `RunPolicyOverrides` (yalnız stored katmanı kullanıyordu),
  `_build_stored`, `_stored_phantom_file`; `geometry.load_breast_phantom` +
  `breast_phantom_mapping` + `VolumeImportConfig`'in `breast_phantom_txt` formatı.
  `caustica-job/1` İKİ kind kaybetti; format numarası DEĞİŞMEDİ (D35: kalkan kind'a özel
  hata yok — pydantic union hatası yeterli, tek karşılaşacak kişi migrasyonu yapan).
- `tests/test_import_direction.py` W0c'den ÖNCE yazıldı ve kırmızı doğdu (planın istediği
  kanıt sırası); şimdi YEŞİL: AST importları + kaba metin taraması ("uwcem" src'de 0 —
  base.py tarihsel yorumu ve W0a/W0b'nin kendi yorumları dahil yeniden ifade edildi).
- Test ayrıştırması: test_job'dan 10 blok + test_geometry'den 1 blok VERBATIM staging'e
  (scratchpad/uwcem_staging/) — W0d'de yeni repoda load_setup→explicit-job kapsamına
  dönüşecek. Kalan test_job'da full-grid uyarı testi medium_volume kind'ına geçirildi
  (gerçek dataset dosyasıyla, yeni kapıdan). Generic `load_labels_txt` testleri yerel
  mapping fonksiyonuyla kaldı.
- D26 penceresi AÇIK: dokuz yerel setup şu an `caustica run` ile koşamaz (stored_setup yok);
  `load_setup` çalışıyor. W0d kapatacak: load_setup explicit job üretecek.

### W0d + W0e + W0f — yeni repo, veri kökü, lisans (2026-08-22)
- **Yeni repo:** `C:\Users\bulbu\Desktop\uwcem-phantom` (yerel git, `45cf16b`; push YOK).
  Taşınanlar: paket, phantom_launcher, phantom_studio (tamamı uwcem-bağımlıydı, 10+ import),
  phantoms.bat/sh, test süiti. `setup_to_job()`/`emit_jobs()` + CLI `--emit-jobs`:
  her stored setup, medium_volume kapısından giren SELF-CONTAINED explicit job üretiyor —
  kalkan `stored_setup` kind'ının migrasyon yolu. `load_breast_phantom` → `legacy_import.py`.
- **KAPANIŞ KAPISI (R11):** dokuz setupta `load_setup` ↔ emitted-job medyaları sha256 BİT-AYNI,
  validate ok, dry-run exit 0 (5:04; test yeni repoda `slow` işaretli). Bonus: dry-run çıktısı
  üretim ayarının 2f0 @ 2.85 ppw uyarısını gösterdi — D31/M10i entegrasyonunun canlı kanıtı.
  Dürüst yorum kaydı: "uçtan uca" = boru hattı + bitwise medya paritesi; dokuz TAM çözüm
  9×2.6 sa CPU (M10i kapısının reddettiği sınıf) — mini tam-çözüm kanıtı medium_volume
  koşu testinde.
- **W0e:** tek veri kökü (arg → CAUSTICA_PHANTOM_DATA → dolu checkout _data → kullanıcı
  önbelleği; platformdirs YOK). dataset/setups aynı kökte. Yerel kök `hifusim\data`:
  kullanıcı-düzeyi env kuruldu; `_data`'nın uwcem/cache/exports klasörleri aynı diskte
  `data/` altına RENAME edildi (295 MB; indirme yok) — 4.5 GB dataset ve setuplar yerinden
  OYNAMADI (plan sözü: "bugünkü gibi"). Taşınan süit + dokuzlu kapı ağsız geçti (T5 kanıtı).
- **W0f:** UWCEM resmi lisans YOK; Instruction Manual'dan verbatim şart README'ye yazıldı
  ("free of charge ... acknowledge the authors ... in any publication derived").
  Muhafazakâr uygulama: git'te fantom baytı yok, türevleri biz dağıtmıyoruz, atıf metadata'da.
- **hifusim temizliği:** taşınanlar git'ten silindi; data/setups + manifestler git'ten çıktı
  (diskte duruyor); .gitignore `data/` bütününe indirgendi; apps/README ve README işaretçilere
  döndü. Kullanıcı görünür değişiklik: `phantoms.bat` artık uwcem-phantom repo'sunda;
  `CAUSTICA_PHANTOM_DATA=C:\Users\bulbu\Desktop\hifusim\data` kullanıcı env değişkeni kuruldu.

### D28 tamamlaması (tam tur beklenirken kendi yakaladığım boşluk)
- Yeni repodaki `PhantomAsset.save` taşıma sonrası hâlâ KENDİ el-yapımı yazıcısını taşıyordu —
  D28 "uwcem repo caustica'nın yazıcısını çağırır, formatın tek kaynağı" der. Rewire edildi
  (`67cbc73`): asset yalnız NEYİN gireceğine karar veriyor (etiketler, yoğun hacimler,
  extent/histogram metadata'sı), baytların düzenini `caustica.io.write_medium_volume` yazıyor.
  Yeni exportlar `caustica-medium-volume/1` etiketi taşıyor; okuyucular eski etiketleri kabul
  etmeye devam ediyor (rebuild yok). Yeni repo süiti 166 passed.

## 2026-08-22 — Operatör yeniden-planı: UWCEM tek dosyada + yalınlaştırma + K15–K17

Kullanıcı yönetim modelini değiştirdi (K17): operatör = Fable 5, kod = Opus 5 alt-ajanları;
PM sistemi MILESTONES.md. Bu tur kod değil, plan/temizlik turuydu:

- **UWCEM tek dosyaya indi**: `docs/uwcem.md` — ayrışım M10k olarak zaten kapanmıştı (W0a–W0f,
  bugün başlık `[x]`'e çekildi ve bayat `load_breast_phantom` kutusu kapatıldı); kalan işler
  (push kararı, bakım) EN SON'a alındı (K16). `library_first_plan.md` W0 gövdesi stub'a indi.
- **Yalınlaştırma #1** (kullanıcı onayı): kök `mtype.txt` (123 MB) + `mtype.txt.labels.npz` +
  `_code_cells.py` + `hifu_pred_dx300_t128.ipynb` silindi (M14 hedefi artık devlog'daki v12
  sayıları — MILESTONES notu düşüldü); bayat `build/` silindi (M10h'deki CRITICAL maskesinin
  nüksü). Hepsi git dışıydı.
- **K15 tam plugin mimarisi** (kullanıcı, erken-soyutlama uyarısına rağmen teyit): beş eksen
  entry-point — solver ✅ / medium kind + array kind (M10m) / backend + report renderer (yeni
  **M10n**). Panzehir kuralı: çekirdek kendi plugin API'sinin birinci müşterisi.
- **uwcem-phantom push kararı**: şimdilik YEREL (tek kopya riski kullanıcıya söylendi, kabul).
- Sıra onaylandı: M10m → M10n → M10j → M10l → M10f → ilk Colab oturumu → M10g → UWCEM kalanları.

## 2026-08-22 — M10m: kendi kurulumunu getir (elements kind + kind registry'leri + şema + dokümanlar)

Kabul sorusu: *hiç tanımadığımız bir araştırmacı kendi problemini koşabiliyor mu?* Bu tur o
sorunun üç ayağını kapattı: transducer tarafında açık eleman tablosu, iki kind ekseninin
registry'ye dönmesi, ve şemayı okumak için pydantic kaynağına inme zorunluluğunun kalkması.

### Yapılanlar
- **`config/kinds.py`** — `KindRegistry` (çözücü registry'sinin kalıbı) + `MediumKindConfig` /
  `ArrayKindConfig` tabanları + `MediumPrep`. Entry-point grupları `caustica.medium_kinds` ve
  `caustica.array_kinds`. Union **kayıt sırasından** kuruluyor (alfabetik DEĞİL) — pydantic
  beklenen-etiket metni bu yüzden bit-aynı kaldı.
- **`elements` array kind'ı** + `arrays/elements.py`: `.npz` (`positions`, opsiyonel `normals`)
  ve 3/6 sütunlu `.csv` okuyucusu, `elements_array()` kurucusu. Normaller opsiyonel — yoksa her
  eleman `(0,0,roc_mm)`'ye bakar. Job tarafı mm, Python tarafı m; 1 m'yi aşan açıklık "unit
  mistake" diye reddediliyor.
- **`caustica schema`** (+ `--kinds`, `--compact`): `job_schema()` pydantic'ten üretiyor.
- **`docs/job_reference.md`** ve **`docs/conventions.md`**; README'ye "Bring your own setup".

### Üç tasarım kararı
1. **`isinstance` yerine seam metodları.** `_build_explicit` artık `provides_grid` +
   `prepare()` (grid'i dosyadan veren kind'lar için) ve `resolve_paths()` üzerinden çalışıyor.
   `MediumPrep.build_medium()` tek atımlık: çağrılınca kind'ın hacme olan referansını BIRAKIYOR
   — eski koddaki `del volume` davranışının seam'e taşınmış hâli (tam boy fantomda GB'lar).
2. **`roc_mm`'yi tabana ALMADIM.** Pydantic taban sınıfta tanımlanan alanı en öne alıyor; bu,
   runner'ın yazdığı normalize `job.json` içindeki anahtar SIRASINI değiştirirdi. Bunun yerine
   taban `focal_length_mm()` metodu istiyor — plugin kendi alan adını seçebiliyor, baytlar aynı.
3. **Ertelenmiş anotasyon (`_LazyKindUnion`).** İlk hâlde union modül-global'iydi; geç kayıt
   (notebook'ta tanımlanan kind, ya da testin zorladığı yeniden tarama) sonrası
   `model_rebuild(force=True)` union'ı GÜNCELLEMİYORDU. Sebep pydantic 2.13'te net:
   `rebuild_model_fields` yalnız `field_info._complete is False` olan alanı yeniden çözüyor —
   ilk seferde başarıyla çözülmüş bir anotasyon bir daha okunmuyor. Çözüm: alan
   `Annotated[Any, _LazyKindUnion(registry)]` ile işaretli; `__get_pydantic_core_schema__` her
   şema üretiminde registry'ye soruyor. Hata metni, JSON Schema `discriminator` haritası ve
   serileştirme aynı kaldı (prototiple doğrulandı, sonra süitle).

### Kanıtlar
- Süit **279 → 311** (309 passed / 2 skipped / 0 failed). Yeni: `test_elements_array.py` (12),
  `test_kind_registry.py` (7), `test_schema_doc.py` (13).
- **Altın karşılaştırma**: M10m öncesi commit bir worktree'ye alındı; üç job (bowl/homojen,
  spiral/scene, spiral/steered) için normalize `job.json` baytları, `derived` anahtar SIRASI ve
  değerleri, `focus_vox`, kaynak voxel sayısı, faz toplamı, `validate` çıktısı ve sekiz farklı
  hata metni iki sürümde üretilip diff'lendi. **Tek fark**: array beklenen-etiket listesi
  `'elements'` kazandı. Medium etiket listesi, PML reddi, scene etiket reddi, `extra_forbidden`,
  format reddi, steered-bowl reddi — hepsi kelimesi kelimesine aynı.
- **Doküman çürümesi mutasyonla sınandı**: `### \`bowl\`` başlığı `bowls` yapılınca ve bir
  parçada `active_fraction` → `activ_fraction` yazılınca süit kırmızıya döndü, sonra geri alındı.
- **`import caustica`**: 210.6 ms → 207.0 ms (medyan, 9 alt-süreç; gürültü içinde). Entry-point
  taraması tek başına 2.9 ms ve YALNIZCA `caustica.config.job` import edilince koşuyor —
  `test_import_caustica_does_not_scan_entry_points` bunu yapısal olarak sabitliyor.

### Yabancı-kullanıcı provası (adım adım)
Repo dışında, `%TEMP%\...\outsider` altında **temiz venv** (sistem Python 3.12), wheel kurulumu
`pip install C:\Users\bulbu\Desktop\hifusim`. Sonra YALNIZCA README + `docs/job_reference.md`
okunarak:
1. `water_bowl.json` elle yazıldı — homojen su (β=3.5, α=0.025 Np/m), dx 0.25 mm,
   20×20×28 mm, PML 2.5 mm, bowl d=12 mm / ROC 16 mm, apex (10,10,4) mm, 1 MHz / 200 kPa,
   harmonics [1,2], westervelt.
2. `caustica validate water_bowl.json` → OK, uyarı yok (ppw 6.00, 80×80×112).
3. `caustica run --dry-run` → planner ~9 s (measured), 0.08 GiB.
4. `caustica run` → 9.1 s, 252 adım, period 19'da yakınsadı, tepe |P| = 1.337 MPa.
5. `caustica report` → **ImportError: matplotlib yok** (kurulumu `[report]` extra'sız yapmıştım;
   mesaj tam olarak ne yapılacağını söyledi). `pip install "caustica[report] @ file://..."`
   sonrası `index.html` + dört figür.
6. Kendi tablomu getirdim: 16 elemanlı halka `my_array.npz` (mm, apex çerçevesi) →
   `{"kind":"elements","file":"my_array.npz","elem_radius_mm":1.0,"roc_mm":16.0}` + steered
   odak → validate OK → 9.7 s koşu → rapor. `run_meta.derived`: `n_elements 16`,
   `elements_represented 16`, `f_number 1.6`.
7. **Pozitif kontrol** (dokümanın L1 kesitinden): `archimedean_spiral(...)` ile 24 elemanlı dizi
   kurulup eleman tablosu `.npz`'ye yazıldı ve AYNI job iki kez koşuldu — bir kez
   `archimedean_spiral` kind'ıyla, bir kez o tabloyu okuyan `elements` kind'ıyla. Sonuç:
   tepe basınç `678863.1 Pa` (ikisinde de, son haneye kadar), aynı voxel, aynı `derived`,
   aynı 995 kaynak voxeli. Yeni kapı doğrulanmış kapıyla bit-aynı alan üretiyor.

**Provanın yan etkisi (dikkat):** `pip install <checkout>` checkout'ta
`build/` klasörünü YENİDEN oluşturdu - 2026-08-22'de silinen, M10h'de wheel testini
maskeleyen artefaktın ta kendisi. Silindi. `test_packaging.py`'nin wheel fixture'ı zaten
pristine bir kopyadan kurduğu için süit bağışık, ama checkout'u pip ile kuran herkes bu
klasörü geri getiriyor: kurulum sonrası `rm -rf build` alışkanlık olmalı (git-ignore'lu
olduğu için `git status` uyarmıyor).

**Kaynak koda inmem gerekmedi.** Tek eksik nokta: pozitif kontrolde `TransducerArray.positions`
/ `.elem_radius` alanlarını okumak istedim, bu ikisi hiçbir dokümanda yazılı değildi (API
şeklinden çıkardım) — `job_reference.md`'nin `elements` bölümüne bir satır eklendi.

**Fizik notu (bug değil):** üç kurulumda da tepe geometrik odağın ÖNÜNDE çıkıyor (bowl: apex'ten
12 mm, odak 16 mm). Bu düşük kazançlı O'Neil davranışı — a/λ = 4, eksenel −6 dB genişliği
15.5 mm. `bowl` (doğrulanmış kind) ve `elements` aynı kaymayı gösteriyor; halka dizide daha
belirgin, çünkü halka eksende her noktaya eşit uzaklıkta olduğundan eksenel odaklama yapmaz,
yalnız 1/r düşüşü kalır. `metrics.json` bunu `hit_ratio` + `displacement_norm_mm` ile dürüstçe
raporluyor.

### M10m gözden geçirme turu — iki mercek, altı gerçek bulgu

İki bağımsız tur koşturuldu: (a) "mevcut davranış sessizce bozuldu mu?" (diferansiyel harness:
M10m öncesi ağaç bir scratch dizine çıkarılıp ~60 senaryo iki sürümde bayt bayt karşılaştırıldı),
(b) şüpheci doğrulayıcı (her iddiayı ÇÜRÜTMEYE çalışan). İkisi de değer üretti.

**Çürütülen iddia (en önemli bulgu).** "Eleman tablosu diskte değişirse yeniden yükleme yakalar"
YANLIŞTI. `derived()`'in sayıları sıra istatistiği: `n_elements`, `r_max_mm`, `shell_depth_mm` —
`elem_radius_mm` config'ten kopya, `f_number` ve `half_angle_deg` ise bu ikisinin cebiri. Doğrulayıcı
asimetrik bir tabloyla diskte dört mutasyon yaptı ve dördü de kontrolden GEÇTİ:

| diskteki mutasyon | kontrol | odak |P| | alan bağıl L2 |
|---|---|---|---|
| x aynalama | KAÇIRDI | 84.68 kPa | %53.0 |
| 37° döndürme | KAÇIRDI | 84.03 kPa | %59.1 |
| r_max hariç hepsini yeniden saçma | KAÇIRDI | 72.65 kPa | %57.9 |
| iki elemanın yarıçapını takas | KAÇIRDI | 83.58 kPa | %47.9 |

(taban 84.64 kPa). Ayrıca yalnız-normal değişiklikleri ve satır sırası + `phases_rad` bileşimi de
görünmezdi. Onarım: `table_sha256` — pozisyon+normal içeriğinin özeti. **Dosyanın değil
GEOMETRİNİN** özeti, çünkü aynı dizinin inline / `.npz` / `.csv` hâlleri aynı özeti vermeli, ve
geometrisi değişmeden baytları değişen bir dosya sapma değildir. `check_derived` sayısal olmayan
değerler için tam eşitlik dalı kazandı. Diğer array kind'larına özet EKLENMEDİ: onların geometrisi
zaten job'daki birkaç sayıdan üretiliyor, açıklığı sabitlemek transducer'ı sabitliyor (ve eklemek
mevcut `run_meta.derived` baytlarını değiştirirdi).

**Diğer beş bulgu.**
1. `f_number: Infinity` — eksen üstü tabloda `run_meta.json` JSON olmayan bir token yazıyordu
   (Python okur; `JSON.parse`, `jq`, Go, serde reddeder). Artık anahtar yazılmıyor.
2. **Plugin kurmak caustica'nın kendi süitinden ALTI testi düşürüyordu** — registry'nin var olma
   sebebi, onu etkinleştiren kütüphanenin süitini bozuyordu. Üçü registry'nin tam içeriğini,
   üçü de "referans her kayıtlı kind'ı belgelemeli"yi iddia ediyordu (bir yabancının paketini
   bizim dokümanımız belgeleyemez). `core_kinds()` (tanımlandığı modüle bakar) ile onarıldı;
   gerçek bir plugin `PYTHONPATH`'te iken tam süit koşturularak doğrulandı.
3. `importlib.reload(caustica.config.job)` patlıyor ve modülü YARI değiştirilmiş bırakıyordu
   (`DriveConfig` yeni, `ExplicitJobConfig` bayat; reload sonrası bir örnek artık `isinstance`
   değil, ve hiçbir yerde hata yok). Sebep: kimlik tabanlı çakışma kontrolü, reload'un ürettiği
   YENİ sınıfı aynı etiketle görüp "kendisiyle çakışıyor" diyordu. `%autoreload 2` = notebook
   akışı, yani tam hedef kitlemiz. modül+qualname eşleşmesi = yeniden tanım kuralıyla onarıldı;
   `on_change` de aynı kuralla tekilleştiriliyor (yoksa her reload bir kanca daha yığardı).
4. `validate_job` hâlâ `isinstance(job.medium, MediumVolumeConfig)` ile dallanıyordu — üçüncü
   taraf bir grid-veren kind, hiçbir şeye mal olmama sözü veren `validate` içinde GB'ları
   materyalize ederdi. Kind'a soruluyor artık.
5. `register()` `discover()`'ın try'ı İÇİNDEydi: kanca patlarsa kind registry'de kalıp modeller
   bayat kalıyordu — `available()` ve `caustica schema` şemanın reddettiği bir kind'ı ilan
   ederken log "plugin failed to load" diyerek tutarsızlığı açıklanmış gösteriyordu. Kayıt artık
   ya tamamen olur ya hiç (geri alır ve yeniden fırlatır).

**Doküman düzeltmeleri** (hepsi kaynağa karşı doğrulandı): `phasor_convention` → **`phase_convention`**
(yanlış anahtar adı; dokümanı izleyen `KeyError` alırdı) + kaynak metnin taşıdığı "mutlak faz sıfırı
kaynak-referanslı değil" uyarısı geri kondu; `amplitude` iddiaları abartılıydı (süit native yolu
−%10/+%12 ile sınırlıyor, değişmezliği yalnız `c_max`/`dt` ekseninde iddia ediyor, k-Wave çapraz
doğrulaması normalize ettiği için mutlak genliği HİÇ iddia etmiyor — ölçülen ile iddia edilen
ayrıldı); `elements_represented` kısmi kaybı raporlayamaz (voxelize önce reddediyor), makbuz olduğu
yazıldı; `volume_import` npz'si `labels`+`dx`+**`origin`** ister (eksik anahtar hatası opak,
`LabelVolume.save_npz` gösterildi); `homogeneous` parçası (β=3.5) altındaki "varsayılan β=0" düz
yazısıyla çelişir okunuyordu; tam-grid kayıt uyarısı açık verilmiş büyük bölge için ÇALIŞMIYOR;
`archimedean_spiral` parçası 100 mm üretim dizisi ve sayfanın kendi örnek grid'inde odağı dışarı
düşüyor (söylendi + sığan bir parça eklendi). Sonuncusu artık testli:
`test_every_array_kind_has_a_snippet_that_fits_the_documented_grid` — bir parçanın şema-geçerli
olması yetmiyor, dokümanın kendi job'ında KOŞMASI gerekiyor.

**Çürütülüp doğrulanan (yani bulgu ÇIKMAYAN) yerler**, kayda değer olanlar: `_build_explicit`
reddetme SIRASI vaka vaka aynı (f0-uyuşmazlığı + grid-dışı apex birlikte verildiğinde ikisi de f0'ı
raporluyor); bellek davranışı sadece eşdeğer değil ÖLÇÜLDÜ — gerçek 188 Mvox fantomda tepe
**3.71 GiB / 3.59 GiB yerleşik, iki sürümde de iki ondalığa kadar aynı**, weakref probe'u hacmin
`build_job` dönerken ölü olduğunu gösteriyor; `derived` anahtar sırası ve içerikleri korunmuş;
taşınan dallardaki hata metinleri karakter-aynı; yol çözümlemesinde ne çift ne eksik çözüm (job'a
göre/iç içe/mutlak/eksik dosya/`base_dir=None` hepsi aynı, ve `resolve_paths` kopya döndürdüğü için
`dump_job` hâlâ göreli yolu yazıyor); ertelenmiş anotasyon ile kapalı union arasında
serileştirme/eşitlik/`model_copy`/pickle/deepcopy/strict mod/`extra=forbid`/alan sırası/JSON Schema
farkı YOK (tek fark: alanın *bildirilen* anotasyonu `Any` — kodda belgelendi, repoda kimse okumuyor);
import döngüsü yok; dokuz `data/setups/s1-*.json` iki sürümde bayt-aynı çıktı veriyor.

---

## 2026-08-22 — Oturum M10n (Opus 5 alt-ajanı): plugin mimarisi, beş eksen entry-point

### Yapılanlar
Beş eksenin de aynı seam'i taşıması (K15, PLAN §2 kural 6). Dal `library-first`, yedi yerel commit.

- **`src/caustica/registry.py` (YENİ)** — ortak şekil TEK yerde: `PluginRegistry` (lazy entry-point
  taraması, reload'a dayanıklı çakışma kontrolü, ya-hep-ya-hiç kayıt + geri alma, `on_change`
  kancaları, kayıtlı adları listeleyen `UnknownPluginError`) ve `FactoryRegistry` (implementasyonu
  düz bir callable olan eksenler). T9'un talimatı buydu: **yeniden türetme, genelle.** M10m'in
  `KindRegistry`'si artık bunun alt sınıfı (pydantic union'ı ve `kind` alanından anahtar okuma
  onda kaldı — yalnız pydantic ekseninin ihtiyacı), solver registry'si de öyle.
- **Backend ekseni** — `get_backend` isim→fabrika kaydına dönüyor; `numpy` ve `cupy` iki KAYITLI
  fabrika, bir `if`'in iki dalı değil. `"auto"` bir backend DEĞİL, onların üstünde bir politika
  (kullanılabilirse cupy, değilse numpy) ve bilinçli olarak çekirdekte kaldı: üçüncü taraf bir
  backend adıyla seçilir, tahmin edilmez. cupy importu fabrikanın İÇİNDE — kayıt hiçbir şeye
  dokunmuyor.
- **Report renderer ekseni (YENİ `src/caustica/report/renderers.py`)** — `caustica report` bir
  renderer'ı isimle çözüyor; caustica'nın matplotlib implementasyonu `"matplotlib"` adıyla kayıtlı
  ve AYNI kapıdan geçiyor. Modül stdlib-only, yani T6 korundu: renderer'ları listelemek matplotlib
  import ETMİYOR (testli).
- **Job şeması açıldı** — `backend` alanı kapalı `Literal`'dan registry'nin doğruladığı `str`'e.
  Üçüncü taraf bir backend job dosyasından erişilemiyordu; `elements`'in array'ler için kapattığı
  çıkmazın aynısı. `run --backend` argparse `choices`'ı da kalktı.
- **Entry-point grup adları donduruldu** (`ENTRY_POINT_GROUPS`, testle çivilendi):
  `caustica.solvers` · `caustica.medium_kinds` · `caustica.array_kinds` · `caustica.backends` ·
  `caustica.report_renderers`.
- **`docs/extending.md`** — beş eksenin her biri için tarif + kopyala-yapıştır kurulabilir iskelet
  paket (pyproject + tek modül, beşini birden kuruyor). İki test dokümanı çürümekten koruyor:
  ilan ettiği gruplar donmuş beşle AYNI olmalı, ve iskeleti `ast.parse`'tan geçip beş parçayı da
  hâlâ tanımlamalı.
- **`tests/test_kind_registry.py` → `tests/test_plugins.py`** — M10m'in fixture'ı BÜYÜTÜLDÜ (ikinci
  fixture kurulmadı): aynı sahte dağıtım artık beş grubu da ilan ediyor.

### Kanıt
- Süit **325 → 339** (337 passed + 2 skipped; +14 test). `ruff check/format src tests` temiz.
- `test_entry_point_plugin_extends_all_five_axes`: iki mini koşu + bir render, hepsi plugin'in
  ortamında. Her eksen ÇALIŞTIĞINI **kendi sayacıyla** kanıtlıyor — job dosyasından yankılanan bir
  damga yeterli sayılmadı. Doğrulayıcı bunu üç "impostor" sabotajıyla sınadı (isimler doğru kalıp
  seam kırılıyor): üçü de test'i kırmızıya düşürdü, yani sayaçlar taşıyıcı.
- Koşu A (plugin çözücü) ile koşu B (plugin backend + native çözücü) BİLEREK ayrı: runner
  `backend=`'i yalnız `_NATIVE_SOLVERS`'a geçiriyor (T3), yani üçüncü taraf bir çözücü varsayılan
  backend'de çözer. Test bunu yazıyor, üretilmemiş bir damgayı iddia etmiyor.
- `import caustica`: medyan **257.5 → 258.8 ms** (+%0.8). Bağımsız ölçüm (worktree ile M10n öncesi
  commit'e karşı, serpiştirilmiş): **−%2.2**. `entry_points()` casuslandı: `import caustica`
  sırasında **sıfır** çağrı.
- `caustica report` çıktısı M10n ÖNCESİ CLI ile karşılaştırıldı: REPORT.md + index.html + üç PNG
  **sha256 aynı**. M10n öncesi yazılan bir checkpoint HEAD ile resume ediliyor.
- Dokuz `data/setups/*.json` bayt-aynı.

### Review turu — beş gerçek bulgu (mercek: "sessizce bozdu mu?")
1. **CLI açılışı iki katına çıkmıştı** (216 → 465 ms) — HER komut için, `--help` dahil.
   `build_parser()` tek bir varsayılan ismi okumak için `caustica.report.renderers`'ı import
   ediyordu; o da `caustica/report/__init__.py`'yi, o da numpy metrics + preview'ı. Üstündeki
   yorum tam tersini iddia ediyordu. Artık `--renderer` varsayılanı `None`, çözüm `report`
   dalında; parser kurulumu `import caustica` maliyetine döndü (263 vs 259 ms).
2. **Backend fabrikası registry anahtarına bağlanmıyordu.** `"x"` adına kayıtlı bir fabrika
   `Backend("numpy", ...)` döndürebiliyordu; aşağısı hep `Backend.name` okuyor — `run_meta`
   damgası, `result.h5` attr'ı ve **resume'un aynı koşu olup olmadığına karar veren checkpoint
   parmak izi**. Yani sahte bir numpy koşusu ayırt edilemezdi. Kapalı `Literal` bunu erişilemez
   kılıyordu, registry kılmıyor. `get_backend` artık hem isim uyuşmazlığını hem de `Backend`
   olmayan bir dönüşü reddediyor.
3. **Lambda'lar çakışma kontrolünü deliyordu:** modül düzeyindeki her lambda `<lambda>` qualname'ini
   taşıyor, `same_definition` ikisini "aynı tanım" sayıp ikincisinin birincisini sessizce
   değiştirmesine izin veriyordu. Anonim qualname artık yeniden-tanım iddia edemiyor.
4. **Solver çakışma metni bir kelime kaybetmişti** ("solver 'linear'" vs "solver name 'linear'").
   Kind metinleri korunmuştu, solver yolu korunmamıştı. Geri kondu ve artık substring yerine
   baştan-eşleşmeyle test ediliyor.
5. **`--backend` yazım hatası artık medium kurulduktan SONRA reddediliyordu** (argparse `choices`
   kalkınca). Önce kontrol ediliyor; test `build_job`'a ulaşılırsa kırmızı.

### Doğrulayıcının beş boşluğu (hepsi kapatıldı)
1. **Belgelenen seam soğuk import'ta yanlış cevap veriyordu.** `from caustica.config.kinds import
   medium_kinds; medium_kinds.available()` — dokümanın yazdırdığı import — `()` döndürüyordu,
   çünkü çekirdek kind'ları `config/job.py` import edilirken kaydediyor ve bunu kimse zorlamıyordu.
   Plugin kuruluysa daha kötüsü: yalnız kendi kind'ları görünüyordu. Kind registry'leri artık
   SORULDUĞUNDA job.py'yi kendisi import ediyor (import anında değil). Yeniden-giriş bedava:
   job.py çalışırken zaten `sys.modules`'te, yani çağrı tam olması gereken anda no-op.
2. Beş-eksen testi plugin MEDIUM'unun koştuğunu iddia etmiyordu — su döndüren bir `build()` de
   geçerdi. Plugin artık build'ini de kaydediyor, test jel olduğunu iddia ediyor.
3. Sahte renderer `metrics["peak"]["p_max_pa"]` okuyordu — böyle bir anahtar YOK, hep
   `peak_pa: None` yazıyordu. Gerçek anahtar (`p_pa`) ve testte iddia.
4. `plugin_on_path` `_loaded`'ı sıfırlıyor ama `_JOB_ADAPTER`'ı yeniden kurmuyordu: plugin ancak
   test önce `available()` yokladığı için keşfediliyordu. Fixture artık girişte keşfediyor.
5. İki docstring "pydantic `importlib.metadata`'yı kendi import anında yükler" diyordu; yüklemiyor
   — `pydantic.plugin._loader` üzerinden, bir model sınıfı kurulduğunda geliyor.

### Ayrıca (M10n'in kendi kusuru değil, ama rename onu mayına çevirirdi)
`test_reloading_the_job_module_still_works` teardown'ında `importlib.reload(jobmod)` **reload'u geri
almıyordu** — ÜÇÜNCÜ bir sınıf kümesi üretiyor, oysa başka test modülleri collection anında aldıkları
İLK kümeyi tutuyor. Sonrası `isinstance`'ta düşüyor, hem de alakasız bir dosyada. Tam süit bunu hiç
görmedi: `test_job.py` hem eski hem yeni dosya adından ÖNCE sıralanıyor — süit alfabe sayesinde
yeşildi. Dosya adını değiştirmek bunu mayına çevireceği için (adı `test_job`'dan önce gelen bir
seçim beş testi sebepsiz kırmızıya düşürürdü) devralınmadı, onarıldı: teardown modül ad alanını
birebir geri koyuyor ve İLK sınıfları yeniden kaydediyor — çakışma kontrolünün zaten
"yeniden tanım" saydığı şey, yani `%autoreload 2`'yi hayatta tutan kuralın ta kendisi.

### Bilerek YAPILMAYANLAR (belgelendi, MILESTONES'ta "BİLİNEN SINIR")
- **Runner'ın `_NATIVE_SOLVERS` beyaz listesi.** `backend=` ve `checkpoint=` yalnız `linear` ve
  `westervelt`'e geçiyor (kwave adaptörü bilinmeyen kwarg reddediyor — T3). Üçüncü taraf bir çözücü
  varsayılan backend'de çözer ve checkpoint almaz. `SolverCaps.backends` alanı bunu zaten ilan
  ediyor ama `src/` içinde onu OKUYAN kimse yok — M10n bu ölü alanı ilk kez anlamlı kılan şey.
  Yetenek sorusuna çevirmek ayrı bir iş: aynı bayrak planner'ı da kapılıyor ve iki farklı konu.
- **`Backend.is_gpu` hâlâ `name == "cupy"`.** Üçüncü taraf bir GPU backend'i `cupyx.scipy.fft`
  yerine `scipy.fft` alır, `synchronize` no-op olur, slow-CPU kapısı ve VRAM kapısı atlanır — hiçbiri
  patlamaz, hiçbiri çalışmaz. Yani backend ekseni bugün yalnız CPU benzerleri için gerçekten
  kullanılabilir. `is_gpu`'yu backend'in İLAN ettiği bir alan yapmak `Backend` değer nesnesini
  değiştirir; M10n kapsamı dışı, `docs/extending.md` açıkça yazıyor.
- `status.json`'daki `error` alanı artık `KeyError:` yerine `UnknownPluginError:` diyebiliyor
  (mesaj tırnaksız ve okunur oldu). M10l sözleşmeyi dondururken bakılacak.
- `caustica validate`'in bilinmeyen backend hatasında pydantic `type`'ı `literal_error` →
  `value_error` oldu (Literal'ın açılmasının kaçınılmaz sonucu). Şema alanı artık `description` +
  `examples` taşıyor, yani editör ipucu enum yerine oradan geliyor.

### Açık uçlar / sonraki adım
- Sıradaki: **M10j** (facade + ilerleme). `progress=` T1/T2/T3 tuzaklarına dokunuyor.
- Bir kez görülen kırılganlık: `tests/test_io.py::test_killed_writer_leaves_no_visible_file` (gerçek
  SIGKILL yarışı) iki review ajanı süiti aynı anda koştururken bir kez düştü; tek başına ve tam
  süitte tekrar tekrar yeşil. Yük altında zamanlamaya duyarlı, kayda geçsin.

## 2026-08-22 — M10j kapandı: facade + ilerleme (kapanış operatörce — ajan bağlantı hatasıyla düştü)

Beş commit (`5fb77ac..0b7d6eb`): `caustica.simulate()` (kapalı girdi listesi, AYNI `build_job`,
ikinci kod yolu yok; `out=None` bellekte ama plan-önce + M10i kapıları aynen), `progress=` kancası
periyot sınırında ve checkpoint'ten BAĞIMSIZ (T1), heartbeat payload'un TÜKETİCİSİ oldu,
`caustica.progress` sunumu (tqdm opsiyonel; önizleme 8 periyotta bir, varsayılan açık).

Şüpheci doğrulayıcının mutasyon turu: T1/T3/kapı/hata-yolu mutasyonlarının beşi de testleri
kırıyor (taşıyıcılık kanıtı); bit-aynılık ve status.json/run_meta sözleşmesi bağımsız repro
edildi (beş yol: taze + iki resume + max_hours=0 + t_end tabanı). İki gerçek boşluk kapatıldı:
önizleme KADANSI testsizdi (8→1 ve 8→100000 süiti yeşil bırakıyordu — artık sabitli) ve heartbeat
periyot sayacı. Hız ölçütü dürüstçe `[~]`: ölçüm var (+%0.1–0.9, %5 kapısının çok altında),
CI zamanlama kapısı BİLEREK yok (paylaşılan koşucuda yanlış kırmızı üretir).

Süit 339 → **379 toplanan (377 passed / 2 skipped / 0 failed, 84.9 s)** — operatör ölçümü.
Ajan devlog yazarken API hatasıyla düştü; MILESTONES işaretleri çalışma ağacında doğruydu,
operatör doğrulayıp commit'ledi.

## 2026-08-22 — M10l kapandı: GUI sözleşmesi donduruldu (GUI kodu YOK)

Beş commit (`8a73464..02069c2`). Milestone'un tamamı "GUI ne zaman gelirse gelsin bunun üstüne
otursun" işiydi: yeni bağımlılık yok, `gui` extra'sı yok, ikinci repo yok, `src/caustica` içinde
GUI'yi bilen tek satır yok.

### Kapanan iki gerçek boşluk
**İptal sinyali.** `grep -rn "cancel" src/caustica` bugüne kadar boştu: GUI'nin "Durdur" düğmesinin
yazacağı hiçbir yer yoktu, süreci öldürmek TEK yoldu ve o da koşuyu kaybediyordu. Kanca zaten
duruyordu — `CheckpointSpec.stop_when`, periyot sınırında çağrılıyor. Eksik olan tek şey dosya
yoklamasıydı. Eklendi: çıktı klasöründe `cancel` görülünce checkpoint yazılır, çıkış 5.

Asıl karar dosyanın SONRASIYDI. İlk yazımda dosya bırakılıyordu; bu haliyle `--resume` ilk periyot
sınırında kendini iptal ederdi ve "resume kesintisizle bit-aynı" ölçütü hiç kapanamazdı. Şimdi
dosya TÜKETİLİYOR. İkinci sıra sorun: cancel görülüp henüz onurlandırılmadan öldürülen bir süreç
dosyayı bırakır ve o klasördeki HER resume'u sonsuza kadar iptal eder — bu yüzden bayat `cancel`
bir sonraki denemenin başında temizleniyor. Bunun kabul edilen bedeli sayfada da yazılı: `cancel`
KOŞAN bir işe sinyaldir, ön-iptal aracı değildir.

**Yapılandırılmış hata.** Koşu başlamadan olan hatalar (`hb` daha yaratılmamış) `return EXIT_CONFIG`
ile çıkıyordu: `status.json` HİÇ oluşmuyordu ve GUI'ye stderr ayrıştırmak kalıyordu. Artık çıktı
klasörüne `error.json` düşüyor — `{format, stage, exit_code, error_class, message, advice[],
written_at}`, altı `stage`. Planner'ın `est.advice`'i bugüne kadar yalnız ekrana basılıyordu; artık
dosyaya da giriyor. Tek kopya olması için `Refusal` yeniden kuruldu: `lines` alanı gitti, yerine
`headline` + `advice` listesi geldi ve `lines` bir property oldu.

Sözleşme EKLENTİ olarak tasarlandı, ikame olarak değil: çıkış kodları, stderr metinleri,
`status.json` alanları, `run_meta` ve checkpoint parmak izi DEĞİŞMEDİ. Bunu göz kararıyla değil
ölçerek doğruladım — ana ağaçtaki `runner.py` geçici olarak ebeveyn commit'inkiyle (`96e6330`)
değiştirilip yedi hata yolu (vram reddi · cpu kapısı · bozuk job · bilinmeyen gpu · bilinmeyen
backend · kesinti · checkpoint çakışması) iki kez koşturuldu: stdout ve stderr, geçici klasör
adları ve `warnings.warn`'un satır numarası dışında BAYT-AYNI. `error.json` yazılamazsa da hiçbir
şey değişmiyor: aynı kod, aynı mesaj, bir uyarı logu (`test_a_write_failure_for_error_json_changes_nothing`).

### Sözleşme sayfası ve çürüme testi
`docs/gui_contract.md` yüzeyin tamamını yazıya döküyor ve başında onu SONLU kılan cümle duruyor:
listelenmeyen hiçbir şey sözleşme değildir. Sonunda da tersi var — modül düzeni, stdout nesri,
`checkpoint.npz` içi, log metinleri ve herhangi bir IPC bilinçli olarak sözleşme DEĞİL.

Alan listeleri elle kopyalanmadı. `tests/test_gui_contract.py` dört gerçek koşu üretiyor (başarılı ·
kesilmiş · VRAM-reddi · store-çökmesi) ve her listeyi tarif ettiği şeyle karşılaştırıyor: klasör
listesi gerçek klasörle, `status.json` alanları ÜÇ gerçek status'un kesişimiyle (durum-bağımlı
ekler farkla türetiliyor — elle sayılmıyor), ilerleme anahtarları gerçek payload'la, çıkış kodları
ve format etiketleri koddaki sabitlerle. Sayfadaki CLI satırları gerçek argparse'tan geçiyor.

### Review turu — iki mercek, on bulgu
Mercek: (1) cancel yoklaması adım maliyetine sızdı mı, (2) error.json mevcut hata sözleşmesini
değiştirdi mi, (3) gui_contract gerçekle çelişiyor mu. Yanına bir de mutasyon turu.

**Q1 — sızmadı.** `stop_when` yalnız `_period_boundary`'den (settle döngüsü + `t_end` doldurma
döngüsü) ve kayıt penceresi öncesindeki TEK yoklamadan çağrılıyor; `step()` içinde hiçbir çağrı
yok ve `engine.py` bu milestone'da hiç değişmedi. Ölçüm: 68 adım / 15 periyot → 16 stat, yani
tam olarak `N+1`. Sayfa "periyot başına bir" diyordu; `+1` (kayıt penceresi öncesi) eklendi.

**Q2 — değiştirmedi, ölçülerek.** Ana ağaçtaki `runner.py` geçici olarak ebeveyn commit'inkiyle
(`96e6330`) değiştirilip yedi hata yolu iki kez koşturuldu: stdout ve stderr, geçici klasör adları
ve `warnings.warn`'un satır numarası dışında BAYT-AYNI. `Refusal`'ın `lines` alanından
`headline` + `advice`'e taşınması iki kapının da metnini kılına dokunmadan koruyor.

**GERÇEK bulgular (hepsi onarıldı):**
1. `docs/gui_contract.md` GUI'ye ön-koşu reddi için `caustica.SimulationRefused` yakalamasını
   söylüyordu. Böyle bir sınıf YOK — facade hem kapılar hem çöken çözüm için
   `SimulationError(mesaj, exit_code)` atıyor. `runner.py`'deki `Refusal` docstring'i de aynı
   yanlış adı taşıyordu (M10i'den beri). Delik kapatıldı:
   `test_every_caustica_name_on_the_page_actually_exists` sayfadaki her `caustica.AD`'ı ve noktalı
   yolu gerçek pakette çözüyor; mutasyonla doğrulandı.
2. **`--dry-run` önceki koşunun `error.json`'ını SİLİYORDU.** Klasörü temizleme skip-guard'dan ve
   dry-run dönüşünden ÖNCE duruyordu, yani bir "sığar mı?" sorgusu GUI'nin gösterdiği teşhisi yok
   ediyordu. Dry-run artık bir PROBE: `error.json`'a da `cancel`'a da hiçbir yönde dokunmuyor
   (`record_failure` sarmalayıcısı + temizlemenin gerçek koşuya taşınması) —
   `test_dry_run_never_touches_the_failure_record_or_the_cancel_file`,
   `test_dry_run_of_a_broken_job_writes_no_error_json`.
3. **skip-guard `cancel`'ı yiyordu.** Tamamlanmış bir klasörde ikinci bir süreç, birincisine
   gönderilmiş Durdur isteğini siliyordu. Skip-guard artık yalnız `error.json` temizliyor —
   `test_the_skip_guard_clears_the_stale_error_but_not_a_cancel`.
4. **`cancel` bir DİZİN ise kilitlenme.** `unlink` dizinde `OSError` atıyor ve yutuluyor, `exists()`
   sonsuza dek `True` → o klasördeki her koşu (resume dahil) periyot 1'de duruyordu. Yoklama artık
   `is_file()` — `test_a_cancel_directory_cannot_livelock_the_folder`.
5. **`_error_outdir` `ensure_dir_verified`'ı atlıyordu.** Hata yolu, dosyanın geri kalanının
   kullandığı Drive-FUSE sertleştirmesi olmadan klasör yaratıyordu. Artık aynı yardımcıdan geçiyor.
6. Ufaklar: `_now_iso()` "asla maskeleme" bloğunun DIŞINDAydı (içeri alındı); store-aşaması
   advice'i `native`'e bakıp `ck_path.exists()`'e bakmıyordu (solve-aşamasıyla hizalandı).

**ABARTILAR (hepsi daraltıldı):** "`--dry-run` çıkış 0" — VRAM reddi dry-run'da da 3 döndürüyor ve
bu sorunun CEVABI, arıza değil (CPU kapısının bilinçli istisnası ayrıca yazıldı; ikisi de artık
`test_the_documented_dry_run_exit_codes_are_the_real_ones` ile çivili) · "`error.json` ancak ve
ancak son deneme başarısızsa vardır" — öldürülen süreç HİÇBİR ŞEY yazmaz, o yüzden yokluğu
"sınıflandırılmış hata kaydedilmedi" demek, "koşu iyi" demek değil · `resolved_backend` "`auto`'nun
seçeceği" değil, koşunun GERÇEKTEN çözdüğü backend · "preview + metrics birlikte ≤10 MB" — 10 MB
bütçesi preview'un ve yalnız onun sıkıştırılmış baytları üzerinde ölçülüyor · "`--dry-run` yalnız
job/plan yazar" NATIVE hal · çıkış kodu kümesi "kapalı" deniyordu ama sınıflandırmadan kaçan bir
istisna hâlâ 1 döndürüyor — sayfaya yazıldı · preview paketinin içeriğinde `p_max` düzlemi eksikti.

**ÇÜRÜTÜLEN şüpheler** (araştırıldı, bulgu ÇIKMADI): `Refusal`'ın `_CLASSES`'ı dataclass alanı
olmuş olabilir (annotation yok, `@dataclass` görmüyor) · `_error_outdir` asıl hatayı maskeleyebilir
(gövde tümüyle `try/except Exception -> None`) · error.json başarılı koşuda yazılabilir (başarı
yolunda çağrı yok) · sayfadaki `caustica-result/1` attr listesi uydurma (gerçek `result.h5`'e karşı
denetlendi; 18 kök attr + 6 damga + 2 nesir sözleşmesi tam) · "resume bit-aynı" abartı (tersine,
ölçüldüğünden GÜÇLÜ: iptal → resume sonrası `result.h5` SHA-1'i kesintisiz koşununkiyle aynı) ·
`env_report()` istisna atabilir (her olgu ayrı ayrı korumalı) · çıkış kodu anlamları
(CPU reddi gerçekten 2, store çökmesi gerçekten 4) · sayfadaki dört CLI alt komutu ve bayrakları.

**Süreç bulgusu.** Çürüme testi alan LİSTELERİNİ sıkı çiviliyor — ve hiçbir liste yanlış değildi.
Turun BÜTÜN bulguları listelerin arasındaki NESİR cümlelerdeydi, yani liste karşılaştırmasının
yapısal olarak ulaşamadığı sınıf. Ucuz olan nesir iddiaları (dry-run çıkış kodları, `caustica.AD`
varlığı) artık kendi testlerini taşıyor.

### Bilerek yapılmayanlar
- **Kesinti (çıkış 5) `error.json` YAZMAZ.** Durmak başarısızlık değildir; `status.json` "interrupted"
  diyor ve kod 5. İptal ile çökme aynı dosyaya düşseydi ikisi karışırdı.
- **kwave iptal edilemez** ve bunu stdout'ta SÖYLÜYOR. Checkpoint almıyor, yani durulacak sınır yok;
  onu "iptal etmek" öldürmek olurdu, ki dosyanın var oluş sebebinin tam tersi.
- **VRAM reddi `--dry-run`'da da 3 döndürüyor.** 0'a çevirmek çıkış kodu davranışını değiştirirdi
  (M10l'in kuralı: kodlar donuk). Bunun yerine sayfaya yazıldı: 3 sorunun cevabıdır.
- **Öldürülen sürecin kaydı YOK.** `status.json` "solving"de kalır. Bunu düzeltmek bir kalp atışı
  zaman aşımı sözleşmesi ister (kuyruk işi, M10g) — burada uydurulmadı, sınır olarak yazıldı.
- Altıncı çıkış kodu EKLENMEDİ: iptal, mevcut 5'i kullanıyor. Kod kümesi kuyruğun API'si.

Süit 379 → **417 (415 passed / 2 skipped / 0 failed)**; `ruff check .` + `format --check` temiz;
`git status --porcelain data/setups/` boş — dokuz kurulum dosyası + manifest bayt-aynı.

---

## 2026-08-22 — M10f: Colab köprüsü (`caustica.colab`) + bayt-donuk notebook — Colab kapıları AÇIK

Kod tarafı bitti, milestone `[~]`: iki ölçüt CANLI Colab oturumu istiyor ve CPU kanıtıyla
işaretlenmedi (aşağıda "Kapanmayanlar"). Süit 417 → **449 (447 passed / 2 skipped)**;
`ruff check src tests` + `format --check src tests` temiz; `git status --porcelain data/setups/`
boş (dokuz kurulum bayt-aynı).

### Köprünün NE OLMADIĞI (asıl tasarım kararı)
`caustica.colab` L4: nerede koştuğuna dair fikri olabilen TEK katman. Ama alt katmanların hiçbirini
değiştirmiyor ve hiçbirini yeniden yazmıyor. Üç şey ekliyor:

1. **Ortam hükmü, HİÇBİR ŞEY hazırlanmadan önce.** `env_report()` basılır, sonra GPU ZORUNLU
   kılınır. Sıra kasıtlı: uygunsuz runtime bir mesaja mal olur — indirme yok, klasör yok,
   multi-GB medium build yok. (`test_a_gpu_less_runtime_is_refused_before_anything_is_prepared`
   `_fetch` ve `run_job_file`'ı patlayıcıyla değiştirip klasörün yaratılmadığını da ölçüyor.)
2. **`/content` altında varsayılan çıktı klasörü.**
3. **Bitmeyen koşu için okunur hüküm**, runner'ın KENDİ `error.json`'ından toplanmış — GUI'nin
   yönlendireceği mesaj ve advice satırlarının aynısı, burada uydurulmuş ikinci bir teşhis değil.

**VRAM kapısı köprüde TEKRARLANMADI.** Tek kopya `runner.check_gates`'te duruyor: plan-first ve
BOŞ VRAM'e karşı. Köprüde bir kopyası olsaydı yapabileceği tek şey ondan sapmaktı. Köprünün
eklediği şey runner'ın bilerek yapMAdığı kontrol: `backend="auto"` GPU'suz makinede sessizce
numpy'a düşer, ki Colab'da bu "kimsenin istemediği saatlerce CPU koşusu" demek.

### İki ret, iki mesaj (K6)
`caustica.env.require_gpu` "cupy kurulu değil" ile "cihaz yok"u AYIRMIYOR — makineye göre ayırıyor.
Colab'da cupy gerçekten yoksa mevcut mesaj "bu Colab runtime'ında CUDA cihazı yok (GPU runtime'ları
zaten cupy ile gelir)" diyor ki yanıltıcı. Köprü env.py'yi DEĞİŞTİRMEDEN eksik ayrımı ekliyor:
`importlib.util.find_spec("cupy")` ile "kurulu mu" sorulur (import edilmez), kurulu DEĞİLSE köprü
kendi mesajını atar; kuruluysa `require_gpu()` ÇAĞRILIR — yani "cihaz yok" cümlesi hâlâ tek yerde,
env.py'de yaşıyor. Test: `test_missing_cupy_and_a_cpu_runtime_are_two_different_messages`
(a'da "no CUDA device" GEÇMEZ, b'de "pip install cupy" GEÇMEZ).

Bilinen ve kabul edilen: BOZUK bir cupy kurulumu (metadata var, import patlıyor) ikinci mesaja
düşer, çünkü önce cihaz probu başarısız olur. Kurulum komutu zaten bir satır yukarıda.

### Notebook: 5 hücre, tek düzenlenen satır
`notebooks/colab_run.ipynb` — markdown + `pip install` + `CONFIG` + `run_job(CONFIG)` +
`show(outdir)`. Kilit `tests/test_colab.py`'de: hücre sayısı, hücre tipleri ve hücre içerikleri
BAYT BAYT sabit şablonla karşılaştırılıyor. Şablon test dosyasında DURUYOR (paylaşılan bir
sabitten okunsaydı test totoloji olurdu) — yani notebook'u düzenlemek testi kırar, ve düzeltmek
şablonu bilerek değiştirmeyi gerektirir. Yanında üç kilit daha: saklı çıktı/execution_count yok,
tek literal atama `CONFIG`, ve notebook'ta `run_job`/`show` dışında çağrı, `def`, döngü, ikinci
import yok (AST ile).

### Drive: sıfır satır, ve ölçüt cümlesi düzeltildi
**Ölçüt düzeltmesi (operatör, 2026-08-22):** MILESTONES'ta yazan "`grep -ri drive src/caustica`
boş" LAFZEN yanlıştı — job şemasındaki `drive` bölümü AKUSTİK sürüştür (`f0`, `amplitude`) ve her
yerde geçer. Niyet Google Drive; kontrol şu hale getirildi:

```
grep -rniE "drive\.mount|/content/drive|google\.colab" src/caustica --include=*.py
src/caustica/colab.py:38   ``google.colab`` is never imported either. ...   (docstring nesri)
src/caustica/colab.py:116  ``google.colab`` itself is never imported —      (docstring nesri)
src/caustica/env.py:102    "google.colab" in sys.modules or ...             (M10i, _on_colab)
src/caustica/progress.py:189  "ipykernel" in sys.modules or "google.colab" in sys.modules  (M10j)
```

Yani: Drive deseni SIFIR; `import google` SIFIR; `google.colab` yalnız `sys.modules` YOKLAMASI ve
iki nesir cümlesi. Operatörün "yalnız colab.py'de eşleşir" beklentisi tam olarak KARŞILANAMADI ve
karşılanmamalıydı da: `env.py:102` ile `progress.py:189` M10i/M10j'den beri duruyor, ikisi de
sözleşmesi dondurulmuş dosya, ve köprü onları DEĞİŞTİRMEDEN yeniden kullanıyor. `colab.on_colab()`
kendi probunu tanımlamıyor, `env._on_colab`'i çağırıyor — ikinci bir tanım `require_gpu`'nun aynı
makine için seçtiği mesajla çelişebilirdi (`test_the_bridge_reuses_the_one_colab_probe`).

### Runner sözleşmesine dokunulmadığının kanıtı
Köprüden geçen koşular gerçek runner yolunu koşuyor (GPU kapısı test içinde `cupy_available`
yamalanarak geçiliyor; koşunun kendisi numpy'da GERÇEKTEN çalışıyor, mock değil):
`test_run_job_produces_the_ordinary_run_folder_and_returns_it` (job/plan/status/result/metrics
yerinde, `error.json` YOK) · `test_a_failed_run_raises_with_the_runners_own_exit_code` (çıkış 2,
`error.json` `caustica-error/1`, mesaj birebir alıntı) · `test_a_cancelled_run_reports_exit_5_and_how_to_continue`
(cancel → 5 → checkpoint var, `error.json` YOK, sonra `resume=True` koşuyu bitiriyor) ·
`test_an_oom_refusal_adds_the_colab_lever` (3 + runner'ın kendi başlığı + Colab'a özgü kol) ·
`test_run_job_passes_dry_run_through_untouched`.

### Bilerek yapılanlar / yapılmayanlar
- **CPU kaçış deliği YOK.** `run_job` "GPU koşusu" demektir; CPU'da denemek isteyen `caustica run`
  veya `caustica.simulate` kullanır. Bir `require_gpu=False` bayrağı, kapının var oluş sebebini
  sessizce iptal ederdi.
- **`progress` varsayılanı `"auto"`** (kütüphane varsayılanı sessiz kalıyor): köprü bir giriş
  noktası ve karşısında hücreye bakan bir insan var — facade'ın D21/K11 kararının aynısı.
- **Varsayılan klasör adı job DOSYASININ stem'inden**, job'un `name` alanından DEĞİL: alanı okumak
  job'u burada ayrıştırmak demekti, ve ayrıştırılamayan bir job runner'dan (çıkış 2 + `error.json`)
  düşmeli, köprüden traceback olarak değil. Bu yüzden `run_job` her zaman açık bir `out` geçiyor;
  runner'ın kendi varsayılanı buradan hiç devreye girmiyor (tek çağrıda tek kural).
- **URL desteği** stdlib `urllib` ile eklendi (yeni bağımlılık yok, notebook CONFIG satırı "yol
  VEYA URL" sözleşmesi bunu istiyor). Bayt kaydeder, içine BAKMAZ — job olmayan bir dosya yine
  runner'ın çıkış koduyla düşer. Sınır belgelendi: URL'den gelen job SELF-CONTAINED olmalı, çünkü
  göreli yollar job dosyasına göre çözülür (T4) ve o artık indirme klasörü.
- **`show()`** raporu `caustica.report.render_report` ile çiziyor — `caustica report <folder>`'ın
  TA KENDİSİ. Hücredeki figürlerle klasördeki figürler tek eser; matplotlib yoksa sayılar yine
  basılıyor, figürler atlanıyor.

### Kapanmayanlar (CANLI Colab oturumu ister — CPU kanıtıyla işaretlenmez)
- Uçtan uca Colab kapısı: repodan açılan notebook → `/content` altında koşu → sonucu indirip
  lokalde `caustica report`.
- Üç kapının birlikte kapanması: M7 parite + tam boy OOM'suz koşu · M8 VRAM ±%10 ve kalibre süre
  ±%25 · bu E2E. Ölçüm altyapısı hazır: `run_meta.json` `planner` ve `actual`'ı yan yana taşıyor,
  `caustica.colab.summary()` ikisini tek satırda basıyor.
- README'deki "Open in Colab" rozeti ve notebook'un varsayılan URL'leri `library-first`
  master'a MERGE edilene kadar 404 — aşağıdaki inceleme turu S6'ya bakın (M10e ön koşulu).

### İnceleme turu (hafif review + şüpheci doğrulayıcı, 2026-08-22) — 16 bulgu, 13'ü düzeltildi

İki ajan paralel koştu: biri "köprü alt katman sözleşmelerine sızdı mı / notebook şablonu gerçekten
kilitli mi" merceğiyle, biri iddiaları ÇÜRÜTMEYE çalışarak. İkisi de aynı iki gerçek sızıntıyı
buldu.

**S1 — Mesaj var olmayan bir dosyayı işaret ediyordu.** `_failure_message` her ret için
"error.json carries the same verdict" satırını basıyordu. Ama sözleşme İKİ durumda hiç
`error.json` YAZMAdığını söylüyor: kesinti (durmak başarısızlık değil) ve `--dry-run` (prob deneme
değil). Yani mesaj, sayfanın "olmayacak" dediği bir dosyaya yolluyordu — ve kendi testimiz
(`assert not (out / ERROR_FILE).exists()`) bunu ispatlayıp yine de geçiyordu. Üstelik dry-run VRAM
reddinde runner'ın asıl başlığı yalnızca stderr'e gidiyor, istisnaya hiç girmiyordu. Şimdi: kayıt
VARSA `stage` ve `error_class` alıntılanıyor (sayfanın "bir program bunlara bakar" dediği iki
alan), YOKSA neden yok olduğu söyleniyor ve `plan.json`'daki `advice` satırları çekiliyor.

**S2 — Çıkış kodu 2'nin açıklaması uydurmaydı.** `_EXIT_MEANING[2]` "config error — the job would
not load or build" diyordu; bu sayfadaki `stage: config` AÇIKLAMASI, çıkış kodunun anlamı değil.
Sonuç: bir CPU-süre reddi (çıkış 2, ama `stage: gate`, `error_class: CpuTimeRefusal`) ekrana
"job yüklenemedi" diye yazılıyordu — aynı blokta `error.json` bunun tersini söylerken. Şimdi
metinler sayfanın tablosundan birebir alınıyor ve **test tabloyu ayrıştırıp karşılaştırıyor**
(`test_the_exit_code_glosses_are_the_documented_ones`); eski metin bu testi geçemiyor (ölçüldü).

**S3 — Job'un kendi `output.folder`'ı sessizce eziliyordu.** Köprü her zaman açık bir `out`
geçtiği için runner'ın kuralı hiç devreye girmiyor; aynı job `caustica run` ile başka yere,
köprüyle başka yere düşüyordu. Kuralı köprüde ÇOĞALTMAK (göreli yol çözümü dahil) daha kötü bir
sızıntı olurdu, o yüzden karar: ezmeye devam ama SESSİZ DEĞİL — job bir klasör adı veriyorsa
ekrana "bu job şunu istiyor, köprü şuraya yazıyor, `out=` ile kendin seç" notu düşüyor. JSON'a
bakış tamamen savunmacı (`try/except: return`), yani ayrıştırılamayan job yine runner'dan düşüyor.

**S4 — Hatalı bir keyword indirmeye mal oluyordu.** `_fetch` `RunnerOptions` doğrulamasından ÖNCE
koşuyordu: `run_job(url, nonesuch=1)` dosyayı indirip sonra `TypeError` atıyordu. Modülün var
oluş sebebi tam olarak bu sıra. Düzeltildi + testli (`test_a_bad_keyword_costs_no_download`).

**S5 — Notebook kilidinin 9 deliği.** Bayt-bayt şablon karşılaştırması GERÇEKTİ (hücre ekleme,
sıra değişimi, tek boşluk, CONFIG değişimi, markdown düzenlemesi hepsi yakalanıyordu). Ama
şablonu "bilerek" güncelleyen biri için semantik testler çok zayıftı. En kötüsü: `!` içeren hücre
TÜMÜYLE atlanıyordu, yani kurulum hücresi hiçbir AST kontrolünden geçmiyordu — oraya
`os.environ['CAUSTICA_CPU_LIMIT_MIN'] = '9999'` (belgeli bir kapıyı sessizce kapatan satır),
`import os`, döngü ve fazladan `!wget` konulabiliyordu ve altı kilit testi de geçiyordu. Ayrıca
attribute çağrısı, lambda, ternary, comprehension "mantık" sayılmıyordu; çağrı ARGÜMANLARINA hiç
bakılmıyordu (`run_job(CONFIG, allow_slow_cpu=True)` serbestti); notebook metadata'sı
(`accelerator: GPU` silinebiliyordu) ve hücre metadata'sı (`cellView: form` — Colab'da hücre
kaynağını GİZLER ama yine çalışır) tamamen serbestti.

Hepsi kapatıldı: magic satırları atlanmıyor BOŞALTILIYOR (yani kurulum hücresinin geri kalanı
ayrıştırılıyor), mantık düğüm listesi 20 tipe genişledi, argümanlar isim olmak zorunda, `!`
satırı tektir ve `!pip install` ile başlar, notebook metadata'sı ve her hücrenin BOŞ metadata'sı
çivilendi. Aynı 9 mutasyon yeniden koşuldu: **9/9 yakalanıyor**, değiştirilmemiş notebook geçiyor.

**S6 — `master`'a bakan üç şey bugün kırık.** `origin/master` hâlâ f0bff2f'te, yani yeniden
adlandırma öncesinde: rozet 404, varsayılan `CONFIG` ham URL'si 404 ve — en sinsisi — notebook'un
`pip install git+https://github.com/ebx0/caustica` satırı VARSAYILAN dalı kurar, o dal hâlâ
`hifusim` paketini taşır, dolayısıyla `from caustica.colab import run_job` ModuleNotFoundError
verirdi. "Push" yetmez; `library-first` master'a MERGE olmalı. Colab kapısı bu yüzden bugün
DENENEMEZ. MILESTONES'ta o madde `[x]` iken `[~]`e çekildi.

**S7 — abartılar daraltıldı.** "prepares NOTHING" → "writes nothing to disk" (auto-fallback
uyarısı, cupy prob önbelleği ve GPU'lu makinede CUDA context'i gerçek yan etkiler — runner da
aynılarını yapıyor, ama iddia dar olmalı) · "The install command is still one line above it"
Colab'da YANLIŞTI (env'in Colab mesajında çalıştırılabilir bir kurulum komutu yok; docstring
düzeltildi ve bunun env politikasının işi olduğu yazıldı) · dönen klasör içeriği listesi
`preview_only`/`dry_run`/`kwave` istisnalarıyla nitelendi · "an https URL" → `http(s)` ·
`show()`'un `None` dönüşü artık üç ayrı sebebi de basıyor ve `caustica report`'un aynı hatada
çıkış 2 verdiğini söylüyor · MILESTONES'taki "testli" (Drive mount hikâyesi) daraltıldı: test
yalnızca "açık `out=` kazanır"ı kanıtlıyor, geçici bir dizinle — mount değil.

**S8 — `require_gpu_here` ile `env.require_gpu` arasındaki metin ikizi çivilendi.** Kurulum
cümlesi artık tek bir sabit (`colab.INSTALL_ADVICE`) ve bir test env'in mesajının hâlâ onu
içerdiğini doğruluyor.

**S9 — sözleşme sayfası.** `docs/gui_contract.md`'nin "Explicitly not contract" listesine
`caustica.colab` eklendi: köprü bilerek FİKİRLİ bir katman, bir GUI ona değil `caustica run` /
`caustica.simulate` + sayfaya dayanmalı. Aynı yerde "Drive üzerinden senkronize olan Colab
oturumu" cümlesi "KULLANICININ mount ettiği Drive klasörü; caustica kendisi mount etmez" diye
netleştirildi. `docs/library_first_plan.md` W6'daki "planner VRAM tahmini — hiçbir şey
hazırlanmadan reddet" cümlesi de düzeltildi: o yarı bilerek YAPILMADI (kapı runner'da tek kopya),
ve runner'ın VRAM reddi medium kurulduktan sonra gelir.

### ÇÜRÜTÜLEMEYEN iddialar (şüpheci ajan denedi, bulgu ÇIKMADI)
"caustica hiçbir şey kurmaz" (tüm `src/caustica`'da tek shell çağrısı `git rev-parse`) · "Drive
yok / `google.colab` import edilmez" (dolaylı yollar da tarandı: `__import__` yok, çalışma
zamanında kurulan modül adı yok, subprocess ile mount yok) · dört ret mesajının gerçekten farklı
olması · çıkış kodlarının yeniden eşlenmemesi · süit sayıları (bağımsızca 449 collected, JUnit
XML'den 447 passed / 2 skipped) · GPU kutusunun CPU kanıtıyla işaretlenmemiş olması.

### Düzeltilmeyenler (bilerek)
- **Bozuk CUDA yığını olan GERÇEK bir Colab GPU runtime'ı** `env.require_gpu`'nun "runtime GPU
  değil, Runtime menüsünden GPU seç" mesajını alır — yani zaten üstünde olduğu şeye geçmesi
  söylenir. Doğru düzeltme env.py'de (M10i) ve bu milestone'un işi değil; köprüye tahmine dayalı
  ÜÇÜNCÜ bir mesaj eklemek daha kötü olurdu. Docstring'e ve buraya yazıldı.
- **GPU reddi `RuntimeError`, `SimulationError` değil.** Hiçbir şey koşmadı, yani sınıflandırılacak
  bir koşu yok — ve `require_gpu`'nun tipini yeniden etiketlemek env sözleşmesine dokunmak olurdu.
  Tuzak belgelendi: `SimulationError` zaten `RuntimeError`'ın alt sınıfı, o yüzden `.exit_code`
  okuyan bir `except` önce `SimulationError`'ı yakalamalı.
- **`_fetch` boyut sınırı yok.** Bir job JSON'u; keyfi bir üst sınır uydurmak politika icat etmek
  olurdu. Atomik yazıma geçirildi (yarım inen ama yine ayrıştırılabilen bir job daha kötü).

---

## 2026-08-22 — M10l sertleştirme turu: mutasyon-doğrulayıcısının dört test boşluğu + bir yanlış nesir

M10l'in mutasyon-doğrulayıcısı davranış hatası bulmadı — **kapsama borcu** buldu: dört yerde
test, iddia ettiği şeyi ölçmüyordu. Bu tur onları kapattı. Her boşluk için doğrulayıcının
mutasyonu **izole worktree'de** yeniden koşuldu; aşağıdaki "önce yeşil / sonra N kırmızı"
sayıları ölçümdür, tahmin değil. Süit **460 → 465** (463 passed, 2 skipped).

**B1 — cancel yoklama sayacı yanlış şeyi sayıyordu.** `test_cancel_poll_is_one_stat_per_period`
yalnız `Path.is_file`'ı enstrümante ediyordu. Aynı regresyon `os.path.isfile(cancel_path)` diye
yazılıp adım döngüsüne taşındığında runner süitinin **36/36'sı yeşil kaldı** (ölçüldü: sözleşmenin
~16'ya izin verdiği yerde 77 yoklama). Sebep Windows/py3.12'ye özgü: `os.path.exists`/`isfile`
orada `nt._path_exists`/`nt._path_isfile`, yani `os.stat`'a hiç inmeyen C kısayolları — primitifi
izlemek yetmiyor. Çözüm: yoklama runner'da **tek bir fonksiyona** toplandı (`_cancel_requested`)
ve test o fonksiyonun çağrı sayısını sayıyor (hangi primitifi kullandığından bağımsız), ayrıca
`Path.is_file/is_dir/exists/stat`, `os.path.isfile/isdir/exists/lexists` ve `os.stat/lstat`'ı
"helper'ı atlayan yoklama" için ayrı izliyor. `polls > 0` da eklendi: eski `assert len(polls) <=
boundaries + 1` sıfırda BOŞLUKTA-DOĞRUYDU, yani hiç yoklamayan bir stop butonu da yeşildi.
Mutasyon yeniden koşuldu → **1 kırmızı** (77 bypass yoklaması adlarıyla raporlanıyor).

**B2 — boş `advice` görünmüyordu.** `assert all(isinstance(a, str) ... for a in advice)` boş
listede boşlukta-doğru. Yazım noktasında tüm advice demetlerini boşaltmak HEAD'de **36/36 yeşil**
bıraktı. Artık on hata sınıfından dokuzu listenin BOŞ OLMADIĞINI iddia ediyor; onuncusu
(checkpoint'siz solver çökmesi) söyleyecek eyleme dönük bir şeyi olmadığı için BİLEREK boş iddia
ediyor — oraya cümle uydurmak sessizlikten kötü olurdu. Ayrı bir test iki kapının stderr
render'ını da çiviledi: `Refusal.lines`'tan `  -> ` satırlarını silmek eskiden görünmezdi; test
artık basılan satırların ve `error.json`'daki `advice[]`'in **aynı liste** olduğunu doğruluyor
(dosya için tutulan bir kopya sürüklenen bir kopyadır). → **11 kırmızı** + **2 kırmızı**.

**B3 — çıkış kodu sabitlerinin gerçek değerleri çivisiz.** `EXIT_OOM = 6` yapmak
`tests/test_runner.py`'de hiçbir şey kırmıyordu, çünkü her assert sabiti kendisiyle
karşılaştırıyordu. (Sayfa testi yakalardı; dosyanın kendi kendine yetmemesi asıl borçtu.) Beş kod
artık literal sayıya çivili, artı ayrıklık, artı 1'in kümede OLMAMASI ("sınıflandırmadan kaçan
istisna" için ayrılmış), artı çıkış kodu sayı olarak karşılaştırılan iki gerçek koşu. →
**1 kırmızı**.

**B4 — yanlış nesir, ama kod gereksiz DEĞİL.** `RunInterrupted` handler'ındaki
`_clear_stale(cancel_path)` yorumu "leaving it would cancel the --resume too... forever" diyordu;
aynı cümle `docs/gui_contract.md`'deydi. GÜNCEL koda karşı (d3da3f1 sonrası) yeniden ölçüldü:
satır silindiğinde iptal 5 veriyor, resume 0 veriyor ve sonuç **bit-aynı** — iddia YANLIŞ; resume'u
kurtaran şey her gerçek koşunun BAŞINDAKİ temizlik. Ama satır **tutuldu**, çünkü başka ve gerçek
bir işi var (sayfanın zaten söz verdiği bir iş): süreç çıkar çıkmaz durmuş bir klasör, kimsenin
yerine getirmeyeceği bir durdurma isteğini ilan etmiyor; klasörü yoklayan bir GUI yerleşmiş bir
durum görüyor. Yorum ve sayfa artık hangisinin yük taşıdığını (baştaki) hangisinin kemer-üstü-askı
olduğunu (handler'ınki) söylüyor, ve **iki yarı da testle** sabit: handler'ın temizliği silinince
→ 1 kırmızı; baştaki temizlik silinince → 2 kırmızı (yeni test iptal sonrası `cancel` dosyasını
GERİ koyup resume'un yine de bit-aynı bitmesini şart koşuyor — eski yorumun ters çevirdiği iddia).

**B5 — sayfa nesri sözleşme değil, artık öyle diyor.** `docs/gui_contract.md`'deki her liste,
tablo ve literal koda karşı test ediliyor; aradaki nesir hiç edilmiyordu ve sayfa bunu
söylemiyordu — yani bir cümle, bir tablo satırıyla aynı ağırlıkta bir söz gibi okunuyordu. M10l
incelemesinin TÜM bulguları ve bu turun B4'ü birer cümleydi. Sayfa artık kapsamını başta beyan
ediyor ve beraberlik bozucuyu adlandırıyor: **nesir ile liste çelişirse sözleşme listedir.**

### Yan bulgu: M10f'ten kalma test-izolasyon hatası
`pytest tests/test_gui_contract.py` **tek başına 28996ac'de KIRMIZIYDI**:
`test_every_caustica_name_on_the_page_actually_exists` `hasattr(caustica, "colab")` kullanıyor,
ama bir ALT MODÜL ancak biri onu import ettikten sonra paketin attribute'u olur. Tam koşuda
yeşildi çünkü `tests/test_colab.py` önce çalışıp import ediyordu. Attribute olmayan adlar artık
import edilerek çözümleniyor.

### Süreç: mutasyon-doğrulayıcı olayı ve yeni kural
Eşzamanlı mutasyon testi sırasında bir mutasyon ana ağaçta kalıp **`02069c2`'ye süpürüldü**;
**`2b42e3f`** geri aldı. HEAD bayt-doğru (iki bağımsız doğrulama). **UYARI: bu iki commit
tarihçede ayrılamaz** — bu dal squash edilirse sorun yok, ama seçmeli rebase / cherry-pick
YAPILMAZ.

**Yeni kural (bu turda uygulandı):** mutasyon-doğrulayıcılar baştan **İZOLE worktree'de** çalışır;
ana ağaçta mutasyon yasak. Uygulaması ucuz: `git worktree add <scratch>/mut HEAD`, sonra
`PYTHONPATH=<scratch>/mut/src ./.venv/Scripts/python.exe -m pytest` — editable kurulum düz bir
`.pth` olduğu için PYTHONPATH onu geçersiz kılıyor, yani ana ağaç hiç dokunulmadan mutasyon
koşuyor. Bu turdaki altı mutasyonun hepsi orada koştu.

## 2026-08-22 — Büyük yeniden planlama: "piyasanın en iyisi" hedefi, K18–K21

Kullanıcıyla 24 soruluk karar turu + üç araştırma raporu (research/landscape_2026.md + ITRUSST/ML
ve differentiable/GPU alt-raporları). Kritik landscape bulguları: k-wave-python v0.6 saf-CuPy
çözücü çıkardı ("saf Python" farkımız kapandı); j-Wave 23 aydır bayat (differentiable NONLİNEER
niş BOŞ); ITRUSST self-serve ve donmuş (giriş = 18 permütasyon + kendi karşılaştırma makalen);
Stride/Openwater AGPL (ticari gömülemez — MIT/LGPL tarafı bize açık); "en doğru" iddiası
kazanılamaz, ayrışma kafatası haritalamada.

Kararlar: kimlik = sözleşmeli çok-motor çatı + native aile (K18) · HIFU-önce, doğruluk-önce
(K20) · v0.1 = ITRUSST dokuzunun tümü akustik-yalnız + JOSS + PyPI (K19) · görüntü köprüsü
entegrasyon-önce [imaging] extra (K21) · KZK ertelendi (AS öne) · M12→M11, M13+M14+M10g→M29
birleşti · adjoint/ML fizibilitesi M28 olarak ikinci fazda (kullanıcı, boş-niş bulgusuna rağmen
v0.1 odağını korudu) · dataset vitrini en sona.

Yeni sıra: CANLI COLAB OTURUMU (ilk iş) → M11 → M18 → M15 → M16 → M25/M26 → M27 → M19 →
M21→v0.1 → ikinci faz → vitrin. M9 KZK ertelendi-damgalı; eski M21 "−6dB 0.2–0.6 mm" kriteri
kaynaktan doğrulanamadığı için gerçek 2022 koridorlarıyla değiştirildi.

---

## 2026-08-23 — İlk Colab oturumunun üç düzeltmesi + GPU kapı süiti (`caustica.validation`)

Bu oturumun girdisi bir ÖLÇÜMDÜ: operatör 2026-08-22'de `water_bowl_mini`'yi gerçek bir
NVIDIA A100-SXM4-40GB'de koşturdu (`colab-run-results/`). Sonuç metrik seviyesinde kusursuzdu —
aynı job CPU'da koşuldu, tepe basınç bağıl fark 1.8e-7, geometri/−6dB/hacim birebir, yakınsama
yörüngesi aynı (periyot 11, 104 adım). Ama üç şey kırıktı ve hiçbiri fizikle ilgili değildi.

### [A] Üç düzeltme

**A1 — `git_commit: "unknown"` (commit `fb47b12`).** Colab wheel'den kuruyor; wheel'in yanında
checkout yok; runner'ın `git rev-parse`'ı boşa düşüyordu. İzlenebilirlik deliği tam da
tekrarlanması en zor makinedeydi. Çözüm build anında damgalamak: `setup.py` (proje metadata'sı
DEĞİL, yalnız bu kanca) `build_stamp.stamp()` çağırıyor, o da `src/caustica/_build_info.py`ye
VERSION/COMMIT/BUILT_AT yazıyor. `setuptools_scm` bilerek EKLENMEDİ: yeni bağımlılık getirir ve
elle tutulan sürüm türetimini de devralır.

Runtime tarafı `caustica.env.git_commit()`: önce canlı checkout (bir sonraki commit'ten sonra da
doğru kalan tek kaynak), yoksa gömülü damga, o da yoksa `"unknown"`. İki incelik:
- **Kuşatan repo reddediliyor.** `git rev-parse` yukarı yürür, dolayısıyla başka birinin
  repo'su içine kurulmuş bir venv o projenin HEAD'ini her koşuya damgalayabilirdi.
  `--show-toplevel == root` kontrolü hem build hem runtime kopyasında var; ikisi
  `test_env_and_build_stamp_agree_on_this_checkouts_head` ile birbirine çivili (build zamanı
  `caustica.env`'i import EDEMEZ — numpy henüz yok, o yüzden iki kopya var).
- **Sdist kuralı:** git yoksa ve damga zaten oradaysa ÜZERİNE YAZILMIYOR. Aksi hâlde sdist'ten
  wheel üretmek, o dağıtımın sahip olacağı tek provenance'ı `"unknown"` ile silerdi.

Kanıt: `tests/test_packaging.py`e 5 test — biri geçici bir git checkout'undan wheel kurup
`--target`'a yükleyerek repo'nun GÖRÜNMEDİĞİ bir cwd'den gerçek koşu yapıyor ve
`run_meta.git_commit == HEAD` diyor. CI'ın wheel bacağı aynı şeyi temiz venv'de tekrarlıyor.

**A2 — `t_step_measured_s` ısınmayı gizliyordu (commit `87573b9`).** Damga 26.6 ms/adım diyordu,
planner probu AYNI SÜREÇTE aynı şekil için 1.03 ms — 25.9×. Ama CPU kontrolü 0.96×. Bu ikisi
birlikte teşhisi veriyor: muhasebe doğru, model EKSİK. 2.77 s'lik bir koşuda ~2.66 s tek seferlik
cuFFT-plan + kernel-JIT + ilk tahsis maliyeti 104 adıma bölünüyordu.

- Planner artık `t_expected = warmup + steps·t_step` hesaplıyor. `model.GPU_WARMUP_S = 3.0`
  (o oturumdan, yukarı yuvarlanmış), `calibrate()` cihazda bir ısınma ölçüp saklıyor,
  `planner.record_warmup()` gerçek bir koşunun ÖDEDİĞİNİ geri yazıyor (prob tam bir çözüm
  değil — property map'leri kurmaz, kaynağı serpmez — bu yüzden eksik sayar).
- Runner tarafında `_StepTiming`, motorun ZATEN yaydığı periyot-sınırı payload'ını tüketiyor:
  yeni enstrümantasyon yok, ekstra device sync yok. Kararlı hız, sınırlar ARASI aralıkların
  medyanı; koşunun başından ilk sınıra kadarki aralık asla dahil değil, çünkü ısınmayı ödeyen
  tam olarak o aralık. Üçten az sınır varsa cevap `None` — destekleyemediği bir sayı yerine
  "ölçülmedi" demek.
- **Hiçbir anahtar değişmedi.** `t_step_measured_s` aynı tanımla duruyor (M8'in Colab kapıları
  ve `gui_contract` onu okuyor); eklenenler `warmup_s`, `t_step_steady_s`, `steady_samples` ve
  plan.json'da `warmup_s`. `gui_contract` testi zaten kırıldı ve doküman güncellendi — sözleşme
  sayfası makine-kontrollü olduğu için EKLEME bile sessizce geçemiyor (tasarım gereği).

**A3 — CI Colab'ın Python'unu test etmiyordu (commit `297752b`).** 3.10 (requires-python tabanı)
ve 3.12 (dev ortamı) vardı; her gerçek koşu 3.13.15'te oluyordu. ubuntu/3.13 bacağı eklendi.

### [B] `caustica.validation` — GPU kapı süiti (M11'in ilk parçası)

M7 ve M8 aynı yerde takılıydı: kalan ölçütleri ancak gerçek bir GPU cevaplayabilir ve **tek
koşu yetmez** (M8 "≥2 grid boyutu" ve "≥2 senaryo" diyor, M7 tam boy + parite + baseline
istiyor). Bu bir koşu değil PROTOKOL; o yüzden protokol olarak yazıldı:

    python -m caustica.validation gpu-gates

Sırayla: `env_report()` → GPU yoksa eyleme geçirilebilir mesajla çıkış 2 → hedef cihazda
`planner.calibrate()` (M8 "kalibrasyon SONRASI" diyor: süit o durumu VARSAYMIYOR, ÜRETİYOR) →
VRAM merdiveni (A100'de ~2/8/14/28 GiB; 14 GiB basamağı tam olarak M7'nin 512³ @ dx=0.30
sınıfı; büyükler `--preview-only`) → cihaza SIĞMAYAN EN KÜÇÜK şekil (çıkış 3 + öneri metni
kanıt olarak) → numpy/cupy paritesi → `benchmarks/reports/gpu_gates/<gpu>-<tarih>/` altına
damgalı MD+JSON + her basamağın KENDİ çıktı klasörü.

**Yanlış PASS'a karşı üç kural** (incelemenin merceği buydu):
1. İki tarafı da olmayan bir kontrol `SKIP` — asla PASS. (`predicted is None`, `actual == 0`,
   ölçülmemiş alan… hepsi.)
2. Kapı, milestone'un istediği SAYIDA geçen kontrol ister; bir tane bile FAIL varsa FAIL;
   hiç kontrolü yoksa `INCOMPLETE`, PASS değil. `required=0` bile PASS kaçıramıyor.
3. Adım sayısı planla uyuşmayan bir koşunun süre karşılaştırması SKIP — o karşılaştırma
   zamanlama modelini değil yakınsama sezgiselini ölçerdi. (Süit `min_settle == max_settle`
   sabitleyerek bunu zaten deterministik yapıyor; kontrol yine de var.)

**Parite kapısının ölçüm noktası operatör ölçümüyle DÜZELTİLDİ.** İlk oturumun `result.h5`i
yerel CPU koşusuyla relL2 3.6e-5 / relL∞ 4.883e-4 farkla uyuşuyordu — ve 4.883e-4 tam olarak
2^-11, yani **bir float16 ULP'u** (p_max'ta %99.17 voxel bit-özdeş, 517 voxel 1 ULP, >1 ULP
sıfır; büyük bağıl sapmaların hepsi sıfır civarı gürültü tabanında işaret biti). Alanlar
dosyanın çözünürlüğünün ALTINDA uyuşuyor. Kapıyı `result.h5` round-trip'i üzerine kurmak
kusursuz bir çözücüye 3.6e-5 okutup 1e-5 ölçütünde YANLIŞ FAIL verirdi. Kapı artık aynı süreçte
numpy ve cupy koşup **bellekteki fp32 `SolverResult` dizilerini** karşılaştırıyor; depolama
tabanı raporda `stored_float16_reference` başlığı altında ayrıca gösteriliyor ki okuyan kişi
~5e-4'ü regresyon sanmasın. Sentetik testi bu sayıyı yeniden üretiyor
(`test_the_parity_gate_is_measured_on_fp32_fields_not_on_a_stored_file`).

**CPU'da ne testlendi:** `Harness` dikişi — süitin dünyaya dokunduğu her yer tek bir
değiştirilebilir nesnede. Sahte bir cihaz, işleri planner'ın KENDİ envanteriyle fiyatlıyor
(yani model kendi kendini kayıramıyor), sonra `vram_error`/`time_error` kadar sapmış bir "ölçüm"
döndürüyor. Böylece merdiven, VERDICT cebri, OOM dalı, adım-sayısı uyuşmazlığı, rapor şeması,
M19 baseline'ı ve ısınmanın kapılardan SONRA geri yazılması (yoksa döngüsel olurdu) CPU'da
koşuyor. **GPU sayısı üretilmedi; M7/M8'in cihaza bağlı kutuları `[ ]` kaldı.**

Bir gerçek hata bu testlerden çıktı: OOM basamağı "boş VRAM'in 1.15 katının altındaki en büyük
şekil" olarak boyutlanıyordu ve 80 GiB'lık bir cihazda bu 75 GiB seçiyor — yani SIĞIYOR. Kapı,
reddedilmesi hiç istenmemiş bir koşuyla notlanacaktı. Şimdi "cihazı AŞAN en küçük şekil"
(`side_above`), testi de 15/39/79/140 GiB için parametrik. İkinci hata: `gpu_key_for` token'ları
küçültmeden noktalama siliyordu, `"A100"` → `"100"` ve `"SXM"` → `""` oluyordu; boş token her
şeyin alt dizesi olduğu için her cihaz eşleşiyordu (A100-80GB yerine H100-SXM dönüyordu).

Notebook: `notebooks/gpu_gates.ipynb` (4 hücre) — bakım notebook'u, kullanıcı akışı değil;
`colab_run.ipynb`in bayt-donuk sözleşmesine DOKUNULMADI. Kendi sözleşmesi daha sıkı: kod
hücrelerinde Python satırı OLAMAZ, sadece `!` ve `#`.

### Kanıt
- Tam süit: **465 → 514 test** yeşil, ruff temiz (`ruff check` + `format --check`).
- Yeni testler: `tests/test_validation_gpu_gates.py` (36), `tests/test_packaging.py` (+5),
  `tests/test_planner.py` (+4), `tests/test_runner.py` (+4).
- Açık kalan: M7'nin üç kutusu ve M8'in üç kutusu — **ikinci Colab oturumunda** ölçülecek.
