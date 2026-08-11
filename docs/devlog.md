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
