# caustica — Milestone Merdiveni ve Başarı Kriterleri

> Kural: **Bir milestone'un TÜM başarı kriterleri sağlanmadan bir sonrakine geçilmez.**
> Kriterler ölçülebilir yazılır; "geçti" kararı test çıktısı/rapor ile belgelenir (docs/devlog.md).
> Durumlar: `[ ]` başlanmadı · `[~]` devam ediyor · `[x]` TAMAMLANDI (kriter kanıtıyla) · `[!]` bloklu
> GUI bu merdivenin kapsamı dışında (kullanıcı kararı 2026-08-10; teyit 2026-08-19 ve 2026-08-21):
> Colab entegrasyon katmanı (M10b–M10g) GUI'nin ileride üstüne oturacağı kontratları hazırlar,
> **M10l** bunları yazıya döküp katmanlamayı testle kilitler. GUI **ayrı repoda** olacak
> (`caustica-gui`) ve teknolojisi seçilmedi. Planner/CLI çıktıları metin+figür tabanlı kalır.
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
- Not: proje "hifusim" çalışma adıyla başladı; 2026-08-21'de (M10e, kullanıcı kararı) PyPI/GitHub çakışma kontrolüyle **caustica** olarak yeniden adlandırıldı.

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

### M4 — Lineer k-space PSTD çözücüsü (numpy; 1D/2D/3D) `[x]` (2026-08-10)
Boyut-agnostik ilk tam dalga çözücü; CW + steady-state fazor çıkarımı.
- [x] `solvers.base.SolverBase` + yetenek deklarasyonu + `solvers.registry` (entry-point plugin desteğiyle)
- [x] 1. mertebe kuple denklemler (p, u), k-space gradyan/diverjans, kappa sinc düzeltmesi, exact-period dt, Gaussian sponge PML, CW kaynak enjeksiyonu (ramp'li), tek-bin DFT fazor + p_max
- FİZİK DÜZELTMESİ (testin yakaladığı, notebook'tan sapma): üstel absorpsiyon artık p VE u'ya simetrik uygulanıyor. Yalnız p sönümlenince uzamsal sönüm α/2 çıkıyor (dispersiyon analizi + enerji eşbölüşümü) — yani KAYNAK NOTEBOOK'UN dataseti etikettekinin YARISI kadar absorpsiyon gördü (dataset içi tutarlı; fiziksel yorum için kayda geçirildi)
- UYARI eklendi: PML'siz grid → periyodik sınır (dalga sarmalar); çözücü loud warning veriyor
- Başarı kriterleri:
  - Düzlem dalga (1D/2D/3D, periyodik yönde): faz hızı hatası < %0.1 @ 4 ppw, 50 periyot
  - Üstel absorpsiyon: ölçülen α, konfigüre α'dan < %1 sapar
  - PML: normal geliş yansıma genliği < giriş genliğinin %3'ü (≈ −30 dB)
  - Küçük 3D odaklı çanak (su, lineer) vs O'Neil: normalize eksenel profil r > 0.99; −6dB eksenel/lateral genişlik farkı < %5; odak konumu < 1 voxel
  - Fazor çıkarımı: saf CW sinüs girişinde genlik hatası < %0.5 (sızıntısızlık)
  - Kararlılık: 200 periyot koşuda enerji patlaması yok (peak drift < %1)

### M4b — `kwave` çözücü adaptörü (CPU/OMP) `[x]` (2026-08-10)
Kullanıcı kararı (2026-08-10, tur 4): k-Wave, registry'de DOĞRUDAN bir çözücü seçeneğidir
(`hs.solvers.get("kwave")`) ve doğrulama zincirinin MERKEZİ referanslarından biridir —
analitik süit (O'Neil/Rayleigh/Fubini) + k-Wave çapraz karşılaştırması birlikte "doğru" tanımıdır.
- [x] `k-wave-python` opsiyonel bağımlılık (`pip install caustica[kwave]`); yokken registry kaydı
      duruyor, `run()` eyleme geçirilebilir hata veriyor; testler auto-skip (importorskip + binary-yok skip)
- [x] Adaptör: Grid/Medium/CWSource → kWaveGrid/kWaveMedium/kSource+kSensor eşlemesi;
      CW sürüş sinyali sentezi; kayıt penceresinden tek-bin DFT fazor çıkarımı (bizim kontratla aynı; Fortran-order maskeleme testli)
- [x] Birim dönüşümleri AÇIK ve testli: alpha Np/m ↔ dB/(MHz^y cm) (y=0 frekans-bağımsız),
      beta ↔ B/A (BonA = 2(beta-1))
- Başarı kriterleri:
  - Küçük 2D su senaryosu: `kwave` çözücüsü koşar, fazor alanı döner (şekil kontratı bizimkiyle aynı)
  - Aynı senaryoda `linear` vs `kwave`: normalize fokal desen r > 0.99 (ilk çapraz doğrulama)
  - k-Wave kurulu değilken tüm süit yeşil kalır (skip'ler raporlanır)
  - Binary tipi/sürümü sonuç meta'sına damgalanır

### M5 — Westervelt nonlineerlik + p_max/2f0 `[x]` (2026-08-10)
- [x] β terimi, p_max takibi, harmonik fazorlar (`harmonics=(1,2,3,...)` tek geçişte), `westervelt` registry'de; `linear` ile ortak `kspace/engine.py` (tek numerik yüzey)
- Not: Fubini kapısı ppw=16'da geçer (A2/A1 %0.85–3.2). ppw=8'de 3f0 aliaslanıp A2'yi ~%10 şişirir — çözünürlük kuralı devlog'da; A3/A1 ikincil kapı <%10
- Başarı kriterleri:
  - β=0 ⇒ `westervelt` ≡ `linear` — BİREBİR aynı (aynı kod yolu; array_equal) ✓
  - Fubini ön-şok rejimi: A2/A1 < %5 sapma, σ ∈ [0.06, 0.61] beş noktada ✓ (kapı σ≤0.3'ten geniş)
  - amp ≤ p_max · (1/cos(π/spp)) diskre tavan değişmezi >%10 bandında her voxel'de ✓
  - Hafif-σ noktasında amp/p_max ∈ [0.85, 1.0] (notebook bandı) ✓

### M6 — Kaynak modeli + transducer arrays `[x]` (2026-08-10)
- [x] `arrays.archimedean_spiral` (notebook portu, parametre-generik) + `TransducerArray` (genel taban: pozisyon/normal/eleman yarıçapı); DAS fazlama; `voxelize()` eleman→voxel kabuk (element sahiplik haritasıyla); faz haritası (sin/cos) + boyut seçici; `rayleigh_preview()` (GUI'siz anlık beam önizleme)
- KEŞİF: üretim 128-spirali 32×32 faz haritasına SIĞMIYOR (95 ofset ihlali) → notebook runtime'da 64×64 fallback'ine düşüyordu; dataset'in gerçek phase_map_size'ı 64. Test bunu regresyon olarak sabitler
- Kapı düzeltmeleri (fizik gereği): DAS kapısı "tepe==hedef" değil — yer-değiştirme farkı (yanal <λ/2, eksenel <λ; odak kayması sistematiği farkta iptal olur) + hedefte genlik ≥3× artış. Entegrasyon kapısı eksenel pencereyi O'Neil-öngörülü tepe ile sınırlar
- Başarı kriterleri:
  - Eleman sayısı/eleman yarıçapı notebook değerleriyle birebir (128, r=3.205 mm) ✓
  - Kaynak voxel'leri: tüm elemanlar temsil edilir (32/32 test dizisinde; kayıp eleman → hata) ✓
  - DAS: yanal yönlendirme < λ/2, eksenel < λ (yer-değiştirme farkı), hedefte ≥3× genlik ✓
  - Entegrasyon: voxelize spiral + linear çözücü su içinde; yanal ≤1 voxel, eksenel [O'Neil tepe−1, geo+1] ✓

### M6b — Geometri sistemi: CSG + import + yeniden örnekleme `[x]` (2026-08-11, araya alınan iş — kullanıcı talebi)
COMSOL-vari geometri kurulumu; materyallerden AYRI (Scene etiket üretir, MaterialDB etiketi yorumlar).
- [x] `geometry.shapes`: primitifler (Ball, Box, Cylinder, Ellipsoid, HalfSpace) — 2D/3D,
      konum/boyut parametreli, `translated/rotated/scaled` dönüşümleri
- [x] `geometry.csg`: boolean cebir — `|` (union/OR), `&` (intersection/AND), `-` (difference),
      `~` (complement/NOT); keyfi derinlikte ağaç
- [x] `geometry.scene`: Scene(ndim, axisymmetric, background) + boyama sıralı etiket ataması +
      `rasterize(grid, supersample)` (süperörnekleme + çoğunluk oyu ile kenar kalitesi) +
      `add_volume` (import edilen hacmi sahneye yerleştirme) + `to_medium(grid, db)`
- [x] `geometry.volumes`: LabelVolume (heterojen çok-sınıflı etiket hacmi, dx+origin'li);
      mtype-tarzı text import (genel eşleme kuralları + meme fantomu preseti + npz önbellek),
      npz IO; `resample(dx_new, method="nearest"|"smooth")` (smooth = one-hot lineer + argmax)
- [x] `geometry.configs`: pydantic tagged-union CSG ağacı JSON'da; import dosya REFERANSI
      JSON'da (yol + format + eşleme); round-trip + build() == elle kurulum
- Başarı kriterleri:
  - Küre/disk hacim doğruluğu: rasterize hacmi analitikten < %2 (makul dx'te); supersample=3,
    supersample=1'den ölçülebilir daha iyi
  - CSG cebiri: örneklenmiş maskelerde numpy boolean eşdeğerliğiyle BİREBİR
  - Axisymmetric sahne (r,z): r≥0 yarı-düzlem doğrulaması; 2D makineyle aynı yol
  - Import: mtype-format round-trip (yaz→oku→eşle), NaN→background, Fortran order;
    gerçek mtype.txt varsa yerelde yüklenip yeniden örnekleniyor (yoksa skip)
  - Resample: 0.5→0.3 mm (gerçek kullanım oranı) etiket kümesi korunur; arayüz konumu ≤ 1 voxel;
    smooth ile nearest karşılaştırılır
  - Config: JSON round-trip; build sonucu elle kurulan sahneyle aynı id_map
  - Entegrasyon: Scene→Medium→linear çözücü smoke testi

### M6c — UWCEM fantom alt modülü + Phantom Studio `[x]` (2026-08-18, araya alınan iş — kullanıcı talebi)
Gerçek anatomiyi simülasyona sokan yol: depo dosyası → TEK import edilebilir dosya.
- [x] `phantoms.catalog`: dokuz UWCEM fantomunun kataloğu, atomik indirici + zip CRC doğrulaması,
      atıf metni her export'un içine yazılır
- [x] `phantoms.reader`: vektörize bayt seviyesi `mtype`/`pval` çözücüleri (18× / 4×), varsayımını
      gerçek baytlarda doğrulayıp tutmazsa yavaş referansa düşer; int8 npz önbelleği; sıfır-kopya
      Fortran reshape
- [x] `phantoms.orientation`: kanonik (x, y, z) çerçevesi (z = hüzme ekseni, z=0 transdüser tarafı);
      ilerleme ekseni kas slab'ından TÜRETİLİR, varsayılmaz; elle-yönlülük korunur
- [x] `phantoms.tissue`: on sınıf için akustik tablo (c, rho, alpha güç yasası, B/A→beta) +
      `detailed`/`grouped`/`simple` doku modelleri (`simple` = `breast_default()` id uzayı);
      doğrulanmış renk paleti
- [x] `phantoms.processing`: kırpma (offset takipli), pad, FFT-dostu boyut, sınıf birleştirme,
      ada temizleme, delik doldurma, çoğunluk yumuşatma, yeniden örnekleme (LabelVolume'a delege)
- [x] `phantoms.heterogeneity`: `pval` interpolasyonu (yalnız pval'i OLAN sınıflara) + tohumlanmış,
      fiziksel korelasyon uzunluklu saçıcı gürültüsü (sıfır ortalama, birim varyans)
- [x] `phantoms.spec` / `builder`: pydantic tarif + sabit sıralı boru hattı + `plan()` (build
      yapmadan boyut/bellek) + fizik değiştiren adımlar için uyarılar
- [x] `phantoms.asset`: `.npz` export — `LabelVolume.load_npz` ile de `load_phantom` ile de açılır
- [x] `phantoms.cli` + `apps/phantom_studio` (bağımlılıksız web GUI: WebGL2 hacim + slider'lı kesitler)
- Başarı kriterleri:
  - Hızlı çözücü ≡ yavaş referans, aynı baytlarda birebir (CRLF + eksik son satırsonu dahil)
  - Dokuz fantomun hepsinde 5.0 mm tam-alan kas slab'ı, s1'in yüksek indeks ucunda
  - `plan(spec).shape == build(spec).shape` (farklı dx / standoff / kırpma modlarında)
  - Export HEM `LabelVolume.load_npz` HEM `load_phantom` ile açılır; `to_medium()` Medium
    invariyantlarını sağlar (float32, C-contiguous, sonlu)
  - pval yalnız pval'i olan sınıflara uygulanır (deri/kas/banyo orta noktada kalır)
  - Gürültü: istenen std ±%15, sıfır ortalama, pozitiflik korunur, tohum tekrarlanabilir
  - Spec JSON round-trip; bilinmeyen anahtar hata

### M6d — `uwcem_phantoms` bağımsız paket + standart hizalı dataset `[x]` (2026-08-19, kullanıcı talebi)
Fantom modülü yan pakete taşındı; dokuz fantom TEK ortak gridde simülasyona hazır.
- [x] Taşıma: `src/caustica/phantoms/` → `uwcem_phantoms/` (repo kökü; wheel dışı — caustica'i
      tüketir, caustica onu import etmez); tüm importlar/launcher/studio/testler/dokümanlar güncel
- [x] `dataset` modülü: survey (native ölçüm ×0.5/dx + emniyet + FFT-dostu birleşim kutusu,
      builder'ın fitted peak-RAM'iyle boş-RAM kapısı) → build → hizala (ön yüz z=front_gap;
      x/y'de çıkıntı-yapan-meme bbox merkezi — göğüs duvarı slab'ı bilinçli dışlanır) → su
      dolgusu (etiket=coupling id, özellikler=suyun mid değerleri) → manifest; alt-küme rebuild
      mevcut gride birleşir, farklı tarif reddedilir; `--verify` diskten bağımsız yeniden ölçer
- [x] Üretim: `data/phantoms/` — 9 dosya, 540×700×625 @ 0.25 mm (135×175×156.25 mm, 236 Mvox),
      toplam 5.66 GB, `detailed` doku modeli, f0=1 MHz, pval AÇIK, gürültü kapalı, sıfır uyarı
- [x] Adversarial review turu (5 mercek + bulgu başına şüpheci, 27 ajan): 22 aday → 17 doğrulandı
      → hepsi düzeltildi (NaN-körü doğrulama, RAM rayı, deflate seviyesi, 236 Mvox ölçek
      maliyetleri, manifest ezme/öksüzler, merkezleme dejenerasyonu, CLI kabloları — devlog)
- Başarı kriterleri (hepsi `dataset --verify` ile diskten ölçüldü, 9/9 dosya):
  - Dokuz dosya AYNI grid ve dx'te; deri ön yüzü hepsinde tam z=20 voxel (5 mm su)
  - Meme bbox merkezi kutu merkezinde ±1 voxel (x ve y)
  - pval kontratı: her voxel kendi MEDYA NUMARASININ [lo, hi] bandında; pval'siz sınıflar
    orta noktada; pval'li sınıflarda sınıf-içi std > 0; NaN/Inf yok
  - Su dolgusu coupling ortamının kendi mid değerlerini taşır; manifest format tag'i doğru;
    manifest'in listelemediği dataset dosyası yok
  - `load_phantom(...).to_medium()` doğrudan çalışır (dx=1 mm uçtan uca testte kapılı)
  - Alt-küme rebuild manifest'i birleştirir + mevcut gridi benimser; farklı tarif `--force`'suz
    reddedilir (testli); tam suite + 32 dataset testi yeşil

### M6e — dataset derinlik tavanı + transducer bütçesi `[x]` (2026-08-19, kullanıcı talebi)
Kullanıcı: "göğüs duvarının ilerisi boş; o boşluğu saklamak istemiyorum — hepsi aynı ızgarada
kalsın, derinlik sınırı 100 mm". Ölçüm: kırpılmamış birleşim 156.25 mm derin ve arkasının
2.75–51.75 mm'si saf su, ardından her fantomda kesiti TAMAMEN kaplayan düz göğüs duvarı slab'ı
(21 voxel kas + yağ) geliyor — HIFU odağının hiç ulaşmadığı bölge.

Tavan uygulandıktan sonra ikinci bir ölçüm gerekti: 5 mm'lik ön su payı bir PML'in tam kendisi
(0.25 mm'de 20 voxel) olduğu için **kullanılabilir su 0 mm** kalıyordu ve odaklı hiçbir çanak
domain'e sığmıyordu (üretim 128-elemanlı spiralin kabuğu tek başına 11.6 mm derin). Kullanıcı
kararıyla ön pay 20 mm'ye, tavan 120 mm'ye çıkarıldı: doku kapsamı 100 mm'de sabit kalıyor.
- [x] `depth_limit_mm` (varsayılan 120 mm; `--depth`, `0` = tavan yok) tüm zincirde:
      `plan_dataset` → `build_dataset` → CLI → launcher. Tavan bir TAVAN olduğu için z ekseni
      `prev_fft_friendly` ile AŞAĞI yuvarlanır (diğer iki eksen yukarı) — istenen mm hiç aşılmaz
- [x] `FRONT_GAP_MM` 5 → 20 mm: ön pay artık transducer bütçesi. PML sünger grid'in İÇİNDE
      olduğu için "5 mm su + 5 mm PML" serbest su bırakmıyordu; 20 mm üretim spiraline
      (kabuk 11.6 mm) ~3 mm boşluk bırakıyor ve ROC 60'a kadar F/1 çanakları da alıyor
- [x] **Sessiz hata kapatıldı:** `SolverBase.validate` artık TÜM voxelleri süngerin içinde
      kalan kaynağı reddediyor (`check_source_clears_pml`). Böyle bir koşu hata vermeden
      yakınsıyor ve sessizce yanlış alan döndürüyordu; k-Wave adaptöründe kontrol vardı,
      yerel k-space yolunda yoktu. Kısmi örtüşme (tam genişlikli düzlem kaynağın yanal
      uçları) kasıtlı olarak serbest — o normal düzlem-dalga kurulumu
- [x] Kırpma YIKICI ve bu bilinçli: `_align_into_common` arka yüzden kesiyor, kesileni sınıf
      sınıf sayıp `truncated_tissue_vox` / `truncated_by_class` / `back_trim_mm` olarak manifest'e
      yazıyor; `--dry-run` maliyeti build'den ÖNCE fantom fantom söylüyor; tavan YOKken aynı
      taşma survey hatası sayılıp reddediliyor (sessiz kırpma yok)
- [x] Transvers merkez artık KALAN dilim üzerinden ölçülüyor (survey ve build aynı tanım):
      meme tabana doğru genişlediği için atılacak dilimleri saymak kutuyu yanlış boyutlandırıyordu
      — düzeltme x'i 540 → 560'a çıkardı
- Başarı kriterleri (hepsi `dataset --verify` ile diskten ölçüldü, 9/9 dosya):
  - Dokuz dosya 560×700×480 @ 0.25 mm = 140×175×120 mm; z ekseni tam 120 mm (tavan aşılmıyor)
  - Deri ön yüzü hepsinde tam z=80 voxel (20.0 mm su); meme bbox merkezi kutu merkezinde ±1 voxel
  - Ön pay > PML: 20 mm su, tipik 5 mm PML → 15 mm serbest su; üretim spirali (kabuk 11.6 mm)
    apex'i PML'in dışında olacak şekilde yerleşiyor (ölçüldü)
  - Manifest'in kesim kaydı dosyayla çelişemiyor: "kesildi" diyen fantomun SON z düzleminde
    doku olmak ZORUNDA (verify bunu ayrı test ediyor)
  - M6d'nin tüm kriterleri (pval bandları, su dolgusu, öksüz dosya, format tag) geçerliliğini
    koruyor; farklı `depth_limit_mm` aynı dizine `--force`'suz reddediliyor (testli)
  - Uçtan uca kanıt: fantom → Grid+PMLSpec → çanak → Westervelt koşusu yakınsıyor ve odak
    dokuda; heterojenlik odağı beklenen yönde öne kaydırıyor (yağ sudan yavaş)
  - Tam suite (296 test) + `ruff check src uwcem_phantoms apps tests` temiz

### M6f — depolanmış kurulumlar: dokuz koşu, yüklemeye hazır `[x]` (2026-08-19, kullanıcı talebi)
Dataset ortamdan ibaret; bir koşu ayrıca sığan bir transducer, onu yutmayan bir sınır ve dokuya
düşen bir odak istiyor. Bu karar artık `data/setups/`ta yazılı (fantom başına ~1.8 KB JSON, git'te).
- [x] `uwcem_phantoms/setup.py`: `ArraySpec` (tarif) + `build_setups` / `load_setup` /
      `verify_setups` / `setup_names`; CLI `setup [--list|--verify|--json|--amplitude|--pml]`
- [x] Standart dizilim **S1**: 64 elemanlı Arşimet spirali, D=60 mm (iç 26.4), ROC=60 mm,
      F/1.0, yarı-açı 30°, eleman r=2.718368 mm, kabuk 6.5611 mm. Apex dokuz dosyada da
      **z = 5.50 mm** (5 mm PML'in 2 voxel ötesinde), odak dizilimin KENDİ geometrik odağı
      **z = 65.50 mm**, tüm fazlar sıfır — elektronik yönlendirme yok
- [x] Hiçbir şey pişirilmiyor: eleman konumları ve 23.283 kaynak voxeli yükleme anında
      TÜRETİLİYOR ve dosyanın kaydettiği değerlere karşı sınanıyor; dizilim kurulumu değişirse
      koşu sessizce başka bir transducer'la devam etmiyor, yükleme patlıyor
- [x] Kurulum düzeyinde daha SIKI PML kuralı (`_require_source_fully_clear_of_pml`): çözücüdeki
      genel kapı kısmi örtüşmeye izin vermek zorunda (tam genişlikli düzlem kaynak), depolanmış
      bir kurulumun ise böyle bir mazereti yok — tek voxel süngerdeyse yazılmıyor
- Başarı kriterleri (hepsi `setup --verify` ile diskten ölçüldü, 9/9 kurulum):
  - Dokuz kurulum aynı grid, aynı dizilim, aynı apex, aynı odak voxeli (280, 350, 262)
  - Kabuk–deri geçişi 9.25–14.50 mm, hepsi pozitif; çakışan kurulum YAZILAMIYOR
  - Odak dokuz fantomda da dokuda (sınıf 3/5/6/7), suya düşen kurulum YAZILAMIYOR
  - Dataset'in pişirdiğinden farklı bir f0 ile kurulum reddediliyor (alpha yanlış olurdu)
  - `load_setup(...)` → `westervelt.validate()` geçiyor; `linear` doğru şekilde reddediyor
  - Kurcalanan dosya (format, türetilmiş geometri, voxel sayısı, apex, geçiş payı) beş ayrı
    yerden yakalanıyor (testli); tam suite + 11 setup testi yeşil

### M7 — CuPy backend (CUDA) `[ ]` — Colab oturumu gerektirir
- [ ] ElementwiseKernel'ların portu; aynı çözücü kodu iki backend'de; fp32 yolu
- Başarı kriterleri:
  - numpy↔cupy parite: mini 3D senaryoda fazor/p_max rel fark < 1e-5 (fp32 toleransı belgelenir)
  - Colab T4 VE A100'de tam boy (dx=0.30, 512³ FFT sınıfı) koşu OOM'suz tamamlanır
  - Adım süresi ölçülür ve `benchmarks/`e damgalanır (baseline; M19 bunu referans alır)
  - GPU yokken testler otomatik SKIP (CI kırılmaz)

### M8 — Planner v1 (süre + VRAM tahmini) `[~]` — yerel yarısı tamam (2026-08-11), Colab kapıları açık
- [x] Statik VRAM modeli (tampon dökümü + cuFFT workspace payı + %15 marj); süre modeli a·N·logN + b·N; `gpu_db.json` (T4/L4/V100/A100/H100); cihazda kalibrasyon (~20 adım) → `~/.caustica/calibration.json`; `planner.estimate(gpu=...)` + `planner.compare(...)` — `src/caustica/planner/`, 11 test (`tests/test_planner.py`)
- Başarı kriterleri:
  - [ ] VRAM tahmini, Colab'da ölçülen mempool tepe değerinin ±%10'u içinde (≥2 farklı grid boyutunda) — **Colab kapısı, M7 oturumunda ölçülecek**
  - [ ] Kalibrasyon SONRASI süre tahmini gerçekleşenin ±%25'i içinde (aynı cihaz, ≥2 senaryo) — **Colab kapısı** (mekanik yerelde testli: cpu kalibrasyonu → fit → estimate zinciri)
  - [x] Tahmin kaynağı raporda etiketli: `db` | `calibrated` | `measured` (testli; cpu kalibrasyonu GPU anahtarıyla asla eşleşmez)
  - [x] OOM öngörüsünde eyleme geçirilebilir öneri metni (dx büyüt ×m hesaplı / AOI küçült / linear'a geç / daha büyük cihaz) — testli
- Not: dt/spp ve time-of-flight türetimi motordan `cw_discretization`/`cw_tof_periods` fonksiyonlarına çıkarıldı (tek doğruluk kaynağı; planner==engine testli). VRAM envanteri engine.py tampon listesini birebir aynalar — motora yeni kalıcı tampon eklersen `test_memory_inventory_matches_hand_count` kırılır (bilerek).

### M9 — KZK çözücüsü `[ ]`
- [ ] Operator splitting: difraksiyon = angular spectrum (VERIFY: gemini2 önerisi; CN alternatifi karşılaştırılacak), absorpsiyon, nonlineerlik = zaman-uzayı Burgers (şok emniyeti); Rayleigh ile başlangıç düzlemi projeksiyonu; registry'de `kzk`
- Başarı kriterleri:
  - Lineer odaklı piston (paraksiyel, F-number ≥ 2): eksenel profil O'Neil'e r > 0.99
  - Fubini düzlem-dalga limiti: A2/A1 < %5 sapma (σ ≤ 0.3)
  - Çapraz doğrulama: zayıf odaklı su senaryosunda kzk vs westervelt fokal basınç farkı < %5, odak konumu < 1 voxel (paraksiyel geçerlilik bölgesinde)
  - Süre: aynı senaryoda full-wave'e karşı ≥ 50× hız (beklenti ~100×; ölçülüp belgelenir)

---

## Faz Grubu C — Veri, IO, Colab entegrasyonu (v1 çizgisi)

Omurga (2026-08-21'de kütüphane-önce kararlarıyla revize edildi — ayrıntı: PLAN.md §0.2 ve
docs/library_first_plan.md): lokalde `job.json` yaz → Colab'da değişmeyen notebook koşar →
çıktı `/content` altına düşer → lokalde raporla. **Drive kütüphanenin işi değil** (K12).
**M10 → M10b → M10c → M10d** (hepsi CPU'da yazılır ve testlenir, Colab beklemez) → **M10e**
public → **M10k** UWCEM ayrışımı ‖ **M10h** paketleme ‖ **M10i** ortam ve güvenlik politikası (üçü paralel;
M10h/M10i `config/job.py`'ye dokunmaz) → **M10m** dışarıdan kullanılabilirlik →
**M10j** facade + ilerleme → **M10l** GUI sözleşmesi → **M10f** Colab köprüsü →
**ilk Colab oturumu** (M7 + M8 kapılarıyla birleşik) → **M10g** kuyruk.
M10m/M10j `config/job.py`'ye dokunduğu için M10k'yı bekler. M12–M14 bu omurganın üstünden koşar.

### M10 — IO: HDF5 kontratı + atomik yazım + resume + koşu-içi checkpoint `[x]` (2026-08-19)
Colab omurgasının temeli: runner, kuyruk ve rapor bu kontratın üstüne oturur.
- [x] `caustica.io` paketi: `atomic` (tmp→`os.replace` + debris süpürme), `quantize` (dinamik
      float16 + ölçülen hata kontratı), `store` (`caustica-result/1` HDF5 kontratı + `ResultStore`
      — DriveResilientStore'un portu: doğrulanmış mkdir, write-probe, resume skip-guard),
      `checkpoint` (koşu-içi durum). h5py lazy (PEP 562) — `import caustica.solvers` h5py yüklemez
- [x] Koşu-içi checkpoint: her N periyotta tam alan durumu (p + u_i, sayaçlar, geçmiş) atomik
      `.npz`; motor parmak izi (grid/medium-sha/source-sha/dt/spec/çözücü/backend/kayıt bölgesi)
      tutmayan checkpoint'i İSİMLE reddeder; kayıt penceresi öncesi "record" anlık görüntüsü
      kayıt sırasındaki ölümü de karşılar; `stop_when` kancası M10c'nin `--max-hours` temeli;
      başarıda dosya silinir. kwave adaptörü `checkpoint=`i açıkça reddeder (harici binary)
- Başarı kriterleri (hepsi testli — `tests/test_io.py` 16 + `tests/test_checkpoint.py` 7 test):
  - [x] Round-trip: float16 max norm hata ≤ 1e-3 ölçülüp doğrulanıyor; 1e-5 kontratında float32'ye düşüş birebir
  - [x] Kesinti simülasyonu: GERÇEK subprocess SIGKILL yazım ortasında → görünür bozuk dosya YOK; .tmp cesedi store açılışında süpürülüyor
  - [x] Resume: 10 örneklik mini sette ortadaki dosya silinince `missing()` YALNIZ onu döndürüyor (tek yeniden üretim, sayaçla doğrulandı)
  - [x] Checkpoint-resume: kesilen koşu devam edince fazor/p_max/geçmiş kesintisiz koşuyla BİREBİR AYNI (bitwise; belgelenen band rel < 1e-6 — 1D linear, 2D westervelt, zincirleme kesinti, record-aşaması dahil)
  - [x] Faz konvansiyonu + absorpsiyon modeli attr'ları her dosyada (kök + output; testle zorlanıyor)
- Kanıt: tam suite **330 test yeşil** (307 + 23 yeni), ruff temiz (devlog 2026-08-19 oturum 10)

### M10b — JobConfig: tam simülasyon şeması + `caustica validate` `[x]` (2026-08-19; kullanıcı kararı: TAM genişletme)
Tek JSON bir koşunun TAMAMINI tarif eder — GUI'siz dönemde elle yazılır, GUI geldiğinde aynı
şemayı üretir. Format `caustica-job/1`; pydantic, `extra="forbid"`, mm/MHz kullanıcı birimleri,
voxel her zaman türetilir (mevcut config sözleşmesi).
- [x] `src/caustica/config/job.py` — `caustica-job/1`; `medium` tagged union: `phantom_dataset`
      (npz'den grid + etiketler; pml_mm tek kullanıcı seçimi) | `scene` (SceneConfig + malzeme
      tablosu; eksik etiket kurulumda hata) | `volume_import` (tek-import'lu scene olarak aynı
      yoldan) | `homogeneous`
- [x] TASARIM SAPMASI (bilinçli): `stored_setup` source-union'ında değil JOB seviyesinde ayrı
      kind — depolanmış kurulum medium+grid+yerleşim+run'ı BİRLİKTE sabitler; onu source'a koymak
      farklı bir medium'la eşleşmesine izin verir ve M6f garantilerini kırardı. `source` union'ı:
      `array` reçetesi (archimedean_spiral | bowl; `apex_mm`; odak natural | steered hedef;
      fazlar zeros | DAS(c0=1500 su) | açık liste; bowl steer/faz reddeder)
- [x] `drive` (f0_mhz/amplitude_kpa/ramp), grid kuralı: dataset medium'u grid bölümünü REDDEDER
      (dosya sabitler), diğerleri GridConfig İSTER; `run` (CWRunSpec + harmonikler +
      record_region_vox), `solver`, `backend`, `output` (klasör/kuantizasyon politikası)
- [x] Override katmanı (`StoredSetupOverrides`): genlik/harmonik/run-policy/steering; f0 override
      ancak dosyadakine EŞİTSE geçer, farklıysa alpha gerekçesiyle RED (M6f korunur); steering
      voxel kümesini değiştirmez, yalnız fazları (DAS) ve odak voxelini değiştirir
- [x] `python -m caustica validate job.json [--fast]` (`src/caustica/__main__.py`): şema + dosya +
      kaynak-PML + odak-dokuda (dataset sınıf 0 = su reddi; steered stored-setup için etiketler
      yüklenip sınanır) + ppw (medium yokken yaklaşık c_min etiketiyle); exit 0/2
- Başarı kriterleri (hepsi testli — `tests/test_job.py`, 35 test):
  - [x] Her düğümde JSON round-trip (12 model parametrize + union + dump/load); typo → hata (üst + iç içe)
  - [x] Parite: stored job == `load_setup` (grid/indeks/faz/genlik/f0/ramp/spec/bölge/odak birebir)
  - [x] Scene yolu: SceneConfig→Medium→mini CPU koşusu (odakta alan canlı, top medium'a işlenmiş)
  - [x] Serbest array: bowl + spiral; `check_derived()` — kurcalanan f_number İSİMLE reddediliyor
  - [x] `validate`: dokuz stored-setup job'ı geçer; scene + volume-import job'ları kurulur; bozuk
    referans / typo / PML'e gömülü kaynak / suya düşen odak / bilinmeyen çözücü / lineer çözücüde
    nonlineer medium ayrı ayrı yakalanıyor; `--fast` medium kontrollerini ertelediğini SÖYLÜYOR
- Kanıt: 365 test yeşil (330 + 35 yeni); ruff temiz (devlog 2026-08-19 oturum 10)

### M10c — Runner: `python -m caustica run job.json` `[x]` (2026-08-19)
Colab hücresinin çağıracağı TEK giriş noktası; lokalde numpy ile de aynı komut.
- [x] `src/caustica/runner.py` + `__main__.py run`: job yükle → planner yazdır + `plan.json`/
      `plan.txt` kaydet → VRAM sığmıyorsa KOŞMADAN reddet (önerilerle) → çöz → M10 store → damga.
      Çıktı düzeni deterministik (job dosyasına göreli — CWD kayması resume'u bozamaz):
      job.json kopyası, plan, status.json, checkpoint.npz, result.h5, run_meta.json
- [x] Bayraklar: `--dry-run` `--resume` (checkpoint varsa resume AÇIKÇA istenmeli; yoksa yüksek
      sesli not) `--max-hours` (0 dahi geçerli: ilk periyot sınırında zarif duruş) `--backend`
      `--gpu` `--no-measure` `--checkpoint-every` `--status-interval` `--vram-limit-gib`
- [x] `status.json` kalp atışı: `stop_when` yoklamasından türetilen periyot sayacı (motor değişikliği
      yok); state/step k-N/ETA(ölçülen kadans)/written_at; Drive senkronuyla lokalden izlenir
- [x] Damga (`run_meta.json` + h5 attr'ları): job kopyası, git commit, ortam (GPU/driver/cupy),
      planner tahmini vs gerçekleşen (M8 Colab kapıları buradan ölçülür), türetilmiş geometri
- [x] Ayrık exit kodları: 0 başarı/zaten-tam · 2 config · 3 OOM reddi · 4 çözücü/store ·
      **5 kesildi-resumable** (kuyruk M10g bunları okuyacak)
- Başarı kriterleri (testli — `tests/test_runner.py`, 14 test):
  - [x] `--dry-run`: plan dosyaları var; result/checkpoint/status YOK (testli)
  - [x] OOM reddi: koşmadan önerilerle exit 3; config hataları (typo, eksik dosya, bilinmeyen
    --gpu) exit 2; hepsi sınıflı (testli)
  - [x] Mini job numpy ile saniyelerde uçtan uca: M10 kontratı + damga alanları tam (testli)
  - [x] Kesinti → `--resume`: çift üretim yok; fazor kesintisiz koşuyla BİREBİR (bant rel<1e-6);
    resume'suz yeniden koşu REDDEDİLİR; store çökmesi bile çözümü KAYBETTİRMEZ
    (checkpoint store başarısına dek yaşar — yalnız kayıt penceresi yinelenir)
  - [x] status.json koşu sırasında güncelleniyor (kesinti anında periods_done gözlemlendi)
- **Adversarial review turu (2026-08-19, M10+M10b+M10c üzerinde):** 5 boyutlu tarama → 14 bulgu →
  12 düzeltildi-testlendi, 1 belgelendi (heartbeat pre-record ±1, kozmetik), 1 çürütüldü (steering
  apex-çerçevesi voxelize okunarak doğru bulundu). Kritik düzeltmeler: kwave job'ları `backend=`
  kwarg'ıyla çakılıyordu (kwarg'lar artık yalnız native); **explicit phantom_dataset yolu M6f
  f0-alpha korumasını atlıyordu** (kapandı, testli); save_result çökmesi bitmiş çözümü yutuyordu
  (keep_on_success: checkpoint'i runner store'dan SONRA siler); atomik yazım yazar-benzersiz tmp
  adlarına geçti (iki oturumun yarışı torn dosya üretemez), süpürme yalnız bayat tmp'leri alır,
  os.replace Windows kilidinde retry + son çare tmp korunur; steered stored su-odağı reddi
  build'e taşındı (run==validate); dataset job'da etiket kontrolü GB'lik medium kurulumundan önce;
  validate tam-grid kayıt uyarısı verir
- Kanıt: tam suite 383 test yeşil; ruff temiz (devlog 2026-08-19 oturum 10)

### M10d — Rapor + önizleme: `caustica report` `[x]` (2026-08-21)
Tam alan dosyası (0.5–0.8 GB) inmeden "koşu başarılı mı" sorusuna 10 saniyede cevap.
- [x] Koşu sonunda ~5–10 MB önizleme paketi (`caustica-preview/1`): tepe dilimleri (3 eksen, her
      harmonik + p_max) + kabalaştırılmış amp hacmi (blok-ortalama, dinamik float16 + scale) +
      metrics.json — runner her başarılı koşuda result'ın YANINA yazar (ileride GUI'nin sonuç
      sekmesi doğrudan bunu okur; önizleme çökmesi koşuyu ASLA düşürmez, yalnız uyarır).
      Bilinçli sapma: "orta dilim" yerine GERÇEKLEŞEN tepe voxel'inden geçen dilimler (daha
      bilgilendirici; meta_json'a not düşülüyor)
- [x] `caustica report <out-dir>`: result.h5'ten HTML + figürler LOKALDE; `--preview` yalnız
      önizleme paketinden hızlı görünüm (result hiç okunmaz). Figür/metrik/render kodu
      `caustica.report` paketine çıkarıldı (metrics+preview numpy-only ve eager; figures
      matplotlib'i, store h5py'ı lazy yükler — çıplak kurulumda runner önizleme yazabilir);
      focus_study analysis/figures/report ince adaptör oldu. result.h5'e apex_vox/focus_vox
      damgası eklendi (rapor apex-çerçeveli mm konumlarını job'suz üretebilsin)
- Başarı kriterleri:
  - [x] Önizleme ≤ 10 MB TAM gridde ölçülür: 256³ sentetik alan → paket ölçüldü ≤ 10 MB;
        kabalaştırma adımı bütçeden hesaplanır + yazım ÖNCESİ bellekte ölçülüp gerekirse
        büyütülür; report yalnız önizlemeyle hızlı görünüm veriyor (ikisi de testli)
  - [x] Metrik tanımları focus_study ile TEK doğruluk kaynağı: `caustica.report.metrics
        .focus_metrics` — `analyze()` delege eder; iki yol aynı sayıyı verir (testli: bölüm
        bölüm dict eşitliği + h5-roundtrip yakınlık; isppa dürüst istisna — result dosyası
        medium taşımaz, None döner)
  - [x] focus_study regresyonu yok: water_bowl + layered_tissue (dx=0.6) refactor öncesi/sonrası
        koşuldu — REPORT.md ve index.html BAYT-AYNI; metrics.json'da tek fark `--out` yolunu
        içeren `command` alanı (beklenen)
- Kanıt: tam suite **393 test yeşil** (383 + 10 yeni `tests\test_report.py`), ruff temiz
  (devlog 2026-08-21)

### M10e — Public'leşme `[~]` (isim + rename + tarama tamam 2026-08-21; commit/push bekliyor)
Colab notebook'u public repodan clone eder: token/secret yönetimi tamamen ortadan kalkar.
`v0.1` tag'i M11'de KALIR; bu milestone yalnız repoyu görünür yapar.
- [x] İsim kararı (2026-08-21, kullanıcı): **caustica** — ilk tercih "kymata" PyPI'da doluydu
      (Cambridge Kymata Atlas); `caustica` PyPI'da boş, GitHub'daki aynı adlı görünür proje
      farklı ekosistem (Java/Minecraft). Rename UYGULANDI: `src/caustica`, format etiketleri
      `caustica-*` (5.66 GB dataset için belgeli legacy alias — rebuild yok), 9 setup yeni
      etiketle yeniden üretildi, GitHub repo `gh repo rename` ile **ebx0/caustica**
- [x] Geçmiş taraması (2026-08-21): en büyük blob 231 KB (rapor PNG'si) — >5 MB dosya YOK;
      secret taraması temiz (yalnız "token" kelimesinin masum kod kullanımları); mtype/dataset/
      kaynak notebook hiç commit'lenmemiş (doğrulandı)
- [~] README/LICENSE/atıf gözden geçirme: README rename'le güncellendi ("working name" notu
      kalktı), UWCEM bölümü + atıf mekanizması yerinde; janitor turunda wheel'e `gpu_db.json`
      eklendi (pip kurulumunda planner çökerdi — düzeltildi, wheel'den canlı doğrulandı)
- Başarı kriterleri:
  - [x] Geçmişte > 5 MB dosya yok, secret yok (tarama çıktısı devlog 2026-08-21)
  - [ ] Repo public (✓ — zaten public'ti, adı değişti); CI public repoda yeşil + temiz ortamda
        `pip install git+https://github.com/ebx0/caustica` → **commit/push bekliyor**
        (commit kuralı: kullanıcı istemeden atılmaz; wheel içeriği lokalde doğrulandı)
  - [ ] UWCEM atıf yükümlülüğü README'de ve export'larda korunuyor (push sonrası son kontrol).
        Not: M10k ile bu yükümlülük `uwcem-phantom` repo'suna taşınıyor
  - [ ] **`CITATION.cff`** (bugün YOK): public bir araştırma kütüphanesi alıntılanabilir olmalı.
        v0.1 tag'inde Zenodo DOI'si alınır ve README'ye "How to cite" bölümü eklenir

### M10k — UWCEM ayrışımı: kütüphane UWCEM'siz olur `[ ]` — **DEVİR PAKETİ 3/4**
Lisans riski ve kendi katmanlama kuralımızın ihlali aynı kökten geliyor: `config/job.py` bugün
DÖRT yerden `uwcem_phantoms` import ediyor. Karar (PLAN.md K7–K10): kütüphane tamamen UWCEM'siz
olur, UWCEM işleri ayrı repoya taşınır, yerine genel `medium_volume` kind'ı gelir.
- [ ] `medium_volume` genel medium kind'ı: etiket haritası + malzeme db'si VEYA voxel-başına
      özellik hacimleri (c, ρ, α, β); **grid dosyadan gelir** (explicit `grid` reddedilir — mevcut
      `phantom_dataset` kuralı ve hata metni aynen taşınır). MEVCUT 4.5 GB dosyaları OLDUĞU GİBİ
      okur (`caustica-phantom-dataset/2` + `hifusim-*` legacy alias) — yeniden üretim gerektiren
      format değişikliği BAŞARISIZ sayılır
- [ ] Format hem OKUNUR hem YAZILIR: `write_medium_volume(...)` public — numpy dizilerinden
      (etiket haritası VEYA c/ρ/α/β hacimleri) + dx + origin ile dosya üretir. `uwcem-phantom`
      repo'su da BU fonksiyonu çağırır (tek kaynak). Yazıcı olmadan "kendi hacmini getir" vaadi
      çalışmaz — dışarıdan gelen kullanıcının tek giriş kapısı budur
- [ ] `geometry/volumes.py::load_breast_phantom()` taşınır: UWCEM üretim fantomunun şekli
      (310×355×253, 0.5 mm) ve `breast_phantom_mapping` kaynağa özgüdür. `load_labels_txt`
      GENELDİR (mapping callable'ı kaynağa özgü kısmı dışarıda tutuyor) — o KALIR
- [ ] Literatür akustik doku değerleri `caustica.materials`e taşınır (`breast_default()` yanına,
      ölçüm/interpolasyon ayrımı koruyan yorumlarıyla); UWCEM media-numarası eşlemesi taşınmaz
- [ ] `_require_uwcem`, `PhantomDatasetMediumConfig`, `StoredSetupJobConfig` SİLİNİR;
      `caustica-job/1` iki kind kaybeder — **kırıcı şema değişikliği**, devlog'a yazılır, format
      numarası değişmez (v1.0 öncesi stabilite garantisi yok, PLAN.md K14)
- [ ] `uwcem-phantom` ayrı (private) repo: catalog / reader / builder / dataset / setup / spec /
      processing / heterogeneity / orientation / paths / cli + `apps/phantom_launcher.py` +
      137 test taşınır. caustica'ya bağımlıdır; ürettiği şey `medium_volume` dosyası ve
      **explicit job JSON**'dur (`load_setup(...)` artık explicit job döndürür)
- [ ] Veri kökü çözümü: açık argüman → `CAUSTICA_PHANTOM_DATA` → mevcut checkout `_data` →
      kullanıcı önbelleği (`~/.cache/caustica/phantoms`). `platformdirs` bağımlılığı EKLENMEZ.
      Hiçbir dosya yeniden İNDİRİLMEZ
- [ ] UWCEM lisans/şart incelemesi yeni repo public olmadan ÖNCE; atıf zinciri export'lara ulaşır
- Geçiş penceresi: kullanıcı kararı (2026-08-21) — ayrışım sırasında dokuz yerel setup'ın geçici
  olarak çalışmaması KABUL; bitişte çalışır olması ŞART
- Başarı kriterleri:
  - `grep -ri uwcem src/` boş (solvers/base.py'deki tarihsel yorum yeniden yazılır)
  - Import-yönü AST testi geçer (`src/caustica` → `apps`/`uwcem_phantoms`/`caustica_gui*` YASAK).
    Bu test M10l'de yazılır ve M10k inene kadar KIRMIZI kalır — ayrışımın kanıtı odur
  - Aynı `.npz` dosyası `medium_volume` üzerinden `phantom_dataset`'in ürettiğiyle **BİT-AYNI**
    `Medium` verir
  - Round-trip: `write_medium_volume(...)` ile yazılan dosya okunduğunda bit-aynı `Medium` verir
  - Dokuz yerel setup ayrışım sonrası uçtan uca koşar (`load_setup` → explicit job →
    `caustica run`) ve ortamlar bit-aynı; 4.5 GB yeniden ÜRETİLMEZ
  - Taşınan 137 test yeni repoda yeşil (silinmeden ÖNCE doğrulanır)

### M10h — Kütüphane paketleme + temiz ortam kapısı `[x]` (2026-08-21; CI kanıtı 2026-08-22) — **DEVİR PAKETİ 1/4**
> **Borç kapandı (2026-08-22):** gözden geçirenin `[~]` kararı üzerine dal pushlandı (kullanıcı
> onayı), CI'ı tetiklemek için **draft PR #1** açıldı (`library-first` → `master`, merge YOK:
> <https://github.com/ebx0/caustica/pull/1>). İlk koşuda `wheel` + windows yeşil; iki ubuntu
> ayağı taşınan M6c kodundaki platform hatasında düştü (export adında `\` POSIX'te meşru —
> korkuluk her platformda reddedecek şekilde düzeltildi, `efc86dd`). İkinci koşu **4/4 YEŞİL**:
> <https://github.com/ebx0/caustica/actions/runs/32529033382> (wheel 21 sn, ubuntu 3.10 + 3.12,
> windows 3.12).
`pip install` deyip checkout'suz koşabilmek. Wheel içeriği artık testle sabitlenir.
- [x] `[project.scripts] caustica = "caustica.__main__:main"`, `src/caustica/py.typed`
      (+ package-data girişleri; wheel içeriği `tests/test_packaging.py` ile sabit)
- [x] `src/caustica/examples/water_bowl_mini.json` — dış veri gerektirmeyen sentetik örnek
      (`tests/test_runner.py::mini_job` şablonu; CPU'da ≤30 sn — ölçüldü: çözüm 0.2 sn).
      Bilinçli sapma: dx 0.75 → 0.5 mm — şablonun dx'i ppw 2.00'da "under-resolved" uyarısı
      bastırıyordu; ilk karşılaşmada uyarıyla açılan quickstart olmaz (dx=0.5 → ppw 3.00, temiz)
- [x] **Örnek YERİNDE koşturulmaz**: `caustica example <ad> [--to DIR]` kopyalar (üzerine yazmayı
      reddeder), adsız çağrı listeler; `caustica.examples.path()/copy()` Python'dan aynı kapı;
      README quickstart kopyayı koşuyor
- [x] matplotlib `[report]` extra'sında kaldı; README quickstart kurulum satırı
      `pip install "caustica[report] @ git+..."` — rapor adımı temiz kurulumda patlamaz
- [x] CI'da temiz-venv wheel işi (`wheel` job'ı): build → repo DIŞINDA taze venv'e kurulum →
      `import caustica` + `caustica --version` + `example` + `validate` + `run --dry-run`
- [x] `network` pytest markası eklendi; CI test ayağı `-m "not kwave and not network"`
- Başarı kriterleri:
  - [x] Checkout'suz temiz ortamda kurulum + import + plan üretimi çalışır — **CI'da doğrulandı
        (2026-08-22)**: `wheel` ayağı yeşil, 21 sn —
        <https://github.com/ebx0/caustica/actions/runs/32529033382> (draft PR #1 üzerinden;
        adımlar: build → repo dışında taze venv → import + `--version` + `example` +
        `validate` + `run --dry-run`). Yerel prova kaydı (2026-08-21): aynı adımlar + TAM koşu
        (1.5 sn, sekiz çıktı dosyası). Örnek ayrıca eşikten çekildi: f0 1.0 → 0.8 MHz,
        ppw 3.00 → 3.75 (`60b73c1` — tam eşikte float-hassasiyet tesadüfüne dayanmasın)
  - [x] Wheel içeriği testle sabit: `py.typed` + `gpu_db.json` + örnek job + entry point +
        yan-paket sızıntısı yok + src'de olmayan dosya (hayalet) yok (`tests/test_packaging.py`,
        9 test). Mutasyonla kanıtlı: gpu_db/examples package-data satırı ya da örnek dosya
        silinince KIRMIZI (fixture pristine kopyadan build eder — bayat `build/` maskesi
        kapatıldı, devlog 2026-08-21). Not: `py.typed` satırı silinse de wheel onu içerir
        (setuptools ≥69 otomatik dahil ediyor) — test dosyanın varlığını sabitler, satırı değil
  - [x] `caustica example` ile kopyalanan job salt-okunur `site-packages` senaryosunda koşar —
        `test_copied_example_runs_without_touching_the_install_dir`: kurulum dizini içerik
        anlık-görüntüsü koşu öncesi/sonrası BİREBİR, çıktı kopyanın yanına düşüyor (T4)

### M10i — Ortam ve güvenlik politikası `[x]` (2026-08-22) — **DEVİR PAKETİ 2/4**
Ortak tema: **kullanıcı sessizce yanmasın.** Yanlış backend, yanlış çözünürlük, görünmeyen uyarı,
tek çekirdekte sürünen CPU — hepsi "çalışıyor gibi görünüp yanlış/yavaş sonuç veren" sınıfından.
`config/job.py`'nin yalnızca `validate` raporlama kısmına dokunur; şemaya dokunmaz.
- [x] `caustica.env_report()` — `_gpu_environment` `caustica.env`'e taşındı, runner AYNI
      fonksiyon üzerinden damgalıyor (test: `test_run_meta_environment_composes_env_report`);
      tarihsel damga anahtarları korunuyor (caustica/python/numpy/platform + GPU alanları),
      yalnız EKLEME yapıldı (scipy/pydantic/h5py/resolved_backend); asla exception atmaz (testli)
- [x] `caustica.require_gpu()` — pip çağırmaz; Colab'da (COLAB_* / google.colab tespiti)
      "Runtime → Change runtime type → GPU" mesajı (pip'ten hiç bahsetmez), yerelde
      `pip install cupy-cuda12x` — iki mesaj ayrı ve testli
- [x] **CPU FFT `workers` — ÖLÇÜM D32'Yİ DEVİRDİ (kullanıcı yetkisi "ölçüm karara üstün gelir",
      2026-08-22):** tesisat kuruldu (`Backend.fft` sarmalayıcısı, tek nokta; cupyx `workers`
      almaz — CuPy docs'tan doğrulandı) ama **varsayılan 1 kaldı**: iki ölçüm turu (i5-13450HX,
      10 çekirdek; %14–22 arka plan yükü belgelendi) 1–26 Mvox motor şekillerinde HİÇBİR worker
      sayısından TEKRARLANABİLİR kazanç bulamadı — ilk turun hücre sinyalleri (96³'te 0.73×
      regresyon, 26 Mvox'ta 1.48× kazanç) teyit turunda kayboldu (tablolar devlog'da). İsteyen
      `CAUSTICA_CPU_WORKERS` / `set_cpu_fft_workers()` ile açar. Sıra korundu: workers →
      kalibrasyon → eşik (üç ayrı commit: `6fe6f6f`, `4db495d`, `d2f3c75`)
- [x] CPU kapısı VRAM reddinin hemen ardında: > 5 dk (`CAUSTICA_CPU_LIMIT_MIN`) → EXIT_CONFIG(2),
      yeni kod YOK; mesaj tahmini + `est.source` etiketini alıntılıyor; `--allow-slow-cpu` aynı
      commit'te. `--no-measure` yolunda kapı GPU db sayısına DEĞİL kalibre cpu girdisine bakar
      (db A100 58.7 sn derken cpu gerçeği 2.6 saatti — kanıt koşusu devlog'da); ikisi de yoksa
      "kapı yargılayamıyor" uyarısı basar, sessiz geçmez
- [x] Kritik olaylar `warnings.warn` ile — kategori **`CausticaWarning(UserWarning)`** (public,
      `__all__`'da; yalnız bizim uyarılar filtrelenebilir): backend auto→numpy düşüşü süreç
      başına BİR kez (testli), düşük ppw koşu başına bir toplu uyarı. Kütüphane import'ta handler
      KURMAZ; `caustica run` girişte `logging.basicConfig` açar (facade M10j'de aynısını yapacak)
- [x] Düşük ppw dört yerde, ENGEL DEĞİL: `low_ppw_warnings()` tek kaynak (validate delege) →
      plan.txt/plan.json + her status.json kalp atışı + run_meta.json + raporun BAŞI (full ve
      preview yolları). Dört ayrı test
- [x] Plan çıktısında beklenen `result.h5` boyutu satırı (quantize-farkında, plan.json'da da);
      `--preview-only` bayrağı — varsayılan DEĞİŞMEDİ: tam alan + önizleme (testli)
- [x] EK (gözden geçiren talimatı, 2026-08-22): VRAM reddi artık **boş VRAM**'e bakıyor
      (`vram_free_gib`; CUDA context'i Colab'da 0.8–1.5 GB yer — toplam "sığar" derken koşu
      OOM'lanırdı); mesaj hangi sınırın uygulandığını söylüyor (boş/toplam/`--vram-limit-gib`);
      testli (sahte GPU ortamıyla)
- Başarı kriterleri:
  - [x] GPU'suz tam boy iş reddedildi — kanıt koşusu (560×700×480 homojen, numpy, --no-measure):
        exit 2, "~9509 s (~2.6 h, estimate source: calibrated), over the 5 min CPU limit",
        iki kaçış da mesajda (devlog 2026-08-22); mekanizma `tests/test_env_gate.py` (7 test)
  - [x] Paketli örnek eşiğin altında TAM BİR uyarıyla koştu (test:
        `test_packaged_example_runs_with_exactly_one_warning`; taze süreçte buna süreç-başına-bir
        backend düşüş uyarısı eklenir — iki ayrı ölçütün bileşimi, testte belgeli);
        `allow_slow_cpu=True` reddi geçersiz kılıyor (testli)
  - [x] `env_report()` cupy'siz makinede sözlük döndürüyor, çökmez, JSON-serileşebilir (testli)
  - [x] `workers=1` vs `workers=-1`: fazor alanları **BİT-AYNI** (`assert_array_equal`, linear +
        westervelt; golden toleransı değil sıkı eşitlik — gözden geçirenin ölçümüyle uyumlu:
        pocketfft toplama sırasını değiştirmez). Hızlanma: **1.00× — tekrarlanabilir kazanç YOK**
        (ölçülen; tablolar devlog'da). Varsayılan bu ölçümle 1
  - [x] cupy'siz `backend="auto"` → görünür `CausticaWarning`, süreç başına TAM BİR kez (testli)
  - [x] ppw uyarısı dört yerde — dört test (plan, status, run_meta, rapor başı)

### M10m — Dışarıdan kullanılabilirlik: kendi kurulumunu getir `[ ]` — **DEVİR PAKETİ 4/4**
Kabul sorusu: **hiç tanımadığımız bir araştırmacı, repoyu bulup kendi problemini koşabiliyor mu?**
Bugün cevap HAYIR — job şeması yalnızca spiral+bowl tanıyor, kendi hacmini üretecek yazıcı yok ve
şemayı okumak için pydantic kaynağına inmek gerekiyor. M10k'dan SONRA gelir (`config/job.py`).
- [ ] `elements` array kind'ı: açık eleman pozisyonları + normalleri (JSON içinde satır satır ya
      da `.npz`/`.csv` dosya referansıyla), eleman yarıçapı ve odak uzaklığıyla. `TransducerArray`
      zaten genel — eksik olan yalnızca şema kapısı. `derived()` kalıbı korunur (yeniden yüklemede
      geometri yeniden türetilir, sessiz sapma yakalanır)
- [ ] `caustica schema` komutu: `caustica-job/1` şemasını JSON Schema olarak basar (pydantic'ten
      üretilir — elle yazılmış ikinci bir tanım OLMAZ)
- [ ] `docs/job_reference.md`: her medium kind'ı, her array kind'ı, drive/run/output bölümleri,
      her biri çalışan bir örnek parçasıyla
- [ ] `docs/conventions.md`: fazor konvansiyonu `p(t)=Re{P·e^{-iωt}}`, Np/m ↔ dB/cm, `amplitude`
      alanının ne demek olduğu (kütle-kaynak normalizasyonu sonrası gerçekleşen genlik), koordinat
      çerçevesi (+z ışın ekseni, apex frame), PML'in grid'e dahil olduğu. Bunlar bilinmezse sonuç
      SESSİZCE yanlış yorumlanır
- [ ] README'de Colab quickstart: `pip install git+...` → paketli örnek → `caustica report`,
      dış veri olmadan uçtan uca
- Başarı kriterleri:
  - Kendi eleman pozisyonlarını `.npz`'den okuyan bir `elements` job'ı uçtan uca koşar ve
    `derived()` yeniden yüklemede eşleşir
  - `caustica schema` çıktısı geçerli JSON Schema; şemadaki kind listesi ile
    `docs/job_reference.md` başlıkları test ile karşılaştırılır (doküman sessizce eskimez)
  - **Yabancı-kullanıcı provası**: repo dışında, temiz ortamda, YALNIZCA README + job_reference
    okunarak kendi çanak+su senaryosu yazılıp koşulur (adım adım devlog'a yazılır)

### M10j — Notebook ergonomisi: facade + ilerleme `[ ]`
M10k'dan SONRA gelir: facade da job şemasına dokunur (`stored_setup` kalkmadan yazılırsa iki kez
elden geçer).
- [ ] `caustica.simulate(...)` — girdisini job'a çevirip AYNI `build_job`'dan geçer; `out=None`
      plan-önce disiplinini KAPATMAZ; `out=<yol>` `run_job_file`'a delege eder (göreli yol job
      dosyasına göre çözülür kuralı korunur)
- [ ] `progress=` kancası: `_period_boundary` checkpoint'ten BAĞIMSIZ hale getirilir,
      linear+westervelt+engine üçlüsüne eklenir, kwave adaptörüne GEÇİRİLMEZ; `_Heartbeat` bu
      payload'un TÜKETİCİSİ olur (ikinci implementasyon değil); koşu-içi önizleme VARSAYILAN AÇIK
      (8 periyotta bir odak kesiti); callback hatası koşuyu düşürmez
- Başarı kriterleri:
  - `simulate(job)` ile `caustica run job.json` bit-aynı alan üretir
  - Checkpoint'SİZ mini koşu periyot başına tam bir kez callback çağırır
  - `progress=` verilen kwave işi ÇÖKMEZ; hata fırlatan callback koşuyu düşürmez
  - `out=None` diske hiçbir şey yazmaz

### M10l — GUI sözleşmesinin dondurulması `[ ]` — GUI kodu YOK
GUI ayrı repoda olacak ve teknolojisi seçilmedi (PLAN.md K13). Bu milestone yalnızca GUI'nin
üstüne oturacağı yüzeyi yazıya döker ve katmanlamayı testle kilitler.
- [ ] `docs/gui_contract.md`: `caustica-job/1` + `validate`, runner çıkış kodları, `status.json`
      alanları, `caustica-preview/1`, `caustica-result/1`, `env_report()`, ilerleme payload'u.
      Listelenmeyen hiçbir şey sözleşme değildir
- [ ] `tests/test_import_direction.py`: `src/caustica` altındaki her modül AST ile taranır;
      `apps` / `uwcem_phantoms` / `caustica_gui*` importu YASAK
- [ ] **İPTAL SİNYALİ** (bugün YOK — `grep -rn "cancel" src/caustica` boş): çıktı klasöründe bir
      `cancel` dosyası görülürse koşu periyot sınırında temiz durur, checkpoint yazar ve çıkış
      kodu 5 (interrupted-resumable) ile çıkar. `stop_when` kancası zaten orada — eksik olan
      dosya yoklaması. GUI'nin "Durdur" düğmesinin yazacağı yer budur; süreci öldürmek tek yol
      olarak kalmamalı
- [ ] **YAPILANDIRILMIŞ HATA** (bugün YOK): koşu başlamadan oluşan hatalar (bozuk job, OOM reddi,
      checkpoint çakışması) `hb` yaratılmadan `return EXIT_CONFIG` ile çıkıyor — `status.json` hiç
      oluşmuyor, GUI'ye stderr metnini ayrıştırmak kalıyor. Çıktı klasörüne `error.json` yazılsın:
      `{stage, exit_code, error_class, message, advice[]}`. Planner zaten `est.advice` üretiyor;
      şu an yalnızca ekrana basılıyor
- Başarı kriterleri:
  - Import-yönü testi yeşil (M10k inmeden yeşile dönemez — kasıtlı)
  - Hiçbir GUI bağımlılığı veya `gui` extra'sı eklenmemiş
  - Koşan bir işe `cancel` dosyası atılınca: periyot sınırında durur, checkpoint bırakır, çıkış 5;
    `--resume` ile tamamlanır ve sonuç kesintisiz koşuyla BİT-AYNI olur
  - Yedi hata sınıfının her biri (M10b'nin `validate` sınıfları + OOM + checkpoint çakışması)
    `error.json` üretir; hiçbir GUI yolu stderr ayrıştırmak zorunda kalmaz

### M10f — Colab köprüsü: `caustica.colab` + değişmeyen notebook `[ ]` — Colab kapısı içerir
"Değişmeyen dosya" şartı mantığın notebook'ta DEĞİL repoda yaşamasıyla sağlanır: notebook 4–5
hücre, tek düzenlenen satır CONFIG yolu; gerisi `from caustica.colab import run_job`. Güncelleme
`git pull` ile gelir, .ipynb'ye dokunulmaz.
- [ ] `caustica.colab`: ortam kontrolü (GPU adı, cupy, boş VRAM planner tahminine yetiyor mu —
      uygunsuzsa HİÇBİR ŞEY hazırlamadan, aksiyon önerisiyle dur), koşu, çıktı `/content` altında
- [ ] Dataset staging KALDIRILDI (PLAN.md K3/K7/K9): kütüphanede anatomik veri yolu yok. Colab
      kullanıcısı ya paketli sentetik örneği ya kendi `medium_volume` dosyasını getirir; UWCEM
      fantomu isteyen `uwcem-phantom` repo'sunu kullanır
- [ ] Drive KALDIRILDI (PLAN.md K12): `caustica.colab` Drive mount ETMEZ, Drive yollarını bilmez,
      Drive'a özgü yeniden deneme mantığı taşımaz. Kalıcılık isteyen kullanıcı Drive'ı KENDİ mount
      edip `--out` ile oraya yazdırır (runner bunu zaten destekler). Kabul edilen risk: oturum
      çökerse `/content` gider — checkpoint oturum-içi restart'ı kurtarır, VM teardown'ı kurtarmaz
- [ ] `notebooks/colab_run.ipynb` repoda; Colab'ın open-in-GitHub linkiyle açılır (M10e ön koşul)
- Başarı kriterleri:
  - Notebook sözleşmesi: CONFIG satırı dışında düzenleme gerektirmez; mantık değişikliği notebook
    diff'i SIFIR olacak şekilde repodan gelir (kontrat testi: hücre içerikleri sabit şablonla karşılaştırılır)
  - `caustica.colab` içinde `google.colab`/Drive dışı hiçbir ortam varsayımı yok; `grep -ri drive
    src/caustica` boş
  - Colab kapısı: repodan açılan notebook → `/content` altında koşu → sonuç indirilip lokalde
    `caustica report` ile açılır (uçtan uca)
  - İlk Colab oturumu üç kapıyı birden kapatır: M7 parite + tam boy OOM'suz koşu, M8 VRAM ±%10 ve
    kalibre süre ±%25, bu E2E — runner damgası ölçümleri zaten topluyor

### M10g — Kuyruk: paylaşılan klasör `jobs/` protokolü `[ ]` — GUI'nin "Run in Colab" altyapısı
Notebook tek job yerine klasör izler: lokalde job at → oturum açıkken kendiliğinden koşar.
Protokol **paylaşılan bir klasör yolu** alır; Drive onun bir örneğidir ve kütüphane bunu bilmez
(PLAN.md K12). Runner çıkış kodları kuyruğun API'sidir: DEĞİŞTİRİLMEZ, YENİSİ EKLENMEZ.
- [ ] Protokol: `jobs/pending/` → claim (kilit dosyası: oturum id + zaman damgası, atomik rename) →
      `jobs/running/` → `jobs/done/` | `jobs/failed/`; her job kendi çıktı klasörü + status.json
- [ ] Ölü oturum devri: `running/`de kalıp kalp atışı eskiyen job, yeni oturumca `--resume` ile
      devralınır (checkpoint M10'dan)
- [ ] İzleme döngüsü: idle'da N sn'de bir yoklama; `--once` (mevcutları bitir, çık) ve `--watch` modları
- Başarı kriterleri:
  - Çift oturum simülasyonu (CPU, yerel klasör): aynı job İKİ KEZ KOŞMAZ (claim yarışı testli).
    Yerel klasörle geçen testler ağ dosya sistemine güvenmeden ÖNCE koşulur — Drive'ın rename
    semantiği POSIX'ten zayıftır
  - Kill edilen oturumun job'ı ikinci oturumda resume ile tamamlanır; çift üretim yok
  - 3 job'lık mini kuyruk sırayla biter; failed job kuyruğu TIKAMAZ (kenara alınır, log'lu)
  - Colab kapısı: canlı oturumda 2 job'lık kuyruk uçtan uca

### M11 — Study/Report + doğrulama harness'i `[ ]`
- [ ] `study.Study`: config + koşu + sonuç + figürler; `report()` → Markdown (+JSON); ortam/GPU/git-hash damgası; `Study.sweep(...)`; analitik doğrulama süiti tek komutla rapor üretir
- Başarı kriterleri:
  - Tek komut: `python -m caustica.validation run-analytic` → damgalı JSON+MD rapor `benchmarks/reports/` altına düşer
  - Rapor, planner tahmini vs gerçekleşen tablosunu içerir
  - Sweep: 3-noktalı p0 taraması uçtan uca koşar, birleşik rapor üretir
  - **v1 ön-etiketi**: M4–M11 kriterlerinin tamamı yeşil → `v0.1` tag (isim kararı + GitHub public M10e'ye öne çekildi, 2026-08-19)

### M12 — k-Wave karşılaştırma harness'i (M4b adaptörünün üstünde) `[ ]`
- [ ] M4b'deki `kwave` çözücüsünü kullanan sistematik harness; T0 sanity kapısı (homojen su — GPU binary'yi her ortamda önce bundan geçir; kalırsa "environment-broken" damgası); karşılaştırma metrikleri (relL2, Pearson, odak konum/amp, −6dB genişlikler, sidelobe)
- Başarı kriterleri:
  - Küçük grid lineer su çanağı: caustica vs k-Wave CPU → relL2 < %3, r > 0.99, odak konumu < 1 voxel
  - Heterojen küçük fantom (2 doku): relL2 < %5, odak basınç farkı < %10 (ITRUSST koridoru)
  - Nonlineer küçük senaryo: 2f0/f0 oranı farkı < %10
  - Rapor damgalı; k-Wave sürümü/binary tipi (CPU/GPU) kayıtlı
  - VERIFY: k-Wave GPU binary'nin Colab durumu yeniden test edilir (notebook v11 bulgusu + gemini "eski mimari desteği çekildi" iddiası)

### M13 — Dataset pipeline (Katman C) `[ ]`
- [ ] `pipelines.DatasetGenerator`: dondurulmuş LHS (seed→checksum), üretim döngüsü, background save, ETA (planner'dan), metadata/timing CSV, disk kontrolü — M10c runner + M10g kuyruk üzerine oturur
- Başarı kriterleri:
  - Mini dataset (N=3, küçük grid): iki ayrı koşuda AYNI checksum'lar (LHS dondurma değişmezi)
  - Kesinti+resume testi: 2. örnekte kill → yeniden başlatma kaldığı yerden, çift üretim yok
  - metadata.csv satır bütünlüğü (append-only, kolon şeması sabit)
  - ETA log'ları planner modelini kullanır (start→start cadence ölçümü korunur)

### M14 — Colab üretim doğrulaması `[ ]` — Colab oturumu gerektirir
- [ ] Tam boy dx=0.30 senaryosu Colab A100'de `caustica run` job akışıyla (M10c/M10f) koşar; notebook v12 ile karşılaştırma
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

- M6d + M6e kapandı (2026-08-19): fantomlar `uwcem_phantoms/` yan paketine taşındı; standart
  hizalı 0.25 mm dataset `data/phantoms/`ta (9 dosya, tek grid **560×700×480 = 140×175×120 mm**,
  verify 9/9). z ekseni: 20 mm transducer suyu + 100 mm doku, 120 mm'de tavanlı; göğüs duvarı
  arkasındaki su ve slab atıldı, kesilen doku fantom başına manifest'te sayılı. Colab'a
  geçişte dataset Drive'dan taşınır ya da `python -m uwcem_phantoms dataset` ile yeniden üretilir.
- **Simülasyona hazırlık durumu (2026-08-19):** zincir uçtan uca koştu (fantom → PML → çanak →
  Westervelt, yakınsadı, 2. harmonik var) ve M6f ile dokuz koşu `data/setups/`ta depolandı —
  `load_setup("s1-012304")` doğrudan çözücüye veriliyor. Planner: A100'de 20.36/38.88 GiB,
  1890 adım, ~3.1 dk. Kalan boşluklar: `apps/focus_study`'de fantom senaryosu YOK; cupy bu
  makinede kurulu değil (çözücü döngüsü backend-generic, Colab'da koşmalı — M7 yalnız kernel
  füzyonu/hız).
- **Colab entegrasyon planı işlendi (2026-08-19, kullanıcı kararları):** (1) repo public'leşmesi
  M11'den M10e'ye öne çekildi (Colab clone token'sız olur); (2) dataset Drive birincil + fallback
  yerinde üretim (ölçüm: 9 fantom yerel ~8 dk, CPU işi > "H100'de 5 dk" eşiği); (3) kuyruk ayrı
  milestone (M10g — GUI'nin "Run in Colab" altyapısı); (4) job şeması TAM genişletme (M10b:
  scene + serbest array + volume import, sadece dokuz stored setup değil). GUI kapsam dışı kaldı;
  M10b şeması + M10d önizlemesi + M10g kuyruğu GUI'nin sonradan oturacağı kontratlar.
- M10 kapandı (2026-08-19, aynı gün): `caustica.io` — atomik yazım, float16 kontratı,
  `caustica-result/1`, ResultStore + skip-guard, koşu-içi checkpoint (bitwise resume). 330 test.
- M10b kapandı (2026-08-19, aynı gün): `caustica-job/1` — stored_setup JOB-seviyesi kind (bilinçli
  sapma, M6f bütünlüğü) + explicit tam ağaç (4 medium × 2 array); override katmanı f0'ı alpha
  gerekçesiyle reddediyor; `python -m caustica validate` 7 hata sınıfını yakalıyor. 365 test.
- M10c kapandı (2026-08-19, aynı gün): `caustica run` — plan-önce, ayrık exit kodları (0/2/3/4/5),
  status.json kalp atışı, tam damga, store-çökmesine dayanıklı resume. Ardından M10+M10b+M10c
  üzerinde adversarial review turu: 14 bulgu → 12 düzeltme (kwave kwarg, explicit-dataset f0-alpha
  deliği, save-çökmesinde çözüm kaybı, tmp yarışları, CWD-outdir). 383 test.
- M10d kapandı (2026-08-21): `caustica.report` — metrik tek-kaynak (focus_study delege eder),
  ≤10 MB önizleme paketi (runner her koşuda yazar), `caustica report` (result'tan tam figür seti,
  `--preview` ile paketten hızlı görünüm). focus_study regresyonsuz (raporlar bayt-aynı). 393 test.
- İsim + rename tamam (2026-08-21, kullanıcı kararı): **caustica** — `src/caustica`, format
  etiketleri `caustica-*` (5.66 GB dataset legacy-alias'la rebuild'siz), GitHub **ebx0/caustica**
  (zaten public). Janitor turu #1 aynı gün: 44 bulgu tarandı, ~20 düzeltme + 9 yeni koruma testi
  (wheel'e gpu_db.json — pip kurulumunda planner çökerdi), açık işler `janitor/` klasöründe.
- **Kütüphane-önce dönüşümü planlandı (2026-08-21, kullanıcı kararları):** proje "gerçek public
  kütüphane" hedefine oturtuldu. Kararlar PLAN.md §0.2'de (K1–K14) ve docs/library_first_plan.md'de
  (D1–D22, uygulama detayı) kayıtlı; vazgeçilenler PLAN.md §0.3'te. Özet: üretilmiş dataset hiç
  dağıtılmaz · kütüphane tamamen UWCEM'siz olur (ayrı repo) · genel `medium_volume` kind'ı gelir ·
  public API üç katman (facade → nesneler → job JSON) · GPU'suz 5 dk üstü koşu reddedilir · cupy
  asla otomatik kurulmaz · ilerleme + odak önizlemesi varsayılan açık · Colab'da Drive
  KULLANILMAZ (`/content`) · GUI ayrı repo, şimdilik yalnızca sözleşme dondurulur.
- **Şimdi (devredilen paket):** **M10h + M10i + M10k + M10m** (2026-08-21 kullanıcı kararları).
  Sıra: M10h paketleme → M10i ortam ve güvenlik politikası → M10k UWCEM ayrışımı → M10m dışarıdan
  kullanılabilirlik. M10h/M10i `config/job.py`'ye dokunmadığı için ayrışımla çakışmaz; M10m
  ayrışımı bekler. Ayrışım sırasında dokuz setup'ın geçici olarak çalışmaması KABUL, bitişte
  bit-aynı koşmalı. **M10j (facade) bilerek dışarıda** — sonraki tur.
- **Paketin kabul sorusu:** bu dört milestone kapandığında *hiç tanımadığımız bir araştırmacı*
  repoyu bulup, README'yi okuyup, Colab'da ya da yerelde **kendi** problemini koşabiliyor olmalı.
  M10m'in "yabancı-kullanıcı provası" kriteri bunu ölçer.
- **Sonra:** M10j facade + ilerleme → M10l GUI sözleşmesi → M10f Colab köprüsü → **ilk Colab
  oturumu tek seferde üç kapıyı kapatır** (M7 parite/tam-boy + M8 VRAM ±%10 ve süre ±%25 +
  M10f E2E) → M10g kuyruk. M14 bu akışın üstünden koşar.
  Not: GPU paritesi (M7) o oturuma kadar DOĞRULANMAMIŞ kalır (kullanıcı kararı 2026-08-21);
  README'de GPU iddiaları bu nedenle açıkça "doğrulanmadı" işaretli durur.
- **M10e kalan kalem:** commit + push (KULLANICI ONAYI) → public CI yeşili (3.10 taban ayağı
  dahil) + temiz-ortam `pip install git+.../caustica` + UWCEM atıf son kontrol (`janitor/06`).
  Not: UWCEM atıf yükümlülüğü M10k ile `uwcem-phantom` repo'suna taşınıyor.
- **Çalışma dalı:** `library-first` — her iş kalemi sonunda YEREL commit, push yok, `master`
  dokunulmaz (kullanıcı 2026-08-21: karar bende).
- M8 yerel yarısı kapandı (2026-08-11): `caustica.planner` — VRAM envanteri (engine birebir),
  a·N·log2N+b·N süre modeli, gpu_db.json (7 cihaz), cpu/cuda kalibrasyon + calibration.json,
  `estimate`/`compare`, kaynak etiketi db|calibrated|measured, OOM önerileri; 11 test.
- Geometri adversarial review turu yapıldı (2026-08-11): 9 bulgu → 5'i düzeltildi-testlendi
  (resample zoom hizalama kayması → tam-pozisyon örnekleme; cache argüman parmak izi; add_volume
  volume.origin; axisym |r| aynalama; chunk böleni s^ndim; majority tie=son boyanan; __eq__;
  HalfSpace+Transform config'leri). Fizik motoru review'u aynı gün: çekirdek fizik temiz;
  4 sınır-kontratı düzeltmesi — kaynak genliği kütle-kaynak normalizasyonu (2c·dt/dx),
  kütüphane çapı fazor konvansiyonu p(t)=Re{P e^{-iωt}}, kwave pml_size=grid.pml_vox +
  kaynak-PML çakışma reddi, settle_capped/ramp koruması (ayrıntı: docs/devlog.md).
- M6b kapandı (2026-08-11): 22 geometri testi; kriter notları — küre hacmi <%2 hem s=1 hem s=3'te
  (hacim hataları istatistiksel dengelenir; süperörnekleme kapısı s=5 referansına yakınsama olarak
  ölçülür), 0.5→0.3 mm resample arayüz ≤1 yeni-voxel, config build == elle kurulum (birebir id_map).
- M0–M6 kriter kanıtları: 90 test yeşil (devlog 2026-08-10). k-Wave canlı çapraz doğrulama r>0.99;
  Fubini A2/A1 < %5; O'Neil 3D kapıları; DAS/voxelizasyon kapıları.
