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

### M7 — CuPy backend (CUDA) `[ ]` — ikinci Colab oturumu gerektirir
- [ ] ElementwiseKernel'ların portu; aynı çözücü kodu iki backend'de; fp32 yolu
- **Ölçüm protokolü HAZIR (2026-08-23):** `python -m caustica.validation gpu-gates` tek komutta
  kalibrasyon → VRAM merdiveni → OOM reddi → numpy/cupy paritesi → damgalı MD+JSON rapor
  (`benchmarks/reports/gpu_gates/<gpu>-<tarih>/`). GPU yoksa eyleme geçirilebilir mesajla temiz
  çıkış (kod 2 = CI'da SKIP). Merdiven kurulumu, VERDICT cebri, rapor şeması, OOM dalı ve
  parite ölçüm noktası CPU'da 36 testle çivili (`tests/test_validation_gpu_gates.py`).
  **Aşağıdaki kutular yine de `[ ]`: sayılar cihazda ölçülmeden işaretlenmez.**
- Başarı kriterleri:
  - [ ] numpy↔cupy parite: mini 3D senaryoda fazor/p_max rel fark < 1e-5 (fp32 toleransı
        belgelenir) — süitte `M7.parity` kapısı. **Ölçüm noktası düzeltildi (operatör ölçümü,
        2026-08-23):** kapı BELLEKTEKİ fp32 alanlar üzerinden ölçülür, `result.h5` round-trip'i
        üzerinden DEĞİL. İlk oturumun dosyası yerel CPU koşusuyla relL2 3.6e-5 / relL∞ 4.883e-4
        farkla uyuşuyordu ve 4.883e-4 tam olarak 2^-11 = **bir float16 ULP'u** (p_max'ta %99.17
        voxel bit-özdeş, 517 voxel 1 ULP, >1 ULP sıfır). Yani alanlar dosyanın çözünürlüğünün
        ALTINDA uyuşuyor; 1e-5 kapısını float16 depolamaya bakarak kurmak kusursuz bir çözücüye
        YANLIŞ FAIL verirdi. Depolama tabanı raporda ayrı başlıkta (`stored_float16_reference`,
        bilgi amaçlı, kapıya girmez)
  - [ ] Colab T4 VE A100'de tam boy (dx=0.30, 512³ FFT sınıfı) koşu OOM'suz tamamlanır —
        süitte `M7.fullsize` kapısı; merdivenin 14 GiB basamağı tam olarak 512³/dx=0.30
  - [ ] Adım süresi ölçülür ve `benchmarks/`e damgalanır (baseline; M19 bunu referans alır) —
        süit `step_time_baseline` bölümünü yazar; ısınma AYRIŞTIRILMIŞ (`t_step_steady_s`)
  - [x] GPU yokken testler otomatik SKIP (CI kırılmaz) — süit de dahil:
        `test_the_real_cli_skips_cleanly_on_a_machine_without_a_gpu`

### M8 — Planner v1 (süre + VRAM tahmini) `[~]` — yerel yarısı tamam (2026-08-11), Colab kapıları açık
- [x] Statik VRAM modeli (tampon dökümü + cuFFT workspace payı + %15 marj); süre modeli a·N·logN + b·N; `gpu_db.json` (T4/L4/V100/A100/H100); cihazda kalibrasyon (~20 adım) → `~/.caustica/calibration.json`; `planner.estimate(gpu=...)` + `planner.compare(...)` — `src/caustica/planner/`, 11 test (`tests/test_planner.py`)
- Başarı kriterleri:
  - [ ] VRAM tahmini, Colab'da ölçülen mempool tepe değerinin ±%10'u içinde (≥2 farklı grid boyutunda) — **Colab kapısı, ikinci oturumda ölçülecek**; süitte `M8.vram`: merdiven ≥2 basamak üretir ve tek iyi ölçüm PASS saydırmaz (`test_one_measurement_is_not_two`)
  - [ ] Kalibrasyon SONRASI süre tahmini gerçekleşenin ±%25'i içinde (aynı cihaz, ≥2 senaryo) — **Colab kapısı** (mekanik yerelde testli: cpu kalibrasyonu → fit → estimate zinciri); süitte `M8.time`: süit kendi `planner.calibrate()`ini koşarak "kalibrasyon sonrası" şartını ÜRETİR, varsaymaz
  - [ ] OOM reddi cihazda kanıtlanır: merdivenin bir üstü (cihaza SIĞMAYAN en küçük şekil) çıkış 3 ile reddedilir ve öneri metni rapora kanıt olarak girer — süitte `M8.oom` (**ikinci Colab oturumu**)
  - [x] Tahmin kaynağı raporda etiketli: `db` | `calibrated` | `measured` (testli; cpu kalibrasyonu GPU anahtarıyla asla eşleşmez)
  - [x] OOM öngörüsünde eyleme geçirilebilir öneri metni (dx büyüt ×m hesaplı / AOI küçült / linear'a geç / daha büyük cihaz) — testli
- **Süre modeli düzeltildi (fix A2, 2026-08-23):** `t_expected = warmup + steps·t_step`. İlk Colab
  oturumu `t_step_measured_s`i 26.6 ms okudu, planner probu aynı süreçte aynı şekilde 1.03 ms —
  25.9× "sapma". Aynı iş CPU'da 0.96×. Muhasebe doğruydu, EKSİK OLAN SABİT terimdi: ~2.66 s'lik
  tek seferlik cuFFT-plan/JIT/ilk-tahsis maliyeti 104 adıma yayılıyordu. `model.GPU_WARMUP_S`
  (3.0 s, o oturumdan), `calibrate()` cihazda ısınma ölçüp saklıyor, `planner.record_warmup()`
  gerçek koşunun ödediğini geri yazıyor. run_meta.actual'a EKLENEN alanlar: `warmup_s`,
  `t_step_steady_s`, `steady_samples` — mevcut anahtarların hiçbiri değişmedi
  (`t_step_measured_s` dahil; M8'in Colab kapıları ve gui_contract onu okuyor)
- Not: dt/spp ve time-of-flight türetimi motordan `cw_discretization`/`cw_tof_periods` fonksiyonlarına çıkarıldı (tek doğruluk kaynağı; planner==engine testli). VRAM envanteri engine.py tampon listesini birebir aynalar — motora yeni kalıcı tampon eklersen `test_memory_inventory_matches_hand_count` kırılır (bilerek).

### M9 — KZK çözücüsü `[ERTELENDİ 2026-08-22]`
Kullanıcı kararı: M15 (AS) öne geçti — AS, KZK'nın kullanım alanının çoğunu TAM fizikle
kapatıyor ve GPU'da tam-dalga zaten hızlanacak. KZK native ailenin uzun-vade vizyonunda
duruyor (kullanıcı 2026-08-22: "westervelt, AS, kzk vs."); talep/manzara değişirse eski
kriterleriyle (git geçmişinde) geri döner. FDA HITU karşılaştırması o gün için not.

---

## Faz Grubu C — Veri, IO, Colab entegrasyonu (v1 çizgisi)

Omurga (2026-08-21'de kütüphane-önce kararlarıyla revize edildi — ayrıntı: PLAN.md §0.2 ve
docs/library_first_plan.md): lokalde `job.json` yaz → Colab'da değişmeyen notebook koşar →
çıktı `/content` altına düşer → lokalde raporla. **Drive kütüphanenin işi değil** (K12).
**M10 → M10b → M10c → M10d** (hepsi CPU'da yazılır ve testlenir, Colab beklemez) → **M10e**
public → **M10h ✅ + M10i ✅ + M10k ✅ + M10m ✅ + M10n ✅** (2026-08-22 kapandı) → **M10j**
facade + ilerleme → **M10l** GUI
sözleşmesi → **M10f** Colab köprüsü → **ilk Colab oturumu** (M7 + M8 kapılarıyla birleşik) →
**M10g** kuyruk → **UWCEM kalanları** (docs/uwcem.md, EN SON — kullanıcı 2026-08-22).
M12–M14 bu omurganın üstünden koşar.

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

### M10e — Public'leşme `[x]` (2026-08-22 — merge `ef837bf` ile kapandı)
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
  - [x] Repo public; `library-first` → `master` merge edildi (PR #1, normal merge, `ef837bf`;
        kullanıcı onayı 2026-08-22). master CI YEŞİL; temiz venv'de
        `pip install "caustica @ git+https://github.com/ebx0/caustica"` → import + `caustica
        --version` + `from caustica.colab import run_job` ÇALIŞIYOR; CONFIG raw URL 200
        (operatör doğrulaması 2026-08-22)
  - [x] UWCEM atıf yükümlülüğü M10k ile `uwcem-phantom` repo'suna taşındı (docs/uwcem.md).
        Not: M10k ile bu yükümlülük `uwcem-phantom` repo'suna taşınıyor
  - [ ] **`CITATION.cff`** (bugün YOK): public bir araştırma kütüphanesi alıntılanabilir olmalı.
        v0.1 tag'inde Zenodo DOI'si alınır ve README'ye "How to cite" bölümü eklenir

### M10k — UWCEM ayrışımı: kütüphane UWCEM'siz olur `[x]` (2026-08-22)
Gerekçe (K7–K10): lisans riski ve katmanlama ihlali aynı kökten geliyordu — `config/job.py`
dört yerden `uwcem_phantoms` import ediyordu. Ayrışım W0a–W0f olarak kapandı; **UWCEM'e dair
güncel durum + kalan işler TEK dosyada: `docs/uwcem.md`** (kullanıcı 2026-08-22: kalanlar EN SON).
- [x] (W0a, 2026-08-22) `medium_volume` genel medium kind'ı: `caustica.io.medium_volume` —
      etiket haritası + malzeme db'si VEYA voxel-başına özellik hacimleri; **grid dosyadan
      gelir** (explicit `grid` "fixes the grid" hata metniyle reddedilir — testli). MEVCUT
      dosyaları OLDUĞU GİBİ okur: dosya-içi etiketler `caustica-phantom/1` + `hifusim-phantom/1`
      legacy alias kabul (plan metnindeki `*-dataset/2` adları manifest etiketiydi; dosya
      etiketi ground-truth'tan düzeltildi). KANIT: gerçek 560×700×480 dataset dosyasında
      `medium_volume` → Medium, PhantomAsset yolununkiyle sha256-düzeyinde BİT-AYNI
      (4 özellik hacmi + id_map; `test_existing_dataset_file_gives_bit_identical_medium`, 25 sn).
      M6f f0-alpha koruması ve su-odak reddi medium_volume'a genelleşti (`water_label`
      alanıyla kapatılabilir; ikisi de testli)
- [x] (W0a, 2026-08-22) Format hem OKUNUR hem YAZILIR: `write_medium_volume(...)` public —
      numpy dizilerinden (etiket VEYA c/ρ/α/β) + dx + origin; yeni dosyalar
      `caustica-medium-volume/1` etiketi taşır. Round-trip bit-aynı Medium (label + continuous
      modda testli); `uwcem-phantom` repo'su W0d'de bu fonksiyona bağlanacak
- [x] `geometry/volumes.py::load_breast_phantom()` taşındı (W0c'de geometry'den çıktı, W0d'de
      yeni repo `legacy_import.py`'de — kutu güncellemesi unutulmuştu, 2026-08-22 kapatıldı).
      `load_labels_txt` GENELDİR (mapping callable'ı kaynağa özgü kısmı dışarıda tutar) — KALDI
- [x] (W0b, 2026-08-22) Literatür akustik doku değerleri `caustica.materials.TISSUE_LIBRARY`'de
      (`AcousticTissue` taşıyıcısı + dB/cm↔Np/m dönüşümleriyle; kaynak/ölçüm yorumları harfiyen,
      isimler dahil — isim MaterialDB JSON'una gömülü olduğundan değişmedi). Beş literatür-çapa
      giriş taşındı (su-37C, deri, kas, fibroglandüler uç, yağ ucu); UWCEM ramp satırları ve
      media-numarası eşlemesi tissue.py'de KALDI ve artık uçları kütüphaneden `is`-aynı nesne
      olarak kullanıyor. KANIT: `test_tissue_library_values_pinned_to_the_digit` (rakamlar
      donmuş literal), `test_uwcem_table_uses_the_moved_library_rows` (tek kaynak), 93 fantom
      testi değişmeden yeşil
- [x] (W0c, 2026-08-22) `_require_uwcem`, `PhantomDatasetMediumConfig`, `StoredSetupJobConfig`
      (+ `StoredSetupOverrides`, `RunPolicyOverrides` — yalnız stored katmanının parçasıydı,
      ölü kod bırakılmadı) SİLİNDİ; `MediumConfig` = homogeneous|scene|volume_import|
      medium_volume; `geometry`'den `load_breast_phantom`/`breast_phantom_mapping` ve
      `VolumeImportConfig`'ten `breast_phantom_txt` formatı çıktı (`load_labels_txt` generik,
      KALDI). **Kırıcı şema değişikliği**: `caustica-job/1` iki kind + bir import formatı
      kaybetti, format numarası değişmedi (K14; devlog'da). `grep -ri uwcem src/` = 0 ve
      import-yönü AST testi YEŞİL (`tests/test_import_direction.py` — W0c'den önce bilerek
      kırmızı yazıldı); UWCEM'e bağlı 11 test bloğu verbatim staging'e alındı (W0d'de yeni
      repoya)
- [x] (W0d, 2026-08-22) `uwcem-phantom` ayrı repo (`C:\Users\bulbu\Desktop\uwcem-phantom`,
      YEREL — push YOK, devir talimatı gereği kullanıcı istemeden pushlanmaz): paket + 
      `apps/phantom_launcher.py` + `apps/phantom_studio` + `phantoms.bat/sh` + test süiti
      taşındı (planın 137 sayısı plan-anı fotoğrafıydı; güncel taşınan süit yeni evde
      **165 passed non-slow** + dokuzlu yavaş kapı). caustica'ya bağımlı; `setup_to_job()`/
      `emit_jobs()` (+ CLI `setup --emit-jobs`) her stored setup'tan **explicit job** üretir
      (medium_volume kapısından). `load_breast_phantom` `legacy_import.py`'de yaşıyor
- [x] (W0e, 2026-08-22) Veri kökü: açık argüman → `CAUSTICA_PHANTOM_DATA` → dolu checkout
      `_data` → kullanıcı önbelleği (`%LOCALAPPDATA%\caustica\phantoms` / `~/.cache/...`),
      elle yazıldı — `platformdirs` YOK. dataset/ ve setups/ aynı kökte (repo varsayımı yok);
      yerelde kök = `hifusim\data` (kullanıcı env değişkeni kuruldu; `_data` alt klasörleri
      aynı diskte oraya taşındı — 4.5 GB dataset ve setup JSON'ları YERİNDEN OYNAMADI).
      Hiçbir dosya yeniden indirilmedi/üretilmedi (taşınan süit + dokuzlu kapı ağsız geçti)
- [x] (W0f, 2026-08-22) UWCEM şartları okundu ve yeni repo README'sine kaydedildi: resmi
      lisans YOK; Instruction Manual verbatim — "free of charge ... reference the online
      repository and acknowledge the authors ... in any publication that is derived".
      Muhafazakâr okuma uygulanıyor: git'te hiçbir fantom baytı yok, türev exportları biz
      dağıtmıyoruz, atıf metni her export'un metadata'sında taşınıyor (`catalog.CITATION`).
      Repo kod-only olduğundan public olabilir — push kararı kullanıcının
- Geçiş penceresi: kullanıcı kararı (2026-08-21) — ayrışım sırasında dokuz yerel setup'ın geçici
  olarak çalışmaması KABUL; bitişte çalışır olması ŞART
- Başarı kriterleri:
  - [x] `grep -ri uwcem src/` boş — testle sürekli zorlanıyor
    (`test_no_uwcem_reference_survives_in_source_text`; base.py yorumu dahil yeniden yazıldı)
  - [x] Import-yönü AST testi geçiyor (`tests/test_import_direction.py` — W0c'den ÖNCE yazıldı,
    kırmızı doğdu, W0c ile yeşile döndü; planın istediği kanıt sırası)
  - [x] Aynı `.npz` → `medium_volume` Medium'u, PhantomAsset yolununkiyle sha256 BİT-AYNI
    (gerçek 560×700×480 dosyada, 4 hacim + id_map; `test_existing_dataset_file_gives_bit_identical_medium`)
  - [x] Round-trip bit-aynı (label + continuous modda testli)
  - [x] Dokuz yerel setup: `load_setup` → `setup_to_job` → caustica build — medya sha256
    BİT-AYNI 9/9 + `caustica validate` ok 9/9 + `caustica run --dry-run` exit 0 9/9
    (`test_all_nine_setups_bit_identical_media_and_dry_run`, 5:04). Hiçbir dosya yeniden
    üretilmedi/indirilmedi. NOT (dürüst yorum): "uçtan uca koşar" burada boru hattının tamamı +
    bitwise medya paritesi demek — dokuz TAM çözüm 9×~2.6 saat CPU'dur ve M10i kapısının tam da
    reddettiği sınıftır; mini ölçekli tam çözüm kanıtı medium_volume koşu testinde var
  - [x] Taşınan süit yeni repoda YEŞİL, buradan silinmeden ÖNCE doğrulandı: 165 passed
    (non-slow) + dokuzlu yavaş kapı 3/3 — plan-anı "137" sayısı o günün fotoğrafıydı (T8)

### M10h — Kütüphane paketleme + temiz ortam kapısı `[x]` (2026-08-21; CI kanıtı 2026-08-22)
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

### M10i — Ortam ve güvenlik politikası `[x]` (2026-08-22)
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

### M10m — Dışarıdan kullanılabilirlik: kendi kurulumunu getir `[x]` (2026-08-22)
Kabul sorusu: **hiç tanımadığımız bir araştırmacı, repoyu bulup kendi problemini koşabiliyor mu?**
Kapanan boşluklar: transducer tarafı artık açık eleman tablosu kabul ediyor, şema komutla
basılıyor, iki referans doküman testle taze tutuluyor. K15 gereği kind kapıları registry'ye döndü.
- [x] `elements` array kind'ı: açık eleman pozisyonları + normalleri (JSON içinde satır satır ya
      da `.npz`/`.csv` dosya referansıyla), eleman yarıçapı ve odak uzaklığıyla. `TransducerArray`
      zaten geneldi — eklenen yalnızca şema kapısı + `arrays/elements.py` okuyucusu/kurucusu.
      `derived()` kalıbı korundu; normaller opsiyonel (yokken her eleman `(0,0,roc_mm)`'ye bakar)
      — `test_elements_job_from_npz_runs_end_to_end`, `test_read_npz_and_csv_agree`,
      `test_missing_normals_aim_at_the_focus`, `test_inline_elements_match_the_same_table_from_file`
- [x] **Kind registry'leri (K15):** `config/kinds.py` — `KindRegistry` + `MediumKindConfig` /
      `ArrayKindConfig` tabanları + `MediumPrep`; union kayıt SIRASINDAN kuruluyor (pydantic
      beklenen-etiket metni bu yüzden değişmedi). Çekirdek yedi kind AYNI dekoratörden geçiyor,
      özel yol yok — `test_core_kinds_register_through_the_same_door`. Sahte bir kurulu dağıtım
      iki grubu da kullanıyor: `test_entry_point_plugin_adds_a_medium_and_an_array_kind`.
      Bozuk plugin öldürmüyor: `test_a_broken_plugin_is_skipped_not_fatal`
- [x] `caustica schema` komutu: `job_schema()` pydantic'ten üretiyor, elle ikinci tanım YOK;
      `--kinds` kayıtlı adları basıyor — `test_schema_is_valid_json_schema` (sarkan `$ref` yok),
      `test_schema_discriminators_match_the_registries`, `test_cli_schema_prints_parseable_json`
- [x] `docs/job_reference.md`: her medium kind, her array kind, source/grid/drive/run/output;
      kind başına çalışan JSON parçası + her varsayılanın GEREKÇESİ —
      `test_reference_documents_exactly_the_registered_kinds` (iki yönlü),
      `test_each_medium_snippet_validates` / `test_each_array_snippet_validates` (7 parametre),
      `test_the_documented_minimal_job_actually_runs_validate` (sıfır uyarı).
      Mutasyonla sınandı: başlık adı bozulunca ve parça anahtarı yanlış yazılınca süit kırmızı
- [x] `docs/conventions.md`: fazor `p(t)=Re{P·e^{-iωt}}` (giden `e^{+ikx}`), Np/m ↔ dB/cm +
      v1 frekans-BAĞIMSIZ alfa uyarısı, `amplitude`'ın `2c·dt/dx` sonrası gerçekleşen genlik
      olduğu, apex/grid çerçeveleri ve job=mm / Python=m ayrımı, PML'in `grid.size_mm` içinde
      olduğu — `test_conventions_covers_the_five_silent_wrongness_traps`
- [x] README'de Colab quickstart + "Bring your own setup" bölümü (`elements` ve `medium_volume`
      üçer satır), iki dokümana bağlantı, `schema` CLI listesinde; bayat
      `ruff check ... uwcem_phantoms` yolu düzeltildi
- Başarı kriterleri:
  - [x] `.npz`'den eleman okuyan `elements` job'ı uçtan uca koşuyor ve `derived()` yeniden
        yüklemede eşleşiyor — `test_elements_derived_matches_on_reload`. **Yanlışlanabilirlik
        gözden geçirmede ÇÜRÜTÜLDÜ ve onarıldı:** özet sayılar (n, r_max, shell_depth) sıra
        istatistikleridir; aynalama, döndürme, satır sırası, iki elemanın yarıçapını takas
        etme ve normalleri değiştirme hepsi bunları KORUYOR ama alanı %48–59 (bağıl L2)
        değiştiriyor — beşi de eski kontrolden geçiyordu. `derived()` artık `table_sha256`
        (pozisyon+normal özeti; dosyanın değil GEOMETRİNİN, böylece inline/.npz/.csv aynı)
        taşıyor — `test_derived_catches_table_changes_that_summaries_miss` (4 mutasyon),
        `test_derived_catches_a_normals_only_change`,
        `test_the_summary_numbers_alone_would_not_catch_these` (mutasyonların gerçekten
        özet-görünmez olduğunu sabitler)
  - [x] `caustica schema` çıktısı geçerli JSON Schema; şemadaki kind listesi ile
        `docs/job_reference.md` başlıkları testle karşılaştırılıyor — `test_schema_doc.py` (13)
  - [x] **Yabancı-kullanıcı provası**: repo DIŞINDA temiz venv, wheel kurulumu, YALNIZCA
        README + job_reference okunarak çanak+su job'ı yazıldı, `validate` → `run` → `report`
        (9.1 s, tepe 1.337 MPa) — sonra kendi 16 elemanlı `.npz` tablosuyla `elements` job'ı
        (adım adım devlog 2026-08-22). Kaynak koda inmek GEREKMEDİ
  - [x] Mevcut davranış bit-değişmez: 279 test → 325 (323 passed / 2 skipped / 0 failed);
        eski commit'e karşı üretilen altın dosya (normalize job.json baytları, `derived` anahtar
        SIRASI + değerleri, focus_vox, kaynak voxel sayısı, faz toplamı, `validate` metni,
        sekiz hata metni) TEK farkla aynı: array beklenen-etiket listesi `'elements'` kazandı
  - [x] `import caustica` yavaşlamadı: 210.6 ms → 207.0 ms (medyan, 9 koşu; entry-point taraması
        tek başına 2.9 ms ve YALNIZ `config.job` import edilince koşuyor) —
        `test_import_caustica_does_not_scan_entry_points`
  - [x] Kayıtsız kind adı aksiyonlu hata: `test_unknown_kind_lists_what_is_registered`
        (kayıtlı adlar + entry-point grubu adı), `test_registration_refusals_teach`
  - [x] **Plugin kurulu ortamda caustica'nın KENDİ süiti yeşil kalıyor** (gözden geçirme
        bulgusu: registry'nin tam içeriğini iddia eden altı test, üçüncü taraf kind kurulunca
        kırmızıya dönüyordu — registry'nin var olma sebebi onu etkinleştiren kütüphanenin
        süitini bozuyordu). Testler artık `core_kinds()` ile caustica'nın kendi kind'larını
        soruyor; gerçek bir plugin `PYTHONPATH`'te iken tam süit doğrulandı
  - [x] `importlib.reload(caustica.config.job)` çalışıyor (`%autoreload 2` = notebook akışı;
        eskiden sınıfın kendisiyle çakıştığını söyleyip modülü yarı-değişmiş bırakıyordu) —
        `test_reloading_the_job_module_still_works`
  - [x] `run_meta.json` geçerli JSON: eksen üstü tabloda `f_number` artık `Infinity` (JSON'da
        böyle bir token yok) yazmıyor — `test_an_on_axis_table_records_no_f_number`
  - [x] Her array kind'ının, dokümanın KENDİ örnek job'ı içinde `validate`'ten geçen en az bir
        parçası var — `test_every_array_kind_has_a_snippet_that_fits_the_documented_grid`
        (spiral bölümü yalnız 100 mm üretim dizisini gösteriyordu: şema-geçerli, odak grid
        dışında)

### M10n — Plugin mimarisi: beş eksen entry-point `[x]` (K15, kullanıcı 2026-08-22)
"Proje her kısmında modüler olabilmeli." Çözücülerdeki kalıp (registry + entry-point +
yetenek deklarasyonu) kalan eksenlere genellenir. Erken-soyutlama riski kullanıcıya söylendi
ve kabul edildi; panzehiri: her seam ÇEKİRDEKTEKİ implementasyonların kendisini de registry'den
geçirmek (özel yol yok — çekirdek, kendi plugin API'sinin birinci müşterisidir).
- [x] Envanter: solver ✅ (var) · medium kind ✅ (M10m) · array kind ✅ (M10m) · **backend**
      (`get_backend` isim→fabrika kaydına döner; numpy/cupy kayıtlı varsayılanlar) · **report
      renderer** (figür/rapor üreticisi seam'i; matplotlib implementasyonu kayıtlı varsayılan)
- [x] Ortak seam TEK yerde: `src/caustica/registry.py` (`PluginRegistry` + `FactoryRegistry`) —
      lazy tarama, reload'a dayanıklı çakışma kontrolü, ya-hep-ya-hiç kayıt, aksiyonlu arama
      hatası. M10m'in `KindRegistry`'si, solver registry'si, backend ve report renderer bunun
      üstünde; T9 gereği yeniden TÜRETİLMEDİ, genellendi
- [x] Job şemasının `backend` alanı kapalı `Literal`'dan registry'nin doğruladığı `str`'e açıldı
      (üçüncü taraf backend job dosyasından erişilemiyordu); `run --backend` argparse
      `choices`'ı kalktı — aynı çıkış kodu (2), aksiyonlu mesaj
- [x] `docs/extending.md`: her eksen için "kendi X'ini ekle" tarifi + çalışan iskelet örneği
      (pyproject + tek modül, beş eksenin hepsini kuran kopyala-yapıştır paket)
- [x] Entry-point grubu adları sabitlenir (`caustica.solvers`, `caustica.medium_kinds`,
      `caustica.array_kinds`, `caustica.backends`, `caustica.report_renderers`)
- Başarı kriterleri:
  - [x] Sahte bir dış paket (test fixture'ı) beş eksenin HER BİRİNE entry-point ile bir uzantı
    ekler ve uçtan uca kullanır (kayıt → keşif → çalıştırma testli) —
    `test_entry_point_plugin_extends_all_five_axes`: M10m'in fixture'ı büyütüldü (ikinci fixture
    YOK), iki mini koşu + bir render; her eksen ÇALIŞTIĞINI kendi sayacıyla kanıtlıyor
    (job dosyasından yankılanan bir damga yeterli sayılmadı)
  - [x] Çekirdek davranış bit-değişmez: süit **325 → 339** (337 passed + 2 skipped; +14 yeni test),
    dokuz `data/setups/*.json` bayt-aynı (sha256 önce/sonra), `ruff check/format src tests` temiz.
    `caustica report` çıktısı doğrulayıcı tarafından M10n ÖNCESİ CLI ile karşılaştırıldı:
    REPORT.md + index.html + üç PNG **sha256 aynı**; M10n öncesi yazılmış bir checkpoint HEAD ile
    kesintisiz resume ediliyor (parmak izi uyumlu)
  - [x] `import caustica` süresi ölçülür ve registry keşfi onu yüzde 10'dan fazla YAVAŞLATMAZ:
    medyan **257.5 ms → 258.8 ms** (15 taze alt-süreç; çıplak yorumlayıcı düşülünce 225.8 → 227.7 ms,
    **+%0.8**). Bağımsız ölçüm (doğrulayıcı, `git worktree` ile M10n öncesi commit'e karşı,
    13 serpiştirilmiş alt-süreç): **−%2.2** (HEAD daha hızlı) — yani gürültü seviyesinde.
    Tarama lazy — `test_import_caustica_does_not_scan_entry_points` beş registry'yi de sınıyor,
    doğrulayıcı `entry_points()` çağrısını casusladı: **sıfır**. GPU'suz makinede `import caustica`
    cupy'ye DOKUNMUYOR (yola sahte bir `cupy` konup import edilmediği pozitif kontrolle gösterildi)
  - [x] Kayıtlı olmayan kind/backend adı, kayıtlı adları listeleyen aksiyonlu hata verir —
    `test_unregistered_names_are_actionable_on_every_axis` (beş eksen), backend adı için ayrıca
    job seviyesinde: `test_a_job_naming_an_unregistered_backend_is_a_config_error`; soğuk
    import'ta kind registry'leri de doğru cevap veriyor (`test_the_kind_registries_answer_from_a_cold_import`)
- [x] Hafif review turu: bir mercek (adversaryal) + şüpheci doğrulayıcı. Yedi kapanış iddiasının
      hepsi doğrulandı; **beş gerçek bulgu** onarıldı (CLI açılışı iki katına çıkmıştı; backend
      fabrikası registry anahtarına bağlanmıyordu — koşuyu ve checkpoint parmak izini yanlış
      etiketleyebilirdi; lambda'lar çakışma kontrolünü delip geçiyordu; solver çakışma metni bir
      kelime kaybetmişti; `--backend` yazım hatası artık medium kurulmadan reddediliyor) +
      doğrulayıcının beş boşluğu. Ayrıntı: devlog 2026-08-22 oturum M10n
- BİLİNEN SINIR (belgelendi, düzeltilmedi — kapsam dışı): runner `backend=`/`checkpoint=` kwarg'ını
  yalnız `_NATIVE_SOLVERS`'a geçiriyor (T3, kwave sözleşmesi), yani üçüncü taraf bir ÇÖZÜCÜ
  varsayılan backend'de çözer ve checkpoint almaz; slow-CPU kapısı ve GPU raporu da `"numpy"` /
  `"cupy"` isimlerine dallanıyor, üçüncü taraf bir backend ikisini de atlar. `docs/extending.md`
  ikisini de yazıyor; yetenek sorusuna çevirmek ayrı bir iş

### M10j — Notebook ergonomisi: facade + ilerleme `[x]` (2026-08-22)
M10k'dan SONRA geldi (o kapandı; `stored_setup` yokken tek kind var, facade şemaya bir kez dokundu).
- [x] `caustica.simulate(...)` (`src/caustica/facade.py`) — KAPALI girdi listesi: job yolu · job
      dict'i · `ExplicitJobConfig` · `BuiltJob`. Dördü de AYNI `build_job`'dan geçer, ikinci
      kurulum yolu yok. Desteklenmeyen tip (çıplak `Grid`/`Medium`/`CWSource`) dördünü sayan ve
      nesne API'sini gösteren TypeError verir — `test_an_unsupported_setup_names_the_accepted_forms`,
      `test_all_four_setup_forms_produce_the_same_run` (beş yazım, tek tepe değeri).
      **Plan çelişkisi kayda geçti:** PLAN.md §4 ve W3 "kurulmuş nesneler"i girdi sayıyor, ama
      W3 kısıt-1 her girdinin `ExplicitJobConfig`'e normalize olmasını şart koşuyor; voxelize
      edilmiş bir `CWSource` job'a geri çevrilemez. `BuiltJob` (= `build_job`'ın DÖNDÜRDÜĞÜ
      kurulmuş nesneler, job'ı üstünde) okuması alındı; çıplak nesneler L1'e yönlendiriliyor
- [x] `out=None` diske hiçbir şey yazmaz AMA planner konuşur, düşük-ppw uyarısı çıkar ve M10i
      kapılarının İKİSİ de uygulanır. Bunu mümkün kılan tek şey kapıların `runner.check_gates()`
      + `Refusal`'a çıkarılması — bellek-içi koşunun daha gevşek kapıları olsaydı, plan-önce
      disiplininin var olma sebebi olan "laptopta çalıştı, Colab'da öldü" hatası geri gelirdi
- [x] `out=<yol>` `run_job_file`'a delege eder; çıktı/damga/resume mantığı ÇOĞALTILMADI. Job
      dosyası olan girdide dosyanın KENDİSİ runner'a verilir (T4: içindeki göreli yollar job
      dosyasına göre çözülür); dosyası olmayan girdide yollar önce CWD'ye sabitlenir, sonra
      geçici bir job dosyası yazılır — `test_a_job_files_relative_paths_still_resolve_against_the_job_file`
- [x] `SimulationRun` `SolverResult`'ı SARAR: `.metrics` `report.metrics` (klasör varsa onun
      `metrics.json`'ından okur — nesne ile klasör asla farklı sayı söyleyemez), `.preview()`
      aynı ≤10 MB paket bellekte (`build_preview`/`decode_preview` ayrımı), `.save()` var olan
      `result.h5`'i KOPYALAR (ikinci kez kuantize etmez) — `test_out_path_produces_the_full_runner_folder`
- [x] `progress=` kancası: `_period_boundary` artık checkpoint'ten BAĞIMSIZ (T1), payload
      linear+westervelt+engine üçlüsünde (T2), kwave adaptörüne GEÇMİYOR (T3). `_Heartbeat`
      payload'un TÜKETİCİSİ (`__call__`); `stop_when` artık tick etmiyor. Koşu-içi önizleme
      VARSAYILAN AÇIK (8 periyotta bir, odaktan geçen kaba kesit; `snapshot` TEMBEL bir
      erişimci olduğu için isteyen periyotta TEK device→host kopya, istemeyende sıfır).
      Callback hatası bir kez uyarır, koşu devam eder
- [x] Sunum çözücünün DIŞINDA (`src/caustica/progress.py`): tqdm varsa ve izleyen bir şey varsa
      (tty ya da notebook çekirdeği) bar, yoksa düz periyodik satır — tqdm runtime bağımlılığı
      DEĞİL. Hepsi stderr'e; stdout'un ayrıştırılabilir sözleşmesi (plan metni, sonuç yolu)
      korunur. CLI aynı payload'dan basar, `--no-progress` susturur
- Başarı kriterleri:
  - [x] `simulate(job_dict)` ile `caustica run job.json` BİT-AYNI phasor/p_max üretir (paketli
        `water_bowl_mini` örneği, `output.quantize=false` — karşılaştırılan çözüm, kodlayıcı
        değil) — `test_simulate_dict_matches_caustica_run_bit_for_bit`
  - [x] Checkpoint'SİZ mini koşu periyot başına TAM BİR kez callback çağırır; settle→record
        geçişi stage değişimi olarak görünür —
        `test_progress_fires_once_per_period_without_a_checkpoint`
  - [x] `progress=` verilen kwave işi ÇÖKMEZ (`test_kwave_job_with_progress_set_does_not_crash`,
        adapter `progress` görürse TypeError atıyor); hata fırlatan callback koşuyu düşürmez ve
        BİR KEZ uyarır — `test_a_throwing_callback_warns_once_and_the_run_completes`
  - [x] `out=None` diske hiçbir şey yazmaz (tmp CWD'de `rglob` ile önce/sonra karşılaştırma) ama
        planner + kapılar çalışır — `test_out_none_writes_absolutely_nothing`,
        `test_out_none_still_applies_the_vram_gate` (çıkış 3),
        `test_out_none_still_applies_the_cpu_time_gate` (çıkış 2),
        `test_allow_slow_cpu_is_the_documented_escape`
  - [x] Kanca sayısal sonucu DEĞİŞTİRMİYOR (testli): phasor ve p_max bit-aynı —
        `test_progress_does_not_change_the_field`, `test_progress_does_not_change_the_runners_result`
  - [~] Kanca hızı değiştirmiyor: **ölçüm var, CI kapısı YOK** (zamanlama testi kasten
        eklenmedi — paylaşılan koşucuda gürültülü olur ve yanlış kırmızı üretir). Ölçümler:
        72×72×96 (336 adım, 40 periyot) kanca kapalı 8.037 s → ConsoleProgress + ASCII önizleme
        8.079 s = **+%0.51** (3 koşunun en iyisi; medyan +%0.93), çıplak callback **+%0.23**.
        Bağımsız doğrulayıcı kendi iki kurulumunda: 64×64×88 **+%0.10**, 32×32×48/50 periyot
        (en kötü hâl: ucuz adım, çok callback) **+%0.89** — hepsi %5 kapısının çok altında
  - [x] status.json alanları ve run_meta anahtarları M10j ÖNCESİYLE aynı. `b675267` (M10j öncesi)
        worktree'sinde ve HEAD'de AYNI iş, BEŞ yolda: taze · settle-checkpoint'inden resume ·
        record-checkpoint'inden resume · `max_hours=0` · `t_end_min_us` tabanı. Beşinde de çıkış
        kodları, status.json anahtar kümesi ve HER değeri (`periods_done`, `steps_done`,
        `steps_expected`, `steps_worst`, `state`, `detail`) ve run_meta anahtar kümesi aynı;
        `preview.npz`, `plan.json`, `job.json` BAYT-aynı; `result.h5`'in tüm dataset'leri
        bit-aynı (tek fark `git_commit` ATTR'ı — dosya bütünü zaten bayt-aynı OLAMAZ).
        Sayacın kendisi ayrıca testle çivili: `test_the_heartbeat_counts_exactly_one_period_per_payload`,
        `test_runner_status_json_matches_the_pre_m10j_contract` (mutasyon `periods += 2` → KIRMIZI)
  - [x] Önizleme kadansının KENDİSİ (8 periyot, D21) testli:
        `test_the_mid_run_preview_fires_every_eight_periods` — 8→1 ve 8→100000 mutasyonlarının
        ikisi de KIRMIZI (gözden geçirmede ikisi de YEŞİLDİ: record-flip önizlemesi kadansı
        maskeliyordu)
  - [x] Facade'ın `progress="auto"` varsayılanı testli — `test_the_facade_default_is_progress_on`
  - [x] Süit 339 → 379 (377 passed / 2 skipped / 0 failed); `data/setups/` on dosya bayt-aynı
  - [x] Gözden geçirme turu: altı gerçek bulgu onarıldı ve her biri KIRMIZIYA düşen bir
        mutasyonla çivilendi — `BuiltJob`'ın `base_dir`'i (T4 ihlali),
        bellek-içi planda eksik `ppw_warnings`, iki çıktı kipinin hatayı FARKLI sınıflaması,
        medium kurulduktan SONRA yakalanan backend yazım hatası, `options=` ile verilen
        `out`'un sessizce düşürülmesi, klasörsüz anlamsız seçeneklerin sessizce yok sayılması

### M10l — GUI sözleşmesinin dondurulması `[x]` (2026-08-22) — GUI kodu YOK
GUI ayrı repoda olacak ve teknolojisi seçilmedi (PLAN.md K13). Bu milestone yalnızca GUI'nin
üstüne oturacağı yüzeyi yazıya döker ve katmanlamayı testle kilitler.
- [x] `docs/gui_contract.md`: `caustica-job/1` + `validate` + `caustica schema`, runner çıkış
      kodları (0/2/3/4/5), `status.json` alanları, `error.json` şeması, `cancel` protokolü,
      `--dry-run`/`plan.json`, `caustica-preview/1`, `caustica-result/1`, `metrics.json`,
      `run_meta.json`, `env_report()`, ilerleme payload'u (dokuz serileştirilebilir anahtar +
      `snapshot`). Başta "Listelenmeyen hiçbir şey sözleşme değildir" + girdi=TEK job dosyası /
      çıktı=TANIMLI klasör; sonda "sözleşme OLMAYANLAR" bölümü (modül düzeni, stdout nesri,
      `checkpoint.npz` içi, log metinleri, herhangi bir IPC)
- [x] Alan listeleri elle kopyalanıp çürümeye bırakılmadı: `tests/test_gui_contract.py` DÖRT
      gerçek koşu üretir (başarılı / kesilmiş / VRAM-reddi / store-çökmesi) ve her listeyi tarif
      ettiği şeyle karşılaştırır — klasör listesi gerçek klasörle, `status.json` alanları ÜÇ
      gerçek status'un kesişimiyle (durum-bağımlı ekler farkla türetilir), ilerleme anahtarları
      gerçek payload'la, `plan.json`/`run_meta.json`/`metrics.json` gerçek dosyalarla, çıkış
      kodları ve format etiketleri koddaki sabitlerle; sayfadaki CLI satırları gerçek
      argparse'tan geçirilir (`test_documented_*`, `test_the_cli_lines_on_the_page_actually_parse`)
- [x] `tests/test_import_direction.py`: M10k'da yazıldı, o tarihten beri YEŞİL — bu milestone
      onu yalnızca kanıtla işaretler: `test_caustica_never_imports_upward` (AST ile her modül;
      `apps`/`uwcem_phantoms`/`caustica_gui*` YASAK) + `test_no_uwcem_reference_survives_in_source_text`
- [x] **İPTAL SİNYALİ**: `stop_when` kancasına dosya yoklaması eklendi — `cancel` görülünce
      checkpoint yazılır ve çıkış 5. Dosya TÜKETİLİR (bırakılsa `--resume` ilk periyot sınırında
      kendini iptal ederdi) ve öldürülmüş bir süreçten kalan bayat `cancel` bir sonraki denemenin
      başında temizlenir. Kabul edilen sonuç, sayfada da yazılı: `cancel` KOŞAN bir işe sinyaldir,
      ön-iptal aracı değildir
- [x] **YAPILANDIRILMIŞ HATA**: `error.json` = `{format, stage, exit_code, error_class, message,
      advice[], written_at}`; altı `stage` — config/plan/gate/checkpoint/solve/store. Planner'ın
      `est.advice`'i artık ekrana VE dosyaya gider (`Refusal` advice'i liste olarak tutuyor, iki
      kopya yok). Kesinti (çıkış 5) error.json YAZMAZ: durmak başarısızlık değildir
- Başarı kriterleri:
  - [x] Import-yönü testi yeşil — `test_caustica_never_imports_upward` (M10k'dan beri)
  - [x] Hiçbir GUI bağımlılığı veya `gui` extra'sı eklenmemiş; yeni runtime bağımlılığı yok —
        `test_freezing_the_contract_added_no_gui_dependency` (`grep -i gui pyproject.toml` boş)
  - [x] Koşan bir işe `cancel` atılınca: periyot sınırında durur (`periods_done == 3`, mid-periyot
        değil), checkpoint bırakır, `result.h5` yazmaz, çıkış 5; `--resume` ile tamamlanır ve
        phasor/p_max/steps_total kesintisiz koşuyla BİT-AYNI —
        `test_cancel_file_stops_at_period_boundary_and_resume_is_bitwise_identical`
  - [x] Yoklama adım maliyetine SIZMADI: periyot sınırı başına bir stat (`polls <= boundaries+1`
        ve `polls*spp <= steps+spp`) — `test_cancel_poll_is_one_stat_per_period_never_per_step`;
        bayat `cancel` bir sonraki koşuyu iptal etmez — `test_a_stale_cancel_file_does_not_cancel_the_next_run`
  - [x] Yedi hata sınıfının her biri şemaya uyan `error.json` üretir — on senaryo, yedi ayrı
        `error_class` (ValidationError · JobError · JSONDecodeError · ValueError · VramRefusal ·
        CpuTimeRefusal · CheckpointConflict) ve altı `stage`'in hepsi:
        `test_every_failure_class_writes_a_conformant_error_json` (parametrik) +
        `test_the_error_table_covers_at_least_seven_distinct_classes`. Hiçbir GUI yolu stderr
        ayrıştırmak zorunda değil: `advice[]` gerçek komutları adlandırıyor
        (`test_config_error_advice_points_at_commands_the_page_documents`)
  - [x] Başarılı koşu error.json ÜRETMEZ ve bayat dosyayı SİLER —
        `test_a_successful_run_writes_no_error_json_and_clears_a_stale_one`; job hiç
        ayrıştırılamasa bile `--out` verildiyse dosya düşer (GUI'nin hali) —
        `test_error_json_lands_even_when_the_job_never_parsed`
  - [x] Mevcut hata sözleşmesi DEĞİŞMEDİ: çıkış kodları, stderr metinleri, `status.json` alanları
        ve `run_meta`/checkpoint parmak izi aynı; error.json yazılamazsa koşu AYNI kodla ve AYNI
        mesajla biter, yalnızca bir uyarı loglanır —
        `test_a_write_failure_for_error_json_changes_nothing`
  - [x] kwave cancel'ı desteklemiyor ve bunu dürüstçe SÖYLÜYOR (checkpoint yok → durulacak sınır
        yok; öldürmek dosyanın var oluş sebebinin tersi) —
        `test_a_non_native_solver_says_cancel_does_nothing`
  - [x] `--dry-run` bir PROBE'dur: `error.json`'a da `cancel`'a da hiçbir yönde dokunmaz (review
        bulgusu — eskiden gerçek bir koşunun hata kaydını SİLİYORDU) —
        `test_dry_run_never_touches_the_failure_record_or_the_cancel_file`,
        `test_dry_run_of_a_broken_job_writes_no_error_json`; skip-guard yalnız `error.json`
        temizler, `cancel` başka bir sürecin olabilir —
        `test_the_skip_guard_clears_the_stale_error_but_not_a_cancel`; dokümante edilen dry-run
        çıkış kodları gerçek — `test_the_documented_dry_run_exit_codes_are_the_real_ones`
  - [x] Sayfanın adlandırdığı her `caustica.AD` gerçekten var (review turunda uydurma bir istisna
        adı bulundu) — `test_every_caustica_name_on_the_page_actually_exists`, mutasyonla doğrulandı
  - [x] `cancel` bir DİZİN ise kilitlenme yok (yoklama `is_file`) —
        `test_a_cancel_directory_cannot_livelock_the_folder`
  - [x] Gözden geçirme turu: iki mercek, altı gerçek bulgu + altı abartı onarıldı, yedi şüphe
        çürütüldü (docs/devlog.md 2026-08-22). "Mevcut hata sözleşmesi değişmedi" iddiası
        ÖLÇÜLEREK kapandı: `runner.py` geçici olarak `96e6330`'unkiyle değiştirilip yedi hata yolu
        iki kez koşturuldu → stdout/stderr bayt-aynı (geçici klasör adları ve `warnings.warn`
        satır numarası hariç)
  - [x] Süit 379 → 417 (415 passed / 2 skipped / 0 failed); `ruff check .` + `ruff format --check`
        temiz; `git status --porcelain data/setups/` boş — dokuz kurulum dosyası + manifest bayt-aynı

### M10f — Colab köprüsü: `caustica.colab` + değişmeyen notebook `[~]` (kod 2026-08-22; Colab kapıları canlı oturum bekliyor)
"Değişmeyen dosya" şartı mantığın notebook'ta DEĞİL repoda yaşamasıyla sağlanır: notebook 5
hücre, tek düzenlenen satır CONFIG yolu; gerisi `from caustica.colab import run_job`. Güncelleme
`pip install -U` ile gelir, .ipynb'ye dokunulmaz.
- [x] `caustica.colab`: ortam kontrolü (`env_report()` basılır + GPU ZORUNLU), koşu, çıktı
      `/content` altında. İki ayrı ret mesajı (K6): "cupy kurulu değil" ile "runtime'da CUDA
      cihazı yok" AYRI cümleler, çünkü çözümleri ayrı — ikincisi `caustica.env.require_gpu`'nun
      kendi mesajı, burada tekrar YAZILMIYOR. **"Hiçbir şey hazırlanmadan" GPU kapısı için
      geçerli** (indirme yok, klasör yok, medium yok — ayrıca hatalı bir keyword de indirmeye mal
      olmuyor; testli). VRAM kapısı için DEĞİL: o kapı runner'da, plan-first ama `build_job`
      medium'u kurduktan ve çıktı klasörü açıldıktan SONRA. Bu runner'ın öteden beri davranışı;
      köprünün belgesinde aksini iddia etmek abartı olurdu (W6 planındaki eski cümle de
      düzeltildi)
- [x] VRAM kapısı köprüde TEKRARLANMADI: tek kopya `runner.check_gates`'te, plan-first ve BOŞ
      VRAM'e karşı. Köprünün eklediği şey runner'ın bilerek yapmadığı kontrol — `auto`
      backend'in GPU'suz makinede sessizce numpy'a düşmesi Colab'da saatlerce CPU koşusu demek
- [x] Dataset staging KALDIRILDI (PLAN.md K3/K7/K9): kütüphanede anatomik veri yolu yok. Colab
      kullanıcısı ya paketli sentetik örneği ya kendi `medium_volume` dosyasını getirir; UWCEM
      fantomu isteyen `uwcem-phantom` repo'sunu kullanır
- [x] Drive KALDIRILDI (PLAN.md K12): `caustica.colab` Drive mount ETMEZ, Drive yollarını bilmez,
      Drive'a özgü yeniden deneme mantığı taşımaz. Kalıcılık isteyen kullanıcı Drive'ı KENDİ mount
      edip `out=` ile oraya yazdırır (runner bunu zaten destekler). Testin KANITLADIĞI şey
      dar: açık bir `out=` her zaman kazanıyor, verilen klasöre yazılıyor
      (`test_an_explicit_out_wins_including_a_folder_the_user_mounted` — geçici bir dizin, mount
      DEĞİL). Gerçek bir Drive mount'u üstünde koşu ilk Colab oturumunun işi. Kabul edilen risk:
      oturum çökerse `/content` gider — checkpoint oturum-içi restart'ı kurtarır, VM teardown'ı
      kurtarmaz
- [~] `notebooks/colab_run.ipynb` repoda ve README'de "Run on Colab" rozeti var; ama link
      HENÜZ ÇALIŞMIYOR ve bu M10e'ye bağlı sert bir engel. `origin/master` hâlâ f0bff2f'te
      (yeniden adlandırma ÖNCESİ): `notebooks/` dizini orada yok, `src/caustica` da yok. Yani
      `master`'a bakan ÜÇ şey birden bugün kırık: (1) rozetin
      `…/blob/master/notebooks/colab_run.ipynb` hedefi 404; (2) notebook'un kurulum hücresi
      `git+https://github.com/ebx0/caustica` ile VARSAYILAN dalı kuruyor, o dal hâlâ `hifusim`
      paketini taşıyor, dolayısıyla `from caustica.colab import run_job` ModuleNotFoundError
      verirdi; (3) varsayılan `CONFIG` ham URL'si 404. Üçü de `library-first` master'a
      MERGE edilince düzelir — "push" yetmez. Colab kapısı bu yüzden denenemez bile
      (aşağıdaki iki açık ölçütün ön koşulu)
- Başarı kriterleri:
  - [x] Notebook sözleşmesi: CONFIG satırı dışında düzenleme gerektirmez; mantık değişikliği
        notebook diff'i SIFIR olacak şekilde repodan gelir. Kontrat testi hücre içeriklerini
        sabit şablonla BAYT BAYT karşılaştırıyor:
        `tests/test_colab.py::test_notebook_cells_match_the_frozen_template` (+ çıktı/execution
        count yok, tek literal atama CONFIG, notebook'ta `run_job`/`show` dışında çağrı yok).
        **İnceleme turu (2026-08-22) kilidin DELİKLERİNİ buldu ve hepsi kapatıldı:** şablon
        karşılaştırması gerçekti (16 mutasyondan 6'sı yakalanıyordu) ama şablonu "bilerek"
        güncelleyen biri için 9 delik açıktı — kurulum hücresi hiç AST'den geçmiyordu (`!`
        satırı yüzünden hücre TÜMÜYLE atlanıyordu; oraya `import os` + gate'i kapatan bir env
        var + döngü gizlenebiliyordu), attribute çağrısı / lambda / ternary / comprehension
        "mantık" sayılmıyordu, çağrı ARGÜMANLARI hiç bakılmıyordu, notebook ve hücre
        METADATA'sı tamamen serbestti (`accelerator: GPU` silinebiliyor, `cellView: form` ile
        hücre gizlenebiliyordu). Şimdi: magic satırları atlanmıyor BOŞALTILIYOR, mantık düğüm
        listesi genişledi, argümanlar isim olmak zorunda, tek `!` satırı var ve `!pip install`
        ile başlıyor, notebook metadata'sı ve her hücrenin boş metadata'sı çivili. Aynı 9
        mutasyon yeniden koşuldu: 9/9 YAKALANIYOR, değiştirilmemiş notebook geçiyor
  - [x] `caustica.colab` içinde `google.colab`/Drive dışı hiçbir ortam varsayımı yok.
        **Ölçüt düzeltmesi (operatör, 2026-08-22):** eski cümle "`grep -ri drive src/caustica`
        boş" idi; bu LAFZEN yanlış — job şemasındaki `drive` bölümü AKUSTİK sürüştür (f0,
        amplitude) ve her yerde geçer. Niyet Google Drive'dır, kontrol şudur:
        `grep -rniE "drive\.mount|/content/drive|google\.colab" src/caustica --include=*.py`
        → Drive deseni SIFIR eşleşme; `google.colab` yalnız `sys.modules` YOKLAMASI olarak,
        asla `import` edilmeden. Ölçülen (2026-08-22): Drive 0 eşleşme, `import google` 0
        eşleşme, `google.colab` 4 satır — `colab.py` (2 satır, yalnız docstring nesri),
        `env.py:102` (`_on_colab`, M10i'den beri), `progress.py:189` (notebook renderer
        seçimi, M10j'den beri). Köprü bu TEK yoklamayı yeniden kullanıyor, ikincisini
        tanımlamıyor. Testle çivili: `test_the_library_has_no_drive_code_and_never_imports_google_colab`
  - [x] **Colab kapısı — İLK OTURUM KOŞULDU (operatör ölçümü, 2026-08-22):** repodan açılan
        notebook → `/content/runs/water_bowl_mini` → sonuç indirilip lokalde `caustica report`
        ile açıldı. Kanıt REPO'DA: `benchmarks/reports/colab_first_session_2026-08-22/`
        (run_meta + plan + metrics + status + job + REPORT.md + index.html; `result.h5` ve
        `preview.npz` girmedi — repo ikili alan verisi taşımıyor). Gerçek
        NVIDIA A100-SXM4-40GB, Python 3.13.15, cupy 14.0.1,
        koşu 2.77 s / 104 adım, periyot 11'de yakınsadı. Metrik seviyesinde parite MÜKEMMEL:
        aynı job CPU'da koşuldu, tepe basınç bağıl fark **1.8e-7**, geometri/−6dB/hacim birebir,
        yakınsama yörüngesi aynı
- **Oturumun ortaya çıkardığı ÜÇ kusur ve düzeltmeleri (2026-08-22/23):**
  - `git_commit: "unknown"` — Colab wheel'den kuruyor, wheel'de checkout yok, damga commit
    kaydedemiyordu. **fix A1:** `build_stamp.py` + `setup.py` build sırasında
    `caustica/_build_info.py` yazıyor; `caustica.env.git_commit()` önce canlı checkout'a bakıyor,
    yoksa gömülü damgaya düşüyor. İkisi de KUŞATAN repo'nun commit'ini reddediyor (`git rev-parse`
    yukarı yürür). Yeni bağımlılık YOK (setuptools_scm eklenmedi). CI wheel bacağı uçtan uca
    kanıtlıyor; `tests/test_packaging.py`e 5 test eklendi (biri temiz kurulumdan gerçek koşu)
  - `t_step_measured_s` ısınmayı gizliyordu → **fix A2** (yukarıda, M8 notunda)
  - CI Colab'ın Python'unu test etmiyordu (3.10 + 3.12 vardı, Colab 3.13.15) → **fix A3:**
    matrise ubuntu/3.13 bacağı eklendi
  - [ ] **İlk Colab oturumu üç kapıyı birden kapatır (ilk Colab oturumunda):** M7 parite + tam
        boy OOM'suz koşu, M8 VRAM ±%10 ve kalibre süre ±%25, bu E2E — runner damgası
        ölçümleri zaten topluyor (`run_meta.json` → `planner` vs `actual`;
        `caustica.colab.summary()` ikisini yan yana basıyor)

### M10g — `[M29'A BİRLEŞTİ 2026-08-22]` Kuyruk
Hiç başlanmamıştı; protokol ve kriterler AYNEN M29'a taşındı (müşterisi DatasetGenerator).

### M11 — `[TAŞINDI 2026-08-22]` → Faz Grubu D'deki yeni M11 bloğu
Yeniden planla genişletildi: doğrulama + ÇOK-MOTOR harness (M12'yi yuttu). Eski v0.1 tetiği
buradan kalktı — v0.1 artık M21 kapısında (K19).

### M12 — `[M11'E BİRLEŞTİ 2026-08-22]` k-Wave karşılaştırma harness'i
Kriterleri M11 (doğrulama + çok-motor harness) taşıyor.

### M13 — `[M29'A BİRLEŞTİ 2026-08-22]` Dataset pipeline
Kriterleri M29 (üretim vitrini) taşıyor; kullanıcı kararı: dataset vitrin, EN SONA.

### M14 — `[M29'A BİRLEŞTİ 2026-08-22]` Colab üretim doğrulaması
Kriterleri (v12 referans sayıları dahil) M29 taşıyor.

---

## Faz Grubu D — Doğrulama çatısı ve HIFU fiziği (yeniden plan 2026-08-22, K18–K21)

> Yeniden planlama: 24 kullanıcı sorusu + research/landscape_2026.md (+iki alt-rapor).
> Kimlik: **sözleşmeli çok-motor çatı + native aile** — "bir API, N motor, damgalı/tekrarlanabilir
> koşular". Landscape gerçeği: k-wave-python v0.6 (2026-03) saf CuPy çözücü yayınladı — "saf
> Python" farkımız kapandı; savunulabilir konum yazılım sözleşmesi + çok-motor çapraz doğrulama.
> HIFU-önce, doğruluk/kredibilite-önce; v0.1 = ITRUSST dokuzunun TÜMÜ (akustik-yalnız) + JOSS.

### M11 — Doğrulama + çok-motor harness `[ ]` (M12'yi yuttu — kullanıcı 2026-08-22)
Tek çatı: analitik süit + N-motor çapraz karşılaştırma + damgalı rapor. Adaptörler (M25/M26)
bu harness'in İÇİNE doğar.
- [x] `caustica.validation` paketi + `python -m caustica.validation` CLI'ı AÇILDI (2026-08-23):
      ilk süit `gpu-gates` — M7/M8'in cihaza bağlı ölçütlerini tek koşuda ölçen protokol
      (kalibrasyon → VRAM merdiveni → OOM reddi → parite → damgalı MD+JSON rapor). Damga:
      ortam/GPU/git + kalibrasyon + her basamağın plan-vs-gerçek satırı + kapı VERDICT'i.
      `Harness` dikişi sayesinde GPU dışındaki HER ŞEY CPU'da testli; sahte ölçümlerle yanlış
      PASS üretemediği gösterildi (SKIP asla PASS sayılmaz; kapı, milestone'un istediği
      SAYIDA geçen ölçüm olmadan PASS vermez)
- [ ] `study.Study`: config + koşu(lar) + sonuç + figürler; `report()` → MD+JSON; ortam/GPU/git
      damgası; `Study.sweep(...)`
- [ ] `python -m caustica.validation run-analytic` → damgalı rapor `benchmarks/reports/`a
- [ ] Çok-motor karşılaştırma: AYNI job N kayıtlı çözücüde → relL2 / r / fokal metrik tablosu.
      Eski M12 kriterleri buraya: kwave T0 sanity kapısı, normalize karşılaştırma,
      "environment-broken" damgası
- Başarı kriterleri:
  - Tek komutla analitik süit raporu; planner tahmin-vs-gerçek tablosu içinde
  - Sweep: 3-noktalı p0 taraması uçtan uca, birleşik rapor
  - Çok-motor: mevcut linear-vs-kwave çaprazları harness'ten yeniden üretilir (r>0.99)

### M18 — Termal modül: Pennes + CEM43 `[ ]` (öne çekildi — HIFU-önce, kullanıcı 2026-08-22)
Ablasyon planlamanın çıktısı basınç değil DOZ. ITRUSST güvenlik konsensüsü (Brain Stim 2025)
eşikleri rapora girer: ≤2 CEM43 beyin / ≤16 kemik / ≤21 deri; ΔT≤2°C.
- [ ] `sensors.HeatingSource` (Q = 2·α·I; harmonik katkılar) → `thermal.PennesSolver` (GPU
      difüzyon+perfüzyon) → CEM43 doz haritası; rapora doz/eşik özeti
- Başarı kriterleri:
  - Analitik nokta-kaynak/Gaussian difüzyon rel err < %2; perfüzyon kararlı-durum < %2
  - Uçtan uca sonication → T(r,t) → CEM43; k-Wave `kWaveDiffusion` çaprazı < %5
  - Tıbbi sorumluluk notu (araştırma amaçlı; klinik karar aracı değil)

### M15 — Eksenel simetri (AS) çözücüsü `[ ]` (KZK'nın önüne geçti — kullanıcı 2026-08-22)
- [ ] VERIFY: güncel CuPy'de DCT/DST; yoksa ayna-genişletme DTT katmanı; WSWA/WSWS
- Başarı kriterleri (değişmedi): AS vs 3D full-wave eksenel r > 0.995, odak < %3;
  ≥50× hızlanma (ölçülür); DTT birim testleri < 1e-10

### M16 — Power-law absorpsiyon (fractional Laplacian) `[ ]` (yerinde — kullanıcı 2026-08-22)
Kriterler değişmedi (α(f)=α0·f^y + KK dispersiyonu; flag-off bit-değişmez; k-Wave çaprazı <%5).
Not: ITRUSST PH1 tek-frekans/lineer — M21 ön koşulu DEĞİL; doku gerçekçiliği/çok-harmonik için.

### M25 — j-Wave adaptörü `[ ]` (YENİ — kullanıcı 2026-08-22, K18)
- [ ] `caustica[jwave]` extra'sı; registry'ye `jwave` (JAX; Linux/Colab). DÜRÜST risk notu
      (landscape 2026): j-Wave PyPI 23 aydır bayat, jaxdf pin'i eski — sürümler SABITLENIR,
      adaptör değeri çapraz-kontrol + gradyan erişimi
- Başarı kriterleri: kwave adaptörü kalıbı (kwarg reddi, caps); su+doku senaryosunda
  native-vs-jwave normalize r > 0.99; extra'sız kurulum etkilenmez

### M26 — Stride adaptörü `[ ]` (YENİ — kullanıcı 2026-08-22, K18)
- [ ] `caustica[stride]` (Devito; FD ailesi = güçlü çapraz). Lisans notu: Stride AGPL —
      adaptör opsiyonel extra olarak kalır, çekirdeğe AGPL bulaşmaz (import eden kullanıcıdır)
- Başarı kriterleri: M25 kalıbı; en az bir ITRUSST BM'inde üç-motor tablo (native/kwave/stride)

### M27 — Görüntü köprüsü: `caustica[imaging]` `[ ]` (YENİ; kapsam DARALTILDI — kullanıcı 2026-08-22)
NIfTI/pseudo-CT → `medium_volume`. **Entegrasyon-önce**: kendi segmentasyon/haritalama AR-GE'si
YOK — SimNIBS/k-Plan-türevi hazır çıktıları almak + 2–3 ADLANDIRILMIŞ literatür haritalaması
(Li 2026 sınıflaması: voxel-linear / üç-katman / tek-katman) uygulamak. Landscape: üretim
araçları tam bu haritalamada ayrışıyor (k-Plan vs BabelBrain) — çoğul-strateji karşılaştırma
raporu ucuz fark alanı.
- [ ] nibabel YALNIZ extra'da (lazy; çekirdek metadata temiz — K15 disiplini)
- Başarı kriterleri: örnek public verisinden uçtan uca medium_volume + koşu; N strateji → N
  medium + fark raporu; extra'sız import etkilenmez (testli)

---

## Faz Grubu E — Performans + ITRUSST → v0.1

### M19 — GPU performans turu `[ ]` (ITRUSST'tan ÖNCE — kullanıcı 2026-08-22)
Kriterler değişmedi (≥1.5× adım; parite korunur; planner rekalibre; TF32 default OFF).
Backlog başlangıcı: research/gemini3_gpu.md §6. Referans nokta (landscape): k-wave-python
native CuPy T4'te 256³/1000 adım = 382 s; C++ CUDA 51 s — ölçülüp yanına konulur.

### M21 — ITRUSST PH1: DOKUZUN TÜMÜ, akustik-yalnız `[ ]` → **v0.1 kapısı** (kullanıcı 2026-08-22)
Landscape (doğrulanmış): ekosistem BEKÇİSİZ ve self-serve — Zenodo 10.5281/zenodo.6020543
(25.3 GB, CC-BY-4.0) + ucl-bug/transcranial-ultrasound-benchmarks (LGPL; 2022'den beri donmuş).
Yol: 18 permütasyon (9 BM × bowl/piston, 500 kHz, LİNEER) → sonuçlar onların .mat düzeninde →
compareTwo/processAll → KENDİ karşılaştırma makalen (şablon: Drainville/CIVA, JASA 2025).
Kafataslı BM'ler AKUSTİK-YALNIZ (akışkan kemik — katılımcı normu); elastik v0.2 (M22).
- [ ] 18 permütasyon `caustica run` akışıyla; .mat dönüştürücü (onların adlandırması)
- Başarı kriterleri (2022 makalesinin GERÇEK koridorları; eski "0.2–0.6 mm −6dB" satırı
  düzeltildi — ana metinde yok, doğrulanamadı):
  - Fokal basınç: kod-medyanlarına fark < %10; fokal konum < 1 mm (tümü < 2.5 mm)
  - FWHM: eksenel medyan < 0.6 mm, lateral < 0.2 mm bandı
  - BM1–2 (su): FOCUS analitiğine L∞ < %1 hedefi
  - Tam-alan L∞/L2 DÜRÜSTÇE raporlanır (ekosistemde %10–100 bandı — saklanmaz)
  - `benchmarks/reports/itrusst/` + karşılaştırma makalesi taslağı
- **v0.1 (M21 kapanınca):** tag + CITATION.cff + Zenodo DOI + **JOSS başvurusu** + PyPI (K2).

---

## Faz Grubu F — İkinci faz (v0.1 sonrası)

### M22 — Viskoelastik/kemik (kayma dalgaları) `[ ]` — v0.2 çekirdeği
Kriterler değişmedi; M27 köprüsü + ITRUSST altyapısıyla BM7–9'un elastik tekrarına bağlanır.

### M17 — Broadband zaman alanı `[ ]` (buraya — kullanıcı 2026-08-22)
Kriterler değişmedi (puls/tone-burst, TOF < dt, CW limiti, pencereli kayıt).

### M28 — Adjoint + ML köprüsü fizibilitesi `[ ]` (YENİ — kullanıcı 2026-08-22; ikinci fazda KALDI)
Landscape: türevlenebilir NONLİNEER akustik + tam-çözücüden faz optimizasyonu İKİ BOŞ HÜCRE
(j-Wave lineer+uykuda; herkes ray/surrogate'te). Kullanıcı kararı: v0.1 odağı bölünmez; bu
niş v0.2 adayı olarak burada bekler.
- [ ] Rapor + prototip: CW elle-adjoint; JAX backend'i plugin ekseninden (K15); M25 üzerinden
      gradyan erişimi
- Başarı kriteri: "yapılır/yapılmaz çünkü" kararı + tek prototip (faz optimizasyonu YA DA
  basit aberasyon düzeltme)

### M20 — Çoklu GPU / çok büyük problemler `[ ]` (fizibilite-önce — kullanıcı 2026-08-22)
Önce fizibilite raporu (cuFFT-Mp/NCCL vs out-of-core); negatif sonuç geçerli çıktı.

### M23 — Tedavi planlama araçları `[ ]`
Kriterler değişmedi; M28 sonucuna bağlanır (time-reversal + kısıtlı optimizasyon).

---

## Faz Grubu G — Vitrin (en son — kullanıcı 2026-08-22: "kütüphane asıl ürün, dataset sonlara")

### M29 — Üretim vitrini: kuyruk + dataset + Colab üretimi `[ ]` (M10g+M13+M14 birleşti)
- [ ] Kuyruk (eski M10g protokolü AYNEN: pending→claim→running→done/failed, ölü-oturum devri,
      claim yarışı GERÇEK subprocess testli; çıkış kodları API) — müşterisi DatasetGenerator
- [ ] `pipelines.DatasetGenerator`: dondurulmuş LHS (seed→checksum), background save, ETA
      (planner), metadata/timing CSV, disk kontrolü
- [ ] Colab tam-boy üretim (eski M14: amp/p_max [0.85,0.95], t_end>110µs, cadence ≤ v12 ~65 s)
- [ ] Learned-Green's dataseti vitrine (landscape: pip'le kurulur ML-surrogate kütüphanesi NİŞİ
      BOŞ; OpenBreastUS analojisi; lisans bilinçli — TFUScapes'in NC tuzağına düşülmez)

### M24 — Backlog: UQ, bulut orkestrasyonu, kalanlar `[ ]`
Her biri fizibilite+prototip raporu; v2 seçimi kullanıcının.

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
- **Yönetim modeli (2026-08-22, kullanıcı):** operatör = Fable 5 (bu oturum), kod = Opus 5
  alt-ajanları. Proje yönetim sistemi = bu dosya (MILESTONES.md); ek araç/MCP gerekmiyor.
  Önemli kararlar kullanıcıya sorulur; gerisi operatörde.
- **Durum (2026-08-22):** M10h ✅ (CI dahil) + M10i ✅ + M10k ✅ + M10m ✅ — kütüphane UWCEM'siz,
  325 test yeşil (M10m öncesi 279); taşınan süit `../uwcem-phantom`'da 165+ yeşil
  (yerel git, push yok — kullanıcı kararı).
  UWCEM'e dair her şey artık TEK dosyada: **docs/uwcem.md** — kalanları EN SON yapılacak.
  Yalınlaştırma turu #1 (2026-08-22): kök mtype.txt (123 MB) + labels.npz + _code_cells.py +
  kaynak notebook silindi (kullanıcı onayı; M14 notu güncellendi); bayat `build/` silindi.
- **Şimdi (sıra — yeniden plan 2026-08-22):** (0) **CANLI COLAB OTURUMU** (ilk iş, kullanıcı:
  "plan biter bitmez"): M7 parite + M8 planner kapıları + M10f E2E tek oturumda; runbook
  operatörden. → (1) **M11** doğrulama+çok-motor harness → (2) **M18** termal → (3) **M15** AS →
  (4) **M16** power-law → (5) **M25** jwave + **M26** stride adaptörleri → (6) **M27** [imaging]
  köprüsü → (7) **M19** GPU perf → (8) **M21 ITRUSST dokuzun tümü → v0.1 + JOSS + PyPI** →
  İkinci faz (M22 elastik, M17 broadband, M28 adjoint fizibilite, M20, M23) → Vitrin (M29, M24).
  Araştırma temeli: research/landscape_2026.md + iki alt-rapor (2026-08-22).
- **Yalınlaştırma (kalan küçük işler; uygun milestone'a iliştirilir):** `data/` kökünün checkout
  dışına taşınması (env var kurulu, acele yok) · `janitor/` defterinin işlenmesi.
  (README'nin yeni kind'larla güncellenmesi M10m'de yapıldı.)
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
