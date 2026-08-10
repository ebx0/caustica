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
