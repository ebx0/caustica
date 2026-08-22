# UWCEM — tek dosya (durum + kalanlar)

> Kullanıcı kararı (2026-08-22): **UWCEM'e dair her şey bu dosyada toplanır ve kalan işleri
> EN SON yapılır.** caustica planlarında UWCEM artık yalnızca buraya işaret eder.

## Durum: ayrışım TAMAMLANDI (2026-08-22)

M10k (W0a–W0f) kapandı; kanıtlar `MILESTONES.md` M10k bloğunda ve devlog'da. Özet:

- **Kütüphane UWCEM'siz**: `grep -ri uwcem src/` = 0, testle sürekli zorlanıyor
  (`test_no_uwcem_reference_survives_in_source_text` + `tests/test_import_direction.py` AST).
- **Genel kapı**: `caustica.io.medium_volume` — okuyucu + yazıcı (`write_medium_volume`),
  `medium_volume` job kind'ı. Mevcut 560×700×480 dataset dosyasında PhantomAsset yoluyla
  **sha256 bit-aynı** Medium; round-trip bit-aynı. Eski dosya etiketleri
  (`caustica-phantom/1`, `hifusim-phantom/1`) okunmaya devam eder.
- **Doku değerleri**: literatür çapaları `caustica.materials.TISSUE_LIBRARY`'de; UWCEM
  media-numarası eşlemesi yeni repoda kaldı ve uçları kütüphaneden `is`-aynı nesne olarak alır.
- **Dokuz yerel setup**: `load_setup` → `setup_to_job` → caustica — medya sha256 bit-aynı 9/9,
  `validate` ok 9/9, `run --dry-run` exit 0 9/9. Hiçbir dosya yeniden indirilmedi/üretilmedi.

## Nerede ne var

| Şey | Yer |
|---|---|
| Yeni repo (yerel git, **push YOK** — kullanıcı kararı 2026-08-22: şimdilik yerel) | `C:\Users\bulbu\Desktop\uwcem-phantom` |
| İçeriği | `uwcem_phantoms/` paketi, `apps/phantom_launcher.py`, `apps/phantom_studio`, `phantoms.bat/.sh`, test süiti (165+ non-slow yeşil + dokuzlu yavaş kapı), README (şart kaydıyla) |
| Veri kökü (env değişkeni kurulu) | `CAUSTICA_PHANTOM_DATA` → `C:\Users\bulbu\Desktop\hifusim\data` (4.5 GB dataset + setups + cache/exports/uwcem — git dışı) |
| Veri kökü çözüm sırası | açık argüman → env → dolu checkout `_data` → `%LOCALAPPDATA%\caustica\phantoms` / `~/.cache/...` |
| `load_breast_phantom` (mtype 310×355×253 sabit importu) | yeni repoda `legacy_import.py` |
| Kök `mtype.txt` + türevleri | **silindi** (2026-08-22, kullanıcı onayı; çözülmüş kopyalar `data/cache`'te, ham dosya UWCEM'den yeniden indirilebilir) |

## Lisans/şart kaydı (W0f, 2026-08-22)

UWCEM'in resmi lisansı YOK. Instruction Manual şartı (verbatim yeni repo README'sinde):
*"free of charge … reference the online repository and acknowledge the authors … in any
publication that is derived."* Muhafazakâr uygulama: git'te hiçbir fantom baytı yok, türev
exportlar dağıtılmıyor, atıf metni her export metadata'sında (`catalog.CITATION`).
Repo kod-only olduğundan public olabilir — karar kullanıcının.

## Kalan işler (EN SON; hiçbiri caustica yol haritasını bloke etmez)

1. **GitHub push** — kullanıcı istediğinde: önce private (kullanıcı boş repo açar, biz
   pushlarız). Tek risk şimdilik: **tek kopya kullanıcının diskinde** (kabul edildi).
2. **Public çıkış öncesi kontrol**: README atıf bölümü son okuma, CI (caustica'ya git+https
   bağımlılığıyla), sürüm etiketi.
3. **Plugin uyumu**: caustica kind registry'leri (M10m/M10n) inince `uwcem-phantom` kendi
   medium kind'ını entry-point ile sunabilir — isteğe bağlı iyileştirme, zorunlu değil.
4. **Bakım kuralı**: `PhantomAsset.save` caustica'nın `write_medium_volume`'una yazar (D28,
   tek kaynak). caustica'nın job şeması değişirse `setup_to_job()` uyumu buradan doğrulanır
   (dokuzlu kapı testi yeni repoda).

## caustica tarafında UWCEM'e dokunan tek bağ

Yön tek taraflı: `uwcem-phantom` → caustica. caustica bu repoyu tanımaz; sınır
`tests/test_import_direction.py` ile kilitli. M14'ün "notebook v12 ile karşılaştırma"
referansı için not: kaynak notebook kökten silindi — karşılaştırma hedefi devlog'daki
yayımlanmış v12 sayılarıdır (amp/p_max bandı, cadence ~65 s/sample A100).
