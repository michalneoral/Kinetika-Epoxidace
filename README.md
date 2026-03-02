<p align="center">
  <img src="icon.png" width="120" alt="FAME EPO Manažer">
</p>

# FAME EPO Manažer

Webové GUI (NiceGUI/Quasar) pro práci s experimentálními daty a jejich kinetickým vyhodnocením (FAME/EPO). Aplikace umožňuje nahrát data z Excelu, vybrat relevantní oblasti tabulek, spouštět připravené i konfigurovatelné (grafové) výpočty a prohlížet výsledky v grafech.

**Autoři:** **Michal Neoral** a **Kristýna Dvořáková**

▸ **Návod na vytvoření instalátoru (Windows):** viz **[README_BUILD_WINDOWS.md](README_BUILD_WINDOWS.md)**

---

## ▣ Obsah

- [▣ Jak aplikaci používat](#-jak-aplikaci-používat)
- [▣ Základní workflow](#-základní-workflow)
- [▣ Záložky experimentu](#-záložky-experimentu)
- [▣ Grafy a export](#-grafy-a-export)
- [▣ Import/Export experimentů](#-importexport-experimentů)
- [▣ Aktualizace aplikace](#-aktualizace-aplikace)
- [▣ Tipy a řešení problémů](#-tipy-a-řešení-problémů)

---

## ▣ Jak aplikaci používat

### ✓ Spuštění
- Po spuštění se aplikace otevře v prohlížeči (lokálně na `http://127.0.0.1:<port>`).
- Aplikace může běžet na pozadí i po zavření tabu v prohlížeči.
- Pro snadné ovládání běhu je k dispozici **ikona v system tray** (Windows).

### ✓ Kde se ukládají data
- Data aplikace (DB, cache, exporty, port soubor, update log) se ukládají do složky určené přes **platformdirs**.
- Cesty najdeš v **Nastavení → O aplikaci**.

---

## ▣ Základní workflow

1) **Home (správa experimentů)**  
   ▸ vytvoř / otevři experiment, případně duplikuj nebo importuj.

2) **Načtení dat**  
   ▸ nahraj Excel, vyber **sheet** a vyznač oblasti tabulek (FAME/EPO apod.).

3) **Zpracování / Tabulky**  
   ▸ z výběrů se vytvoří „processed tables“ (cache podle hashe).

4) **Rychlosti**  
   ▸ spusť výpočty (prebuilt modely i custom ODE), zkontroluj fit a diagnostiku.

5) **Grafy**  
   ▸ nastav vizualizaci (barvy, osy, mřížky, clip_on) a exportuj obrázky.

---

## ▣ Záložky experimentu

### ■ Načtení dat
- **Upload Excelu** (uloží se do DB).
- Výběr **sheetu**.
- Výběr oblastí tabulek (řádky/sloupce) pro FAME/EPO.

### ■ Zpracování / Tabulky
- Sestavení „processed tables“ z vybraných oblastí.
- Cache se invaliduje při změně vstupu.

### ■ Rychlosti
- Spuštění hlavní výpočetní pipeline:
  - `ProcessingConfig` (např. `t_shift`, `optim_time_shift`, …)
  - fit modelů přes `KineticModel.fit(...)`
- Výsledky se vykreslují sjednoceně a jsou připravené pro další práci v „Grafech“.

### ■ Grafy
- Upravuješ vizualizaci výsledků:
  - barvy křivek (ColorPicker),
  - osy (auto/manual) + limity (limity jsou editovatelné jen když je osa v režimu **manual**),
  - mřížka **major/minor**,
  - „**Ořez (clip_on)**“ (vykreslování přes okraje os),
  - export PNG/PDF/SVG.

---

## ▣ Grafy a export

### ✓ Nastavení mřížky
- **Major grid** a **Minor grid** lze zapnout zvlášť.
- Každá mřížka má vlastní volby (osa `x/y/both`, barva).

### ✓ Export obrázků
- Exportuješ z tabu **Grafy**.
- Podporované formáty: **PNG / PDF / SVG**.

---

## ▣ Import/Export experimentů

Na Home stránce:
- **Exportovat** (u experimentu) – stáhne ZIP se všemi daty experimentu (včetně souborů a nastavení).
- **Exportovat vše** – bundle se všemi experimenty + globálním nastavením aplikace.
- **Importovat** – umí jak ZIP jednoho experimentu, tak i “Exportovat vše” bundle.

---

## ▣ Aktualizace aplikace

Aktualizace se berou z GitHub repozitáře:

- `michalneoral/Kinetika-Epoxidace`

Chování:
- Aplikace při spuštění zkontroluje, zda existuje novější verze.
- Pokud ano, zobrazí dialog (nainstalovaná verze vs. dostupná verze) a zeptá se, zda chceš aktualizovat.
- Pokud zvolíš **Teď ne**, aplikace se znovu zeptá při příštím spuštění.
- Pokud zvolíš **Aktualizovat**, stáhne a spustí instalátor.

Ruční kontrola:
- v tray menu: **Zkontrolovat aktualizace**
- nebo ve Start menu: **FAME EPO Manažer - zkontrolovat aktualizace**

---

## ▣ Tipy a řešení problémů

### ■ Aplikace “běží, ale nic nevidím”
- Pokud jsi zavřel tab prohlížeče, použij tray menu **Otevřít**.
- Druhé spuštění EXE otevře nový tab na běžící instanci (single-instance).

### ■ Nejde ukončit
- Použij tray menu **Ukončit** (nebo Start menu: **FAME EPO Manažer - ukončit aplikaci**).

### ■ Export PDF/SVG nefunguje
- U PyInstaller buildu musí být přibalené Matplotlib backendy (PDF/SVG).  
  Pokud se to objeví, je to bug v balení — opravuje se v `.spec` přes hidden imports.

---
