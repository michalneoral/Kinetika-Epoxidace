TODO: add readme

How to compile

```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --clean --collect-all nicegui --collect-data latex2mathml -n kinetika --onefile -i build/icon.ico app/kinetika_webapp.py
```


- [X] není vidět nastavení tabulky, když neprojde check
- [X] k bude přímo u grafu
- [X] zatrhavátku pro modely
- [ ] instalačka a auto update - in progress
- [ ] config - in progress [high priority]
- [ ] potlačení avastu - podepsání dokumentu a udělat ho méně nápadný - tuším, jak udělat
- [X] umisťování věcí do grafu na kliknutí
- [X] aliasy pro kontroly textu (je jedno jestli bude v sloupci C17:0 nebo C17 apod)
- [ ] C18:2 rozlišit tu křivku epoxidů na C18:2 1-EPO a C18:2 2-EPO
- [ ] proč u C18:1 není spočítané "k" a nesedí tam ta simulace? - je to: d[OH]/dt = k*[oxiran]
- [ ] No tam to přejmenovat na správně, že je to C18:3 a tam to je rozdělené

