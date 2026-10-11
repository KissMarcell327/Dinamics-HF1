# BME Dinamika 1. Házi Feladat – Kinematikai Megoldó Webalkalmazás

Önálló, böngészőben futó interaktív **Python Streamlit** alkalmazás a BME Műszaki Mechanikai Tanszék Dinamika (BMEGEMMBXM3) tárgyának 1. házi feladatához: síkbeli összetett mechanizmus kinematikájának analitikus és méretarányos grafikus megoldása tetszőleges bemeneti paraméterekkel.

---

## 🚀 Főbb funkciók

1. **Interaktív paraméterezés (Oldalsáv / Sidebar):**
   - $\phi$ [fok]: a (3) henger szögelfordulása (alapértelmezett: `55.0°`, csúszka `10°` – `85°`)
   - $l_1$ [m]: az (1) henger sugara (alapértelmezett: `0.06 m`)
   - $l_2$ [m]: a (2) rúd hossza (alapértelmezett: `0.21 m`)
   - $l_3$ [m]: a (3) henger sugara (alapértelmezett: `0.05 m`)
   - $R$ [m]: a fix domború körpálya sugara (alapértelmezett: `0.15 m`, kapcsolóval $R = 3 \cdot l_3$)
   - $v_{Cx}$ [m/s]: a $C$ pont vízszintes sebessége (alapértelmezett: `0.96 m/s`)
   - Vektorskálák és jelölések finomhangolása

2. **100% hwchk kompatibilitás:**
   - Eredménytáblázat a hivatalos tanszéki mezőnevekkel:
     - `omega_2z` [rad/s]
     - `epsilon_2z` [rad/s²]
     - `omega_3z` [rad/s]
     - `epsilon_3z` [rad/s²]
     - `v_S2` [m/s]
     - `a_S2` [m/s²]
     - `|alpha_2|` [fok]
     - `|a_S2t|` [m/s²]
     - `|a_S2n|` [m/s²]
     - `rho_S2` [m]
   - Egy kattintással kimásolható mezőlista (hwchk formátum), JSON kimenet, CSV letöltés.

3. **Méretarányos műszaki ábra (Matplotlib):**
   - A (3) és (1) hengerek, a fix domború pálya ($R$), vízszintes talaj és vezeték, (2) rúd.
   - Nevezetes pontok: $E, A, B, C, S_2, P$ és $G \equiv C$.
   - Szaggatott szerkesztővonalak az $E-B-P$ és $C-P$ mentén a $P$ sebességpólushoz.
   - Sebességvektorok (kék nyilak: $\mathbf{v}_C, \mathbf{v}_B, \mathbf{v}_{S2}$).
   - Gyorsulásvektorok (piros nyilak: $\mathbf{a}_B, \mathbf{a}_{S2}$).
   - Komponensek (lila nyilak: $\mathbf{a}_{S2t}, \mathbf{a}_{S2n}$).
   - 300 DPI felbontású PNG letöltési lehetőség.

4. **Lépésről lépésre kidolgozott elméleti levezetés:**
   - Geometria, sebességállapot, sebességpólus, gyorsulásállapot (domború gördülés $a_E$, 2x2 egyenletrendszer), természetes koordináta-rendszer.
   - Minden lépésnél képletek LaTeX formátumban és behelyettesített számértékekkel.

5. **Mozgástartomány & Kinematikai görbék:**
   - Grafikonok a szögsebességek, szöggyorsulások és súlyponti sebesség alakulásáról $\phi$ függvényében.

6. **Letölthető LaTeX jegyzőkönyv (.tex):**
   - Komplett tanszéki stílusú LaTeX forráskód, amely közvetlenül lefordítható Overleaf-ben vagy TeXMaker-ben.

---

## 🛠️ Telepítés és Futtatás

### 1. Követelmények telepítése (ha szükséges):
```powershell
pip install -r requirements.txt
```

### 2. Alkalmazás indítása:
```powershell
streamlit run app.py
```
vagy dupla kattintás a `run_app.bat` fájlra Windows alatt.

A böngésző automatikusan megnyílik a `http://localhost:8501` címen.
