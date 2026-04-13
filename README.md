## v3_thesis — beteendearketyper i frånvaro (EDM)

Det här projektet syftar till att hitta **dolda beteendearketyper** i elevers frånvaromönster med **osuperviserad klustring (K-Means)**. Fokus är på **hur** elever är frånvarande (beteende), inte **hur mycket** frånvaro de har (volym). Volym används i stället som **validering** efter klustring.

Pipeline i korthet:

- `**preprocess.py`**: rå lektionsdata (parquet) → aggregerade elevfeatures (`student_features.csv`)
- `**train_kmeans.py**`: KMeans på beteendefeatures → `clustered_students.csv` + figurer (t.ex. `elbow_plot.png`, `cluster_demographics.png`)
- `**test_kmeans_stability.py**`: stabilitetsanalys över seeds och k + korrelationsmatriser + PCA/diagnostikfigurer

## Projektets filer (hur allt fungerar)

### `preprocess.py`

Syfte: **Aggregerar lektionsrader till elevnivå** och skapar en feature-tabell som kan användas för klustring/analys.

**Indata**

- `--input`: en parquet-fil med lektionsrader (läses med `pyarrow`).

**Centrala antaganden / filter**

- Endast rader med `report_status == "REPORTED"` används. `UNREPORTED` tas bort.
- “Sann frånvaro” (`is_true_absence`) definieras som:
  - `present == 0` och
  - `cause_ext` **inte** är någon av `OTHERACTIVITY`, `WORKBASEDLEARNING` (sanktionerad aktivitet räknas inte som frånvaro).
- Elever med färre än `--min-reported-lessons` rapporterade lektioner exkluderas (default: 100).

**Features som skapas (för klustring)**
Listas i `CLUSTERING_FEATURES` i filen:

- `punctuality_score`: andel lektioner som är `LATEARRIVAL`.
- `morning_absence`: andel “true absence” för lektioner som startar **före 09:00** (lokal tid).
- `afternoon_absence`: andel “true absence” för lektioner som startar **efter 13:00** (lokal tid).
- `subject_variance`: varians i frånvaroandel mellan ämnen per elev (högre = mer ojämn frånvaro mellan ämnen).
- `trend_score`: \text{VT-andel} - \text{HT-andel}, där andel = `sum(true_absence_minutes) / sum(schema_minutes)` per termin.
- `invalid_ratio`: \sum(\text{invalid_absence_minutes} + \text{NOCAUSE-minuter}) / \sum(\text{absence_minutes_total}).
  - Metodval: `cause_ext == NOCAUSE` räknas som “dold ogiltig frånvaro” och inkluderas i numeratorn.
- `fragmentation_index`: andel **partial-day** frånvarodagar bland (partial + full) frånvarodagar på dagsnivå.
  - Dagsnivå: per elev+datum beräknas `day_abs_rate = sum(absence_minutes_total) / sum(schema_minutes)`.
  - Full-day om `day_abs_rate >= 0.9`, partial-day om `0 < day_abs_rate < 0.9`.
  - Robusthet: om eleven har < 3 frånvarodagar sätts index till 0 (för att undvika extrema kvoter).
- `weekday_variance`: varians i elevens **dagliga frånvaroandel** över veckodagar (”Friday effect”).

**“Reservkolumner” (metadata/validering)**
Utöver features sparas bl.a.:

- `reserved_absence_minutes_total`: total frånvarominuter (summa `absence_minutes_total`).
- `reserved_absence_type_none|valid|invalid`: antal lektionsrader per `absence_type` (används senare som “minsta antal lektioner”-proxy).
- metadata: `school_name`, `grade`, `gender`, samt `anon_student_id`.

**Utdata**

- `--output` (default `student_features.csv`): en rad per elev med features + metadata.
- Skriver även en liten logg i terminalen om bortfiltrering (UNREPORTED + låg lektionsmängd).

**Varför den här filen är viktig för beteendearketyper?**

- Här skapas de kvoter/variationer/trender som gör att elever med olika volym kan hamna i samma beteendekluster.

### `train_kmeans.py`

Syfte: **Tränar KMeans** på elevfeatures och producerar klusteretiketter samt enkla valideringsutskrifter/figurer.

**Indata**

- `--input` (default `student_features.csv`) från `preprocess.py`.

**Vilka features används?**
Det styrs av listan `FEATURES` i toppen av filen. I den här versionen är den inställd för **beteende-klustring** (ingen volymfeature som input).

**Rensning innan träning (`load_and_clean`)**

- Säkerställer att kolumnerna i `FEATURES` finns.
- Skapar `_total_lessons` = summa av `reserved_absence_type`_* och filtrerar bort elever med färre än `--min-lessons` (default 100).
- Fyller `NaN` med 0 för rate-features (`morning_absence`, `afternoon_absence`) om de ingår.
- Tar bort “noll-profiler”: rader där **alla** features i `FEATURES` är 0.

**Skalning**

- `StandardScaler` används alltid innan KMeans så att feature-skala inte dominerar avståndet.

**Utdata**

- `clustered_students.csv` (styrt av `--output`): samma elevtabell + `cluster_id`.
- `elbow_plot.png`: inertia för k=1..k_{\max} (styrt av `--elbow-max-k`).
- När exakt **2 features** används: `cluster_2d_validation.png` (styrt av `--scatter-output`) med:
  - punkter färgade per kluster
  - OLS-linje
  - Spearman rho i titeln
- `cluster_demographics.png` (styrt av `--demographics-output`): valideringsfigur `cluster_id × grade` och `cluster_id × gender`.

### `test_kmeans_stability.py`

Syfte: Testar **stabilitet** och **k-sensitivitet** för KMeans, med samma rensning och features som `train_kmeans.py`.

**Indata**

- `--input` (default `student_features.csv`)
- features importeras direkt från `train_kmeans.py` (`from train_kmeans import FEATURES, load_and_clean, plot_2d_analysis`)
  - Viktigt: stabilitetstestet använder alltså exakt samma `FEATURES` som du har valt i `train_kmeans.py`.

**Vad som testas**

- Kör KMeans för flera k (default `--k-list 3,4,5`) och flera seeds (hårdkodat: `SEEDS = (10, 20, 30, 40)`).
- Klusteretiketter **alignas** mot första körningen med:
  - centroid-avstånd (L2)
  - Hungarian algorithm (`linear_sum_assignment`)
- Mått som rapporteras per k:
  - medel `silhouette` (över 4 körningar)
  - centroid-drift MSE (Run 1 vs Run 4 efter alignering)
  - variation i klusterstorlekar (std)
- Skriver också ut **korrelationsmatriser** för dina features:
  - Pearson (linjär, känslig för outliers)
  - Spearman (monotont samband; ofta bättre för skev frånvarodata)

**Figurer**
För “bästa” k (högst mean silhouette, tie-break på lägre MSE och storleks-std):

- `stability_test_pca_k{bestk}.png`
  - Om exakt 2 features: 2D scatter per seed + OLS + Spearman i varje panel (ingen PCA behövs).
  - Annars: PCA-visualisering per seed + utskrift av top-loadings för PC1/PC2 (tolkning).
- `feature_distributions_k{bestk}.png`: boxplots av featurefördelningar per kluster.

## Så här startar/kör du projektet

### 1) Skapa miljö och installera beroenden

Det finns ingen `requirements.txt` i den här versionen, så installera minimalt:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib scipy pyarrow
```

### 2) Preprocess: parquet → `student_features.csv`

```bash
python3 preprocess.py \
  --input "lyckeboskolan_absence_lasaret2425_v6.parquet" \
  --output student_features.csv \
  --min-reported-lessons 180
```

### 3) Träna KMeans + elbow + (valfritt) 2D-samband

```bash
python train_kmeans.py --input student_features.csv --k 3 --min-lessons 180
```

Tips:

- `cluster_2d_validation.png` skapas bara om `FEATURES` har exakt 2 kolumner.
- Demografi-validering skapas som standard: `cluster_demographics.png`.

### 4) Stabilitetstest (k-sensitivitet + korrelationer + figurer)

```bash
python test_kmeans_stability.py --input student_features.csv --min-lessons 180 --k-list 3,4,5
```

Om du vill ha mer detaljer per körning och per k:

```bash
python test_kmeans_stability.py --verbose-all-k
```

## Vanliga utdatafiler i repo-roten

- `**student_features.csv**`: elevtabell med features + metadata (från `preprocess.py`)
- `**clustered_students.csv**`: samma tabell + `cluster_id` (från `train_kmeans.py`)
- `**elbow_plot.png**`: elbow för att välja k
- `**cluster_2d_validation.png**`: 2D scatter + OLS + Spearman (endast när `FEATURES` har exakt 2 kolumner)
- `**cluster_demographics.png**`: validering `cluster_id` vs grade/gender
- `**stability_test_pca_k*.png**`: stabilitetsfigur (2D-läge eller PCA beroende på antal features)
- `**feature_distributions_k*.png**`: boxplots per kluster

## Noteringar för “det här är bara en version”

- Den här versionen är byggd för att hitta **beteendearketyper** och sedan validera dem mot t.ex. `grade` och `gender`.
- Nästa naturliga steg är att iterera på feature-uppsättning och jämföra stabilitet/interpretation via PCA-loadings och boxplots.

