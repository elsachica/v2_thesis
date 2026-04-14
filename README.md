## v3_thesis — beteendearketyper i frånvaro (EDM)

Projektet grupperar elever utifrån **beteendefeatures** (K-means). Volym mäts som **validering**, inte som kluster-input.

### Definitioner

#### Beteendefeatures

Dessa kolumner (se `FEATURES` i `src/train_kmeans.py`) byggs i `preprocess.py` och används till klustring:

- **Morning / afternoon absence:** Andel lektioner med “sann” frånvaro vars starttid är **före 09:00** respektive **efter 13:00** (lokal tid).
- **Fragmentation index:** Andel **strö-/partialdagar** bland frånvarodagar som klassas som partial jämfört med hel dags frånvaro på **dagsnivå** (mäter “hackighet” i frånvaromönster; kräver tillräckligt många frånvarodagar för att vara meningsfullt — se kod).
- **Subject variance:** **Varians** i frånvaroandel mellan **skolämnen** för eleven (ojämn frånvaro över ämnen ger högre värde).
- **Punctuality score:** Andel lektioner som registrerats som **sen ankomst** (`LATEARRIVAL`).
- **Trend score:** Skillnad i **frånvaroandel** mellan **VT och HT** (positiv = relativt högre frånvaro VT).
- **Weekday variance:** **Varians** i elevens dagliga frånvaroandel över **veckodagar** (koncentrerad vs utspridd frånvaro i veckan).

#### Datarensning

- Endast rader med `report_status == REPORTED` används i preprocess (övriga rader slängs).
- **Tröskel 180 lektioner** (`--min-reported-lessons` i preprocess och `--min-lessons` i träning/stabilitet): används för att säkerställa **statistiskt stabila beteendemönster** per elev — tillräckligt många lektioner så att rates (morgon, invalid ratio, m.m.) inte domineras av tillfälligheter hos elever med mycket få rapporterade timmar.

#### Sökvägar i koden

Standardfiler och mappar (`data/raw/…`, `data/processed/…`, `output/plots/…`) definieras centralt i `**src/project_paths.py`**. Övriga skript importerar dessa defaults; override sker via CLI-flaggor (t.ex. `--input`, `--output`). `scripts/run_project.sh` kör Python från projektroten med dessa standarder.

### Mappstruktur

```
v3_thesis/
├── data/
│   ├── raw/              # original-parquet (standardnamn nedan)
│   └── processed/        # student_features.csv, clustered_students.parquet
├── output/
│   └── plots/            # Alla .png från pipeline
├── scripts/
│   └── run_project.sh    # Kör hela kedjan
└── src/                     # Python-moduler
    ├── project_paths.py     # Standardvägar relativt projektrot
    ├── preprocess.py
    ├── train_kmeans.py
    └── test_kmeans_stability.py
```

Standard indata-parquet: `**data/raw/lyckeboskolan_absence_lasaret2425_v6.parquet**`. Om filen saknas: lägg den där med det namnet, eller sätt `PARQUET=/full/sökväg/din.parquet ./scripts/run_project.sh`.

### Miljö

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Plug-and-play (hela pipelinen)

1. Lägg parquet som `**data/raw/lyckeboskolan_absence_lasaret2425_v6.parquet**` (eller sätt `PARQUET`).
2. Från projektroten:

```bash
chmod +x scripts/run_project.sh   # första gången
./scripts/run_project.sh
```

Standard: `--min-reported-lessons 180`, `--min-lessons 180`, `k=3`, stabilitet `--k-list 3`. Miljövariabler: `MIN_LESSONS`, `K`, och vid behov `**PARQUET**` (sökväg till din parquet om den inte ligger på standardplatsen).

### Vad som skrivs ut och sparas (resultat)

Efter en lyckad körning har du följande **filer** (standardnamn):


| Steg         | Filer                                       | Innehåll                                             |
| ------------ | ------------------------------------------- | ---------------------------------------------------- |
| Preprocess   | `data/processed/student_features.csv`       | En rad per elev med beteendefeatures + metadata.     |
| Train KMeans | `data/processed/clustered_students.parquet` | Samma som ovan + `cluster_id`.                       |
| Stabilitet   | `output/plots/feature_distributions_k*.png` | Boxplots per kluster för varje feature.              |
| Stabilitet   | `output/plots/stability_test_pca_k*.png`    | PCA (eller 2D-scatter om bara två features används). |


I **terminalen** skrivs bland annat: datakvalitet från preprocess (antal elever, filtrering), **silhouette** för valt *k*, tabell med volym per kluster (`reserved_absence_minutes_total`, validering), **klusterprofiler** (medelvärde per feature per kluster), Pearson/Spearman-korrelation mellan features, stabilitetsmått (seeds, centroid-drift, PCA-loadings) och en kort **stabilitets-heuristik** (OK/VARNING). **Exakta tal** (antal elever, silhouette, klusterstorlekar) beror på din parquet och dina trösklar — dokumentera dem i uppsatsen när du kör, inte som fasta värden i README.

**Utdata (kort):** `data/processed/student_features.csv`, `data/processed/clustered_students.parquet` + alla relevanta `**output/plots/*.png`**.

### Manuellt (samma innehåll som skriptet)

```bash
cd /sökväg/till/v3_thesis
python3 src/preprocess.py --min-reported-lessons 180
python3 src/train_kmeans.py --k 3 --min-lessons 180
python3 src/test_kmeans_stability.py --min-lessons 180 --k-list 3
```

Alla skript har standardvägar till `data/processed/` och `output/plots/` (se `src/project_paths.py`). Kör alltid från **projektroten** så att sökvägarna stämmer.

### Viktiga filer

- `**src/preprocess.py`**: parquet → features (`invalid_ratio` m.m. finns kvar).
- `**src/train_kmeans.py`**: K-means (lean) → `cluster_id` och Parquet-utdata.
- `**src/test_kmeans_stability.py**`: stabilitet, PCA/boxplots.

