# Clasificarea automată a tipurilor de suprafețe carosabile și a degradărilor drumului

**Road Surface & Pothole Classification for Autonomous Driving**

Sistem de clasificare a stării carosabilului și a defectelor rutiere prin rețele convoluționale și augmentare a datelor
bazată pe modele generative.

Proiect de semestru – Computer Vision 3.

## Sarcina

Clasificare multiclasă a imaginilor captate de o cameră montată pe vehicul (dashcam). Modelul primește o imagine color a
carosabilului și o încadrează într-una din 4 clase:

| # | Clasă            | Descriere                          |
|---|------------------|------------------------------------|
| 0 | `good_asphalt`   | Drum asfaltat în stare bună        |
| 1 | `potholes`       | Drum cu gropi / avarii structurale |
| 2 | `cracks`         | Drum cu fisuri de suprafață        |
| 3 | `unpaved_gravel` | Drum neasfaltat / pietriș          |

Datele se împart strict în 3 seturi independente: 70% train / 15% validation / 15% test.

## Structura proiectului

```
road-surface-classification/
├── data/              # setul de date 
├── checkpoints/       # checkpoint-uri salvate la antrenare 
├── results/           # grafice și metrici
├── src/
│   ├── dataset.py     # Dataset, DataLoader, transformări, împărțire train/val/test
│   ├── model.py       # arhitectura CNN cu conexiuni reziduale
│   ├── train.py       # antrenare + validare + salvare / încărcare checkpoint
│   ├── test.py        # evaluarea pe setul de test
│   └── plots.py       # grafice loss / acuratețe
├── requirements.txt
└── README.md
```

## Instalare

Python 3.11

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Setul de date

Setul de date nu este inclus în repository (vezi `.gitignore`). Se obține din două surse publice:

- [RDD2022](https://figshare.com/articles/dataset/21431547) (CC BY-SA 4.0): imagini dashcam din Cehia și India.
  Adnotările de detecție se transformă în clase: groapă (D40) → `potholes`, doar fisuri (D00/D10/D20) → `cracks`,
  fără adnotări → `good_asphalt`.
- [RSCD](https://figshare.com/articles/dataset/20424582): eșantion aleator din clasele `dry-gravel` și `wet-gravel`
  → `unpaved_gravel`.

```bash
python scripts/download_data.py              # descarcă doar părțile necesare (~0.8 GB) în data/downloads/
python scripts/prepare_data.py --per-class 1500   # construiește data/raw/ (1500 imagini / clasă, seed 42)
```

Rezultatul este câte un folder per clasă:

```
data/raw/
├── good_asphalt/
├── potholes/
├── cracks/
└── unpaved_gravel/
```

La prima rulare, imaginile sunt împărțite stratificat 70/15/15 (seed 42), iar lista fișierelor din fiecare split se
salvează în `data/splits.json`. Rulările următoare refolosesc aceeași împărțire.

## Rulare

Verificarea Dataset-ului și a DataLoader-ului (afișează dimensiunea split-urilor și distribuția claselor, încarcă un
batch și salvează un exemplu augmentat în `results/sample_batch.png`):

```bash
python -m src.dataset --data-dir data/raw
```

Din cod:

```python
from src.dataset import get_dataloaders
loaders, datasets = get_dataloaders("data/raw", batch_size=32, img_size=224)
```

## Status
- [x] S3 – descrierea inițială a proiectului
- [x] S4 – crearea repository-ului
- [x] S6 – setul de date și DataLoader-ul funcționale
- [ ] S9 – rularea end-to-end a modelului
- [ ] S12 – augmentarea cu model generativ
- [ ] S14 – prezentarea finală
