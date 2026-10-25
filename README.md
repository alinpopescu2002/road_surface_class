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

Setul de date nu este inclus în repository inca.

## Rulare

*O sa completez pe parcurs.*

## Status
- [x] S3 – descrierea inițială a proiectului
- [x] S4 – crearea repository-ului
- [ ] S6 – setul de date și DataLoader-ul funcționale
- [ ] S9 – rularea end-to-end a modelului
- [ ] S12 – augmentarea cu model generativ
- [ ] S14 – prezentarea finală
