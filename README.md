# Star Type Classification (Starfinder)
Python ML backend + HTML/CSS/JS frontend for classifying stellar objects into:
- 0: Red Dwarf
- 1: Brown Dwarf
- 2: White Dwarf
- 3: Main Sequence
- 4: Super Giants
- 5: Hyper Giants
The project uses a NASA-inspired dataset in `Stars.csv`.

**Live Demo:** https://stellarinsight.onrender.com/

## Project Structure
- `backend/app.py`: Flask API + static frontend hosting
- `backend/train_model.py`: Train/compare models and save artifacts
- `backend/run_eda.py`: EDA outputs (stats + plots)
- `static/`: pitch-black space-themed UI (HTML/CSS/JS)
- `static/assets/images/`: background + prediction images
- `artifacts/`: saved model + domain spec + metrics
- `reports/`: EDA results (generated)
## Domain-Aware Constraint
This classifier is restricted to the stellar domain present in the dataset.
Before predicting, the backend validates inputs using dataset-derived constraints:
- Numeric features must fall inside the dataset min/max range.
- Categorical features must match known categories from the dataset.
If the input is out-of-domain, the API returns:
`Input parameters lie outside the stellar classification domain and may correspond to a compact object.`
No new labels are invented and compact objects are not forced into a star class.
## Local Setup
Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```
Run EDA (writes to `reports/`):
```bash
python3 -m backend.run_eda
```
Train models and save the best pipeline (writes to `artifacts/`):
```bash
python3 -m backend.train_model
```
Start the web app (API + UI):
```bash
python3 -m gunicorn backend.app:app --bind 0.0.0.0:8000
```
Open:
- http://localhost:8000/
## REST API
### POST `/predict`
Request JSON:
```json
{
  "Temperature": 5800,
  "L": 1.0,
  "R": 1.0,
  "A_M": 4.83,
  "Color": "Yellow",
  "Spectral_Class": "G"
}
```
In-domain response:
```json
{
  "in_domain": true,
  "prediction": {
    "type_id": 3,
    "type_name": "Main Sequence",
    "confidence": 0.98
  }
}
```
Out-of-domain response:
```json
{
  "in_domain": false,
  "message": "Input parameters lie outside the stellar classification domain and may correspond to a compact object."
}
```
## Deploy on Render (GitHub)
1. Push this repository to your GitHub account (e.g. `Saksham-Gupta-GH`).
2. In Render, create a new **Web Service** from the GitHub repo.
3. Render automatically uses `render.yaml`:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn backend.app:app --bind 0.0.0.0:$PORT`
Recommended: keep `artifacts/` committed so deployment does not retrain.
