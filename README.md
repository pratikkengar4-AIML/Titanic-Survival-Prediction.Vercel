# Titanic Survival Prediction Web App

A simple web app that predicts Titanic survival probability using a heuristic model.

## Run the app
1. Open a terminal in `C:\Users\HP\titanic_app`
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Start the server:

```powershell
python app.py
```

4. Open `http://127.0.0.1:5001` in your browser.

## API
- `POST /api/predict`
- JSON body fields:
  - `pclass` (1, 2, 3)
  - `sex` (`male` or `female`)
  - `age` (number)
  - `sibsp` (number)
  - `parch` (number)
  - `fare` (number)
  - `embarked` (`C`, `Q`, `S`)

## Frontend
- `static/index.html`
- `static/app.js`
- `static/styles.css`
