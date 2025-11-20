# diamonds

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small Python web app and example machine learning project that predicts diamond prices using a Random Forest regressor.

## Live demo

The app is available at: [http://diamonds.johnthompson.io](http://diamonds.johnthompson.io)

## Project purpose

This project started as an experiment to estimate the cost of an engagement ring. I collected local jeweler listings for oval diamonds, built a Random Forest regression model (trained in Python after experimenting in Alteryx), and exposed a simple web UI to score new diamonds.

## Contents

- `app.py` — Flask web application that serves the UI and prediction endpoint.
- `diamonds-predict.py` — helper script / example for running predictions from the command line.
- `diamonds-model-creation.py` — notebook-style script used to train the model.
- `random_forest.pkl` — serialized trained RandomForestRegressor used by the app.
- `diamonds.csv` — raw dataset used for model training and experiments.
- `templates/index.html` — web UI template.
- `requirements.txt` — Python dependencies.
- `gunicorn_config.py`, `Procfile` — deployment configuration.

## Quick start (local)

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run locally:

```bash
python app.py
```

By default the Flask app will listen on the port printed to the console. Open [http://localhost:5000](http://localhost:5000) (or the printed address) to use the UI.

3. Or run with Gunicorn for a production-like server:

```bash
gunicorn -c gunicorn_config.py app:app
```

## Usage

The web UI lets you input diamond attributes and get a predicted price. `diamonds-predict.py` shows how to call the model from a script (it loads `random_forest.pkl`). If you want to retrain, inspect `diamonds-model-creation.py`.

## Notes & history

- The original prototype was tested in Alteryx as a proof of concept before being ported to Python.
- Fun note: the real purchase ended up being very close to the model estimate.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

If you have questions, contact John Thompson or open an issue in this repository.
