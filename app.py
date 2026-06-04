"""
app.py - Flask backend za kontinuiranu autentifikaciju pomoću dinamike tipkanja.

Ova verzija je prilagođena stvarnim modelima:
- models/jure_model.pkl
- models/jure_scaler.pkl
- models/jure_threshold.json
- models/mateo_model.pkl
- models/mateo_scaler.pkl
- models/mateo_threshold.json
- models/feature_columns.json

Frontend šalje prozor tipkanja na /predict.
Backend:
1. uzima prijavljenog korisnika iz sessiona
2. iz raw eventa računa feature_vector
3. poziva ml.predict.predict_for_user()
4. vraća rezultat frontendu
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request, session

from ml.features import extract_features_from_window
from ml.predict import predict_for_user, session_status


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-keystroke-auth")

AUTHORIZED_USERS = {"jure", "mateo"}


@app.route("/")
def index():
    """Prikaz glavne stranice."""
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    """
    Prima JSON:
        { "user": "jure" | "mateo" }

    Sprema prijavljenog korisnika u Flask session.
    """
    data = request.get_json(force=True) or {}
    user = str(data.get("user", "")).strip().lower()

    if user not in AUTHORIZED_USERS:
        return jsonify({
            "status": "error",
            "error": "Nepoznat korisnik. Odaberi 'jure' ili 'mateo'."
        }), 400

    session.clear()
    session["logged_in_user"] = user
    session["last_results"] = []

    return jsonify({
        "status": "ok",
        "user": user,
    })


@app.route("/logout", methods=["POST"])
def logout():
    """Briše session."""
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prima prozor tipkanja iz frontenda.

    Očekivani JSON:
        {
            "events": [
                {"key": "a", "keydown": 1000.0, "keyup": 1080.0},
                ...
            ],
            "backspace_count": 1
        }

    Podržan je i stariji frontend format:
        {"key": "a", "type": "keydown", "timestamp": 1000.0}
        {"key": "a", "type": "keyup",   "timestamp": 1080.0}
    """
    logged_in_user = session.get("logged_in_user")
    if not logged_in_user:
        return jsonify({
            "accepted": False,
            "score": 0.0,
            "status": "error",
            "message": "Korisnik nije prijavljen."
        }), 401

    data = request.get_json(force=True) or {}
    raw_events = data.get("events", [])
    backspace_count = int(data.get("backspace_count", 0) or 0)

    if not raw_events:
        return jsonify({
            "accepted": False,
            "score": 0.0,
            "status": "error",
            "message": "Nisu primljeni podaci o tipkanju."
        }), 400

    try:
        feature_vector = extract_features_from_window(
            events=raw_events,
            backspace_count=backspace_count,
            min_events=30,
        )

        prediction = predict_for_user(logged_in_user, feature_vector)

        last_results = list(session.get("last_results", []))
        last_results.append(bool(prediction.get("accepted", False)))
        last_results = last_results[-5:]
        session["last_results"] = last_results

        prediction["session_status"] = session_status(
            last_results,
            max_windows=5,
            warn_if_bad=2,
            lock_if_bad=3,
        )
        prediction["last_results"] = last_results

        return jsonify(prediction)

    except Exception as exc:
        return jsonify({
            "accepted": False,
            "score": 0.0,
            "status": "error",
            "session_status": "warning",
            "message": str(exc),
        }), 400


@app.route("/reset", methods=["POST"])
def reset():
    """Resetira live povijest predikcija, ali ostavlja prijavljenog korisnika."""
    if "logged_in_user" in session:
        session["last_results"] = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
