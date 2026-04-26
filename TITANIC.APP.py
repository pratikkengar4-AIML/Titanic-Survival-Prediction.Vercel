from flask import Flask, request, jsonify
from flask_cors import CORS
from model import titanic_model

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    try:
        passenger = {
            "Pclass": int(data.get("pclass", 3)),
            "Sex": data.get("sex", "male").lower(),
            "Age": float(data.get("age", 30.0)),
            "SibSp": int(data.get("sibsp", 0)),
            "Parch": int(data.get("parch", 0)),
            "Fare": float(data.get("fare", 0.0)),
            "Embarked": data.get("embarked", "S").upper()
        }

        result = titanic_model.predict_survival(passenger)

        return jsonify({
            "survived": result["survived"],
            "probability": result["probability"],
            "confidence": result["confidence"]
        })
    except Exception as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)
