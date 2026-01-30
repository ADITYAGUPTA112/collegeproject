from flask import Flask, render_template, request
import pandas as pd
from model import predict_single, predict_batch

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review")

    if review:
        label, confidence = predict_single(review)
        return render_template(
            "index.html",
            single_result=True,
            review=review,
            label=label,
            confidence=confidence
        )

    return render_template("index.html")


@app.route("/predict_batch", methods=["POST"])
def predict_batch_route():
    reviews = request.form.get("batch_reviews")

    if reviews:
        review_list = [r.strip() for r in reviews.split("\n") if r.strip()]
        results = predict_batch(review_list)

        batch_results = list(zip(review_list, results))

        return render_template(
            "index.html",
            batch_result=True,
            batch_results=batch_results
        )

    return render_template("index.html")


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files["file"]

    if file:
        df = pd.read_csv(file)

        if "review" not in df.columns:
            return render_template("index.html", error="CSV must contain 'review' column")

        reviews = df["review"].astype(str).tolist()
        predictions = predict_batch(reviews)

        df["Prediction"] = [p[0] for p in predictions]
        df["Confidence"] = [p[1] for p in predictions]

        table = df.to_html(classes="table table-striped", index=False)

        return render_template("index.html", table=table)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
