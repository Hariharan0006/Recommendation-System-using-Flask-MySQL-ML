import os
import pandas as pd
import joblib
from flask import Flask, render_template, request
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------
# Database
# -------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

engine = create_engine(DATABASE_URL)

# -------------------------
# Load TF-IDF Model
# -------------------------
try:
    tfidf = joblib.load("matrix")  # Ensure this file exists
except FileNotFoundError:
    raise FileNotFoundError("TF-IDF model file 'matrix' not found!")

# -------------------------
# Global Data Storage
# -------------------------
app.config["ANIME_DF"] = pd.DataFrame()
app.config["MOVIES_LIST"] = []
app.config["COSINE_MATRIX"] = None
app.config["ANIME_INDEX"] = None

# -------------------------
# Create Table if not exists
# -------------------------
def create_table():
    try:
        with engine.begin() as conn:  # auto-commit DDL
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS movies (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    genre TEXT
                );
            """))
    except SQLAlchemyError as e:
        print("Database Table Creation Error:", e)
        raise

# -------------------------
# Load Data
# -------------------------
def load_data():
    try:
        create_table()
        anime = pd.read_sql("SELECT * FROM movies", engine)

        if anime.empty:
            app.config["MOVIES_LIST"] = []
            app.config["ANIME_DF"] = pd.DataFrame()
            app.config["COSINE_MATRIX"] = None
            app.config["ANIME_INDEX"] = None
            print("⚠️ No data found in database.")
            return

        anime["genre"] = anime["genre"].fillna(" ")
        tfidf_matrix = tfidf.transform(anime["genre"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        anime_index = pd.Series(anime.index, index=anime["name"]).drop_duplicates()

        # Save to app config
        app.config["ANIME_DF"] = anime
        app.config["MOVIES_LIST"] = anime["name"].tolist()
        app.config["COSINE_MATRIX"] = cosine_sim_matrix
        app.config["ANIME_INDEX"] = anime_index

        print("✅ Data Loaded Successfully.")

    except SQLAlchemyError as e:
        print("Database Error:", e)
        app.config["MOVIES_LIST"] = []

# -------------------------
# Recommendation Function
# -------------------------
def get_recommendations(name, topN=5):
    anime_index = app.config.get("ANIME_INDEX")
    cosine_sim_matrix = app.config.get("COSINE_MATRIX")
    anime = app.config.get("ANIME_DF")

    if anime_index is None or cosine_sim_matrix is None or anime.empty:
        return pd.DataFrame(columns=["name", "Score"])

    if name not in anime_index:
        return pd.DataFrame(columns=["name", "Score"])

    idx = anime_index[name]
    scores = list(enumerate(cosine_sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:topN+1]  # skip itself
    indices = [i[0] for i in scores]
    values = [i[1] for i in scores]

    return pd.DataFrame({
        "name": anime.loc[indices, "name"].values,
        "Score": values
    })

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    if not app.config["MOVIES_LIST"]:
        load_data()
    return render_template("index.html", movies_list=app.config["MOVIES_LIST"])

@app.route("/recommend", methods=["POST"])
def recommend():
    name = request.form.get("mn")
    topN = request.form.get("tp")

    if not name or not topN:
        return "❌ Invalid input."

    try:
        topN = int(topN)
        if topN <= 0:
            return "❌ Top N must be positive."
    except ValueError:
        return "❌ Top N must be an integer."

    results = get_recommendations(name, topN)
    if results.empty:
        return f"⚠️ No recommendations found for '{name}'."

    html_table = results.to_html(classes="table table-striped", index=False)
    return render_template("data.html", table=html_table)

# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
