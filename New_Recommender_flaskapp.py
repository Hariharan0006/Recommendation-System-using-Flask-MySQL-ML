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
# Load Model
# -------------------------
tfidf = joblib.load("matrix")

anime = None
movies_list = []
cosine_sim_matrix = None
anime_index = None


# -------------------------
# Create table if not exists
# -------------------------
def create_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                id SERIAL PRIMARY KEY,
                name TEXT,
                genre TEXT
            );
        """))
        conn.commit()


# -------------------------
# Load Data
# -------------------------
def load_data():
    global anime, movies_list, cosine_sim_matrix, anime_index

    try:
        create_table()

        anime = pd.read_sql("SELECT * FROM movies", engine)

        if anime.empty:
            movies_list = []
            return

        anime["genre"] = anime["genre"].fillna(" ")

        movies_list = anime["name"].tolist()

        tfidf_matrix = tfidf.transform(anime["genre"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)

        anime_index = pd.Series(anime.index, index=anime["name"]).drop_duplicates()

        print("✅ Data Loaded")

    except SQLAlchemyError as e:
        print("Database Error:", e)
        movies_list = []


# -------------------------
# Recommendation
# -------------------------
def get_recommendations(name, topN):
    if name not in anime_index:
        return pd.DataFrame(columns=["name", "Score"])

    idx = anime_index[name]

    scores = list(enumerate(cosine_sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:topN+1]

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
    if not movies_list:
        load_data()
    return render_template("index.html", movies_list=movies_list)


@app.route("/recommend", methods=["POST"])
def recommend():
    name = request.form.get("mn")
    topN = request.form.get("tp")

    if not name or not topN:
        return "Invalid Input"

    topN = int(topN)

    results = get_recommendations(name, topN)

    if results.empty:
        return "No recommendations found"

    html_table = results.to_html(classes="table table-striped", index=False)

    return render_template("data.html", table=html_table)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

