# Import necessary libraries
import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask application
app = Flask(__name__)

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

engine = create_engine(DATABASE_URL)

# Load ML model
tfidf = joblib.load('matrix')

# Global variables (loaded after DB connection)
anime = None
movies_list = None
cosine_sim_matrix = None
anime_index = None


def load_data():
    global anime, movies_list, cosine_sim_matrix, anime_index

    sql = "SELECT * FROM movies"
    anime = pd.read_sql_query(sql, engine)

    anime["genre"] = anime["genre"].fillna(" ")

    movies_list = anime['name'].to_list()

    tfidf_matrix = tfidf.transform(anime.genre)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicates()


def get_recommendations(Name, topN):
    anime_id = anime_index[Name]

    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

    cosine_scores_N = cosine_scores[0:topN + 1]

    anime_idx = [i[0] for i in cosine_scores_N]
    anime_scores = [i[1] for i in cosine_scores_N]

    anime_similar_show = pd.DataFrame(columns=["name", "Score"])
    anime_similar_show["name"] = anime.loc[anime_idx, "name"]
    anime_similar_show["Score"] = anime_scores
    anime_similar_show.reset_index(inplace=True)

    return anime_similar_show.iloc[1:, ]


@app.route('/')
def home():
    if movies_list is None:
        load_data()
    return render_template("index.html", movies_list=movies_list)


@app.route('/guest', methods=["POST"])
def Guest():
    if movies_list is None:
        load_data()

    mn = request.form["mn"]
    tp = request.form["tp"]

    top_n = get_recommendations(mn, topN=int(tp))

    top_n.to_sql(
        'top_10',
        con=engine,
        if_exists='replace',
        chunksize=1000,
        index=False
    )

    html_table = top_n.to_html(classes='table table-striped')

    return render_template(
        "data.html",
        Y="Results have been saved in your database",
        Z=f"""
        <style>
            .table {{
                width: 50%;
                margin: 0 auto;
                border-collapse: collapse;
            }}
            .table thead {{
                background-color: #39648f;
            }}
            .table th, .table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .table td {{
                background-color: #5e617d;
            }}import os
import pandas as pd
import joblib
from flask import Flask, render_template, request
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# Flask App Initialization
# ----------------------------------
app = Flask(__name__)

# ----------------------------------
# Database Connection
# ----------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

engine = create_engine(DATABASE_URL)

# ----------------------------------
# Load ML Model
# ----------------------------------
tfidf = joblib.load("matrix.pkl")

# Global Variables
anime = None
movies_list = []
cosine_sim_matrix = None
anime_index = None


# ----------------------------------
# Create Table If Not Exists
# ----------------------------------
def create_table_if_not_exists():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                id SERIAL PRIMARY KEY,
                name TEXT,
                genre TEXT
            );
        """))
        conn.commit()


# ----------------------------------
# Load CSV if DB empty
# ----------------------------------
def load_csv_if_empty():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM movies"))
        count = result.scalar()

    if count == 0:
        if os.path.exists("movies.csv"):
            df = pd.read_csv("movies.csv")
            df.to_sql("movies", engine, if_exists="append", index=False)
            print("✅ CSV data inserted into database")
        else:
            print("⚠ movies.csv not found")


# ----------------------------------
# Load Data
# ----------------------------------
def load_data():
    global anime, movies_list, cosine_sim_matrix, anime_index

    try:
        create_table_if_not_exists()
        load_csv_if_empty()

        anime = pd.read_sql("SELECT * FROM movies", engine)

        if anime.empty:
            movies_list = []
            return

        anime["genre"] = anime["genre"].fillna(" ")

        movies_list = anime["name"].tolist()

        tfidf_matrix = tfidf.transform(anime["genre"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)

        anime_index = pd.Series(anime.index, index=anime["name"]).drop_duplicates()

        print("✅ Data Loaded Successfully")

    except SQLAlchemyError as e:
        print("Database Error:", e)
        movies_list = []


# ----------------------------------
# Recommendation Function
# ----------------------------------
def get_recommendations(name, topN):
    if name not in anime_index:
        return pd.DataFrame(columns=["name", "Score"])

    idx = anime_index[name]

    cosine_scores = list(enumerate(cosine_sim_matrix[idx]))
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

    top_scores = cosine_scores[1:topN + 1]

    anime_idx = [i[0] for i in top_scores]
    anime_scores = [i[1] for i in top_scores]

    result = pd.DataFrame({
        "name": anime.loc[anime_idx, "name"].values,
        "Score": anime_scores
    })

    return result


# ----------------------------------
# Routes
# ----------------------------------
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

    try:
        topN = int(topN)
    except:
        return "Top N must be integer"

    results = get_recommendations(name, topN)

    if results.empty:
        return "No recommendations found"

    results.to_sql("top_results", engine, if_exists="replace", index=False)

    html_table = results.to_html(classes="table table-striped", index=False)

    return render_template("data.html", table=html_table)


# ----------------------------------
# Run App
# ----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
            .table tbody th {{
                background-color: #ab2c3f;
            }}
        </style>
        {html_table}
        """
    )


# Render compatible run block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

