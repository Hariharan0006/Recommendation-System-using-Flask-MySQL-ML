# Import necessary libraries
import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote


# Initialize Flask application FIRST
app = Flask(__name__)


# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

engine = create_engine(DATABASE_URL)


# SQL query
sql = 'select * from movies'

anime = pd.read_sql_query(sql, engine)

anime["genre"] = anime["genre"].fillna(" ")

movies_list = anime['name'].to_list()

tfidf = joblib.load('matrix')

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
    return render_template("index.html", movies_list=movies_list)


@app.route('/guest', methods=["POST"])
def Guest():
    if request.method == 'POST':

        mn = request.form["mn"]
        tp = request.form["tp"]

        top_n = get_recommendations(mn, topN=int(tp))

        top_n.to_sql('top_10', con=engine, if_exists='replace', chunksize=1000, index=False)

        html_table = top_n.to_html(classes='table table-striped')

        return render_template("data.html", Y="Results have been saved in your database", Z=f"""
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
                }}
                .table tbody th {{
                    background-color: #ab2c3f;
                }}
            </style>
            {html_table}
        """)


# ✅ Render Compatible Run Block (ONLY ONE)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
