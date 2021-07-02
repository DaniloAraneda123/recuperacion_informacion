from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

app = Flask(__name__)
CORS(app)


def embed_text(text):
    vectors = session.run(embeddings, feed_dict={text_ph: text})
    return [vector.tolist() for vector in vectors]


def consultaEmbedding(query):
    query_vector = embed_text([query.lower()])[0]
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "body"]}
        }
    )
    list = []
    for hit in response["hits"]["hits"]:
        list.append({
            "score": hit["_score"],
            "titulo": hit["_source"]["title"]
        }
        )
    return list


def consultaTradicional(query):
    response = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title.analizado"
                    ]
                }
            },
            "_source": {
                "includes": ["title"]
            }
        }
    )

    list = []
    for hit in response["hits"]["hits"]:
        list.append({
            "score": hit["_score"],
            "titulo": hit["_source"]["title"]
        })
    return list


@app.route("/recuperarInformacion", methods=['POST'])
def actualizarEmpleado():

    texto = request.json['texto']

    a = consultaTradicional(texto)
    b = consultaEmbedding(texto)

    dict = {}
    dict["tradicional"] = a
    dict["embeddings"] = b

    return jsonify(dict)


if __name__ == '__main__':
    INDEX_NAME = "posts"
    BATCH_SIZE = 1000
    SEARCH_SIZE = 10
    GPU_LIMIT = 0.5
    tf.disable_eager_execution()
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    text_ph = tf.placeholder(tf.string)
    embeddings = embed(text_ph)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_LIMIT
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    client = Elasticsearch(timeout=30)

    app.run(host='localhost', debug=True, port=5000)