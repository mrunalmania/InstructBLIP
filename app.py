import gradio as gr
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
embeddings = embeddings.astype("float32")

embedding_size = embeddings.shape[1]
n_clusters = 1
num_results = 1

qunatizer = faiss.IndexFlatIP(embedding_size)

index = faiss.IndexIVFFlat(
    qunatizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT,
)


index.train(embeddings)
index.add(embeddings)

def _search(query):
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding.reshape(1,-1)

    _, indices = index.search(query_embedding, num_results)
    print(indices)
    images = [f"images/{i}.jpg" for i in indices[0]]
    print(images)
    return images

with gr.Blocks() as demo:
    query = gr.Textbox(lines=1, label="search query")
    outputs = gr.Gallery(preview=True)
    submit = gr.Button(value="search")
    submit.click(_search, inputs=query, outputs=outputs)

demo.launch()
