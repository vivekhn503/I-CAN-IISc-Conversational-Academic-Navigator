import openai
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from config.settings import FAISS_INDEX_PATH
from langchain_community.embeddings import OpenAIEmbeddings

EVAL_DATA_PATH = "src/eval_scripts/random_samples.json"
OUTPUT_RESULTS_PATH = "src/eval_scripts/retrieval_scores.json"
K = 5
THRESHOLD = 0.75


embedding = OpenAIEmbeddings()
retriever = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)

def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def get_embedding(text):
    return embedding.embed_query(text)

def evaluate_retriever_openai(queries, retriever, k=5, threshold=0.75):
    hit_at_k = []
    mrr_scores = []
    detailed_results = []

    for q in tqdm(queries):
        question = q["question"]
        ground_truth = q["ground_truth"]

        gt_emb = get_embedding(ground_truth)

        docs = retriever.similarity_search(question, k=k)
        doc_embeddings = [get_embedding(doc.page_content) for doc in docs]
        doc_texts = [doc.page_content for doc in docs]

        similarities = [cosine_sim(gt_emb, doc_emb) for doc_emb in doc_embeddings]

        hit = any(score >= threshold for score in similarities)
        hit_at_k.append(int(hit))

        mrr = 0
        for rank, score in enumerate(similarities):
            if score >= threshold:
                mrr = 1 / (rank + 1)
                break
        mrr_scores.append(mrr)

        detailed_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_docs": doc_texts,
            "similarities": [round(s, 4) for s in similarities],
            "hit": hit,
            "mrr": round(mrr, 4)
        })

    summary = {
        "hit@k": round(np.mean(hit_at_k), 4),
        "mrr@k": round(np.mean(mrr_scores), 4),
        "k": k,
        "threshold": threshold
    }

    result_json = {
        "summary": summary,
        "samples": detailed_results
    }

    with open(OUTPUT_RESULTS_PATH, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\nEvaluation saved to {OUTPUT_RESULTS_PATH}")
    print(f"Hit@{k}: {summary['hit@k']:.3f} | MRR@{k}: {summary['mrr@k']:.3f}")

with open(EVAL_DATA_PATH, "r") as f:
    queries = json.load(f)

evaluate_retriever_openai(queries, retriever)
