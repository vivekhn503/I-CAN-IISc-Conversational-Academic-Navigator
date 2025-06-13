import json
from bert_score import score
import pandas as pd

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def evaluate(data):
    candidates = [item["answer"] for item in data]
    references = [item["ground_truth"] for item in data]

    P, R, F1 = score(candidates, references, lang="en", verbose=True)

    results = []
    for i, item in enumerate(data):
        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": item["ground_truth"],
            "bertscore_f1": round(F1[i].item(), 4)
        })

    return results

def main():
    path = "src/validation/eval_samples.json"
    data = load_data(path)
    results = evaluate(data)

    df = pd.DataFrame(results)
    print(df[["question", "bertscore_f1"]])
    df.to_csv("src/validation/bertscore_results.csv", index=False)
    print("Results saved to `bertscore_results.csv`")

if __name__ == "__main__":
    main()
