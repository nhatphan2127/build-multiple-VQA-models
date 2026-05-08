import numpy as np
import json
import re
import google.generativeai as genai
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import scorer as bert_scorer
from underthesea import word_tokenize  # Thư viện tách từ tiếng Việt

API_KEY = 'AIzaSyCPYialunNcbKahVeTrSguuhZdCfTbEMJc'

def preprocess_vi(text):
    """
    Chuẩn hóa và tách từ tiếng Việt. 
    Ví dụ: "Học sinh học sinh học" -> "Học_sinh học môn sinh_học"
    """
    text = text.lower().strip()
    # Tách từ: "sinh viên" -> "sinh_viên" để BLEU/ROUGE coi là 1 unit
    tokens = word_tokenize(text, format="text")
    return tokens

def calculate_vqa_accuracy(preds, gts):
    """
    Độ chính xác VQA cải tiến: Không phân biệt hoa thường, dấu cách thừa
    """
    score = 0
    for p, g in zip(preds, gts):
        p_clean = p.strip().lower()
        g_clean = g.strip().lower()
        if p_clean == g_clean:
            score += 1
    return score / len(preds) if len(preds) > 0 else 0

from collections import defaultdict

def calculate_metrics(preds, gts, all_question, types):
    # 1. Tiền xử lý: Tách từ tiếng Việt
    preds_vn = [preprocess_vi(p) for p in preds]
    gts_vn = [preprocess_vi(g) for g in gts]

    # --- Tính toán score cho từng mẫu ---
    
    # BLEU
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for p, g in zip(preds_vn, gts_vn):
        score = sentence_bleu([g.split()], p.split(), smoothing_function=smoothie)
        bleu_scores.append(score)

    # ROUGE-L
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = []
    for p, g in zip(preds_vn, gts_vn):
        rouge_scores.append(r_scorer.score(g, p)['rougeL'].fmeasure)

    # BERTScore
    b_scorer = bert_scorer.BERTScorer(lang="vi", rescale_with_baseline=False)
    P, R, F1 = b_scorer.score(preds, gts)
    bert_scores = F1.tolist()

    # VQA Accuracy (Exact Match cho từng câu)
    vqa_acc_scores = []
    for p, g in zip(preds, gts):
        vqa_acc_scores.append(1.0 if p.strip().lower() == g.strip().lower() else 0.0)

    # LLM Judge (Trả về list scores)
    llm_scores = llm_judge(preds=preds, gts=gts, questions=all_question, api_key=API_KEY)

    # --- Gom nhóm theo Type ---
    type_results = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(preds)):
        t = types[i]
        type_results[t]['bleu'].append(bleu_scores[i])
        type_results[t]['rougeL'].append(rouge_scores[i])
        type_results[t]['bert_score'].append(bert_scores[i])
        type_results[t]['vqa_accuracy'].append(vqa_acc_scores[i])
        type_results[t]['llm_judge'].append(llm_scores[i])

    # --- Tính toán trung bình ---
    final_metrics = {"per_type": {}, "overall": {}}
    
    # List để tính overall
    all_metrics_flat = defaultdict(list)

    for t, scores_dict in type_results.items():
        final_metrics["per_type"][t] = {}
        for metric_name, values in scores_dict.items():
            avg_val = np.mean(values)
            final_metrics["per_type"][t][avg_val_name := f"avg_{metric_name}"] = avg_val
            all_metrics_flat[metric_name].extend(values)
            
        print(f"Type {t}: Acc: {final_metrics['per_type'][t]['avg_vqa_accuracy']:.4f}, BLEU: {final_metrics['per_type'][t]['avg_bleu']:.4f}")

    # Tính Overall (Trung bình của tất cả các mẫu)
    for metric_name, values in all_metrics_flat.items():
        final_metrics["overall"][f"avg_{metric_name}"] = np.mean(values)

    return final_metrics

def llm_judge_batch(batch_preds, batch_gts, batch_questions, api_key, model_name="gemini-3-flash-preview"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Constructing a structured prompt for the batch
    samples_text = ""
    for i, (p, g, q) in enumerate(zip(batch_preds, batch_gts, batch_questions)):
        samples_text += f"ID: {i}\nQ: {q}\nGT: {g}\nPred: {p}\n---\n"

    prompt = f"""
    Bạn là một giám khảo VQA chuyên nghiệp. Hãy chấm điểm các cặp câu trả lời sau trên thang điểm từ 0 đến 10.
    
    Tiêu chí:
    1. Chính xác so với Ground Truth (GT).
    2. Tự nhiên và dễ hiểu.

    Dữ liệu cần chấm điểm:
    {samples_text}

    Yêu cầu: Trả về kết quả dưới dạng JSON list các con số, ví dụ: [8.5, 5.0, 10.0]. 
    Không giải thích, chỉ trả về JSON array.
    """

    try:
        response = model.generate_content(prompt)
        # Extracting the JSON list using regex
        match = re.search(r"\[.*\]", response.text.replace("\n", ""))
        if match:
            scores = json.loads(match.group())
            return scores
        else:
            print("Could not parse JSON from response.")
            return [0.0] * len(batch_preds)
    except Exception as e:
        print(f"Error: {e}")
        return [0.0] * len(batch_preds)
    

def llm_judge(preds, gts, questions, api_key, batch_size=100):
    all_scores = []
    
    for i in range(0, len(preds), batch_size):
        b_preds = preds[i : i + batch_size]
        b_gts = gts[i : i + batch_size]
        b_qs = questions[i : i + batch_size]
        
        print(f"Processing batch {i // batch_size + 1}...")
        batch_scores = llm_judge_batch(b_preds, b_gts, b_qs, api_key)
        all_scores.extend(batch_scores)
        time.sleep(10)
        
    return all_scores
