i mport logging
import re
import numpy as np
import requests
import json
import pandas as pd
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
import faiss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

###############################################################################
# LOGGING CONFIG: Log to both console and rag_ko.log
###############################################################################
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        # FileHandler: writes log messages to 'rag_ko.log'
        logging.FileHandler("rag_ko.log", mode="w"),
        # StreamHandler: writes log messages to the console
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
# FORCE CPU (REMOVE MPS/GPU USAGE)
###############################################################################
device = torch.device("cpu")
logger.info("Using CPU only.")

# Ensure NLTK data is available
nltk.download('punkt_tab')

# Import METEOR score and ROUGE scorer
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

###############################################################################
# 1. EXCEL LOADING
###############################################################################
def load_disease_definitions(excel_path):
    """
    Excel with 'title' (질병명) and 'definition' (정의)
    -> returns list of { "title":..., "text":... }
    """
    df = pd.read_excel(excel_path)  # requires openpyxl
    corpus = []
    for _, row in df.iterrows():
        disease = str(row["title"]).strip()
        definition = str(row["definition"]).strip()
        corpus.append({
            "title": disease,
            "text": definition
        })
    return corpus

###############################################################################
# 2. TEXT CLEANING (OPTIONAL)
###############################################################################
def clean_korean_text(txt):
    """
    Remove bracketed references and collapse whitespace.
    """
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

###############################################################################
# 3. EMBEDDINGS + FAISS
###############################################################################
def create_ko_embeddings(corpus, model):
    """
    Embed the 'text' field of each document in corpus, returning float32 CPU arrays.
    """
    texts = [clean_korean_text(c["text"]) for c in corpus]
    logger.info(f"{len(texts)}개 문서 임베딩 생성 중...")

    # By specifying convert_to_tensor=True on CPU, we get torch tensors on CPU:
    emb_torch = model.encode(texts, show_progress_bar=False, convert_to_tensor=True).to(device)
    # Convert them to float32 CPU NumPy arrays for FAISS:
    emb_cpu = emb_torch.cpu().numpy().astype("float32")

    logger.info(f"임베딩 shape: {emb_cpu.shape}")
    return emb_cpu

def build_faiss_index(embeddings):
    """
    Build a FAISS index using L2 distance on CPU.
    """
    dim = embeddings.shape[1]
    logger.info(f"FAISS 인덱스 생성 (차원={dim})")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"인덱스 size={index.ntotal}")
    return index

###############################################################################
# 4. RAG QUERY (KOREAN)
###############################################################################
def retrieve_definitions_ko(query, index, corpus, model, top_k=3):
    """
    Retrieve the top_k definition texts based on the query, CPU-based.
    """
    query_emb = model.encode(query, convert_to_tensor=False).astype("float32")  # for FAISS
    distances, ids = index.search(np.array([query_emb]), top_k)

    retrieved = []
    for dist, idx_ in zip(distances[0], ids[0]):
        retrieved.append(corpus[idx_]["text"])
    return retrieved

def build_prompt_ko(query, retrieved_texts):
    system_msg = (
        "당신은 한국어 질병 정보를 제공하는 유용한 어시스턴트입니다. "
        "반드시 한국어로만 답변을 작성하세요. "
        "아래 병(질병) 정의를 참고하여 질문에 성실히 답변해 주세요. "
        "정보가 없으면 '모른다'고 말하세요.\n"
    )
    context_str = "\n\n".join(f"[문맥]\n{txt}" for txt in retrieved_texts)
    final_prompt = f"{system_msg}\n{context_str}\n\n사용자 질문: {query}\n답변(한국어로):"
    return final_prompt

def query_rag_ko(query, index, corpus, model, api_url, top_k=3):
    logger.info(f"[RAG-KO] 질의: {query}")
    top_defs = retrieve_definitions_ko(query, index, corpus, model, top_k=top_k)
    if not top_defs:
        # No retrieval
        final_prompt = (
            "당신은 한국어 질병 정보를 제공하는 유용한 어시스턴트입니다. "
            "현재 문맥(정의)가 없습니다.\n"
            f"질문: {query}\n답변(한국어로):"
        )
    else:
        final_prompt = build_prompt_ko(query, top_defs)

    payload = {
        "model": "llama2:7b-chat",  # 예시 모델 이름
        "prompt": final_prompt,
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        with requests.post(api_url, json=payload, stream=True) as resp:
            if resp.status_code == 200:
                output = ""
                for line in resp.iter_lines():
                    if line:
                        try:
                            decoded = line.decode('utf-8')
                            data = json.loads(decoded)
                            if "response" in data:
                                output += data["response"]
                            if data.get("done"):
                                return output
                        except json.JSONDecodeError:
                            logger.warning("JSON 디코드 오류: %s", line)
            else:
                logger.error(f"RAG-KO LLM 오류 {resp.status_code}: {resp.text}")
                return None
    except requests.exceptions.ConnectionError as ce:
        logger.error(f"RAG-KO 연결 오류: {ce}")
        return None

###############################################################################
# 5. NO-RAG QUERY (KOREAN)
###############################################################################
def query_no_rag_ko(query, api_url):
    system_msg = (
        "당신은 한국어 질병 정보를 제공하는 어시스턴트입니다. "
        "추가 정의 없이 질문만 보고 한국어로 답변해 주세요.\n"
    )
    final_prompt = f"{system_msg}\n질문: {query}\n답변(한국어로):"

    payload = {
        "model": "llama2:7b-chat",
        "prompt": final_prompt,
        "temperature": 0.7,
        "max_tokens": 200
    }
    try:
        with requests.post(api_url, json=payload, stream=True) as resp:
            if resp.status_code == 200:
                output = ""
                for line in resp.iter_lines():
                    if line:
                        try:
                            decoded = line.decode('utf-8')
                            data = json.loads(decoded)
                            if "response" in data:
                                output += data["response"]
                            if data.get("done"):
                                return output
                        except json.JSONDecodeError:
                            logger.warning("JSON 디코드 오류: %s", line)
            else:
                logger.error(f"No-RAG-KO 오류 {resp.status_code}: {resp.text}")
                return None
    except requests.exceptions.ConnectionError as ce:
        logger.error(f"No-RAG-KO 연결 오류: {ce}")
        return None

###############################################################################
# 6. NEW: ROUGE, METEOR, and Semantic Similarity Functions
###############################################################################
def compute_rouge(reference, hypothesis):
    """
    Compute the ROUGE-L F1 score between reference and hypothesis.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def compute_meteor(reference, hypothesis):
    """
    Compute the METEOR score between reference and hypothesis.
    Pre-tokenizes the texts using .split().
    """
    return meteor_score([reference.split()], hypothesis.split())

def compute_sem_score(reference, hypothesis, model):
    """
    Compute semantic similarity (cosine similarity) between the embeddings
    of the reference and hypothesis on CPU.
    """
    ref_emb = model.encode(reference, convert_to_tensor=True).to(device)
    hyp_emb = model.encode(hypothesis, convert_to_tensor=True).to(device)
    return util.cos_sim(ref_emb, hyp_emb).item()

###############################################################################
# 7. VALIDATION WITH BLEU, ROUGE, METEOR, SEMSCORE
###############################################################################
def validate_rag_ko(
    test_data,
    index,
    corpus,
    model,
    api_url,
    top_k=3,
    sim_threshold=0.7,
    compute_bleu=True
):
    """
    RAG-KO 방식 검증: 각 테스트 항목에 대해
      - LLM 응답을 받아 cosine similarity, BLEU, ROUGE, METEOR, Semantic score 계산.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothing = SmoothingFunction().method1

    results = []
    y_true = []
    y_pred = []
    bleu_scores = []
    rouge_scores = []
    meteor_scores_ = []
    sem_scores_ = []

    logger.info("RAG-KO 검증 (cos_sim + BLEU + ROUGE + METEOR + Semantic) ...")
    for item in test_data:
        q = item["query"]
        exp = item["expected_answer"]
        y_true.append(1)

        rag_ans = query_rag_ko(q, index, corpus, model, api_url, top_k=top_k)
        if rag_ans:
            sim_val = util.cos_sim(
                model.encode(exp, convert_to_tensor=True).to(device),
                model.encode(rag_ans, convert_to_tensor=True).to(device)
            ).item()
            pred_label = 1 if sim_val >= sim_threshold else 0

            # BLEU score (naive tokenization)
            if compute_bleu:
                ref_tokens = nltk.word_tokenize(exp)
                hyp_tokens = nltk.word_tokenize(rag_ans)
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            else:
                bleu = 0.0

            rouge_val = compute_rouge(exp, rag_ans)
            meteor_val = compute_meteor(exp, rag_ans)
            sem_val = compute_sem_score(exp, rag_ans, model)
        else:
            sim_val = 0.0
            pred_label = 0
            bleu = 0.0
            rouge_val = 0.0
            meteor_val = 0.0
            sem_val = 0.0

        y_pred.append(pred_label)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_val)
        meteor_scores_.append(meteor_val)
        sem_scores_.append(sem_val)
        results.append({
            "query": q,
            "expected_answer": exp,
            "generated_response": rag_ans if rag_ans else "",
            "similarity": sim_val,
            "label": pred_label,
            "bleu": bleu,
            "rouge": rouge_val,
            "meteor": meteor_val,
            "semantic": sem_val
        })

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_ = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    mean_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
    mean_meteor = np.mean(meteor_scores_) if meteor_scores_ else 0.0
    mean_sem = np.mean(sem_scores_) if sem_scores_ else 0.0

    logger.info(f"[RAG-KO] 정밀도={prec:.3f}, 재현율={rec:.3f}, F1={f1_:.3f}, 정확도={acc:.3f}, "
                f"평균BLEU={mean_bleu:.3f}, 평균ROUGE={mean_rouge:.3f}, "
                f"평균METEOR={mean_meteor:.3f}, 평균Semantic={mean_sem:.3f}")
    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1_,
        "accuracy": acc,
        "mean_bleu": mean_bleu,
        "mean_rouge": mean_rouge,
        "mean_meteor": mean_meteor,
        "mean_semantic": mean_sem,
        "results": results
    }

def validate_no_rag_ko(
    test_data,
    model,
    api_url,
    sim_threshold=0.7,
    compute_bleu=True
):
    """
    NO-RAG-KO 방식 검증: LLM에 직접 질의하고 BLEU, ROUGE, METEOR, Semantic score 계산.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothing = SmoothingFunction().method1

    results = []
    y_true = []
    y_pred = []
    bleu_scores = []
    rouge_scores = []
    meteor_scores_ = []
    sem_scores_ = []

    logger.info("NO-RAG-KO 검증 (cos_sim + BLEU + ROUGE + METEOR + Semantic) ...")
    for item in test_data:
        q = item["query"]
        exp = item["expected_answer"]
        y_true.append(1)

        direct_ans = query_no_rag_ko(q, api_url)
        if direct_ans:
            sim_val = util.cos_sim(
                model.encode(exp, convert_to_tensor=True).to(device),
                model.encode(direct_ans, convert_to_tensor=True).to(device)
            ).item()
            pred_label = 1 if sim_val >= sim_threshold else 0

            if compute_bleu:
                ref_tokens = nltk.word_tokenize(exp)
                hyp_tokens = nltk.word_tokenize(direct_ans)
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            else:
                bleu = 0.0

            rouge_val = compute_rouge(exp, direct_ans)
            meteor_val = compute_meteor(exp, direct_ans)
            sem_val = compute_sem_score(exp, direct_ans, model)
        else:
            sim_val = 0.0
            pred_label = 0
            bleu = 0.0
            rouge_val = 0.0
            meteor_val = 0.0
            sem_val = 0.0

        y_pred.append(pred_label)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_val)
        meteor_scores_.append(meteor_val)
        sem_scores_.append(sem_val)
        results.append({
            "query": q,
            "expected_answer": exp,
            "generated_response": direct_ans if direct_ans else "",
            "similarity": sim_val,
            "label": pred_label,
            "bleu": bleu,
            "rouge": rouge_val,
            "meteor": meteor_val,
            "semantic": sem_val
        })

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_ = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    mean_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
    mean_meteor = np.mean(meteor_scores_) if meteor_scores_ else 0.0
    mean_sem = np.mean(sem_scores_) if sem_scores_ else 0.0

    logger.info(f"[NO-RAG-KO] 정밀도={prec:.3f}, 재현율={rec:.3f}, F1={f1_:.3f}, 정확도={acc:.3f}, "
                f"평균BLEU={mean_bleu:.3f}, 평균ROUGE={mean_rouge:.3f}, "
                f"평균METEOR={mean_meteor:.3f}, 평균Semantic={mean_sem:.3f}")
    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1_,
        "accuracy": acc,
        "mean_bleu": mean_bleu,
        "mean_rouge": mean_rouge,
        "mean_meteor": mean_meteor,
        "mean_semantic": mean_sem,
        "results": results
    }

###############################################################################
# 8. MAIN
###############################################################################
if __name__ == "__main__":
    excel_path = "snuh_medical_data.xlsx"  # 예시 (title, definition 컬럼)
    api_url = "http://localhost:11434/api/generate"  # 로컬 LLM 엔드포인트

    logger.info("엑셀 로드 중...")
    corpus = load_disease_definitions(excel_path)
    if not corpus:
        logger.error("엑셀에서 질병 정의를 불러오지 못함. 종료.")
        exit(1)

    logger.info("한국어 임베딩 모델 로딩 (CPU 전용)...")
    model_name = "jhgan/ko-sroberta-multitask"  # 예시 모델
    # Force CPU use explicitly when loading the model:
    embedding_model = SentenceTransformer(model_name, device=str(device))

    # 임베딩 및 FAISS 인덱스 생성
    embeddings = create_ko_embeddings(corpus, embedding_model)
    index = build_faiss_index(embeddings)

    # 예시 테스트 데이터 (10개 정도)
    test_data = [
        {
            "query": "ABO 신생아용혈성질환이 궁금합니다.",
            "expected_answer": "혈액형이 O형인 임신부가 A형 또는 B형인 아기를..."
        },
        {
            "query": "A형 간염에 대해 설명해 주세요.",
            "expected_answer": "간염 바이러스의 한 종류인 A형 간염 바이러스(HAV)에 의해..."
        },
        {
            "query": "B형 간염은 어떤 병인가요?",
            "expected_answer": "B형 간염은 B형 간염 바이러스(HBV)에 감염된 경우 간에 염증이..."
        },
        {
            "query": "C형 간염이 무엇인지 알고 싶어요.",
            "expected_answer": "C형 간염은 C형 간염 바이러스(HCV)에 감염되어 발생하는 간염..."
        },
        {
            "query": "D형 간염, 어떤 질환인가요?",
            "expected_answer": "D형 간염은 D형 간염 바이러스(HDV)에 감염되어 발생하는 간염이며..."
        },
        {
            "query": "Rh 신생아 용혈성 질환이 궁금합니다.",
            "expected_answer": "Rh 음성 여성이 Rh 양성 아기를 임신했을 때 엄마의 항-D 항체가 적혈구..."
        },
        {
            "query": "WPW 증후군이 뭔가요?",
            "expected_answer": "부전도로(accessory pathway)를 통한 조기 흥분 증후군을 WPW 증후군이라 부른다..."
        },
        {
            "query": "가스 괴저병은 어떤 질병인가요?",
            "expected_answer": "클로스트리디움 종 세균이 근육층을 침범하여 조직을 괴사시키고 가스를..."
        },
        {
            "query": "가와사키병 설명 부탁드립니다.",
            "expected_answer": "소아에서 발생하는 원인 불명의 급성 열성 혈관염으로, 전신에 다양하게..."
        },
        {
            "query": "각기병이 뭔지 알려주세요.",
            "expected_answer": "각기병(beriberi)은 비타민 B1(티아민) 결핍으로 인해..."
        }
    ]

    # 1) Validate RAG-KO (BLEU, ROUGE, METEOR, Semantic)
    rag_metrics = validate_rag_ko(
        test_data,
        index,
        corpus,
        embedding_model,
        api_url,
        top_k=3,
        sim_threshold=0.7,
        compute_bleu=True
    )

    # 2) Validate NO-RAG-KO (BLEU, ROUGE, METEOR, Semantic)
    no_rag_metrics = validate_no_rag_ko(
        test_data,
        embedding_model,
        api_url,
        sim_threshold=0.7,
        compute_bleu=True
    )

    # 결과 출력
    logger.info("=== RAG-KO METRICS ===")
    logger.info(f"Precision={rag_metrics['precision']:.3f}, Recall={rag_metrics['recall']:.3f}, "
                f"F1={rag_metrics['f1_score']:.3f}, Accuracy={rag_metrics['accuracy']:.3f}, "
                f"Mean BLEU={rag_metrics['mean_bleu']:.3f}, Mean ROUGE={rag_metrics['mean_rouge']:.3f}, "
                f"Mean METEOR={rag_metrics['mean_meteor']:.3f}, Mean Semantic={rag_metrics['mean_semantic']:.3f}")

    logger.info("=== NO-RAG-KO METRICS ===")
    logger.info(f"Precision={no_rag_metrics['precision']:.3f}, Recall={no_rag_metrics['recall']:.3f}, "
                f"F1={no_rag_metrics['f1_score']:.3f}, Accuracy={no_rag_metrics['accuracy']:.3f}, "
                f"Mean BLEU={no_rag_metrics['mean_bleu']:.3f}, Mean ROUGE={no_rag_metrics['mean_rouge']:.3f}, "
                f"Mean METEOR={no_rag_metrics['mean_meteor']:.3f}, Mean Semantic={no_rag_metrics['mean_semantic']:.3f}")

    # 엑셀로 저장
    rag_df = pd.DataFrame(rag_metrics["results"])
    rag_df.to_excel("rag_ko_bleu_results.xlsx", index=False)

    no_rag_df = pd.DataFrame(no_rag_metrics["results"])
    no_rag_df.to_excel("no_rag_ko_bleu_results.xlsx", index=False)

    logger.info("비교 완료. 'rag_ko_bleu_results.xlsx' / 'no_rag_ko_bleu_results.xlsx' 생성됨.")