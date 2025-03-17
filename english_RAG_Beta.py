###############################################################################
# validation_beta2_fixed.py
# -------------------------
# Example of using MPS on Apple Silicon for RAG pipeline, with fallback enabled
###############################################################################

import os
# 1) Enable MPS fallback if any ops aren't fully supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
import re
import numpy as np
import requests
import json
import pandas as pd
import torch

# Sentence Transformers for embeddings and similarity computations
from sentence_transformers import SentenceTransformer, util
import faiss
import wikipediaapi
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import NLTK for BLEU and METEOR score calculation.
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')

# Import ROUGE scorer
from rouge_score import rouge_scorer

###############################################################################
# DETECT AND SET THE TORCH DEVICE (MPS IF AVAILABLE)
###############################################################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU.")

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("english_log.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
# 1) Fetch Wikipedia Articles
###############################################################################
def fetch_wikipedia_articles(page_titles):
    """
    Fetch Wikipedia pages using the wikipediaapi module.
    Returns a list of dictionaries with page title and page object.
    """
    logger.info("Initializing Wikipedia API (English).")
    wiki = wikipediaapi.Wikipedia(language='en', user_agent="MyRAGApp/1.0")
    articles = []
    for t in page_titles:
        logger.info(f"Fetching page: '{t}'")
        page_py = wiki.page(t)
        if page_py.exists():
            logger.info(f"Page '{t}' found.")
            articles.append({"title": t, "page_obj": page_py})
        else:
            logger.warning(f"Page '{t}' does not exist.")
    return articles

###############################################################################
# 2) Clean and Sentence Split
###############################################################################
def clean_wikipedia_text(text):
    """
    Remove bracketed references and extra whitespace from text.
    """
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def naive_sentence_split(text):
    """
    A naive sentence splitting: split on period followed by whitespace.
    """
    sents = re.split(r"\.\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

###############################################################################
# 3) Near-Duplicate Filter
###############################################################################
def near_duplicate_filter(sentences, model, sim_threshold=0.9):
    """
    Filter out near-duplicate sentences based on cosine similarity.
    Ensures all embeddings go to the same device (MPS or CPU).
    """
    accepted_sents = []
    accepted_embs = []
    for s in sentences:
        # Force embeddings onto the correct device:
        s_emb = model.encode(s, convert_to_tensor=True).to(device)

        if not accepted_embs:
            accepted_sents.append(s)
            accepted_embs.append(s_emb)
        else:
            # Stack the accepted embeddings, which are also on 'device':
            stacked_embs = torch.stack(accepted_embs)
            sims = util.cos_sim(s_emb, stacked_embs)[0]
            max_sim = float(sims.max().item())
            if max_sim < sim_threshold:
                accepted_sents.append(s)
                accepted_embs.append(s_emb)

    return accepted_sents

###############################################################################
# 4) Build Sentence Corpus for a Wikipedia Article
###############################################################################
def build_sentence_corpus(page_obj, model, sim_threshold=0.9):
    """
    Processes a Wikipedia page by cleaning text, splitting into sentences,
    and filtering near-duplicates.
    """
    logger.info(f"Processing page '{page_obj.title}'")
    full_text = clean_wikipedia_text(page_obj.text)
    raw_sents = naive_sentence_split(full_text)
    logger.info(f"Page '{page_obj.title}' => {len(raw_sents)} raw sentences.")
    dedup_sents = near_duplicate_filter(raw_sents, model, sim_threshold=sim_threshold)
    logger.info(f"After near-dup filter => {len(dedup_sents)} sentences remain.")
    corpus = []
    for s in dedup_sents:
        corpus.append({
            "title": page_obj.title,
            "sentence_text": s
        })
    return corpus

###############################################################################
# 5) Create Sentence Embeddings and Build a FAISS Index
###############################################################################
def create_sentence_embeddings(sentence_corpus, model):
    """
    Encode a list of sentences into embeddings using the provided model.
    Returns float32 numpy arrays for FAISS indexing (CPU-based).
    """
    texts = [sc["sentence_text"] for sc in sentence_corpus]
    logger.info(f"Encoding {len(texts)} sentences.")
    # We encode to a torch tensor, then move to CPU .numpy() for FAISS:
    emb_torch = model.encode(texts, show_progress_bar=False, convert_to_tensor=True)
    emb_torch = emb_torch.to("cpu")  # FAISS expects CPU-based NumPy array
    emb = emb_torch.numpy().astype("float32")
    logger.info(f"Embeddings shape: {emb.shape}")
    return emb

def build_faiss_index(embeddings):
    """
    Build a FAISS index (L2 distance) using the sentence embeddings (CPU).
    """
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index of dim={dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"Index built, size={index.ntotal}")
    return index

###############################################################################
# 6) Multi-sentence Retrieval for Dataset Generation
###############################################################################
def retrieve_best_sentences(query, local_index, local_corpus, model,
                            top_k=5, top_n=3, min_sim=0.4):
    """
    Retrieve the best matching sentences for a given query.
    1) We do a FAISS search with CPU-based embeddings.
    2) We re-rank top results with Torch-based cos_sim on device.
    """
    # First step: encode the query for FAISS (CPU-based).
    # We can do a separate embedding that stays as a NumPy float32 array:
    query_emb_cpu = model.encode(query, convert_to_tensor=False)
    query_emb_cpu = query_emb_cpu.astype("float32")

    # Search top_k in FAISS:
    distances, ids = local_index.search(np.array([query_emb_cpu]), top_k)

    candidate_sents = []
    for dist, idx_ in zip(distances[0], ids[0]):
        candidate_sents.append(local_corpus[idx_]["sentence_text"])
    if not candidate_sents:
        return ""

    # Second step: re-encode on the MPS/CPU device for cos_sim:
    cand_embs = model.encode(candidate_sents, convert_to_tensor=True).to(device)
    q_emb2 = model.encode(query, convert_to_tensor=True).to(device)

    sims = util.cos_sim(q_emb2, cand_embs)[0]
    sorted_idxs = torch.argsort(sims, descending=True)
    final_sentences = []
    for idx in sorted_idxs:
        i_ = idx.item()
        if float(sims[i_].item()) >= min_sim:
            final_sentences.append(candidate_sents[i_])
            if len(final_sentences) >= top_n:
                break

    if final_sentences:
        return ". ".join(final_sentences)
    else:
        return ""

###############################################################################
# 7) Generate Dataset with Multi-line Expected Answers
###############################################################################
def generate_dataset(articles, sentence_corpus_map, sentence_index_map,
                     model, num_questions=5, top_k=5, top_n=3, min_sim=0.4):
    """
    For each article, generate question-answer pairs.
    """
    question_templates = [
        "What is the definition of {topic}?"
    ]
    dataset = []
    logger.info("Generating dataset with multi-sentence approach.")
    for art in articles:
        title = art["title"]
        local_index = sentence_index_map[title]
        local_corpus = sentence_corpus_map[title]
        for template in question_templates[:num_questions]:
            query = template.format(topic=title)
            snippet = retrieve_best_sentences(
                query,
                local_index,
                local_corpus,
                model,
                top_k=top_k,
                top_n=top_n,
                min_sim=min_sim
            )
            dataset.append({
                "query": query,
                "expected_answer": snippet,
                "article_title": title
            })
    logger.info(f"Dataset created with {len(dataset)} Q/A pairs.")
    return dataset

###############################################################################
# 8) "No RAG" Query
###############################################################################
def query_no_rag(query, api_url):
    """
    Query the local LLM endpoint without any retrieval augmentation.
    """
    system_msg = "You are a helpful assistant. Please answer the user's question directly.\n"
    final_prompt = f"{system_msg}\nUser question: {query}\nAnswer:"
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
                            logger.warning("JSON decode error: %s", line)
            else:
                logger.error(f"No-RAG LLM error {resp.status_code}: {resp.text}")
                return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"No-RAG connection error: {e}")
        return None

###############################################################################
# 9) Multi-sentence RAG Query
###############################################################################
def build_prompt(query, top_sents):
    """
    Build the prompt for the RAG approach by incorporating context sentences.
    """
    context_str = "\n\n".join(f"[Context]\n{c['text']}" for c in top_sents)
    system_msg = (
        "You are a helpful assistant. Use these context sentences to answer the user's question. "
        "If not found, say so.\n"
    )
    final_prompt = f"{system_msg}\n{context_str}\n\nUser question: {query}\nAnswer:"
    return final_prompt

def query_rag(query, combined_index, combined_corpus, model, api_url,
              top_k=3, top_n=3, min_sim=0.4):
    """
    Perform a Retrieval-Augmented Generation (RAG) query.
    """
    logger.info(f"[RAG] Query: {query}")

    # 1) Search in FAISS with CPU embeddings
    query_emb_cpu = model.encode(query, convert_to_tensor=False).astype("float32")
    distances, ids = combined_index.search(np.array([query_emb_cpu]), top_k)

    candidate_sents = []
    for dist, idx_ in zip(distances[0], ids[0]):
        candidate_sents.append(combined_corpus[idx_]["sentence_text"])

    if not candidate_sents:
        final_prompt = f"You are a helpful assistant. No context.\nUser question: {query}\nAnswer:"
    else:
        # 2) Re-rank on MPS or CPU device via cos_sim:
        cand_embs = model.encode(candidate_sents, convert_to_tensor=True).to(device)
        q_emb2 = model.encode(query, convert_to_tensor=True).to(device)
        sims = util.cos_sim(q_emb2, cand_embs)[0]
        sorted_idxs = torch.argsort(sims, descending=True)

        top_contexts = []
        for idx__ in sorted_idxs:
            i_ = idx__.item()
            s_ = candidate_sents[i_]
            sim_ = float(sims[i_].item())
            if sim_ >= min_sim:
                top_contexts.append({"text": s_, "similarity": sim_})
                if len(top_contexts) >= top_n:
                    break
        final_prompt = build_prompt(query, top_contexts)

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
                            logger.warning("JSON decode error: %s", line)
            else:
                logger.error(f"RAG LLM error {resp.status_code}: {resp.text}")
                return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"RAG connection error: {e}")
        return None

###############################################################################
# 10) Compute ROUGE, METEOR, and Semantic Similarity Scores
###############################################################################
def compute_rouge(reference, hypothesis):
    """
    Compute the ROUGE-L F1 score between reference and hypothesis.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def compute_sem_score(reference, hypothesis, model):
    """
    Compute a semantic similarity score using cosine similarity between embeddings.
    """
    # Encode on same device for cos_sim:
    ref_emb = model.encode(reference, convert_to_tensor=True).to(device)
    hyp_emb = model.encode(hypothesis, convert_to_tensor=True).to(device)
    return util.cos_sim(ref_emb, hyp_emb).item()

###############################################################################
# 11) Validation: RAG vs. NO-RAG (with BLEU, ROUGE, METEOR, SemScore)
###############################################################################
def validate_rag(test_data, combined_index, combined_corpus, model, api_url,
                 top_k=3, top_n=3, min_sim=0.4, sim_threshold=0.7):
    results = []
    y_true = []
    y_pred = []
    bleu_scores = []
    rouge_scores_list = []
    meteor_scores_list = []
    sem_scores_list = []

    logger.info("Validating: RAG multi-sentence approach.")
    smoothing = SmoothingFunction().method1

    for row in test_data:
        query = row["query"]
        exp = row["expected_answer"]
        y_true.append(1)

        rag_ans = query_rag(query, combined_index, combined_corpus, model, api_url,
                            top_k=top_k, top_n=top_n, min_sim=min_sim)
        if rag_ans:
            # Compare embeddings on device:
            sim = util.cos_sim(
                model.encode(exp, convert_to_tensor=True).to(device),
                model.encode(rag_ans, convert_to_tensor=True).to(device)
            ).item()
            pred = 1 if sim >= sim_threshold else 0

            bleu = sentence_bleu([exp.split()], rag_ans.split(), smoothing_function=smoothing)
            rouge_val = compute_rouge(exp, rag_ans)
            meteor_val = meteor_score([exp.split()], rag_ans.split())
            sem_val = compute_sem_score(exp, rag_ans, model)
        else:
            sim = 0.0
            pred = 0
            bleu = 0.0
            rouge_val = 0.0
            meteor_val = 0.0
            sem_val = 0.0

        y_pred.append(pred)
        bleu_scores.append(bleu)
        rouge_scores_list.append(rouge_val)
        meteor_scores_list.append(meteor_val)
        sem_scores_list.append(sem_val)
        results.append({
            "query": query,
            "expected_answer": exp,
            "generated_response": rag_ans if rag_ans else "",
            "similarity": sim,
            "label": pred,
            "bleu_score": bleu,
            "rouge_score": rouge_val,
            "meteor_score": meteor_val,
            "semantic_score": sem_val
        })

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_ = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_rouge = np.mean(rouge_scores_list) if rouge_scores_list else 0.0
    avg_meteor = np.mean(meteor_scores_list) if meteor_scores_list else 0.0
    avg_sem = np.mean(sem_scores_list) if sem_scores_list else 0.0

    logger.info(
        f"[RAG] Precision={prec:.3f}, Recall={rec:.3f}, F1={f1_:.3f}, Accuracy={acc:.3f}, "
        f"Average BLEU={avg_bleu:.3f}, Average ROUGE={avg_rouge:.3f}, "
        f"Average METEOR={avg_meteor:.3f}, Average Semantic={avg_sem:.3f}"
    )

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1_,
        "accuracy": acc,
        "average_bleu": avg_bleu,
        "average_rouge": avg_rouge,
        "average_meteor": avg_meteor,
        "average_semantic": avg_sem,
        "results": results
    }

def validate_no_rag(test_data, model, api_url, sim_threshold=0.7):
    results = []
    y_true = []
    y_pred = []
    bleu_scores = []
    rouge_scores_list = []
    meteor_scores_list = []
    sem_scores_list = []

    logger.info("Validating: No-RAG approach.")
    smoothing = SmoothingFunction().method1

    for row in test_data:
        query = row["query"]
        exp = row["expected_answer"]
        y_true.append(1)

        direct_ans = query_no_rag(query, api_url)
        if direct_ans:
            sim = util.cos_sim(
                model.encode(exp, convert_to_tensor=True).to(device),
                model.encode(direct_ans, convert_to_tensor=True).to(device)
            ).item()
            pred = 1 if sim >= sim_threshold else 0

            bleu = sentence_bleu([exp.split()], direct_ans.split(), smoothing_function=smoothing)
            rouge_val = compute_rouge(exp, direct_ans)
            meteor_val = meteor_score([exp.split()], direct_ans.split())
            sem_val = compute_sem_score(exp, direct_ans, model)
        else:
            sim = 0.0
            pred = 0
            bleu = 0.0
            rouge_val = 0.0
            meteor_val = 0.0
            sem_val = 0.0

        y_pred.append(pred)
        bleu_scores.append(bleu)
        rouge_scores_list.append(rouge_val)
        meteor_scores_list.append(meteor_val)
        sem_scores_list.append(sem_val)

        results.append({
            "query": query,
            "expected_answer": exp,
            "generated_response": direct_ans if direct_ans else "",
            "similarity": sim,
            "label": pred,
            "bleu_score": bleu,
            "rouge_score": rouge_val,
            "meteor_score": meteor_val,
            "semantic_score": sem_val
        })

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_ = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_rouge = np.mean(rouge_scores_list) if rouge_scores_list else 0.0
    avg_meteor = np.mean(meteor_scores_list) if meteor_scores_list else 0.0
    avg_sem = np.mean(sem_scores_list) if sem_scores_list else 0.0

    logger.info(
        f"[NO-RAG] Precision={prec:.3f}, Recall={rec:.3f}, F1={f1_:.3f}, Accuracy={acc:.3f}, "
        f"Average BLEU={avg_bleu:.3f}, Average ROUGE={avg_rouge:.3f}, "
        f"Average METEOR={avg_meteor:.3f}, Average Semantic={avg_sem:.3f}"
    )

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1_,
        "accuracy": acc,
        "average_bleu": avg_bleu,
        "average_rouge": avg_rouge,
        "average_meteor": avg_meteor,
        "average_semantic": avg_sem,
        "results": results
    }

###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    # 1) Define Wikipedia pages to fetch.
    pages_to_fetch = [
        "Strategy", "Leadership", "Innovation", "Planning", "Organization",
        "Efficiency", "Coordination", "Analytics", "Automation", "Integration",
        "Governance", "Collaboration", "Security", "Performance", "Transformation",
        "Optimization", "Technology", "Data", "Communication", "Digitization"
    ]
    # Define the local LLM endpoint URL.
    api_url = "http://localhost:11434/api/generate"  # local LLM endpoint

    # 2) Fetch Wikipedia articles.
    articles = fetch_wikipedia_articles(pages_to_fetch)
    if not articles:
        logger.error("No valid Wikipedia pages found, exiting.")
        exit(1)

    # 3) Load the SentenceTransformer model onto the chosen device (MPS or CPU).
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    embedding_model.max_seq_length = 128
    # 4) Build a sentence corpus and FAISS index for each article.
    sentence_corpus_map = {}
    sentence_index_map = {}
    for art in articles:
        title = art["title"]
        page_py = art["page_obj"]
        sent_corpus = build_sentence_corpus(page_py, embedding_model, sim_threshold=0.9)
        sentence_corpus_map[title] = sent_corpus
        emb = create_sentence_embeddings(sent_corpus, embedding_model)
        s_index = build_faiss_index(emb)
        sentence_index_map[title] = s_index

    # 5) Combine all sentences from all articles into one corpus and build a combined FAISS index.
    combined_sentence_corpus = []
    for t, cset in sentence_corpus_map.items():
        combined_sentence_corpus.extend(cset)
    combined_emb = create_sentence_embeddings(combined_sentence_corpus, embedding_model)
    combined_index = build_faiss_index(combined_emb)

    # 6) Generate a dataset of Q/A pairs using the multi-sentence approach.
    test_data = generate_dataset(
        articles,
        sentence_corpus_map,
        sentence_index_map,
        model=embedding_model,
        num_questions=5,
        top_k=5,
        top_n=3,
        min_sim=0.4
    )

    # 7) Validate the RAG approach (includes BLEU, ROUGE, METEOR, and Semantic scores).
    rag_metrics = validate_rag(
        test_data,
        combined_index,
        combined_sentence_corpus,
        embedding_model,
        api_url,
        top_k=3,
        top_n=3,
        min_sim=0.4,
        sim_threshold=0.7
    )

    # 8) Validate the direct (No-RAG) approach.
    no_rag_metrics = validate_no_rag(
        test_data,
        embedding_model,
        api_url,
        sim_threshold=0.7
    )

    # 9) Save both validation results to Excel files for later comparison.
    rag_df = pd.DataFrame(rag_metrics["results"])
    rag_df.to_excel("rag_multi_sents.xlsx", index=False)

    no_rag_df = pd.DataFrame(no_rag_metrics["results"])
    no_rag_df.to_excel("no_rag_multi_sents.xlsx", index=False)

    logger.info("Done. Compare 'rag_multi_sents.xlsx' vs 'no_rag_multi_sents.xlsx'.")