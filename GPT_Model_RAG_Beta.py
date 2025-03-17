import logging
import re
import os
import numpy as np
import json
import pandas as pd
import torch
import openai

import wikipediaapi
import faiss
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import NLTK and its BLEU score functionality for evaluation.
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

###############################################################################
# LOGGING SETUP (for both pipelines)
###############################################################################
# Create a logger instance for the module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the log level to INFO.
# Define the log message format.
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# Create and add a console handler to print logs to the console.
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create and add a file handler to save logs into 'gpt_log.log'.
file_handler = logging.FileHandler("gpt_log.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

###############################################################################
# COMMON FUNCTIONS
###############################################################################

def build_faiss_index(embeddings):
    """
    Build a FAISS index using L2 distance for the provided embeddings.
    Logs the dimension and total number of entries in the index.
    """
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index with dimension {dim}.")
    index = faiss.IndexFlatL2(dim)  # Create a flat (brute-force) L2 index.
    index.add(embeddings)  # Add embeddings to the index.
    logger.info(f"FAISS index built with {index.ntotal} entries.")
    return index

###############################################################################
# OPENAI GPT HELPER (used by both pipelines)
###############################################################################
# Set your OpenAI API key either directly here or via the environment variable.
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def query_gpt(system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=200):
    """
    Make a generic call to OpenAI's ChatCompletion API.
    Passes a system message and a user prompt, and returns the response.
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Return the generated content after stripping any surrounding whitespace.
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI GPT error: {e}")
        return None

###############################################################################
# 1) KOREAN PIPELINE (Using Excel data and GPT)
###############################################################################
def load_korean_excel(excel_path):
    """
    Load an Excel file containing Korean medical data.
    The file is expected to have columns "title" and "definition".
    Returns a list of dictionaries.
    """
    df = pd.read_excel(excel_path)
    corpus = []
    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        definition = str(row["definition"]).strip()
        corpus.append({"title": title, "text": definition})
    return corpus

def clean_korean_text(txt):
    """
    Clean the Korean text by removing bracketed references and extra whitespace.
    """
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def create_ko_embeddings(corpus, model):
    """
    Generate embeddings for a list of Korean texts after cleaning them.
    """
    texts = [clean_korean_text(c["text"]) for c in corpus]
    emb = model.encode(texts, show_progress_bar=False).astype("float32")
    return emb

def retrieve_kor_definitions(query, index, corpus, model, top_k=3):
    """
    Retrieve the top_k most similar Korean definitions from the FAISS index.
    """
    q_emb = model.encode(query).astype("float32")
    distances, ids = index.search(np.array([q_emb]), top_k)
    # Retrieve the corresponding text from the corpus using the returned indices.
    retrieved = [corpus[idx]["text"] for idx in ids[0]]
    return retrieved

def build_kor_prompt(query, retrieved_texts):
    """
    Build a prompt for the Korean pipeline by including a system message,
    the retrieved context texts, and the user's question.
    """
    system_msg = ("당신은 한국어 질병 정보를 제공하는 어시스턴트입니다. "
                  "아래 병(질병) 정의를 참고하여 질문에 한국어로 답변해 주세요.")
    context_str = "\n\n".join(f"[문맥]\n{text}" for text in retrieved_texts)
    user_msg = f"질문: {query}\n답변:"
    return system_msg + "\n" + context_str + "\n\n" + user_msg

def query_rag_kor(query, index, corpus, model, top_k=3):
    """
    Retrieval-augmented generation for the Korean pipeline:
    Retrieve context from the corpus and build a prompt to query GPT.
    """
    ret = retrieve_kor_definitions(query, index, corpus, model, top_k=top_k)
    if not ret:
        # If no context is found, fall back to a prompt without additional information.
        system_prompt = "한국어 질병 정보가 없습니다."
        user_prompt = f"질문: {query}\n답변:"
        return query_gpt(system_prompt, user_prompt)
    else:
        prompt = build_kor_prompt(query, ret)
        system_prompt = "당신은 한국어 질병 정보를 제공하는 어시스턴트입니다."
        return query_gpt(system_prompt, prompt)

def query_no_rag_kor(query):
    """
    Directly query GPT for Korean questions without using retrieval augmentation.
    """
    system_prompt = ("당신은 한국어 질병 정보를 제공하는 어시스턴트입니다. "
                     "추가 정보 없이 질문만 보고 답해주세요.")
    user_prompt = f"질문: {query}\n답변:"
    return query_gpt(system_prompt, user_prompt)

def tokenize_text(sentence):
    """
    Naively tokenize text using nltk.word_tokenize.
    (For Korean, consider using a more robust tokenizer if needed.)
    """
    return nltk.word_tokenize(sentence)

def validate_kor_rag(test_data, index, corpus, model, top_k=3, sim_threshold=0.7):
    """
    Validate the retrieval-augmented generation (RAG) for Korean data.
    For each test item:
      - Query GPT using retrieval augmentation.
      - Compute cosine similarity and BLEU score against the expected answer.
    Returns overall metrics and detailed results.
    """
    smoothing = SmoothingFunction().method1
    results = []
    y_true, y_pred, bleu_scores = [], [], []
    
    for item in test_data:
        query = item["query"]
        exp = item["expected_answer"]
        y_true.append(1)
        gen = query_rag_kor(query, index, corpus, model, top_k=top_k)
        if gen:
            # Compute similarity between expected and generated text.
            sim_val = util.cos_sim(model.encode(exp, convert_to_tensor=True),
                                   model.encode(gen, convert_to_tensor=True)).item()
            pred_label = 1 if sim_val >= sim_threshold else 0
            # Tokenize texts and compute BLEU score.
            ref_tokens = tokenize_text(exp)
            hyp_tokens = tokenize_text(gen)
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        else:
            sim_val, pred_label, bleu = 0.0, 0, 0.0
        y_pred.append(pred_label)
        bleu_scores.append(bleu)
        results.append({
            "query": query,
            "expected_answer": exp,
            "generated_response": gen if gen else "",
            "similarity": sim_val,
            "label": pred_label,
            "bleu": bleu
        })
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "mean_bleu": sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0.0,
        "results": results
    }

def validate_kor_no_rag(test_data, model, sim_threshold=0.7):
    """
    Validate the direct (No-RAG) approach for Korean data.
    For each test item:
      - Query GPT without retrieval augmentation.
      - Compute cosine similarity and BLEU score.
    """
    smoothing = SmoothingFunction().method1
    results = []
    y_true, y_pred, bleu_scores = [], [], []
    
    for item in test_data:
        query = item["query"]
        exp = item["expected_answer"]
        y_true.append(1)
        gen = query_no_rag_kor(query)
        if gen:
            sim_val = util.cos_sim(model.encode(exp, convert_to_tensor=True),
                                   model.encode(gen, convert_to_tensor=True)).item()
            pred_label = 1 if sim_val >= sim_threshold else 0
            ref_tokens = tokenize_text(exp)
            hyp_tokens = tokenize_text(gen)
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        else:
            sim_val, pred_label, bleu = 0.0, 0, 0.0
        y_pred.append(pred_label)
        bleu_scores.append(bleu)
        results.append({
            "query": query,
            "expected_answer": exp,
            "generated_response": gen if gen else "",
            "similarity": sim_val,
            "label": pred_label,
            "bleu": bleu
        })
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "mean_bleu": sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0.0,
        "results": results
    }

###############################################################################
# 2) ENGLISH PIPELINE (Multi-sentence retrieval with GPT)
###############################################################################
# (a) Wikipedia Article Fetching and Text Processing
def fetch_wikipedia_articles(page_titles):
    """
    Fetch Wikipedia pages for the provided titles using wikipediaapi.
    Returns a list of dictionaries with title and page object.
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

def clean_wikipedia_text(text):
    """
    Clean Wikipedia text by removing bracketed references and extra whitespace.
    """
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def naive_sentence_split(text):
    """
    Naively split text into sentences using a period followed by whitespace.
    """
    sents = re.split(r"\.\s+", text)
    return [s.strip() for s in sents if s.strip()]

# (b) Near-Duplicate Filtering and Building a Sentence Corpus
def near_duplicate_filter(sentences, model, sim_threshold=0.9):
    """
    Filter out near-duplicate sentences using cosine similarity.
    Only add a sentence if its maximum similarity with already accepted sentences is below the threshold.
    """
    accepted_sents = []
    accepted_embs = []
    for s in sentences:
        s_emb = model.encode(s, convert_to_tensor=True)
        if not accepted_embs:
            accepted_sents.append(s)
            accepted_embs.append(s_emb)
        else:
            sims = util.cos_sim(s_emb, torch.stack(accepted_embs))[0]
            if float(sims.max().item()) < sim_threshold:
                accepted_sents.append(s)
                accepted_embs.append(s_emb)
    return accepted_sents

def build_sentence_corpus(page_obj, model, sim_threshold=0.9):
    """
    Process a Wikipedia page:
      - Clean the text.
      - Split it into sentences.
      - Remove near-duplicate sentences.
    Returns a list of dictionaries with title and sentence text.
    """
    logger.info(f"Processing page '{page_obj.title}'")
    full_text = clean_wikipedia_text(page_obj.text)
    raw_sents = naive_sentence_split(full_text)
    logger.info(f"Page '{page_obj.title}' => {len(raw_sents)} raw sentences.")
    dedup_sents = near_duplicate_filter(raw_sents, model, sim_threshold=sim_threshold)
    logger.info(f"After near-duplicate filter => {len(dedup_sents)} sentences remain.")
    corpus = [{"title": page_obj.title, "sentence_text": s} for s in dedup_sents]
    return corpus

def create_sentence_embeddings(sentence_corpus, model):
    """
    Create embeddings for each sentence in the corpus using the provided model.
    """
    texts = [sc["sentence_text"] for sc in sentence_corpus]
    logger.info(f"Encoding {len(texts)} sentences.")
    emb = model.encode(texts, show_progress_bar=False).astype("float32")
    logger.info(f"Embeddings shape: {emb.shape}")
    return emb

# (c) Multi-sentence Retrieval and Dataset Generation
def retrieve_best_sentences(query, local_index, local_corpus, model,
                            top_k=5, top_n=3, min_sim=0.4):
    """
    Retrieve the best matching sentences for the query:
      1) Retrieve top_k candidate sentences from FAISS.
      2) Re-rank them by similarity.
      3) Select up to top_n sentences above the min_sim threshold.
      4) Join them into a single snippet.
    """
    # Encode the query and search in the FAISS index.
    query_emb = model.encode(query).astype("float32")
    distances, ids = local_index.search(np.array([query_emb]), top_k)
    candidate_sents = [local_corpus[idx]["sentence_text"] for idx in ids[0]]
    if not candidate_sents:
        return ""
    # Re-encode candidate sentences to compute similarity with the query.
    cand_embs = model.encode(candidate_sents, convert_to_tensor=True)
    q_emb2 = model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(q_emb2, cand_embs)[0]
    sorted_idxs = torch.argsort(sims, descending=True)
    final_sentences = []
    # Select sentences with similarity above the min_sim threshold.
    for idx in sorted_idxs:
        i = idx.item()
        if float(sims[i].item()) >= min_sim:
            final_sentences.append(candidate_sents[i])
            if len(final_sentences) >= top_n:
                break
    return ". ".join(final_sentences) if final_sentences else ""

def generate_dataset(articles, sentence_corpus_map, sentence_index_map,
                     model, num_questions=5, top_k=5, top_n=3, min_sim=0.4):
    """
    For each article, generate question-answer pairs:
      - Create a query using a question template.
      - Retrieve context sentences to form the expected answer.
    Returns a dataset as a list of dictionaries.
    """
    question_templates = [
        "What is the definition of {topic}?"
    ]
    dataset = []
    logger.info("Generating English dataset (multi-sentence approach).")
    for art in articles:
        title = art["title"]
        local_index = sentence_index_map[title]
        local_corpus = sentence_corpus_map[title]
        for template in question_templates[:num_questions]:
            query = template.format(topic=title)
            snippet = retrieve_best_sentences(query, local_index, local_corpus, model,
                                              top_k=top_k, top_n=top_n, min_sim=min_sim)
            dataset.append({
                "query": query,
                "expected_answer": snippet,
                "article_title": title
            })
    logger.info(f"Dataset created with {len(dataset)} Q/A pairs.")
    return dataset

# (d) English Query Functions Using GPT
def query_no_rag_eng(query):
    """
    Query GPT directly (without context) for an English question.
    """
    system_prompt = "You are a helpful assistant with no additional context. Answer in English."
    user_prompt = f"Question: {query}\nAnswer:"
    return query_gpt(system_prompt, user_prompt)

def query_rag_eng(query, combined_index, combined_corpus, model,
                  top_k=3, top_n=3, min_sim=0.4):
    """
    Query GPT using a retrieval-augmented approach:
      - Retrieve context sentences from the combined corpus.
      - Build a prompt that includes the context.
      - Call GPT to generate an answer.
    """
    logger.info(f"[RAG] English query: {query}")
    q_emb = model.encode(query).astype("float32")
    distances, ids = combined_index.search(np.array([q_emb]), top_k)
    candidate_sents = [combined_corpus[idx]["sentence_text"] for idx in ids[0]]
    if not candidate_sents:
        final_prompt = f"Question: {query}\nAnswer:"
    else:
        cand_embs = model.encode(candidate_sents, convert_to_tensor=True)
        q_emb2 = model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(q_emb2, cand_embs)[0]
        sorted_idxs = torch.argsort(sims, descending=True)
        top_contexts = []
        for idx in sorted_idxs:
            i = idx.item()
            if float(sims[i].item()) >= min_sim:
                top_contexts.append(candidate_sents[i])
                if len(top_contexts) >= top_n:
                    break
        context_str = "\n\n".join(f"[Context]\n{s}" for s in top_contexts)
        final_prompt = f"{context_str}\n\nQuestion: {query}\nAnswer:"
    system_prompt = ("You are a helpful assistant that uses the provided context to answer the question. "
                     "If no relevant context is available, answer the question to the best of your ability.")
    return query_gpt(system_prompt, final_prompt)

# (e) Validation Functions for the English Pipeline
def validate_eng_rag(test_data, combined_index, combined_corpus, model, top_k=3, top_n=3,
                     min_sim=0.4, sim_threshold=0.7):
    """
    Validate the retrieval-augmented generation (RAG) approach for English questions.
    For each test item:
      - Retrieve context and query GPT.
      - Evaluate the response using cosine similarity and BLEU score.
    Returns overall metrics and detailed results.
    """
    smoothing = SmoothingFunction().method1
    results = []
    y_true, y_pred, bleu_scores = [], [], []
    logger.info("Validating English RAG (with context).")
    
    for row in test_data:
        query = row["query"]
        exp = row["expected_answer"]
        y_true.append(1)
        gen = query_rag_eng(query, combined_index, combined_corpus, model,
                            top_k=top_k, top_n=top_n, min_sim=min_sim)
        if gen:
            sim = util.cos_sim(model.encode(exp, convert_to_tensor=True),
                               model.encode(gen, convert_to_tensor=True)).item()
            pred = 1 if sim >= sim_threshold else 0
            bleu = sentence_bleu([exp.split()], gen.split(), smoothing_function=smoothing)
        else:
            sim, pred, bleu = 0.0, 0, 0.0
        y_pred.append(pred)
        bleu_scores.append(bleu)
        results.append({
            "query": query,
            "expected_answer": exp,
            "generated_response": gen if gen else "",
            "similarity": sim,
            "label": pred,
            "bleu_score": bleu
        })
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "average_bleu": np.mean(bleu_scores) if bleu_scores else 0.0,
        "results": results
    }

def validate_eng_no_rag(test_data, model, sim_threshold=0.7):
    """
    Validate the direct (No-RAG) approach for English questions.
    For each test item:
      - Query GPT without context.
      - Evaluate the response using cosine similarity and BLEU score.
    """
    smoothing = SmoothingFunction().method1
    results = []
    y_true, y_pred, bleu_scores = [], [], []
    logger.info("Validating English No-RAG (direct question).")
    
    for row in test_data:
        query = row["query"]
        exp = row["expected_answer"]
        y_true.append(1)
        gen = query_no_rag_eng(query)
        if gen:
            sim = util.cos_sim(model.encode(exp, convert_to_tensor=True),
                               model.encode(gen, convert_to_tensor=True)).item()
            pred = 1 if sim >= sim_threshold else 0
            bleu = sentence_bleu([exp.split()], gen.split(), smoothing_function=smoothing)
        else:
            sim, pred, bleu = 0.0, 0, 0.0
        y_pred.append(pred)
        bleu_scores.append(bleu)
        results.append({
            "query": query,
            "expected_answer": exp,
            "generated_response": gen if gen else "",
            "similarity": sim,
            "label": pred,
            "bleu_score": bleu
        })
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "average_bleu": np.mean(bleu_scores) if bleu_scores else 0.0,
        "results": results
    }

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    ##############################
    # A) KOREAN PIPELINE
    ##############################
    logger.info("=== Starting Korean Pipeline ===")
    # Set the path to the Excel file containing Korean medical data.
    kor_excel_path = "snuh_medical_data.xlsx"  # Adjust as needed.
    # Load the Korean corpus from the Excel file.
    kor_corpus = load_korean_excel(kor_excel_path)
    
    # Load the Korean SentenceTransformer model.
    kor_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    # Create embeddings for the Korean corpus.
    kor_emb = create_ko_embeddings(kor_corpus, kor_model)
    # Build a FAISS index from the Korean embeddings.
    kor_index = build_faiss_index(kor_emb)
    
    # Define test queries and expected answers for the Korean pipeline.
    kor_test_data = [
        {
            "query": "ABO 신생아용혈성질환이 궁금합니다.",
            "expected_answer": "혈액형이 O형인 임신부가 A형 또는 B형인 아기를 가졌을 때 엄마가 가지고 있던 면역글로불린 G 유형(IgG type)의 항A, B항체(anti-A, B)가 태반을 건너가 아기의 적혈구 항원과 결합하여 비장에 있는 마크로파지(macrophage)에 의해 적혈구를 파괴되는 질환을 말한다."
        },
        {
            "query": "A형 간염에 대해 설명해 주세요.",
            "expected_answer": "간염 바이러스의 한 종류인 A형 간염 바이러스(hepatitis A virus, HAV)에 의해 발생하는 간염으로 주로 급성 간염의 형태로 나타난다."
        },
        # Add additional test items as needed.
    ]
    
    # Validate the Korean pipeline using retrieval augmentation (RAG).
    kor_rag_results = validate_kor_rag(kor_test_data, kor_index, kor_corpus, kor_model, top_k=3, sim_threshold=0.7)
    # Validate the Korean pipeline using direct questioning (No-RAG).
    kor_norag_results = validate_kor_no_rag(kor_test_data, kor_model, sim_threshold=0.7)
    
    logger.info("=== KOREAN RAG RESULTS ===")
    logger.info(kor_rag_results)
    logger.info("=== KOREAN NO-RAG RESULTS ===")
    logger.info(kor_norag_results)
    
    # Save the Korean pipeline results to Excel files.
    pd.DataFrame(kor_rag_results["results"]).to_excel("korean_rag_results.xlsx", index=False)
    pd.DataFrame(kor_norag_results["results"]).to_excel("korean_no_rag_results.xlsx", index=False)
    
    ##############################
    # B) ENGLISH PIPELINE
    ##############################
    logger.info("=== Starting English Pipeline ===")
    # Define Wikipedia page titles to fetch.
    pages_to_fetch = [
        "Strategy",
        "Leadership",
        "Innovation",
        "Planning",
        "Organization",
        "Efficiency",
        "Coordination",
        "Analytics",
        "Automation",
        "Integration",
        "Governance",
        "Collaboration",
        "Security",
        "Performance",
        "Transformation",
        "Optimization",
        "Technology",
        "Data",
        "Communication",
        "Digitization"
    ]
    # Fetch Wikipedia articles.
    articles = fetch_wikipedia_articles(pages_to_fetch)
    if not articles:
        logger.error("No valid Wikipedia pages found for English. Exiting.")
        exit(1)
    
    # Load the English SentenceTransformer model.
    eng_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Build sentence corpora and FAISS indices for each article.
    sentence_corpus_map = {}
    sentence_index_map = {}
    for art in articles:
        title = art["title"]
        page_obj = art["page_obj"]
        # Process the page to build a sentence corpus.
        sent_corpus = build_sentence_corpus(page_obj, eng_model, sim_threshold=0.9)
        sentence_corpus_map[title] = sent_corpus
        # Create embeddings and build a FAISS index for the sentences.
        emb = create_sentence_embeddings(sent_corpus, eng_model)
        s_index = build_faiss_index(emb)
        sentence_index_map[title] = s_index
    
    # Combine all sentences from all articles for multi-sentence retrieval.
    combined_sentence_corpus = []
    for corpus in sentence_corpus_map.values():
        combined_sentence_corpus.extend(corpus)
    combined_emb = create_sentence_embeddings(combined_sentence_corpus, eng_model)
    combined_index = build_faiss_index(combined_emb)
    
    # Generate a dataset for English questions.
    eng_test_data = generate_dataset(
        articles,
        sentence_corpus_map,
        sentence_index_map,
        model=eng_model,
        num_questions=5,
        top_k=5,
        top_n=3,
        min_sim=0.4
    )
    
    # Validate using RAG (with context) and No-RAG (direct question) for English.
    eng_rag_results = validate_eng_rag(eng_test_data, combined_index, combined_sentence_corpus, eng_model,
                                       top_k=3, top_n=3, min_sim=0.4, sim_threshold=0.7)
    eng_norag_results = validate_eng_no_rag(eng_test_data, eng_model, sim_threshold=0.7)
    
    logger.info("=== ENGLISH RAG RESULTS ===")
    logger.info(eng_rag_results)
    logger.info("=== ENGLISH NO-RAG RESULTS ===")
    logger.info(eng_norag_results)
    
    # Save the English pipeline results to Excel files.
    pd.DataFrame(eng_rag_results["results"]).to_excel("english_rag_results.xlsx", index=False)
    pd.DataFrame(eng_norag_results["results"]).to_excel("english_no_rag_results.xlsx", index=False)
    
    logger.info("All pipelines complete. Compare results in the output Excel files.")
