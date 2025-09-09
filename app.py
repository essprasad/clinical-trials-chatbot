import os
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2

HF_API = os.environ.get('HUGGINGFACEHUB_API_TOKEN')  # Set in Space Secrets
DEFAULT_MODEL = os.environ.get('HF_MODEL', 'gpt2')  # Update to preferred model

embedder = SentenceTransformer('all-MiniLM-L6-v2')

DOCS = []  # {'id','text','source'}
EMBS = None

def extract_text_from_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(file_path)
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def add_pdf_file(uploaded_file):
    global DOCS, EMBS
    path = uploaded_file.name
    text = extract_text_from_pdf(path)
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    start = len(DOCS)
    for i, c in enumerate(chunks):
        DOCS.append({'id': f'doc_{start+i}', 'text': c, 'source': uploaded_file.name})
    new_embs = embedder.encode(chunks, convert_to_numpy=True)
    global EMBS
    if EMBS is None:
        EMBS = new_embs
    else:
        EMBS = np.vstack([EMBS, new_embs])
    return len(chunks)

def search_similar(query, top_k=4):
    global EMBS, DOCS
    if EMBS is None or len(DOCS)==0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    norms = np.linalg.norm(EMBS, axis=1) * np.linalg.norm(q_emb) + 1e-12
    sims = (EMBS @ q_emb) / norms
    idx = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in idx]

def call_hf_model(prompt, model=DEFAULT_MODEL, max_tokens=256):
    if not HF_API:
        return "HUGGINGFACEHUB_API_TOKEN not set in Space secrets."
    headers = {"Authorization": f"Bearer {HF_API}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature":0.2}}
    url = f"https://api-inference.huggingface.co/models/{model}"
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        return f"Model call failed: {resp.status_code}: {resp.text[:400]}"
    data = resp.json()
    if isinstance(data, list) and len(data)>0 and 'generated_text' in data[0]:
        return data[0]['generated_text']
    if isinstance(data, dict) and 'generated_text' in data:
        return data['generated_text']
    return str(data)

def build_rag_prompt(query, contexts):
    ctx_text = '\n\n'.join([f"Source {i+1}: {c[:800]}" for i,c in enumerate(contexts)])
    prompt = f"You are a helpful clinical trials assistant. Use the context excerpts below when answering and cite the source numbers.\n\nContext:\n{ctx_text}\n\nQuestion: {query}\n\nAnswer:"
    return prompt

def answer_query(query, model):
    hits = search_similar(query, top_k=4)
    if not hits:
        return call_hf_model(query, model=model), []
    contexts = [DOCS[i]['text'] for (_s,i) in hits]
    sources = [DOCS[i]['source'] for (_s,i) in hits]
    prompt = build_rag_prompt(query, contexts)
    resp = call_hf_model(prompt, model=model)
    return resp, list(zip(sources, contexts))

with gr.Blocks() as demo:
    gr.Markdown("""# Clinical Trials RAG Chatbot\nUpload clinical trial PDFs and ask questions. Set `HUGGINGFACEHUB_API_TOKEN` in Space secrets and optionally `HF_MODEL` env var.""")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot()
            txt = gr.Textbox(placeholder='Ask clinical-trials questions...')
            model_box = gr.Textbox(label='HF model (optional)', value=DEFAULT_MODEL)
            ask_btn = gr.Button('Ask')
        with gr.Column(scale=1):
            upload = gr.File(file_count='multiple', file_types=['.pdf'])
            info = gr.Markdown('No documents indexed.')
            clear_btn = gr.Button('Clear docs')
    def on_upload(files):
        total = 0
        for f in files:
            added = add_pdf_file(f)
            total += added
        info.value = f"Indexed chunks: {len(DOCS)}"
        return info.value
    def on_clear():
        global DOCS, EMBS
        DOCS = []
        EMBS = None
        return 'Cleared.'
    def on_ask(inp, model):
        resp, sources = answer_query(inp, model or DEFAULT_MODEL)
        src_text = '\n'.join([f"- {s}" for s,_c in sources])
        reply = resp + "\n\nSources:\n" + (src_text or 'None')
        chat.append((inp, reply))
        return chat, ''
    upload.change(on_upload, inputs=[upload], outputs=[info])
    clear_btn.click(on_clear, outputs=[info])
    ask_btn.click(on_ask, inputs=[txt, model_box], outputs=[chat, txt])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
