# Clinical Trials RAG Chatbot

This is a Hugging Face Spaces-ready Gradio app for **clinical trials Q&A**.

Upload PDFs of clinical trial protocols, reports, or publications, and ask natural language questions.  
The bot retrieves context from the documents and generates answers using an open-source LLM via the Hugging Face Inference API.

---

## 🚀 One-Click Deploy to Hugging Face

[![Deploy to Spaces](https://img.shields.io/badge/Deploy%20to%20Hugging%20Face%20Spaces-blue)](https://huggingface.co/new-space?template=your-username/clinical-trials-chatbot)

👉 Replace `essprasad` with your GitHub username in the link above.

---

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

---

## Hugging Face Setup

1. Create a new Space (Gradio).  
2. Upload this repo's files.  
3. In Space **Settings → Variables and secrets**, add:  
   - `HUGGINGFACEHUB_API_TOKEN` = your HF token  
   - `HF_MODEL` = e.g. `tiiuae/falcon-7b-instruct`  
4. Restart the Space.  

Now you have your **Clinical Trials chatbot** online!
