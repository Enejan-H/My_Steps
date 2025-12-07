# RAG Pipeline ile Local LLM Kullanımı

Bu rehber, M2 Mac üzerinde **Pinecone + BGE-small + Flan-T5** kullanarak **RAG (Retrieval-Augmented Generation) pipeline** kurulumunu ve kullanımını adım adım gösterir. Artık tamamen ücretsiz ve local LLM ile çalışıyor.

---

## 🚀 1. Ortamı Hazırlama (M2 Mac + Python venv)

Mevcut environment’ı kapatıp, temiz bir virtual environment oluştur:

```bash
deactivate
python3 -m venv my_steps_env
source my_steps_env/bin/activate
````

Gerekli paketleri yükle:

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio      # PyTorch ≥2.6, M2 Metal uyumlu
pip install "numpy<2"                         # NumPy uyumlu
pip install sentence-transformers transformers python-dotenv pinecone grpcio protobuf googleapis-common-protos
pip install openai                             # opsiyonel
```

**Notlar:**

* PyTorch ≥2.6 → M2 Metal GPU uyumlu
* NumPy <2 → sentence-transformers uyumu
* Pinecone v3 + gRPC çalışması için protobuf ve grpcio gerekli

---

## 🔑 2. Pinecone API Key

`.env` dosyasına ekle:

```env
PINECONE_API_KEY=senin_api_key
```

Script’te çek:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
```

---

## 📦 3. Pinecone Index Oluşturma

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=api_key)

if "rag-demo" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-demo",
        dimension=384,         # BGE-small embedding boyutu
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
```

> Index, embedding’leri saklayacağın veri tabanıdır. Dimension, embedding modeli ile aynı olmalı.

---

## 📝 4. Embedding Modelini Yükleme

```python
from sentence_transformers import SentenceTransformer
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en", device=device)
```

* BGE-small → metinleri vektöre çevirir
* M2 Metal GPU varsa kullanır, yoksa CPU fallback

---

## 📄 5. Dokümanları Embed Edip Pinecone’a Upsert Etme

```python
docs = [("doc1", "FastAPI is a Python web framework."),
        ("doc2", "Docker containers are used for packaging apps.")]

vectors = [(doc_id, model.encode(text).tolist(), {"text": text}) for doc_id, text in docs]
pc.upsert(index_name="rag-demo", vectors=vectors)
```

* Metadata ile metin içeriği saklanır
* Artık query ile eşleşme bulabilir hâle geldi

---

## 🔍 6. Query ve Benzer Dokümanlar

```python
query = "What is FastAPI?"
q_vec = model.encode(query).tolist()
res = pc.query(index_name="rag-demo", vector=q_vec, top_k=2, include_metadata=True)
print(res)
```

* `score` → dokümanın query ile benzerliği
* `metadata` → orijinal metin

---

## 🤖 7. RAG Pipeline: Context + LLM Yanıtı

Local LLM olarak **Flan-T5** kullan:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

# Context oluştur
context = "\n".join([match['metadata']['text'] for match in res['matches']])
prompt = "Answer based on context:\n" + context + "\nQuestion: " + query

# LLM response
answer = llm_pipe(prompt)[0]['generated_text']

print("\n--- Context from Pinecone ---\n", context)
print("\n--- LLM Answer ---\n", answer)
```

**Sonuç:**

* Pinecone’dan top_k dokümanları aldın
* Context oluşturup prompt hazırlandı
* Flan-T5 modelinden yanıt üretildi
* Tamamen ücretsiz ve local LLM ile RAG pipeline çalışıyor

---

## ⚡ 8. Bundan Sonra Kullanım

1. **FastAPI ile API oluştur:**
   Kullanıcı query gönderir → embedding → Pinecone → top_k → context → Flan-T5 → yanıt döner

2. **Dockerize et:**
   Tek script’i container içine koy → RAG service olarak deploy

3. **Dokümanları artır ve test et:**
   Farklı namespace’ler ile Pinecone limitlerini aşmadan büyüt

---

## 🔹 Özet

| Adım      | Kullanılan Teknoloji                                           |
| --------- | -------------------------------------------------------------- |
| Ortam     | Python venv, M2 Metal GPU                                      |
| Pinecone  | Vektör depolama & retrieval                                    |
| Embedding | BGE-small                                                      |
| LLM       | Flan-T5 (local, ücretsiz)                                      |
| Pipeline  | query → embedding → Pinecone → context → prompt → LLM response |

Pipeline **çalışıyor ve test edilebilir** durumda.

---

### 📸 Önerilen Görseller (Opsiyonel)

1. `pipeline.png` → RAG pipeline akış diyagramı
2. `pinecone_query.png` → Pinecone sorgu çıktısı
3. `flan_t5_answer.png` → LLM yanıt ekran görüntüsü

