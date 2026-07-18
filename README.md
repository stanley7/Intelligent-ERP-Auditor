# Intelligent-ERP-Auditor

# 🤖 AI ERP Audit Assistant

An AI-powered ERP auditing system that combines rule-based audit analytics with Retrieval-Augmented Generation (RAG) to help auditors identify financial anomalies and interactively investigate ERP transactions using natural language.

Built for the OpenAI Hackathon.

---

# Features

- ERP transaction parsing
- Rule-based audit engine
- Knowledge Graph representation
- Automated anomaly detection
- Retrieval-Augmented Generation (RAG)
- GPT-5 powered audit assistant
- OpenAI Embeddings
- FAISS vector search
- Interactive Streamlit interface
- Professional audit report generation

---

# System Architecture

```text
                             ERP DATASET
                                   │
                                   ▼
                    +-------------------------------+
                    |      ERP File Parser          |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    |      ERP Data Model           |
                    | Vendors • Customers • GL      |
                    | Assets • Transactions         |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    |      Knowledge Graph          |
                    | Nodes + Relationships         |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    |      Audit Engine V2          |
                    +-------------------------------+
                     │        │        │        │
                     ▼        ▼        ▼        ▼
             Large Payments  Weekend  Duplicate  Split
               Detection     Posting   Payments  Payments
                     │        │        │        │
                     └────────┴────────┴────────┘
                                   │
                                   ▼
                    +-------------------------------+
                    | Structured Audit Findings     |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    | Convert Findings to Text      |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    | OpenAI text-embedding-3-small |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    |         FAISS Index           |
                    +-------------------------------+
                                   │
                         User Natural Language Query
                                   │
                                   ▼
                    +-------------------------------+
                    | Hybrid Retrieval              |
                    | • ERP ID Lookup               |
                    | • Vector Search               |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    | GPT-5                         |
                    | AI Audit Assistant            |
                    +-------------------------------+
                                   │
                                   ▼
                    +-------------------------------+
                    | Streamlit Web Interface       |
                    +-------------------------------+
```

---

# Audit Rules Implemented

The audit engine currently detects:

- Large Payments
- Weekend Postings
- Duplicate Payments
- Split Payments

Each finding contains:

- Rule
- Severity
- Description
- Entity ID
- Supporting ERP Transactions

---

# AI Capabilities

The assistant supports natural language questions such as:

- Why is vendor 200003 considered risky?
- Show duplicate payments.
- Explain the weekend postings.
- Summarize the highest risk findings.
- Which vendors have multiple suspicious transactions?

Responses are generated using GPT-5 and grounded using retrieved audit evidence.

---

# Technologies Used

| Component | Technology |
|----------|------------|
| Language | Python |
| UI | Streamlit |
| LLM | GPT-5 |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | FAISS |
| Knowledge Representation | Graph-based ERP Model |
| AI Framework | OpenAI API |

---

# Project Structure

```
.
├── app.py
├── ERP_Audit_Assistant.ipynb
├── requirements.txt
├── README.md
└── assets/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-ERP-Audit-Assistant.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Set your OpenAI API key

```bash
export OPENAI_API_KEY=your_api_key
```

Run the application

```bash
streamlit run app.py
```

---

# Example Workflow

```text
ERP Files
     │
     ▼
Parse Transactions
     │
     ▼
Knowledge Graph
     │
     ▼
Audit Engine
     │
     ▼
Generate Findings
     │
     ▼
Create Embeddings
     │
     ▼
FAISS Retrieval
     │
     ▼
GPT-5
     │
     ▼
Interactive Audit Assistant
```

---

# Example Questions

```text
Why is vendor 200003 considered risky?

Show duplicate payments.

Explain the split payment findings.

Which vendor has the largest suspicious transaction?

Summarize all high-risk findings.
```

---

# Future Improvements

- Additional ERP fraud detection rules
- Multi-company support
- Financial statement analytics
- Interactive graph visualization
- LangGraph-based AI agents
- Multi-step reasoning workflows
- PDF audit report export

---

# Acknowledgements

Developed for the OpenAI Hackathon.

Built using OpenAI GPT-5, OpenAI Embeddings, FAISS, and Streamlit.

---

# License

MIT License
