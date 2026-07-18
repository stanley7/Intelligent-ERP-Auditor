"""
AI ERP Audit Assistant — Streamlit UI
Rebuilt from Untitled113.ipynb (RAG audit engine over ERP transaction data).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import csv
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List, Optional
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------
# Optional heavy deps — imported lazily so the page can render even if
# they're missing, and we can show a friendly install hint instead of
# crashing.
# ----------------------------------------------------------------------
try:
    import faiss
except ImportError:
    faiss = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ========================================================================
# DATA MODELS
# ========================================================================

@dataclass
class Vendor:
    vendor_id: str
    tax_id: str
    address: str
    postal_code: str
    city: str
    country: str
    name: str
    vendor_group: str
    classification: str
    currency: str


@dataclass
class VendorTransaction:
    vendor_id: str
    document_number: str
    posting_date: datetime
    document_date: Optional[datetime]
    value_date: datetime
    description: str
    amount: float
    currency: str
    local_amount: float
    reference: str
    external_document: str
    journal: str
    posted: bool


@dataclass
class Customer:
    customer_id: str
    tax_id: str
    address: str
    postal_code: str
    city: str
    country: str
    name: str
    customer_group: str
    classification: str
    currency: str


@dataclass
class CustomerTransactionRecord:
    customer_id: str
    document_number: str
    posting_date: datetime
    document_date: Optional[datetime]
    value_date: datetime
    description: str
    amount: float
    currency: str
    local_amount: float
    reference: str
    external_document: str
    journal: str


@dataclass
class GLAccount:
    account_number: str
    account_name: str
    account_type: str
    reconciliation_account: bool
    cost_center_required: str
    posting_type: str
    financial_statement_type: str


@dataclass
class GLTransaction:
    account_number: str
    period: str
    cost_center: str
    document_type: str
    posting_key: str
    debit: bool
    credit: bool
    amount: float
    currency: str
    local_amount: float
    description: str
    posting_date: datetime
    document_number: str
    document_date: datetime
    reference: str
    status: str
    assignment: str
    partner: str
    user: str
    external_document: str
    company_code: str
    entry_date: datetime
    entry_time: str
    posted: bool


@dataclass
class Asset:
    asset_id: str
    asset_name: str
    gl_account: str
    asset_type: str
    location: str
    cost_center: str
    serial_number: str
    manufacturer: str
    status: str


@dataclass
class AssetTransaction:
    gl_account: str
    posting_date: datetime
    transaction_type: str
    amount: float
    currency: str
    local_amount: float
    journal: str
    reference: str
    asset_account: str
    status: str


@dataclass
class ERPDataModel:
    vendors: List[Vendor]
    vendor_transactions: List[VendorTransaction]
    customers: List[Customer]
    customer_transactions: List[CustomerTransactionRecord]
    gl_accounts: List[GLAccount]
    gl_transactions: List[GLTransaction]
    assets: List[Asset]
    asset_transactions: List[AssetTransaction]


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    data: Any


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str


@dataclass
class AuditFinding:
    rule: str
    severity: str
    title: str
    description: str
    entity_id: str
    evidence: Any


# ========================================================================
# PARSERS
# ========================================================================

class BaseParser(ABC):

    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        records = []
        with open(self.file_path, "r", encoding="latin1") as file:
            reader = csv.reader(file, delimiter=";", quotechar='"')
            for row in reader:
                if not row:
                    continue
                records.append(self.parse_row(row))
        return records

    @abstractmethod
    def parse_row(self, row):
        pass


def _parse_date(value):
    value = value.strip()
    if value == "":
        return None
    return datetime.strptime(value, "%d.%m.%Y")


def _parse_amount(value):
    value = value.replace(".", "")
    value = value.replace(",", ".")
    return float(value)


def _parse_bool_yes(value):
    return value.strip().lower() == "yes"


def _parse_bool_ja(value):
    return value.strip().lower() == "ja"


class VendorParser(BaseParser):
    def parse_row(self, row):
        return Vendor(
            vendor_id=row[0], tax_id=row[1], address=row[2], postal_code=row[3],
            city=row[4], country=row[5], name=row[6], vendor_group=row[7],
            classification=row[9], currency=row[10],
        )


class VendorTransactionParser(BaseParser):
    def parse_row(self, row):
        return VendorTransaction(
            vendor_id=row[0], document_number=row[1],
            posting_date=_parse_date(row[2]), document_date=_parse_date(row[3]),
            value_date=_parse_date(row[4]), description=row[5],
            amount=_parse_amount(row[6]), currency=row[7],
            local_amount=_parse_amount(row[8]), reference=row[9],
            external_document=row[10], journal=row[11],
            posted=_parse_bool_yes(row[12]),
        )


class CustomerParser(BaseParser):
    def parse_row(self, row):
        return Customer(
            customer_id=row[0], tax_id=row[1], address=row[2], postal_code=row[3],
            city=row[4], country=row[5], name=row[6], customer_group=row[7],
            classification=row[9], currency=row[10],
        )


class CustomerTransactionParser(BaseParser):
    def parse_row(self, row):
        return CustomerTransactionRecord(
            customer_id=row[0], document_number=row[1],
            posting_date=_parse_date(row[2]), document_date=_parse_date(row[3]),
            value_date=_parse_date(row[4]), description=row[5],
            amount=_parse_amount(row[6]), currency=row[7],
            local_amount=_parse_amount(row[8]), reference=row[9],
            external_document=row[10], journal=row[11],
        )


class GLAccountParser(BaseParser):
    def parse_row(self, row):
        return GLAccount(
            account_number=row[0], account_name=row[1], account_type=row[2],
            reconciliation_account=_parse_bool_ja(row[3]), cost_center_required=row[4],
            posting_type=row[5], financial_statement_type=row[6],
        )


class GLTransactionParser(BaseParser):
    def parse_row(self, row):
        return GLTransaction(
            account_number=row[0], period=row[1], cost_center=row[2],
            document_type=row[3], posting_key=row[4],
            debit=_parse_bool_ja(row[5]), credit=_parse_bool_ja(row[6]),
            amount=_parse_amount(row[7]), currency=row[8],
            local_amount=_parse_amount(row[9]), description=row[10],
            posting_date=_parse_date(row[11]), document_number=row[12],
            document_date=_parse_date(row[13]), reference=row[14], status=row[15],
            assignment=row[16], partner=row[17], user=row[18],
            external_document=row[19], company_code=row[20],
            entry_date=_parse_date(row[21]), entry_time=row[22],
            posted=_parse_bool_ja(row[23]),
        )


class AssetParser(BaseParser):
    def parse_row(self, row):
        return Asset(
            asset_id=row[0], asset_name=row[1], gl_account=row[2], asset_type=row[3],
            location=row[4], cost_center=row[5], serial_number=row[6],
            manufacturer=row[7], status=row[8],
        )


class AssetTransactionParser(BaseParser):
    def parse_row(self, row):
        return AssetTransaction(
            gl_account=row[0], posting_date=_parse_date(row[1]),
            transaction_type=row[2], amount=_parse_amount(row[3]), currency=row[4],
            local_amount=_parse_amount(row[5]), journal=row[6], reference=row[7],
            asset_account=row[8], status=row[9],
        )


# German filenames used in the dataset export, mapped to a parser + field name
FILE_MAP = {
    "Lieferanten.txt": ("vendors", VendorParser),
    "Lieferantenbuchungen.txt": ("vendor_transactions", VendorTransactionParser),
    "Kunden.txt": ("customers", CustomerParser),
    "Kundenbuchungen.txt": ("customer_transactions", CustomerTransactionParser),
    "Sachkonten.txt": ("gl_accounts", GLAccountParser),
    "Sachkontobuchungen.txt": ("gl_transactions", GLTransactionParser),
    "Anlagen.txt": ("assets", AssetParser),
    "Anlagenbuchungen.txt": ("asset_transactions", AssetTransactionParser),
}


def load_erp_data(extract_dir: Path) -> ERPDataModel:
    file_registry = {f.name: str(f) for f in extract_dir.rglob("*") if f.is_file()}

    missing = [name for name in FILE_MAP if name not in file_registry]
    if missing:
        raise FileNotFoundError(
            "Missing expected file(s) in the uploaded ZIP: " + ", ".join(missing)
        )

    parsed = {}
    for filename, (field, parser_cls) in FILE_MAP.items():
        parsed[field] = parser_cls(file_registry[filename]).parse()

    return ERPDataModel(**parsed)


# ========================================================================
# KNOWLEDGE GRAPH
# ========================================================================

class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def add_edge(self, edge):
        self.edges.append(edge)
        self.outgoing[edge.source].append(edge)
        self.incoming[edge.target].append(edge)

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def neighbors(self, node_id):
        return self.outgoing.get(node_id, [])

    def incoming_edges(self, node_id):
        return self.incoming.get(node_id, [])


def build_graph(erp_data: ERPDataModel) -> KnowledgeGraph:
    graph = KnowledgeGraph()

    for vendor in erp_data.vendors:
        graph.add_node(GraphNode(f"vendor:{vendor.vendor_id}", "Vendor", vendor))

    for customer in erp_data.customers:
        graph.add_node(GraphNode(f"customer:{customer.customer_id}", "Customer", customer))

    for account in erp_data.gl_accounts:
        graph.add_node(GraphNode(f"gl:{account.account_number}", "GLAccount", account))

    for asset in erp_data.assets:
        graph.add_node(GraphNode(f"asset:{asset.asset_id}", "Asset", asset))

    vendor_lookup = {v.vendor_id: f"vendor:{v.vendor_id}" for v in erp_data.vendors}
    for i, tx in enumerate(erp_data.vendor_transactions):
        node_id = f"vendor_tx:{i}"
        graph.add_node(GraphNode(node_id, "VendorTransaction", tx))
        src = vendor_lookup.get(tx.vendor_id)
        if src:
            graph.add_edge(GraphEdge(src, node_id, "HAS_TRANSACTION"))

    customer_lookup = {c.customer_id: f"customer:{c.customer_id}" for c in erp_data.customers}
    for i, tx in enumerate(erp_data.customer_transactions):
        node_id = f"customer_tx:{i}"
        graph.add_node(GraphNode(node_id, "CustomerTransaction", tx))
        src = customer_lookup.get(tx.customer_id)
        if src:
            graph.add_edge(GraphEdge(src, node_id, "HAS_TRANSACTION"))

    gl_lookup = {a.account_number: f"gl:{a.account_number}" for a in erp_data.gl_accounts}
    for i, tx in enumerate(erp_data.gl_transactions):
        node_id = f"gl_tx:{i}"
        graph.add_node(GraphNode(node_id, "GLTransaction", tx))
        src = gl_lookup.get(tx.account_number)
        if src:
            graph.add_edge(GraphEdge(src, node_id, "HAS_TRANSACTION"))

    asset_lookup = {a.gl_account: f"asset:{a.asset_id}" for a in erp_data.assets}
    for i, tx in enumerate(erp_data.asset_transactions):
        node_id = f"asset_tx:{i}"
        graph.add_node(GraphNode(node_id, "AssetTransaction", tx))
        src = asset_lookup.get(tx.asset_account)
        if src:
            graph.add_edge(GraphEdge(src, node_id, "HAS_TRANSACTION"))

    return graph


# ========================================================================
# AUDIT ENGINE
# ========================================================================

class AuditEngine:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def find_large_vendor_payments(self, threshold=50000):
        findings = []
        for node in self.graph.nodes.values():
            if node.node_type != "VendorTransaction":
                continue
            tx = node.data
            if abs(tx.amount) >= threshold:
                findings.append(AuditFinding(
                    rule="Large Vendor Payment", severity="Medium",
                    title="Large vendor payment detected",
                    description=f"Vendor {tx.vendor_id} has a payment of {tx.amount:,.2f} {tx.currency}",
                    entity_id=tx.vendor_id, evidence=tx,
                ))
        findings.sort(key=lambda f: abs(f.evidence.amount), reverse=True)
        return findings

    def find_weekend_postings(self):
        findings = []
        for node in self.graph.nodes.values():
            if node.node_type != "VendorTransaction":
                continue
            tx = node.data
            if tx.posting_date and tx.posting_date.weekday() >= 5:
                findings.append(AuditFinding(
                    rule="Weekend Posting", severity="Medium",
                    title="Transaction posted on weekend",
                    description=f"Vendor {tx.vendor_id} transaction {tx.document_number} was posted on a weekend.",
                    entity_id=tx.vendor_id, evidence=tx,
                ))
        findings.sort(key=lambda f: f.evidence.posting_date)
        return findings

    def find_duplicate_vendor_payments(self):
        groups = defaultdict(list)
        for node in self.graph.nodes.values():
            if node.node_type != "VendorTransaction":
                continue
            tx = node.data
            key = (tx.vendor_id, tx.document_number, tx.amount, tx.currency)
            groups[key].append(tx)

        findings = []
        for (vendor_id, document, amount, currency), transactions in groups.items():
            if len(transactions) < 2:
                continue
            findings.append(AuditFinding(
                rule="Duplicate Vendor Payment", severity="High",
                title="Possible duplicate vendor payment",
                description=f"Document {document} for vendor {vendor_id} appears {len(transactions)} times.",
                entity_id=vendor_id, evidence=transactions,
            ))
        return findings

    def find_split_payments(self, approval_limit=20000, minimum_transactions=2):
        groups = defaultdict(list)
        for node in self.graph.nodes.values():
            if node.node_type != "VendorTransaction":
                continue
            tx = node.data
            if abs(tx.amount) >= approval_limit:
                continue
            sign = "positive" if tx.amount > 0 else "negative"
            key = (tx.vendor_id, tx.posting_date.date() if tx.posting_date else None, sign)
            groups[key].append(tx)

        findings = []
        for (vendor_id, posting_date, sign), transactions in groups.items():
            if len(transactions) < minimum_transactions:
                continue
            total = sum(abs(tx.amount) for tx in transactions)
            if total < approval_limit:
                continue
            findings.append(AuditFinding(
                rule="Split Payment", severity="High",
                title="Possible split payment detected",
                description=(
                    f"{len(transactions)} {sign} transactions below {approval_limit:,.0f} EUR "
                    f"total {total:,.2f} EUR on {posting_date}."
                ),
                entity_id=vendor_id, evidence=transactions,
            ))
        return findings

    def run_all(self):
        findings = []
        findings.extend(self.find_large_vendor_payments())
        findings.extend(self.find_weekend_postings())
        findings.extend(self.find_duplicate_vendor_payments())
        findings.extend(self.find_split_payments())
        return findings


# ========================================================================
# RAG: DOCUMENTS + EMBEDDINGS + RETRIEVAL
# ========================================================================

def findings_to_documents(findings: List[AuditFinding]) -> List[str]:
    documents = []
    for f in findings:
        evidence = _format_evidence(f.evidence)
        doc = f"""Rule: {f.rule}
Severity: {f.severity}
Title: {f.title}
Description: {f.description}
Entity ID: {f.entity_id}

Evidence:
{evidence}"""
        documents.append(doc.strip())
    return documents


def _format_evidence(evidence, max_items: int = 10, max_chars: int = 3000) -> str:
    """Render a single transaction or a list of transactions compactly so no
    single document blows up the embedding token budget (duplicate/split
    findings can carry dozens of transactions each)."""
    if isinstance(evidence, list):
        lines = []
        for tx in evidence[:max_items]:
            lines.append(str(tx))
        if len(evidence) > max_items:
            lines.append(f"... and {len(evidence) - max_items} more transaction(s)")
        text = "\n".join(lines)
    else:
        text = str(evidence)

    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    return text


def build_index(client, documents: List[str]):
    """Embed all documents in token-budgeted batches and build a FAISS L2 index.

    OpenAI's embeddings endpoint caps requests at 300k tokens total. We
    estimate ~4 chars/token and batch conservatively (target 100k tokens,
    i.e. ~400k chars per request) so large finding sets never blow the limit.
    """
    all_embeddings = []
    progress = st.progress(0.0, text="Creating embeddings...")

    max_chars_per_batch = 400_000  # ~100k tokens, well under the 300k cap
    max_docs_per_batch = 200

    batches = []
    current_batch = []
    current_chars = 0

    for doc in documents:
        # A single oversized doc still needs to fit under the hard cap on its own.
        doc_chars = min(len(doc), 1_000_000)
        if current_batch and (
            current_chars + doc_chars > max_chars_per_batch
            or len(current_batch) >= max_docs_per_batch
        ):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(doc)
        current_chars += doc_chars

    if current_batch:
        batches.append(current_batch)

    done = 0
    for batch in batches:
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
        done += len(batch)
        progress.progress(min(1.0, done / len(documents)),
                           text=f"Embedded {done}/{len(documents)}")

    progress.empty()

    embeddings = np.array(all_embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(client, index, documents: List[str], query: str, k: int = 5):
    ids = re.findall(r"\b\d{6}\b", query)
    if ids:
        matches = [doc for doc in documents if ids[0] in doc]
        if matches:
            return matches[:k]

    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embedding = np.array([response.data[0].embedding], dtype="float32")
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]


def ask_auditor(client, index, documents: List[str], question: str, model: str = "gpt-5"):
    docs = retrieve(client, index, documents, question)
    context = "\n\n-----------------------------\n\n".join(docs)

    prompt = f"""You are an expert ERP auditor.

Use ONLY the provided audit findings to answer.

If the answer is not contained in the findings, say that no supporting evidence was retrieved.

Audit Findings:

{context}

Question:
{question}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an ERP audit assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content, docs


# ========================================================================
# STREAMLIT UI
# ========================================================================

st.set_page_config(page_title="AI ERP Audit Assistant", layout="wide")

st.title("🤖 AI ERP Audit Assistant")
st.caption("GPT-5 • FAISS • OpenAI Embeddings — RAG over automated audit findings")

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
    )

    model = st.selectbox("Chat model", ["gpt-5", "gpt-4o", "gpt-4o-mini"], index=0)

    uploaded_zip = st.file_uploader("ERP dataset (.zip)", type=["zip"])

    st.caption(
        "Expected files inside the ZIP: Lieferanten.txt, Lieferantenbuchungen.txt, "
        "Kunden.txt, Kundenbuchungen.txt, Sachkonten.txt, Sachkontobuchungen.txt, "
        "Anlagen.txt, Anlagenbuchungen.txt"
    )

    build_clicked = st.button("Build Knowledge Base", type="primary", use_container_width=True)

if build_clicked:
    if not uploaded_zip:
        st.sidebar.error("Upload a dataset ZIP first.")
    elif not api_key:
        st.sidebar.error("Enter your OpenAI API key first.")
    elif faiss is None or OpenAI is None:
        st.sidebar.error("Missing dependencies. Run: pip install faiss-cpu openai")
    else:
        with st.spinner("Extracting and parsing ERP data..."):
            tmp_dir = Path(tempfile.mkdtemp())
            zip_path = tmp_dir / "data.zip"
            zip_path.write_bytes(uploaded_zip.getvalue())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            try:
                erp_data = load_erp_data(tmp_dir)
            except FileNotFoundError as e:
                st.sidebar.error(str(e))
                st.stop()

            graph = build_graph(erp_data)
            engine = AuditEngine(graph)
            findings = engine.run_all()
            documents = findings_to_documents(findings)

        client = OpenAI(api_key=api_key)

        with st.spinner(f"Embedding {len(documents)} findings and building index..."):
            index = build_index(client, documents)

        st.session_state.kb_ready = True
        st.session_state.erp_data = erp_data
        st.session_state.findings = findings
        st.session_state.documents = documents
        st.session_state.index = index
        st.session_state.client = client
        st.session_state.model = model

        st.sidebar.success("Knowledge base built!")

if st.session_state.kb_ready:
    erp_data = st.session_state.erp_data
    findings = st.session_state.findings

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vendors", len(erp_data.vendors))
    col2.metric("Customers", len(erp_data.customers))
    col3.metric("GL Accounts", len(erp_data.gl_accounts))
    col4.metric("Audit Findings", len(findings))

    st.divider()

    tab_chat, tab_findings = st.tabs(["Ask the Auditor", "Browse Findings"])

    with tab_chat:
        question = st.text_input(
            "Ask a question",
            value="Which vendors have suspicious large payments?",
        )

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                answer, docs = ask_auditor(
                    st.session_state.client,
                    st.session_state.index,
                    st.session_state.documents,
                    question,
                    model=st.session_state.model,
                )

            st.success("Analysis Complete")
            st.markdown("## AI Response")
            st.write(answer)

            st.markdown("## Retrieved Evidence")
            for i, d in enumerate(docs):
                with st.expander(f"Evidence {i + 1}"):
                    st.text(d)

    with tab_findings:
        severity_filter = st.multiselect(
            "Severity",
            options=sorted({f.severity for f in findings}),
            default=sorted({f.severity for f in findings}),
        )
        rule_filter = st.multiselect(
            "Rule",
            options=sorted({f.rule for f in findings}),
            default=sorted({f.rule for f in findings}),
        )

        rows = [
            {
                "Rule": f.rule,
                "Severity": f.severity,
                "Entity ID": f.entity_id,
                "Description": f.description,
            }
            for f in findings
            if f.severity in severity_filter and f.rule in rule_filter
        ]
        st.dataframe(rows, use_container_width=True, height=500)
else:
    st.info("Upload your ERP dataset ZIP and click **Build Knowledge Base** in the sidebar to get started.")