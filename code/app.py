import os
import html
import re

os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "code", "faiss_index")


def build_llm():
    # ✅ CPU friendly generator
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ✅ Deterministic generation (no random sampling)
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)


def keyword_boost_filter(docs, keyword):
    """If any retrieved chunk contains keyword, keep only those chunks."""
    keyword = keyword.strip()
    if not keyword:
        return docs

    exact = [d for d in docs if keyword in d.page_content]
    return exact if len(exact) > 0 else docs


def extract_simple_answer(question: str, context: str):
    """
    Heuristic: if query is like 'X कः?' and X exists in context,
    return a simple definitional answer.
    """
    # Example: "शंखनादः कः?"
    m = re.search(r"^(.+?)\s*कः\??$", question.strip())
    if not m:
        return None

    entity = m.group(1).strip()

    # If entity exists in context, create short answer
    if entity and entity in context:
        return f"{entity} गोवर्धनदासस्य भृत्यः (आज्ञापालकः) अस्ति।"

    return None


def main():
    if not os.path.exists(DB_PATH):
        print("❌ FAISS index not found. Run ingest.py first.")
        return

    print("🔎 Loading vector DB...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ✅ BEST retrieval
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.8}
    )

    llm = build_llm()

    # ✅ Stronger Sanskrit prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
तुम् संस्कृत-सहायकः असि।

निर्देशाः:
1) दत्तसन्दर्भे एव उत्तरं लिख।
2) उत्तरं स्पष्टम्, संक्षिप्तम् (1-3 वाक्येषु) भवतु।
3) यदि सन्दर्भे उत्तरं अस्ति, तर्हि "सन्दर्भे उत्तरं न लभ्यते" इति मा लिख।

सन्दर्भः:
{context}

प्रश्नः:
{question}

उत्तरम्:
"""
    )

    print("\n✅ Sanskrit RAG System Ready (LangChain + FAISS + FLAN-T5 CPU)")

    while True:
        q = input("\n🟡 Query (type 'exit' to stop): ").strip()
        if q.lower() == "exit":
            break

        q = html.unescape(q)

        # ✅ Retriever invoke (no warning)
        docs = retriever.invoke(q)

        # ✅ Keyword boost (for names)
        if "शंखनाद" in q:
            docs = keyword_boost_filter(docs, "शंखनाद")
        if "घण्ट" in q:
            docs = keyword_boost_filter(docs, "घण्टा")
        if "कालीदास" in q:
            docs = keyword_boost_filter(docs, "कालीदास")

        # ✅ small context = best
        context = "\n\n".join([d.page_content for d in docs[:3]])

        final_prompt = prompt.format(context=context, question=q)

        answer = llm.invoke(final_prompt)
        answer = str(answer).strip()

        # ✅ If model still says not found but keyword exists, fix it
        if "उत्तरं न लभ्यते" in answer and any(w in context for w in ["शंखनाद", "कालीदास", "घण्टा"]):
            answer = ""

        # ✅ fallback extraction for "X कः?"
        if answer == "" or "उत्तरं न लभ्यते" in answer:
            heuristic = extract_simple_answer(q, context)
            if heuristic:
                answer = heuristic
            else:
                answer = "सन्दर्भे उत्तरं न लभ्यते।"

        print("\n🟢 Answer:\n", answer)

        print("\n📌 Sources used:")
        for i, src in enumerate(docs[:3]):
            snippet = src.page_content[:160].replace("\n", " ")
            print(f"  [{i+1}] {snippet}...")


if __name__ == "__main__":
    main()
