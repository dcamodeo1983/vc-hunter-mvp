
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from agents.founder_doc_reader_and_orchestrator import run_orchestration

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup Streamlit
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter – Startup to VC Match Analysis")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set it in a .env file or your environment.")
    st.stop()

# Upload section
st.subheader("📥 Upload Your Founder Document")
uploaded_files = st.file_uploader("Upload .pdf, .txt, or .docx", type=["pdf", "txt", "docx"], accept_multiple_files=True)
if uploaded_files:
    st.session_state["founder_docs"] = uploaded_files

# Run section
if st.button("🚀 Run VC Analysis"):
    if "founder_docs" not in st.session_state or not st.session_state["founder_docs"]:
        st.warning("Please upload your white paper or concept document before running the analysis.")
    else:
        with st.spinner("Running full VC landscape analysis..."):
            try:
                results = run_orchestration(st.session_state["founder_docs"])
                st.session_state["results"] = results
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                logger.exception("Error during orchestration")

# Results Display
results = st.session_state.get("results")
if results:
    st.subheader("✅ Summary at a Glance")

    st.markdown("### 🔍 Top 3 VC Matches")
    for match in results["matches"][:3]:
        st.markdown(f"- **{match['vc_url']}** – Match Score: {match['score']}")

    st.markdown("### 🎯 Closest Similar Startups")
    for comp in results["similar_companies"]:
        st.markdown(f"- **{comp['company_name']}** (Backed by {comp['vc_url']}) – Similarity: {comp['similarity']}")

    st.markdown("### 🧠 VC Landscape Insights")
    st.info(results["gap"])

    st.subheader("📊 Cluster Map")
    if "cluster_plot" in results["visuals"]:
        st.image("data:image/png;base64," + results["visuals"]["cluster_plot"])

    st.subheader("📡 VC Relationship View")
    if "relationship_plot" in results["visuals"]:
        st.image("data:image/png;base64," + results["visuals"]["relationship_plot"])

    st.subheader("💬 Ask VC Hunter Anything")
    st.markdown("_Chat context loaded from founder + VC embeddings_")
    st.text_area("Chat context (for future integration)", value=results["chat_context"], height=250)
