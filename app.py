
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from agents.founder_doc_reader_and_orchestrator import run_orchestration, run_chat_agent

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

# Load VC URLs
vc_urls = []
if os.path.exists("vc_urls.txt"):
    with open("vc_urls.txt") as f:
        vc_urls = [url.strip() for url in f.readlines()]
else:
    st.error("Missing vc_urls.txt. Please add VC URLs to analyze.")
    st.stop()

# Run section
if st.button("🚀 Run VC Analysis"):
    if "founder_docs" not in st.session_state or not st.session_state["founder_docs"]:
        st.warning("Please upload your white paper or concept document before running the analysis.")
    else:
        with st.spinner("Running full VC landscape analysis..."):
            try:
                results = run_orchestration(st.session_state["founder_docs"], vc_urls)
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
        st.markdown(
            f"**{match['vc_url']}**  
"
            f"• Match Score: {match['score']}  
"
            f"• Why a Match: _{match['why_match']}_  
"
            f"• Messaging Advice: {match['messaging_advice']}"
        )

    st.markdown("### 🎯 Closest Similar Startups")
    for comp in results["similar_companies"]:
        st.markdown(
            f"**{comp['company_name']}** (Backed by {comp['vc_url']})  
"
            f"• Similarity: {comp['similarity']}  
"
            f"• What They Do: {comp['description']}  
"
            f"• Strategic Insight: _{comp['strategic_insight']}_"
        )

    st.markdown("### 🧠 VC Landscape Insights")
    st.info(results["gap"])

    st.subheader("📊 Cluster Map")
    st.plotly_chart(results["visuals"]["tsne"])

    st.subheader("🔥 Heatmap of Investment Themes")
    st.pyplot(results["visuals"]["heatmap"])

    st.subheader("📡 VC Relationship Graph")
    rel_fig = results["relationships"]
    st.pyplot(rel_fig)

    st.subheader("💬 Ask VC Hunter Anything")
    query = st.text_input("Ask a question about your matches or competitors:")
    if query:
        response = run_chat_agent(results["chat_context"], query)
        st.success(response)
