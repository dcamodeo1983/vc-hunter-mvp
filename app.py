# app.py

import streamlit as st
import os
import logging
from agents.llm_embed_gap_match_chat import generate_chatbot_response
from agents.founder_doc_reader_and_orchestrator import run_full_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
if "OPENAI_API_KEY" not in os.environ:
    st.error("OpenAI API key not found. Please set it as an environment variable 'OPENAI_API_KEY'.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter: Founder Intelligence App")

st.markdown("""
This app analyzes your startup's white paper and matches it with top venture capital firms based on their actual portfolio behavior and investment focus.
""")

# Founder document upload
uploaded_file = st.file_uploader("Upload your startup white paper (PDF or text file)", type=["pdf", "txt"])

# VC URLs: hardcoded list for now
vc_urls = [
    "https://a16z.com", "https://luxcapital.com", "https://foundersfund.com", "https://8vc.com",
    "https://firstround.com", "https://sequoiacap.com", "https://benchmark.com", "https://union.vc",
    "https://cofoundpartners.com", "https://drivecapital.com", "https://lightspeedvp.com", "https://root.vc",
    "https://wing.vc", "https://greylock.com", "https://signalfire.com", "https://accel.com",
    "https://boldstart.vc", "https://initialized.com", "https://craftventures.com", "https://upfront.com"
]

if uploaded_file is not None:
    st.success("White paper uploaded successfully.")
    run_button = st.button("Run Analysis")

    if run_button:
        with st.spinner("Running full founder-to-VC analysis... this may take a few minutes"):
            try:
                founder_bytes = uploaded_file.read()
                results = run_full_pipeline(
                    founder_doc_bytes=founder_bytes,
                    vc_urls=vc_urls
                )
                logger.info("Pipeline executed successfully.")

                st.subheader("📌 Summary of Your Startup")
                st.write(results['founder_summary'])

                st.subheader("🔎 Top Matching VC Firms")
                for match in results['matches']:
                    st.markdown(f"**{match['vc_name']}**")
                    st.markdown(f"🔗 [Website]({match['vc_url']})")
                    st.markdown(f"**Why it matches:** {match['match_reason']}")
                    st.markdown("---")

                st.subheader("🧠 Similar Companies in the VC Landscape")
                for company in results.get("similar_companies", []):
                    st.markdown(f"- **{company['name']}**: {company['description']}, funded by {company['vc']}")

                st.subheader("📊 VC Clusters and Strategic Patterns")
                if results['visuals'].get('clusters'):
                    st.pyplot(results['visuals']['clusters'])
                else:
                    st.warning("No cluster visualization available.")

                st.subheader("🤝 VC Relationships & Competitive Dynamics")
                if results['visuals'].get('relationships'):
                    st.pyplot(results['visuals']['relationships'])
                else:
                    st.warning("No relationship visualization available.")

                st.subheader("🌌 Gap / White Space Analysis")
                st.markdown(results.get('gap', 'No gap analysis available.'))

                st.subheader("💬 Chat With Your Results")
                user_query = st.text_input("Ask about the VC landscape, fit, or competition")
                if user_query:
                    chatbot_response = generate_chatbot_response(
                        query=user_query,
                        founder_summary=results['founder_summary'],
                        vc_summaries=results['vc_summaries']
                    )
                    st.markdown(f"**AI Response:** {chatbot_response}")

            except Exception as e:
                logger.error(f"Error during analysis: {e}", exc_info=True)
                st.error(f"An error occurred: {e}")

