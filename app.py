import streamlit as st
import os
import embed_file
from llm_utils import convert_message, get_rag_chain, get_rag_fusion_chain


if os.path.exists(secrets_file_path):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        else:
            print("OpenAI API Key not found in environment variables")
    except FileNotFoundError:
        print('Secrets file not found')
else:
    print('Secrets file not found')

if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
else:
    if st.sidebar.button("Embed Documents"):
        st.sidebar.info("Embedding documents...")
        try:
            embed_file.embed_all_pdf_docs()
            st.sidebar.info("Done!")
        except Exception as e:
            st.sidebar.error(e)
            st.sidebar.error("Failed to embed documents.")


st.title("annika.llingerie асистент")

chosen_file = st.radio(
    "Choose a file to search", embed_file.get_all_index_files(), index=0
)

if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()


rag_method_map = {
    'Basic RAG': get_rag_chain,
    'RAG Fusion': get_rag_fusion_chain
}
chosen_rag_method = st.radio(
    "Choose a RAG method", rag_method_map.keys(), index=0
)
get_rag_chain_func = rag_method_map[chosen_rag_method]

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()

        retrieval_status = retrival_container.status("**Context Retrieval**")
        queried_questions = []
        rendered_questions = set()


        def update_retrieval_status():
            for q in queried_questions:
                if q in rendered_questions:
                    continue
                rendered_questions.add(q)
                retrieval_status.markdown(f"\n\n`- {q}`")


        def retrieval_cb(qs):
            for q in qs:
                if q not in queried_questions:
                    queried_questions.append(q)
            return qs

        custom_chain = get_rag_chain_func(chosen_file, retrieval_cb=retrieval_cb)

        if "messages" in st.session_state:
            chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

        full_response = ""
        for response in custom_chain.stream(
                {"input": prompt, "chat_history": chat_history}
        ):
            if "output" in response:
                full_response += response["output"]
            else:
                full_response += response.content

            message_placeholder.markdown(full_response + "▌")
            update_retrieval_status()

        retrieval_status.update(state="complete")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})