import streamlit as st
import logging
import ollama
from typing import Dict, Tuple, List, Any

from src.agent import get_graph, run_graph

graph = get_graph()


@st.cache_resource(show_spinner=True)
def get_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:

    model_names = tuple(model["name"] for model in models_info["models"])

    return model_names


def app():
    st.set_page_config(
        page_title="Local OLLAMA RAG Agent",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    st.subheader("Local Ollama RAG Agent", divider="gray", anchor=False)

    # Initialize session state
    if "question" not in st.session_state:
        st.session_state["question"] = []
    if "response" not in st.session_state:
        st.session_state["response"] = []

    message_container = st.container(height=400, border=True)

    # Display questions history
    for i, question in enumerate(st.session_state["question"]):
        avatar = "🤖" if question["role"] == "assistant" else "😎"
        with message_container.chat_message(question["role"], avatar=avatar):
            st.markdown(question["content"])

    for i, answer in enumerate(st.session_state["response"]):
        avatar = "🤖" if answer["role"] == "assistant" else "😎"
        with message_container.chat_message(answer["role"], avatar=avatar):
            st.markdown(answer["content"])

    if prompt := st.chat_input("Enter a question here...", key="chat_input"):
        print(f"this is the prompt: {prompt}")
        try:
            # Add user message to chat
            print(f"previous question: {st.session_state['question']}")
            st.session_state["question"].append({"role": "user", "content": prompt})
            print(f"1")
            with message_container.chat_message("user", avatar="😎"):
                print(f"2")
                st.markdown(prompt)

            # Process and display assistant response
            print(f"3")
            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):

                    inputs = {"question": prompt, "max_retries": 3}
                    response = run_graph(graph=graph, inputs=inputs)
                    st.markdown(response)

            st.session_state["response"].append(
                {"role": "assistant", "content": response}
            )

        except Exception as e:
            st.error(e, icon="⛔️")
            logger.error(f"Error processing prompt: {e}")
    else:
        st.warning("......")


if __name__ == "__main__":
    app()
