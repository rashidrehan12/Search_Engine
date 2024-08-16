import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="LangChain Search Chat", layout="wide")

# Custom CSS for a dark theme with enhanced chat bubbles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background: #121212;
        color: #e0e0e0;
        margin: 0;
    }

    .main {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e1e1e;
        backdrop-filter: blur(10px);
    }

    .sidebar .sidebar-content {
        background-color: #2e3b4e;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }

    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #0056b3;
    }

    .stTextInput>div>div>input {
        background-color: #2c2c2c;
        color: #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }

    .stSlider > div > div > div {
        color: #e0e0e0;
    }

    .stSelectbox > div > div > div > div {
        color: #e0e0e0;
    }

    .stChatMessage {
        border-radius: 15px;
        margin: 10px 0;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        font-size: 16px;
        line-height: 1.5;
        display: flex;
        align-items: center;
        animation: fadeIn 0.5s ease-in;
    }

    .stChatMessage.user {
        background: linear-gradient(135deg, #007bff, #00d2ff);
        color: white;
        justify-content: flex-end;
    }

    .stChatMessage.user:before {
        content: '👤';
        font-size: 20px;
        margin-right: 10px;
    }

    .stChatMessage.assistant {
        background: linear-gradient(135deg, #ff5858, #f09819);
        color: white;
        justify-content: flex-start;
    }

    .stChatMessage.assistant:before {
        content: '🤖';
        font-size: 20px;
        margin-right: 10px;
    }

    .stSidebar .stSidebarContent h3 {
        color: white;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for settings
st.sidebar.title("⚙ Settings")
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key:", type="password")

# Display a message if the API key is not provided
if not api_key:
    st.write(
        "Please enter your Groq API key to use the LangChain Search Chat."
    )
else:
    # Arxiv and Wikipedia Tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    search = DuckDuckGoSearchRun(name="Search")

    # App Title
    st.title("🔎 AI Chat Assistant")

    # Instructions Section
    st.markdown(
        """
        Welcome to the enhanced LangChain Search Chat! This chatbot can search the web and provide answers using multiple sources.
        """
    )

    num_results = st.sidebar.slider("📊 Number of Results", 1, 5, 3)

    # Add Clear Chat History button
    if st.sidebar.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}]
        st.experimental_rerun()

    # Initialize messages if not already done
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f"<div class='stChatMessage {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Handle user input
    if prompt := st.chat_input(placeholder="Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='stChatMessage user'>{prompt}</div>", unsafe_allow_html=True)

        # Initialize LLM and tools
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [search, arxiv, wiki]

        # Select the agent type
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION

        search_agent = initialize_agent(tools, llm, agent=agent, handling_parsing_errors=True)

        # Display a loading spinner while processing
        with st.spinner("Processing your request..."):
            try:
                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(f"<div class='stChatMessage assistant'>{response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, something went wrong while processing your request. Please try again."})
                st.markdown(f"<div class='stChatMessage assistant'>Sorry, something went wrong while processing your request. Please try again.</div>", unsafe_allow_html=True)

    # Optional: Display chat history
    if st.sidebar.checkbox("Show Chat History"):
        st.sidebar.subheader("Chat History")
        for i, msg in enumerate(st.session_state.messages):
            st.sidebar.write(f"{i+1}: {msg['role']} - {msg['content']}")

    # User feedback section
    st.sidebar.subheader("👍 Your Feedback")
    feedback = st.sidebar.radio("Was the response helpful?", ("Yes", "No"))
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.write(f"Thank you for your feedback: {feedback}")
