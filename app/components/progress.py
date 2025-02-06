import streamlit as st

def show_progress_message(message: str, thinking: bool = False):
    """Show a progress message in the chat interface."""
    if thinking:
        with st.empty():
            st.markdown(f"*{message}...*")
    else:
        st.toast(message)
