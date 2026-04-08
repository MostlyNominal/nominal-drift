"""Chat Copilot page."""
import streamlit as st


def render():
    st.title("💬 Chat Copilot")
    st.caption(
        "Natural language interface · Powered by Ollama (local) · Routes to science modules"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about sensitization, diffusion, datasets...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _handle_prompt(prompt)
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})


def _handle_prompt(prompt: str) -> str:
    """Route prompt to tool, or fall back to Ollama."""
    from nominal_drift.core.tool_router import route_intent
    from nominal_drift.templates.factory import template_from_chat_intent

    intent = template_from_chat_intent(prompt)
    tool_response = route_intent(intent, prompt)
    if tool_response:
        return tool_response
    # Fall back to Ollama
    try:
        from nominal_drift.llm.client import OllamaClient

        client = OllamaClient()
        if client.is_available():
            system = (
                "You are Nominal Drift's materials engineering assistant. "
                "You help with sensitization, diffusion, and crystal structure analysis. "
                "Always note that you route quantitative questions to the science modules."
            )
            return client.generate(prompt, system_prompt=system, max_tokens=400)
        return (
            "⚠️ Ollama is not running. Start it with `ollama serve` and pull a model. "
            f"\nDetected intent: **{intent}**"
        )
    except Exception as e:
        return f"⚠️ Could not reach Ollama: {e}\nDetected intent: **{intent}**"
