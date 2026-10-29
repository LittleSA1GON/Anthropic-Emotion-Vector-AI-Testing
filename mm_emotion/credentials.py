"""Read credentials without persisting them to disk or experiment metadata."""
import os


def model_token(explicit=None):
    value = explicit if explicit is not None else os.environ.get("HF_TOKEN")
    return value.strip() if value and value.strip() else None


def load_kaggle_model_secret():
    """Use an attached Kaggle secret, retaining an existing environment token."""
    if model_token():
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return False  # Public models work without authentication.
    if value and value.strip():
        os.environ["HF_TOKEN"] = value.strip()
        return True
    return False


def credential_panel():
    import streamlit as st
    with st.sidebar.expander("Your model access", expanded=False):
        st.text_input("Hugging Face read token (optional for public models)",
                      type="password", key="hf_token",
                      help="Kept in this session; never written to experiment files. Gated models also require license approval.")
        st.caption("Kaggle API credentials are used by your local Kaggle CLI, not model inference. See docs/QUICKSTART.md.")
    return model_token(st.session_state.get("hf_token") or None)
