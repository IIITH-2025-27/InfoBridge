"""Minimal Streamlit entrypoint for InfoBridge."""

import streamlit as st

from frontend.app_shell import run_app


def main() -> None:
    """Configure page settings and run the frontend app shell."""
    st.set_page_config(
        page_title="InfoBridge — Indian Government Services AI",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={},
    )
    run_app()


if __name__ == "__main__":
    main()
