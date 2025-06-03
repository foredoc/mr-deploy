#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Run Streamlit on the port specified by Cloud Run ($PORT),
# listening on all interfaces (0.0.0.0).
# --server.headless=true is useful for running in a container if not already default.
# --server.fileWatcherType none is recommended for production.
streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none \
    --server.headless true

