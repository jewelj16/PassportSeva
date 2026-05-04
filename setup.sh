#!/bin/bash
python3 scripts/build_kb.py

streamlit run app.py --server.port=$PORT --server.address=0.0.0.0