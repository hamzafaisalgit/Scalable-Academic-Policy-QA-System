# File Structure
├── data/
│   ├── handbook.pdf
│   └── chunks.json
│
├── src/
│   ├── ingestion.py        # PDF → text → chunks
│   ├── minhash_lsh.py      # MinHash + LSH
│   ├── simhash.py          # SimHash
│   ├── tfidf.py            # baseline
│   ├── retrieval.py        # combine all methods
│   ├── answer.py           # extractive / LLM
│   ├── evaluation.py       # metrics + comparison
│   └── main.py             # run everything
│
├── app.py                  # Streamlit UI (optional but nice)
├── requirements.txt
└── README.md
