import sys
from frontend.pipeline import load_pipeline
from src.llm.generator import ResponseGenerator

pipeline, error = load_pipeline()
if not pipeline:
    print(error)
    sys.exit(1)

gen = ResponseGenerator(retriever=pipeline["retriever"])

queries = [
    "How do I apply for Ayushman Bharat?",
    "Who is eligible for Ayushman bharat scheme ?",
    "What documents are required for Ayushman Bharat?",
    "how can i know the status of my beneficiary?",
    "How long does it take to be beneficiary after filling the application form?"
]

for q in queries:
    print(f"\n--- QUERY: {q} ---")
    res = gen.generate(q)
    print("ANSWER:")
    print(res["answer"])
    print("SOURCES:", [s['source_file'] for s in res['sources']])
