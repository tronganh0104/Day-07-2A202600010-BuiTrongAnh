from src.chunking import SentenceChunker, RecursiveChunker

with open("data/heart_health/heart_health_01.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Baseline: SentenceChunker (mặc định lấy 3 sentences)
baseline = SentenceChunker(max_sentences_per_chunk=3)
chunks_base = baseline.chunk(text)
avg_base = sum(len(c) for c in chunks_base) / max(1, len(chunks_base))

print("Baseline (SentenceChunker):")
print(f"Count: {len(chunks_base)}")
print(f"Avg Len: {avg_base:.1f}")

# 2. My Tuned Strategy: RecursiveChunker
tuned = RecursiveChunker(
    chunk_size=600,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
)
chunks_tuned = tuned.chunk(text)
avg_tuned = sum(len(c) for c in chunks_tuned) / max(1, len(chunks_tuned))

print("\nTuned (RecursiveChunker):")
print(f"Count: {len(chunks_tuned)}")
print(f"Avg Len: {avg_tuned:.1f}")
