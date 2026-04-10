from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        import re
        parts = re.split(r'([.!?]\s+|\.\n)', text)
        sentences = []
        current = ""
        for i in range(len(parts)):
            if i % 2 == 0:
                current += parts[i]
            else:
                current += parts[i]
                stripped = current.strip()
                if stripped:
                    sentences.append(stripped)
                current = ""
        if current.strip():
            sentences.append(current.strip())
            
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunks.append(" ".join(sentences[i : i + self.max_sentences_per_chunk]))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        separator = "" 
        for s in remaining_separators:
            if s == "":
                separator = s
                break
            if s in current_text:
                separator = s
                break
                
        splits = current_text.split(separator) if separator else list(current_text)
        
        good_splits = []
        next_seps = remaining_separators[remaining_separators.index(separator) + 1:] if separator in remaining_separators else []
        for s in splits:
            if not s:
                continue
            if len(s) > self.chunk_size:
                if next_seps:
                    good_splits.extend(self._split(s, next_seps))
                else:
                    for i in range(0, len(s), self.chunk_size):
                        good_splits.append(s[i:i+self.chunk_size])
            else:
                good_splits.append(s)
                
        final_chunks = []
        current_chunk = []
        current_len = 0
        for s in good_splits:
            add_len = len(s) + (len(separator) if current_chunk else 0)
            if current_len + add_len > self.chunk_size and current_chunk:
                final_chunks.append(separator.join(current_chunk))
                current_chunk = []
                current_len = 0
            
            current_chunk.append(s)
            current_len += len(s) + (len(separator) if len(current_chunk) > 1 else 0)
            
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
        return final_chunks


class HeaderAwareChunker:
    """
    Chunk text by Markdown-style header sections and fallback to paragraph/sentence splits.

    This strategy is useful for documents that use explicit sections (e.g. `##`, `###`) and
    where we want each chunk to preserve a coherent section of content.
    """

    def __init__(self, chunk_size: int = 600) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        import re

        sections = re.split(r'(?=^#{1,6}\s+)', text, flags=re.MULTILINE)
        chunks: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_long_section(section))
        return chunks

    def _split_long_section(self, text: str) -> list[str]:
        import re

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if len(paragraphs) <= 1:
            return self._split_by_sentences(text)

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if not current:
                current = paragraph
                continue
            if len(current) + len(paragraph) + 2 <= self.chunk_size:
                current = f"{current}\n\n{paragraph}"
            else:
                if len(current) <= self.chunk_size:
                    chunks.append(current)
                else:
                    chunks.extend(self._split_by_sentences(current))
                current = paragraph

        if current:
            if len(current) <= self.chunk_size:
                chunks.append(current)
            else:
                chunks.extend(self._split_by_sentences(current))
        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        import re

        parts = re.split(r'([.!?]\s+|\.\n)', text)
        sentences: list[str] = []
        current = ""
        for i, part in enumerate(parts):
            current += part
            if i % 2 == 1 or i == len(parts) - 1:
                stripped = current.strip()
                if stripped:
                    sentences.append(stripped)
                current = ""
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue
            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk), self.chunk_size):
                    final_chunks.append(chunk[i : i + self.chunk_size])
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        results = {}
        fc = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        fc_chunks = fc.chunk(text)
        results['fixed_size'] = {
            'count': len(fc_chunks),
            'avg_length': sum(len(c) for c in fc_chunks) / max(1, len(fc_chunks)),
            'chunks': fc_chunks
        }
        
        sc = SentenceChunker(max_sentences_per_chunk=3)
        sc_chunks = sc.chunk(text)
        results['by_sentences'] = {
            'count': len(sc_chunks),
            'avg_length': sum(len(c) for c in sc_chunks) / max(1, len(sc_chunks)),
            'chunks': sc_chunks
        }
        
        rc = RecursiveChunker(chunk_size=chunk_size)
        rc_chunks = rc.chunk(text)
        results['recursive'] = {
            'count': len(rc_chunks),
            'avg_length': sum(len(c) for c in rc_chunks) / max(1, len(rc_chunks)),
            'chunks': rc_chunks
        }
        return results
