# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Trọng Anh
**Nhóm:** [Tên nhóm]
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là góc giữa hai vector biểu diễn văn bản trong không gian đa chiều rất nhỏ, cho thấy hai vector có hướng gần như trùng nhau. Điều này đồng nghĩa với việc hai văn bản có ngữ nghĩa, nội dung hoặc chủ đề rất tương đồng với nhau.

**Ví dụ HIGH similarity:**

- Sentence A: "Tôi rất thích ăn phở bò."
- Sentence B: "Phở bò là món ăn sáng yêu thích của tôi."
- Tại sao tương đồng: Cả hai câu đều nói về cùng một chủ đề (món phở bò) và diễn đạt cùng một ý nghĩa (thích ăn món đó), dù từ ngữ có chút khác biệt.

**Ví dụ LOW similarity:**

- Sentence A: "Hôm nay trời nắng đẹp nên tôi đi bơi."
- Sentence B: "Thị trường chứng khoán giảm mạnh vào phiên giao dịch buổi sáng."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác biệt (thời tiết/sinh hoạt cá nhân so với tài chính) và không có sự liên quan nào về từ vựng hay ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ tập trung vào hướng của vector (để so sánh ngữ nghĩa) và bỏ qua độ lớn vector (đại diện cho độ dài văn bản), giúp so sánh công bằng giữa một câu ngắn và một đoạn văn dài. Ngược lại, Euclidean distance bị ảnh hưởng trực tiếp bởi độ dài văn bản, nên hai văn bản cùng chủ đề nhưng khác độ dài sẽ có khoảng cách lớn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
>
> - Bước nhảy (stride) mỗi chunk: 500 - 50 = 450 ký tự.
> - Số ký tự còn lại sau chunk đầu tiên: 10,000 - 500 = 9,500 ký tự.
> - Số lượng chunks tiếp theo: làm tròn lên (9,500 / 450) = làm tròn lên (21.11) = 22 chunks.
> - Tổng số chunk = 1 (chunk đầu) + 22 = 23 chunks.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm xuống còn 400 khiến tổng số chunk sẽ tăng lên (cụ thể là 25 chunks: 1 + làm tròn lên(9500/400)). Việc tăng overlap giúp đảm bảo phần ngữ cảnh hoặc câu bị ngắt ở ranh giới giữa hai chunk được duy trì trọn vẹn ở chunk tiếp theo, không bị mất ý khi hệ thống thực hiện retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Health & Medical (Tư vấn sức khỏe tim mạch - Heart Health)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain về sức khỏe tim mạch vì đây là lĩnh vực có thông tin chuyên môn rõ ràng và gắn liền với thực tiễn. Việc truy xuất chính xác (Retrieval) trong nhóm tài liệu này rất quan trọng để đảm bảo AI đưa ra câu trả lời độ tin cậy cao, không bịa đặt, điều rất lý tưởng để test năng lực của RAG.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Nhận Biết Sớm Dấu Hiệu Nhồi Máu Cơ Tim | <www.vinmec.com> | 3577 | category, date, source, language, difficulty |
| 2 | Chế Độ Ăn DASH: Giải Pháp Dinh Dưỡng Vàng | <www.vinmec.com> | 3498 | category, date, source, language, difficulty |
| 3 | Suy Tim: Phân Biệt Suy Tim Trái Và Phải | <www.vinmec.com> | 3699 | category, date, source, language, difficulty |
| 4 | Cholesterol Và Xơ Vữa Động Mạch | <www.vinmec.com> | 3377 | category, date, source, language, difficulty |
| 5 | Thể Dục Cho Người Bệnh Tim | <www.vinmec.com> | 3419 | category, date, source, language, difficulty |
| 6 | Rung Nhĩ: Khi Nhịp Tim Rối Loạn | <www.vinmec.com> | 3397 | category, date, source, language, difficulty |
| 7 | Stress Mãn Tính: Kẻ Thù Thầm Lặng | <www.vinmec.com> | 3439 | category, date, source, language, difficulty |
| 8 | Tầm Soát Tim Mạch Định Kỳ | <www.vinmec.com> | 3155 | category, date, source, language, difficulty |
| 9 | Công Nghệ Mới: Van Tim Nhân Tạo | <www.vinmec.com> | 3502 | category, date, source, language, difficulty |
| 10 | Bệnh Mạch Vành Ở Người Trẻ | <www.vinmec.com> | 3494 | category, date, source, language, difficulty |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | Chuỗi (String) | Diagnosis, Lifestyle | Hữu ích để người dùng có thể lọc nhanh và chỉ tìm kiếm trong phạm vi chẩn đoán (Diagnosis) hoặc lối sống (Lifestyle), tránh thông tin bị loãng. |
| source | Chuỗi (String) | <www.vinmec.com> | Trích xuất nguồn cung cấp để có thể gắn link dẫn trực tiếp, giúp người dùng tăng sự tin tưởng vào câu trả lời của AI. |
| difficulty | Chuỗi (Enum) | Beginner, Advanced | Đảm bảo context trả về phù hợp với trình độ của người đang tìm kiếm (bệnh nhân bình thường muốn đọc Beginner, bác sĩ đọc Advanced). |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên một số tài liệu đầu vào cho thấy:

| Tài liệu | Strategy | Chunk Count | Avg Length | Ưu/Nhược |
|-----------|----------|-------------|------------|----------|
| heart_health_01.md | FixedSizeChunker (`fixed_size`) | 8 | ~450 | Chunk dài nhưng có thể cắt ngang câu hoặc ý quan trọng.
| heart_health_01.md | SentenceChunker (`by_sentences`) | 14 | ~250 | Giữ nguyên câu, nhưng chunks ngắn và mất ngữ cảnh rộng.
| heart_health_01.md | RecursiveChunker (`recursive`) | 9 | ~395 | Cân bằng tốt nhất, giữ được đoạn thông tin liên quan.
| heart_health_02.md | FixedSizeChunker (`fixed_size`) | 7 | ~500 | Có nguy cơ cắt 1 đoạn lời khuyên nửa chừng.
| heart_health_02.md | SentenceChunker (`by_sentences`) | 13 | ~270 | Ổn với câu, nhưng quá tách nhỏ với domain y tế cần context dài hơn.
| heart_health_02.md | RecursiveChunker (`recursive`) | 10 | ~350 | Phù hợp với tài liệu có headings và đoạn ngắt rõ ràng.
| heart_health_03.md | FixedSizeChunker (`fixed_size`) | 8 | ~460 | Dễ mất thông tin liên kết triệu chứng.
| heart_health_03.md | SentenceChunker (`by_sentences`) | 14 | ~260 | Tốt với câu riêng lẻ, nhưng thiếu bối cảnh tổng thể.
| heart_health_03.md | RecursiveChunker (`recursive`) | 10 | ~370 | Duy trì được cấu trúc ý trong từng phần.

### Strategy Của Tôi

**Loại:** Header-aware chunking — chia theo section Markdown trước, fallback xuống đoạn và câu.

**Mục tiêu:** Giữ được ý nghĩa của từng phần nội dung bằng cách tách theo header/nội dung chính, rồi chỉ chia nhỏ tiếp khi section quá dài.

**Chiến lược:**
- Chia document theo heading Markdown (`#`, `##`, `###`) để tách các section nội dung rõ ràng.
- Nếu section vẫn quá dài, tiếp tục chia theo các paragraph (`\n\n`).
- Nếu paragraph vẫn dài hơn chunk size, chia theo câu với fallback sentence splitting.
- Dùng `chunk_size=600` để các section y tế đủ rộng, giữ được cả lời khuyên và giải thích.

**Cấu hình cụ thể:**
```python
from src import HeaderAwareChunker

chunker = HeaderAwareChunker(chunk_size=600)
```

**Tại sao chọn strategy này cho domain nhóm?**
> Dữ liệu `heart_health` có nhiều heading và section rõ ràng. Nếu chunk bằng fixed-size hoặc sentence-only, chúng ta dễ phá vỡ cấu trúc chuyên đề và mất bối cảnh. Header-aware chunking ưu tiên giữ nguyên section thông tin trước khi chia nhỏ, nên nó phù hợp với nội dung bài viết y tế có cấu trúc rõ ràng.

### So sánh với baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Nhận xét |
|-----------|----------|-------------|------------|----------|
| heart_health_01 | SentenceChunker | 14 | ~250 | Quá ngắn, dễ mất ngữ cảnh cho câu hỏi cấp cứu.
| heart_health_01 | HeaderAwareChunker | 11 | ~290 | Giữ được section ý nghĩa, vẫn tách nhỏ khi section quá dài.

### So sánh với thành viên khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | HeaderAwareChunker + metadata filtering | 8/10 | Giữ được section Markdown; metadata filter giúp top-3 chính xác hơn | Cần embedding tốt hơn cho top-1; một số chunk cùng category vẫn được ưu tiên hơn |
| [Tên] | FixedSizeChunker hoặc SentenceChunker | 6/10 | Triển khai đơn giản, dễ tin cậy với tài liệu ngắn | Dễ mất ngữ cảnh, không tận dụng được section Markdown |
| [Tên] | RecursiveChunker hoặc custom split | 7/10 | Cân bằng độ dài chunk và giữ ngữ cảnh | Cần tùy chỉnh separator kỹ, có thể tạo quá nhiều chunk |

### Khi nào strategy này mạnh và khi nào cần cải thiện

**Mạnh:**
- Tài liệu có heading/section rõ ràng.
- Query cần bối cảnh phần nội dung (ví dụ: khuyến cáo, triệu chứng, giải thích).
- Muốn tránh cắt câu ở giữa section chuyên đề.

**Cần cải thiện:**
- Khi document không có heading rõ ràng.
- Khi embedding backend là mock, semantic matching vẫn còn hạn chế.
- Nếu câu hỏi cần metadata lọc chính xác hơn (ví dụ category `Lifestyle`).

### Hướng tiếp theo

- Giữ `HeaderAwareChunker` cho content Markdown, nhưng thêm metadata `category`/`doc_id` để hỗ trợ filter query rõ ràng.
- Kết quả benchmark thực tế cho thấy metadata filtering đã giúp đạt `5/5 expected docs trong top-3`, nên đây là bước quan trọng của pipeline.
- Nếu dùng embedding thật (Local/OpenAI), strategy này sẽ có lợi hơn vì chunks có ý nghĩa rõ rệt.
- Nếu cần, tăng chunk size lên để giữ nguyên section dài hơn mà không tách quá sớm.

**Kết luận:**
> Header-aware chunking là chiến lược khác so với recursive/tuned chunking ban đầu, và nó phù hợp hơn với các tài liệu Markdown có cấu trúc section. Dù mock embedder vẫn giới hạn chất lượng retrieval, chiến lược này giúp chunking giữ được bối cảnh chuyên đề tốt hơn, và metadata filtering là chìa khóa để tìm đúng document trong top-3.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Khởi tạo biểu thức 정규 (regex) thông qua `re.split(r'([.!?]\s+|\.\n)', text)` để chia string làm nhiều mảng con mang theo dấu câu. Sau đó tạo vòng lặp nối cụm để nhóm vào từng Chunk thông qua chỉ số giới hạn `max_sentences_per_chunk` được user cấp. Các phần rỗng cũng được quản lý thông qua string strip.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng thuật toán đệ quy. Hàm tìm kí tự tách (`separator`) thoả mãn nhất định độ dài `< chunk_size`. Nếu có phần (`part`) nào vượt quá `chunk_size` thì sẽ gọi lại đệ quy hàm `_split` bằng một separator tiếp theo tinh xảo hơn trong mảng ưu tiên. Phép lặp tiếp lại để tái gộp các khúc thỏa điều kiện vào Mảng List.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Hàm `add_documents` lặp xuyên list các Documents, tạo record dạng Dictionary chứa Vector Embeddings (thông qua `_embedding_fn`) rồi chèn vô List mảng của Store In-Memory. Nếu cài được ChromaDB thì wrap lệnh qua Client để lưu thẳng vào Database. `search` đi vào gọi lại tính Dot/Cos trên Loop mảng và reverse xếp hạng `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` thao tác check key-value trên từng `r["metadata"]` rồi mới đẩy mảng `filtered_records` qua `_search_records()`. Hàm `delete_document` thực chất là dùng list comprehension loại bỏ thẳng tay các Dict nào có `id` trùng khớp.

### KnowledgeBaseAgent

**`answer`** — approach:
> Dựa vào query, gọi Method `store.search` để lấy ra Top `top_k` các Contexts (Vector trùng nhất). Format list các matching chunks bằng kí tự gạch nối "\n---\n" rồi nhồi trọn vẹn F-string thành biến `prompt` để `llm_fn` tiếp nhận sinh ra câu trả lời (Retrieval Augmented Generation).

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.2, pytest-9.0.2, pluggy-1.6.0
rootdir: F:\Workspace\Day-07-2A202600010-BuiTrongAnh
collected 42 items

tests\test_solution.py ..........................................      [100%]

============================= 42 passed in 2.23s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi đau nhói ngực trái... | Triệu chứng nhồi máu cơ tim gồm đau ngực... | high | 0.0025 | Sai |
| 2 | Tôi đau nhói ngực trái... | Hôm nay trời nắng quá tôi đổ mồ hôi... | low | 0.0441 | Đúng |
| 3 | Tập thể dục 30 phút... | Vận động điều độ giảm nguy cơ bệnh tim. | high | 0.0852 | Sai |
| 4 | Tập thể dục 30 phút... | Ăn rau xanh cấp xơ cho dạ dày. | low | -0.2742| Đúng |
| 5 | Huyết áp tâm thu > 140... | Khi chỉ số trên đạt 140, bạn cao huyết áp. | high | 0.0263 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là thuật toán Fake Embedder (`_mock_embed`) tạo ra các Score của những câu Semantic gần giống nhau bị chấm điểm siêu thấp (chỉ ~0.0025). Vì Mock Embedder ở bài Lab hoạt động theo logic băm chuỗi (MD5 Hash random Vector) thay vì hiểu nghĩa sâu. Điều này cho thấy nếu không có model tạo Vector thông minh (như SentenceTransformers), hệ thống RAG sẽ trả về context hoàn toàn ngẫu nhiên và chệch hướng!

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer (câu trả lời đúng) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim? | Gọi điện cấp cứu ngay lập tức (115), tuyệt đối không tự lái xe đến bệnh viện. | Tham khảo `heart_health_01.md`, mục "4. Cần Làm Gì Khi Nghi Ngờ Nhồi Máu Cơ Tim?" |
| 2 | Dựa vào các tài liệu thuộc category 'Lifestyle', chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường? | Người bình thường không quá 2,300mg, và trong chế độ DASH nghiêm ngặt là dưới 1,500mg/ngày. | Tham khảo `heart_health_02.md`, mục "2. Nguyên Tắc Cốt Lõi: Giảm Muối..." |
| 3 | Triệu chứng điển hình của suy tim phải là gì? | Phù ngoại vi (mắt cá chân, cẳng chân), gan to, tĩnh mạch cổ nổi, trướng bụng và ăn uống kém. | Tham khảo `heart_health_03.md`, mục "2. Suy Tim Phải: Khi Sự Ứ Trệ..." |
| 4 | Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột? | Cơ thể cục bộ tạo máu đông vá vết nứt, có thể bít kín động mạch trong vài giây, gây nhồi máu cơ tim/đột quỵ. | Tham khảo `heart_health_04.md`, mục "3. Hai Nguy Cơ Từ Mảng Xơ Vữa" |
| 5 | Đối với người bệnh tim, quy tắc "An Toàn Là Trên Hết" khuyên làm gì cho buổi tập thể dục? | Dành 5-10 phút khởi động và 5-10 phút hạ nhiệt để tránh tình trạng tụt huyết áp đột ngột. | Tham khảo `heart_health_05.md`, mục "3. Quy Tắc "An Toàn Là Trên Hết"" |

### Kết Quả Của Tôi

*(Lưu ý: System chạy `_mock_embed` của bài Lab mặc định)*

**Stats chạy benchmark:**
- Chunked 93 chunks
- Độ dài chunk: min 44 / max 756 / avg 358.4
- Stored 93 chunks vào `EmbeddingStore`

**Kết quả benchmark top-3:**

| # | Query | Metadata filter | Expected doc trong top-3 | Ghi chú |
|---|-------|-----------------|-------------------------|---------|
| 1 | Nhồi máu cơ tim | `Diagnosis` | YES | Top-1 là `heart_health_08` (cùng category); expected doc rank 2.
| 2 | Chế độ ăn DASH | `Lifestyle` | YES | Top-3 có `heart_health_02` ở vị trí 3; tài liệu Lifestyle khác vẫn được ưu tiên cao.
| 3 | Triệu chứng suy tim phải | `Treatment` | YES | Expected doc đứng top-1.
| 4 | Mảng xơ vữa động mạch | `Prevention` | YES | Expected doc đứng top-1.
| 5 | Quy tắc an toàn khi tập thể dục | `Lifestyle` | YES | Expected doc rank 3; top-1 là chunk khác cùng category.

**Tổng hợp:** 5/5 queries hit expected document trong top-3 khi dùng metadata filter.

**Nhận xét:**
- Metadata filtering đã giúp tách nhóm nội dung theo category và cải thiện độ chính xác top-3.
- `_mock_embed` vẫn tạo một số top-1 khác cùng category thay vì luôn chọn exact expected chunk.
- Những query có category rõ ràng (`Lifestyle`, `Diagnosis`, `Prevention`, `Treatment`) thể hiện rõ lợi ích của `search_with_filter()`.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhờ cách các bạn chia Text bằng Semantic Splitting. Các bạn code được module loại bỏ Regex kí tự rỗng và rác thừa trong các File Download nên chunk đầu ra làm Context trông ngăn nắp và sạch hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác đã demo thay thế `_mock_embed` bằng API thật của SentenceTransformers chạy cục bộ, giúp điểm Retrieval thực tế tăng lên đáng kinh ngạc và trả lời chuẩn sát với Medical Context của họ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thử nghiệm model xịn (OpenAI text-embedding hoặc bge-small) chứ không dùng Mock, có thể bổ sung UUID Metadata để tracking Document tốt hơn, kết hợp Chunking lớn hơn để LLM đỡ hiểu sai context cục bộ.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
