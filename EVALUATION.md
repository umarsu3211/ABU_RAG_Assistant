# Chapter 3.7 & Chapter 4.2 - System Evaluation - ABU RAG Assistant

## 3.7 Evaluation Metrics
The developed system will be evaluated using real administrative queries obtained from students and staff of Ahmadu Bello University.

### 3.7.1 Accuracy
**Definition:** Measures the correctness of the generated responses relative to the information contained in the source documents.

**How we test:**
- Ask 20 real questions: e.g., "How do I register courses?", "What are graduation requirements?", "Late registration procedure?"
- Compare AI answer with original PDF page
- Formula: Accuracy = (Correct Answers / Total Questions) * 100
- Expected: 85-95% because RAG is grounded in verified docs, reduces hallucination.

**Result in our implementation:**
Implemented with FAISS similarity search + metadata citation (Source + Page). Every answer shows evidence passage.

### 3.7.2 Relevance
**Definition:** Assesses how well the retrieved documents and generated responses address the user's query.

**How we test:**
- Use cosine similarity score from FAISS
- Top k=4 chunks retrieved per query
- Human rating 1-5 on relevance

**Implementation:**
```python
results = vectorstore.similarity_search_with_score(query, k=4)
# Score closer to 0 = more relevant (FAISS L2 distance)
