# RETRIEVAL-AUGMENTED GENERATION (RAG)-BASED ADMINISTRATIVE KNOWLEDGE ASSISTANT FOR AHMADU BELLO UNIVERSITY (ABU), ZARIA
### COEN807 Term Project - MSc Computer Engineering
### Student Project - NOT OFFICIAL ABU WEBSITE - Hosted Independently on GitHub

---

## 1. TITLE PAGE
Development of a Retrieval-Augmented Generation (RAG)-Based Administrative Knowledge Assistant for Ahmadu Bello University

## 2. DECLARATION
I declare that this work was carried out by me... ABU MSc Computer Engineering

## 3. CERTIFICATION/APPROVAL PAGE
Certified by Prof. M.B Muaz...

## 4. ABSTRACT (Page 2 of your doc)
Access to accurate administrative information remains a significant challenge for students and staff in many Nigerian universities. Information relating to course registration, academic regulations, examination procedures, fee payment, and graduation requirements is often dispersed across multiple documents and platforms, resulting in delays, confusion, and misinformation. This study proposes the development of a Retrieval-Augmented Generation (RAG)-based Administrative Knowledge Assistant for Ahmadu Bello University (ABU), Zaria. The aim of the study is to develop an AI-powered system capable of providing accurate, reliable, and source-based responses to administrative queries using official university documents. The methodology involves the collection of relevant administrative documents, document preprocessing, knowledge base development, implementation of a RAG architecture, and integration of a Large Language Model (LLM) for response generation. A web-based user interface will be developed to enable interaction between users and the system. The performance of the developed system will be evaluated using metrics such as accuracy, relevance, response time, retrieval effectiveness, and user satisfaction. The expected outcome is an intelligent administrative assistant that improves access to university information, reduces misinformation, enhances administrative efficiency, and supports digital transformation initiatives within Ahmadu Bello University. The study demonstrates the potential of Retrieval-Augmented Generation technology in addressing administrative information management challenges in higher education institutions.

Keywords: Retrieval-Augmented Generation, Artificial Intelligence, Knowledge Base, Large Language Model, Administrative Information System, Ahmadu Bello University.

## 5. TABLE OF CONTENTS (As per your 10-point structure)
1. Title Page
2. Declaration
3. Certification
4. Abstract
5. Table of Contents
6. Chapter One – Introduction (5-6 pages)
7. Chapter Two – Literature Review (8-10 pages)
8. Chapter Three – Methodology (5-6 pages)
9. Chapter Four – Expected Results (2-3 pages)
10. Chapter Five – Summary, Conclusion and Recommendations (1-2 pages)
11. References

---

## CHAPTER ONE: INTRODUCTION

### 1.1 Background of the Study
Access to accurate administrative information is essential... In many Nigerian universities, such information is distributed across multiple documents, websites, notice boards, handbooks... RAG system first retrieves relevant information from trusted documents and then generates responses based on the retrieved content. This enhances accuracy, transparency, trustworthiness.

### 1.2 Statement of the Problem
Administrative information is scattered across multiple sources... Students face challenges... Existing systems provide static documents... Conventional AI chatbots may generate inaccurate responses because they are not grounded in verified institutional documents.

### 1.3 Aim of the Study
To develop an AI-powered administrative knowledge assistant that provides accurate, reliable, and source-based responses to queries relating to university policies, regulations, and procedures.

### 1.4 Objectives
i. To collect and organize official university administrative documents.
ii. To develop a searchable knowledge base from the collected documents.
iii. To implement a Retrieval-Augmented Generation (RAG) framework for information retrieval and response generation.
iv. To design a user-friendly interface for interacting with the system.
v. To evaluate the performance of the developed system using real administrative queries.

### 1.5 Research Questions
- How can official university documents be organized into a searchable knowledge repository?
- How effective is a RAG framework in answering university administrative questions?
- What level of accuracy and relevance can be achieved?
- How can an AI assistant improve access to administrative information?

### 1.6 Significance of the Study
Benefits to students, staff, university management. Reduces workload, improves dissemination, contributes to knowledge on RAG in education.

### 1.7 Scope
Focuses on administrative information within ABU Zaria, using student handbooks, academic regulations, registration guidelines etc.

### 1.8 Definition of Terms
AI, LLM, Information Retrieval, RAG, Knowledge Base, Chatbot

---

## CHAPTER TWO: LITERATURE REVIEW

### 2.1 Artificial Intelligence in Education
AI tasks: learning, reasoning... Applications: tutoring systems, automated assessment, virtual assistants.

### 2.2 Large Language Models (LLMs)
Examples: GPT, Gemini, Claude, LLaMA. Tasks: QA, summarization, translation. Challenges: hallucination, outdated knowledge.

### 2.3 Information Retrieval Systems
Process: Document Collection, Indexing, Query Processing, Retrieval and Ranking, Result Presentation. Techniques: Keyword, Vector-space, TF-IDF, BM25, Semantic search.

### 2.4 Retrieval-Augmented Generation (RAG)
Components: Retrieval Component, Generation Component. Workflow: User query → Relevant documents retrieved → Provided to LLM → Grounded response. Advantages: Improved factual accuracy, Reduced hallucination, Up-to-date info, Increased transparency, Better user trust.

### 2.5 AI Chatbots in Higher Education
Applications: Admission inquiries, Course registration, Examination info, Fee payment, Campus navigation. Benefits: 24/7 availability, Reduced workload, Faster response, Improved satisfaction. Limitations of traditional rule-based.

### 2.6 Related Studies
- Okonkwo and Ade-Ibijola (2021): AI chatbot for Nigerian universities - improved response times but predefined responses.
- Kasneci et al. (2023): LLMs in higher ed - potential but hallucination concerns.
- Lewis et al. (2020): Introduced RAG framework - improvements in factual accuracy.
- Peng et al. (2023): RAG QA in education - improved relevance and trust.
- Alshahrani and Almutairi (2024): University chatbot with RAG - high accuracy, positive feedback.

### 2.7 Research Gap
1. Many chatbots rule-based, cannot answer complex queries.
2. Standalone LLMs prone to hallucination, not grounded in official docs.
3. Limited research on RAG for Nigerian university administrative info. This study bridges gap.

---

## CHAPTER THREE: METHODOLOGY

### 3.1 Research Design
Design and Development Research (DDR) approach for RAG-based assistant. Collect docs, construct knowledge base, implement retrieval, integrate LLM, evaluate performance.

### 3.2 Data Collection
Official documents: Student Handbook, Academic Regulations, Registration Guidelines, Examination Policies, Fee Payment Procedures, Graduation Requirements, Senate Approved Regulations, Faculty and Departmental Administrative Guidelines, Other official publications. Source: university website, academic planning unit, registry.

### 3.3 Document Preprocessing
#### 3.3.1 Document Conversion: PDF, Word → machine-readable text
#### 3.3.2 Text Cleaning: Remove page numbers, headers, footers, special symbols
#### 3.3.3 Text Segmentation: Divide into smaller chunks (800 chars, 100 overlap) for efficient retrieval
#### 3.3.4 Metadata Creation: Document title, Source department, Publication date, Section title, Document category
#### 3.3.5 Embedding Generation: Transform chunks into vector representations using pre-trained embedding model (Sentence Transformers)

### 3.4 Knowledge Base Development
Process: Organizing docs into collections, Generating embeddings, Storing embeddings in vector database (FAISS/ChromaDB), Indexing documents, Linking passages to original sources for citation. Purpose: Ensure responses grounded in official docs.

### 3.5 RAG Architecture
#### 3.5.1 Retrieval Module: Query → vector embedding → cosine similarity → most relevant chunks
#### 3.5.2 Generation Module: Retrieved chunks → LLM → accurate context-aware response
#### 3.5.3 System Workflow
**User Query → Query Embedding → Vector Search → Relevant Document Retrieval → LLM Processing → Response Generation → User Interface**
This minimizes hallucination, ensures verified info.

### 3.6 System Implementation
**Software Tools Table:**
- Programming Language: Python
- LLM Framework: LangChain
- Vector Database: FAISS/ChromaDB
- Embedding Model: Sentence Transformers
- User Interface: Streamlit
- Database: SQLite
- Document Processing: PyPDF, Docx2txt

**Implementation Steps:**
1. Collection of university administrative documents
2. Document preprocessing and chunking
3. Generation of vector embeddings
4. Construction of vector database
5. Integration of retrieval and generation
6. Development of web-based UI
7. System testing and debugging
8. Deployment for evaluation

UI will allow students to enter questions and receive source-based responses.

### 3.7 Evaluation Metrics (Using real administrative queries)
- **3.7.1 Accuracy:** Correctness relative to source documents
- **3.7.2 Relevance:** How well retrieved docs address query
- **3.7.3 Response Time:** Average time to retrieve and generate
- **3.7.4 User Satisfaction:** Questionnaire 5-point Likert Scale: 5=Strongly Agree, 4=Agree, 3=Neutral, 2=Disagree, 1=Strongly Disagree
- **3.7.5 Retrieval Precision:** Proportion of retrieved docs that are relevant
- **3.7.6 Retrieval Recall:** Proportion of all relevant docs successfully retrieved

---

## CHAPTER FOUR: EXPECTED RESULTS AND DISCUSSION

### 4.1 Expected System Features
#### 4.1.1 Intelligent Question Answering
Examples: How do I register my courses? What are requirements for graduation? What is procedure for late registration? What are examination regulations?

#### 4.1.2 Document-Based Retrieval
Unlike traditional chatbots, will retrieve info from verified ABU documents before generating.

#### 4.1.3 Source Referencing
Provide references to source documents (Document name + Page number) for transparency and verification.

#### 4.1.4 User-Friendly Interface
Web-based interface, simple submission, short response time.

#### 4.1.5 Search and Retrieval Capability
Efficient searching through vector-based retrieval.

#### 4.1.6 Knowledge Base Management
Structured repository, updatable when new docs available.

### 4.2 Expected Performance
- **4.2.1 Response Accuracy:** Highly accurate due to official docs, minimize unsupported responses
- **4.2.2 Retrieval Relevance:** Vector mechanism identifies most relevant sections
- **4.2.3 Fast Response Time:** Vector DB enables few seconds response
- **4.2.4 Improved User Satisfaction:** Timely accurate info reduces frustration
- **4.2.5 Reliability and Trustworthiness:** Official sources + references enhance confidence
- **4.2.6 Reduced Information Retrieval Effort:** No need to manually search lengthy docs

### 4.3 Benefits to ABU
- **4.3.1 Enhanced Access:** Centralized platform
- **4.3.2 Improved Administrative Efficiency:** Reduce workload for repetitive enquiries
- **4.3.3 Reduction in Misinformation:** Responses from verified docs
- **4.3.4 Better Student Experience:** Quick access to registration, exams, fees, graduation info
- **4.3.5 Promotion of Digital Transformation:** AI adoption in higher education
- **4.3.6 Foundation for Future AI Applications:** Framework for admission, advising, library support
- **4.3.7 Institutional Knowledge Preservation:** Structured repository for regulations

---

## CHAPTER FIVE: SUMMARY, CONCLUSION AND RECOMMENDATIONS

### 5.1 Summary
Proposed RAG-based Administrative Knowledge Assistant for ABU motivated by challenges accessing administrative info scattered across multiple docs. Aim, methodology, evaluation criteria as above. Expected outcome: reliable admin info system.

### 5.2 Conclusion
Increasing AI opportunities. Unlike traditional chatbots without evidence, RAG retrieves from verified docs before generating. Improves accuracy, reliability, transparency, trustworthiness. Enhances access, reduces misinformation, supports digital transformation. Demonstrates practical AI application in Nigerian higher education.

### 5.3 Recommendations
1. ABU should support development and deployment of AI-powered administrative knowledge systems.
2. Administrative units should ensure official regulations are regularly updated and made available in digital formats.
3. Future implementations should include multilingual support for local Nigerian languages.
4. Knowledge base should be continuously updated.
5. Future research should investigate voice-based interaction.
6. Similar RAG systems for admissions, academic advising, library support, student affairs.
7. Further studies evaluate long-term effectiveness, scalability, user acceptance in Nigerian universities.

---

## REFERENCES
- Ahmadu Bello University. (2024). Academic regulations and guidelines for undergraduate students.
- Brown et al. (2020). Language models are few-shot learners. NeurIPS.
- Gao et al. (2024). Retrieval-Augmented Generation for LLMs: A survey. ACM Computing Surveys.
- Lewis et al. (2020). Retrieval-Augmented Generation for knowledge-intensive NLP tasks. NeurIPS.
- Manning et al. (2008). Introduction to information retrieval. Cambridge.
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP.
- And others as per your doc page 28.

---

## DISCLAIMER
**STUDENT PROJECT - NOT OFFICIAL ABU WEBSITE**
Not affiliated with abu.edu.ng. Datasets hosted independently on GitHub for academic purpose only. This implementation follows Design and Development Research approach for COEN807.

**Deployment Note:** GitHub hosts code independently. ABU logo and name used for academic identification only.
