from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

MODEL_NAME = "gemini-2.5-flash"

LLM_PROMPT = """
You are an intelligent, fast, and accurate insurance policy document parser.  
Your goal is to produce detailed, natural-sounding answers that mimic how an experienced human insurance advisor would respond to customers.

Treat the provided insurance policy document as the absolute source of truth.  
Base every response strictly on its content.

---

**Guidelines for each answer:**
1. Begin with a direct, natural statement that addresses exactly what the question is asking.
2. Seamlessly weave the relevant policy clause name or number into the explanation.
3. Clearly state any applicable numerical limits, amounts, percentages, or durations using digits (e.g., "30 days", "5%", "2 years").
4. Explain briefly why the decision applies, directly linking it to the policy wording.
5. If relevant, mention any applicable Government Acts or Schemes naturally within the flow.
6. Keep the tone professional, polite, and confident. Avoid robotic phrasing.
7. Avoid overly complex or run-on sentences. Prefer clear, short sentences.
8. Never wrap words in quotation marks unless they are part of the clause title itself.
9. Do not begin every answer mechanically with "Yes", "No", or "Covered with limits". Instead, phrase the decision in a human-like way that still answers directly.
10. Ensure the explanation flows like a conversation with an informed helpdesk officer, not like text read aloud from a document.
11. The first sentence must directly answer the key point of the question.
12. Never include irrelevant details not connected to the query.
13. Never use quotation marks ("") in any sentence or words.
14. Instead of blindly copying the PDF, sound as if an experienced insurance help-desk agent is answering, 
    a person who knows the policy very well.
15. Always prefer using numerical data wherever possible (except currency amounts or costings).
16. For cost incur or amounts, reply with refer to appropriate clause/section.

---

**Formatting rules:**
- Provide the answer as a single coherent paragraph for each question.
- Do NOT use bullet points or numbered lists in the output.
- Always use numerical digits instead of words for numbers (e.g., "30" instead of "thirty").
- Avoid escape sequences (\n).
- Do not use symbols like & for and, / for or, write in words instead.
- Do not wrap the answer in quotation marks.
- Keep answers between 50 and 90 words, (2-3 sentences), balancing completeness with conciseness.
- Answer according to question. Don't elongate answers by responding with un-asked things. Lesser than 50 words is also preferred.


REFER BELOW EXAMPLE AS WELL:
Sample Question: 
"What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"

Sample human-help-desk like answer:
"A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."


"""

# Create a single global genai.Client instance for reuse
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Embedding model for vector-based search
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="RETRIEVAL_QUERY"
)

def extract_keywords(question: str) -> List[str]:
    """Extract 2-3 important keywords using the LLM."""

    keyword_prompt = f"""
    Extract not more than 2-3 important keywords from the question below. 
    These should be the high-value tokens most relevant to insurance policy documents.
    Example: "What is the waiting period of cataract surgery?" → ['waiting-period', 'cataract']

    Question: {question}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=keyword_prompt
        )
        keywords = [kw.strip() for kw in response.text.split(",") if kw.strip()]

        return keywords
    
    except Exception as e:
        print(f"[Keyword Extraction Error] {str(e)}")
        return []
    


def retrieve(question: str, keywords: List[str], vectordb, topK=5):
    try:
        q_vector = embedding_model.embed_query(question)
        results_vec = vectordb.similarity_search_by_vector(q_vector, k=topK*3)  # get more to avoid duplicates

        # Deduplicate by text content
        seen = set()
        unique_results = []
        for doc in results_vec:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_results.append(doc)
            if len(unique_results) == topK:
                break

        #print("\n=== Retrieved Chunks ===")
        #for i, doc in enumerate(unique_results, 1):
        #    print(f"\n--- Chunk {i} ---")
        #    print(doc.page_content)

        return unique_results

    except Exception as e:
        print(f"[Hybrid Retrieval Error] {str(e)}")
        return []
    



def generate_answers(questions: List[str], vectordb, topK=3, num_workers=4):
    """Generate plain string answers for each question."""

    def process_single_question(question_with_index):
        index, question = question_with_index
        try:
            keywords = extract_keywords(question)
            retrieved_docs = retrieve(question, keywords, vectordb, topK=topK)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Ask Gemini without JSON formatting
            prompt = f"{LLM_PROMPT}\nContext:\n{context}\nQuestion: {question}"

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"temperature": 0.2}
            )
            return index, response.text.strip()
        except Exception as e:
            return index, f"[Error answering question] {str(e)}"

    indexed_questions = list(enumerate(questions))
    answers = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(process_single_question, q_with_idx): q_with_idx[0]
            for q_with_idx in indexed_questions
        }
        for future in as_completed(future_to_index):
            index, answer = future.result()
            answers[index] = answer

    return [answers[i] for i in range(len(questions))]