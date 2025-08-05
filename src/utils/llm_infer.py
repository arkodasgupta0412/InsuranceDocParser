from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import os

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
13. Do not include words with quotation marks ("") in any sentence. The helpdesk is not quoting something.
14. Instead of blindly copying the PDF, sound as if an experienced insurance help-desk agent is answering, 
    a person who knows the policy very well.

---

**Formatting rules:**
- Provide the answer as a single coherent paragraph for each question.
- Do NOT use bullet points or numbered lists in the output.
- Always use numerical digits instead of words for numbers (e.g., "30" instead of "thirty").
- Avoid escape sequences (\n).
- Do not wrap the answer in quotation marks.
- Keep answers between 60 and 90 words, balancing completeness with conciseness.

---

**Example transformation:**

**Question:** What is the grace period for premium payment?  
**Good answer:** You have a 30-day grace period to pay your premium and maintain your policy’s continuity, as detailed in clause 2.21. This allows you to renew without losing your accrued benefits, provided the payment is made within this time. However, coverage remains suspended during the grace period until the premium is received.

**Bad answer:** The grace period is 30 days. [Refer: 2.21]

"""



# Create a single global genai.Client instance for reuse
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_keywords(question: str) -> List[str]:
    """
    Extract keywords using a keyword extraction prompt.
    """
    keyword_prompt = f"""
    Extract not more than 2-3 important keywords from the question below. The important keywords are basically
    the high-value tokens in the question that hits the main asking of the question. They are mostly terms that 
    are specific to insurance policy documents. In other words, they are mostly used in insurance policy docs.
    Keyword can be one word/group of words.

    Example: What is the waiting period of cataract surgery ?
    Important keywords: ['waiting-period', 'cataract']

    Question: {question}
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=keyword_prompt
        )
        keywords = [kw.strip() for kw in response.text.split(",") if kw.strip()]
        print(keywords)
        return keywords
    except Exception as e:
        print(f"[Keyword Extraction Error] {str(e)}")
        return []

def generate_answers(questions: List[str], vectordb, num_workers=1):
    """Generate answers using multithreading while maintaining order."""

    def process_single_question(question_with_index):
        index, question = question_with_index

        try:
            # Extract keywords
            keywords = extract_keywords(question)
            focused_query = ", ".join(keywords) if keywords else question
            # print(focused_query)

            # Similarity search
            focused_docs = vectordb.similarity_search(focused_query, k=10)
            global_docs = vectordb.similarity_search(question, k=5)
            combined_docs = list({doc.page_content: doc for doc in focused_docs + global_docs}.values())
            context = "\n\n".join([doc.page_content for doc in combined_docs])

            # Prompt generation
            prompt = f"{LLM_PROMPT}\n context = {context}\n query = {question}"

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"temperature": 0.2}
            )
            return index, response.text.strip()

        except Exception as e:
            return index, f"Error: {str(e)}"

    indexed_questions = list(enumerate(questions))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(process_single_question, q_with_idx): q_with_idx[0]
            for q_with_idx in indexed_questions
        }

        answers = {}
        for future in as_completed(future_to_index):
            index, answer = future.result()
            answers[index] = answer

    return [answers[i].strip(" \n") for i in range(len(questions))]
