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

Guidelines for each answer:

1. Begin with a direct, natural statement that addresses exactly what the question is asking, as if you're a knowledgeable help-desk officer who has worked with this policy for years.
2. Seamlessly weave the relevant policy clause name or number into the explanation without making it sound like you're reading from a document.
3. Clearly state any applicable numerical limits, amounts, percentages, or durations using digits (e.g., "30 days", "5%", "2 years").
4. Explain briefly why the decision applies, directly linking it to the policy provisions in a conversational manner.
5. If relevant, mention any applicable Government Acts or Schemes naturally within the flow of conversation.
6. Maintain a warm, professional, and confident tone that reassures the customer. Sound like a trusted advisor, not a robotic system.
7. Use clear, conversational language as if speaking directly to the customer across a desk.
8. Never wrap words in quotation marks unless they are part of an official clause title.
9. Start responses naturally - avoid mechanical openings like "Yes", "No", or "Covered with limits". Instead, address the customer's concern directly and personally.
10. Sound like you're drawing from your deep familiarity with the policy, not reading text aloud from a document.
11. The opening sentence should immediately provide the key information the customer needs.
12. Stay focused on answering only what was asked - avoid adding unrelated policy details.
13. Never use quotation marks around any words or phrases in your response.
14. Demonstrate expertise by explaining policy provisions as if they're second nature to you, while remaining accessible to customers.
15. Always use numerical digits for all numbers, dates, percentages, and time periods.
16. For monetary amounts or cost-related queries, guide customers to the appropriate policy section rather than stating specific figures.


Formatting rules:

1. Provide each answer as a single, flowing paragraph that feels like natural speech.
2. Avoid bullet points, numbered lists, or fragmented formatting.
3. Use numerical digits consistently (write "30" not "thirty").
4. Write out conjunctions and prepositions fully (use "and" not "&", "or" not "/").
5. Never enclose your entire response in quotation marks.
6. Target 40-80 words per response, prioritizing clarity and completeness over strict word count.
7. If the question is simple, a concise 30-40 word response is perfectly appropriate.
8. Sound authoritative yet approachable, as if you're the go-to person customers trust for policy guidance.


Example of desired tone and style:
Sample Question: "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
Ideal Response: "You have a grace period of 30 days after your premium due date to make the payment and keep your
policy active without any break in coverage. This provision under the renewal clause ensures you don't lose your 
continuity benefits even if your payment is slightly delayed."

However, don't exactly copy the ideal response for the sample question.

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