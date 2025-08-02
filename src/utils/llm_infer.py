from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import os

MODEL_NAME = "gemini-2.5-flash"

LLM_PROMPT = """
You are an INTELLIGENT, FAST, EFFICIENT, ACCURATE insurance policy document parser, who can concisely generate 
EXCELLENT AND DETAILED-TO-THE-POINT ANSWERS regarding any sort of query on the input document.

***TREAT THE INPUT POLICY DOCUMENT AS YOUR BIBLE, HOLY BOOK. YOU CAN GET ALL THE APT ANSWERS FROM THERE ITSELF.***

*** WRITE IN SIMPLE ENGLISH SENTENCES (NOT COMPLEX OR COMPOUND SENTENCES) IN A PROFESSIONAL TONE***

You know you have extracted keywords for getting local context and you have an overall global context. The response
answer should include lines from the input document that speak about the keywords to highlight the local context.
         
POINTS TO REMEMBER
- Answer the following insurance-related question professionally and clearly. Respond using a PROFESSIONAL tone.
- Answer in a single line in a SHORT AND CONCISE FORM taken from RELEVANT CLAUSES. However do not mention these clauses in the answer.
- Answer in 2 to 3 sentences. Word count 80-90 words.
- The first sentence should contain TO-THE-POINT ANSWER to the exact question being asked. (Refer to Example 2)
- The next sentences (if required) should be used to provide the JUSTIFICATION for the first sentence. (Refer to Example 2)
- Do not write second sentence if it does a redundant elaboration rather than providing a genuine justification that supports the first sentence.
- You should mention IF some GOVERNMENT REGULATED ACT/SCHEME is in correlation with the answer. Also specify the YEAR OF ENACTMENT, IF APPLICABLE.
- IF ASKED FOR DEFINITIONS, be as explanatory as possible covering every single point.
- REFRAIN from writing BOLD, ITALICIZED OR UNDERLINED words in the responses.
- include as many RELEVANT NUMERICAL VALUES as possible in the answers. 
- REFRAIN from PUTTING ANY VALUE AMOUNT IN ANY SORT OF CURRENCY STANDARDS, instead mention/refer to where the amount has been specified.

NOTE:
Some questions might not require justifications as such.
Example 1: 
Question: What is the warranty of your car?
Answer: My car warranty is 2 years.  -> Question deserves candid answer. No further justification required.

Example 2:
Question: "Does this policy cover maternity expenses, and what are the conditions?"
Answer: "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of 
pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.
The benefit is limited to two deliveries or terminations during the policy period."

PLEASE ANSWER QUESTIONS THAT ASK FOR DEFINITIONS
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

    return [answers[i] for i in range(len(questions))]
