import os
import json
from openai import OpenAI

def generate_questions(input_documents, output_file):
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    together_client = OpenAI(api_key=together_api_key, base_url='https://api.together.xyz/v1')

    llm_prompt = """
**Task: Question Generation**
=========================================
You are a question generation system that takes a document's text as input and responds with ONLY a valid JSON list of strings.  Based on the input document provided, generate 3 deterministic, fact-based questions that can be answered by the document.
**Example Document:** "RAG stands for retrieval augmented generation, it's a method for giving LLMs access to information outside of their training data.  The core of a RAG pipeline usually includes a vector database and LLM, but other search indices can also be used."
**Example output:** ["What is RAG?", "What does RAG stand for?", "What are the core components of a RAG pipeline?"]
**Input Document:** {input_document}
**Please respond ONLY with a valid list of three strings.** 
"""

    responses = []

    for input_document in input_documents:
        try:
            questions = json.loads(together_client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[{"role": "user", "content": llm_prompt.format(input_document=input_document)}],
                stream=False,
                temperature=0.2,
            ).choices[0].message.content)

            response_object = {
                "doc": input_document,
                "questions": questions
            }

            responses.append(response_object)

            with open(output_file, 'w') as f:
                json.dump(responses, f, indent=2)
        except:
            print(f'LLM hallucinated, skipping this document: "{input_document}"')

    return responses