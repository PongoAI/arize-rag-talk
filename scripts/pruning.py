import os
import json
from openai import OpenAI

def select_best_questions(input_file, output_file):
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    together_client = OpenAI(api_key=together_api_key, base_url='https://api.together.xyz/v1')

    llm_prompt = """
**Task: Question Selection**
=========================================
You are a question selection system that takes a document's text and a list of questions about the document as input and responds with ONLY a valid JSON list containing one string.  Based on the input, select the one question that is most completely answered by the provided document and return it as a string in a JSON list
**Example Document:** "RAG stands for retrieval augmented generation, it's a method for giving LLMs access to information outside of their training data.  The core of a RAG pipeline usually includes a vector database and LLM, but other search indices can also be used."
**Example questions:** ["What is RAG?", "What does RAG stand for?", "What are the core components of a RAG pipeline?"]
**Example output:** ["What does RAG stand for?"]
**Input Document:** {input_document}
**Input Questions:** {input_questions}
**Please respond ONLY with a valid JSON list containing one string.**
"""

    with open(input_file, 'r') as f:
        input_data = json.load(f)

    selected_questions = []

    for item in input_data:
        input_document = item['doc']
        input_questions = item['questions']

        try:
            selected_question = json.loads(together_client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[{"role": "user", "content": llm_prompt.format(input_document=input_document, input_questions=input_questions)}],
                stream=False,
                temperature=0.2,
            ).choices[0].message.content)

            selected_questions.append({
                "doc": input_document,
                "selected_question": selected_question[0]
            })

            with open(output_file, 'w') as f:
                json.dump(selected_questions, f, indent=2)
        except:
            print(f'LLM hallucinated, skipping this document: "{input_document}"')

    return selected_questions