import re
import ollama
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

query = (
    'You are an expert Large Language Model reviewer. Given 4 different LLMs: '
    'Llama 2, Mixtral 8x7B, OpenAI GPT-4, and Google Gemini, please determine '
    'which LLM generated the sentences delimited by triple backticks below (```): \n'
    '```\n'
    '{}\n'
    '```\n\n'
    '#######'
    'Rules:\n'
    '1. You can only repond with one of the 4 choices below:\n'
    '   - "Llama 2"\n'
    '   - "Mixtral 8x7B"\n'
    '   - "OpenAI GPT-4"\n'
    '   - "Google Gemini"\n\n'
    '2. You can only pick one LLM. Give your best guess!\n\n'
    '3. The repond format is:\n'
    '   LLM: <<one of the LLMs listed above>>\n'
    '   Reason: <<Explain your reason of choosing the LLM here>>\n'
)

review_results = []
for model in ['llama2', 'mixtral', 'openai']:
    for index in range(1,4):
        for temp in [0, 0.3, 0.7]:
            temp_str = re.sub(r'\.', r'_', str(temp))
            filename = f'results/{model}_question_{index}_temp_{temp_str}.txt'
            print(f'Reviewing file: {filename}')
            review_str = f'File: {filename}\n'

            f = open(filename, 'r')
            res = f.read()
            f.close()

            for reviewer_model in ['llama2', 'mixtral']:
                response = ollama.chat(
                    model = reviewer_model,
                    options = {
                        'temperature': 0
                    },
                    messages = [
                        {
                            'role': 'user',
                            'content': query.format(res),
                        }
                    ]
                )

                if response:
                    review_str += f'Classifier: {reviewer_model}\n'
                    review_str += f"Verdict: \n{response['message']['content'].strip()}\n"
                    review_results.append(review_str)

            for reviewer_model in ['openai']:
                if model == 'openai':
                    llm = ChatOpenAI( model_name = 'gpt-4', temperature = temp)
                    messages = [
                        HumanMessage(
                            content=question
                        ),
                    ]
                    result = llm.invoke(messages)
                elif model == 'gemini':
                    llm = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = temp)
                    result = llm.invoke(question)
                else:
                    result = None

                if result:
                    review_str += f'Classifier: {reviewer_model}\n'
                    review_str += f"Verdict: \n{result.content.strip()}\n"
                    review_results.append(review_str)

f = open('results/review_results.txt', 'w')
f.write('\n'.join(review_results))
f.close()
