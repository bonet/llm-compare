import re
import ollama
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

questions = [
    'Give me a 5-paragraph summary of Moby-Dick',
    (
        'Write a step-by-step tutorial on how to create and launch a successful '
        'Kickstarter campaign for your creative project'
    ),
    (
        'Write a story about a group of friends who embark on a dangerous journey '
        'to find a legendary treasure hidden deep in the jungle. Along the way, they must '
        'face treacherous obstacles, deadly creatures, and their own personal demons'
    )
]

for model in ['llama2', 'mixtral']:
    for index, question in enumerate(questions):
        q_index = index + 1

        for temp in [0, 0.3, 0.7]:
            print(f'{model} -- Processing question: {q_index} temperature: {temp}')

            response = ollama.chat(
                model = model,
                options = {
                    'temperature': temp
                },
                messages = [
                    {
                        'role': 'user',
                        'content': question,
                    }
                ]
            )

            if response:
                temp_str = re.sub(r'\.', r'_', str(temp))
                filename = f'results/{model}_question_{q_index}_temp_{temp_str}.txt'
                f = open(filename, 'w')
                f.write(response['message']['content'].strip())
                f.close()

for model in ['openai', 'gemini']:
    for index, question in enumerate(questions):
        q_index = index + 1

        for temp in [0, 0.3, 0.7]:
            print(f'{model} -- Processing question: {q_index} temperature: {temp}')

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
                temp_str = re.sub(r'\.', r'_', str(temp))
                filename = f'results/{model}_question_{q_index}_temp_{temp_str}.txt'
                f = open(filename, 'w')
                f.write(result.content.strip())
                f.close()
