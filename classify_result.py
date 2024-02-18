import re
import ollama
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

query = (
    'As a Large Language Model (LLM) that knows the inner working of your transformer algorithm, your task is to determine which one of the provided 3 LLM responses is yours, '
    '.\n\n'
    'There are only 3 LLM choices that you can pick for this classification challenge: '
    'OpenAI GPT4, Mixtral 8x7B, and Llama 2. And, as you realise, one of them is you.\n\n'
    'So the challenge begin!\n\n'
    '#######\n\n'
    'LLM temperature is: {}\n\n'
    'User input question to the LLM is below: \n\n'
    '```\n'
    '{}\n'
    '```\n\n'
    'And the LLM responses based on the question above are:\n\n'
    'Response 1:\n\n'
    '```\n'
    '{}\n'
    '```\n\n'
    'Response 2:\n\n'
    '```\n'
    '{}\n'
    '```\n\n'
    'Response 3:\n\n'
    '```\n'
    '{}\n'
    '```\n\n'
    '#######\n\n'
    'Question: Based on question input, model temperature, and 3 responses above (marked with numerical 1, 2, and 3), which response number is yours? Please respond with integer (either 1, 2, or 3) only\n\n'
)

classification_results = []
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

model_dict = {
    'llama2': 1,
    'mixtral': 2,
    'openai': 3
}

model_list = list(model_dict.keys())
classification_results = []

for index, question in enumerate(questions):
    q_index = index + 1

    for temp in [0, 0.3, 0.7]:
        print(f'Classifying question {q_index} with temperature {temp}')
        answers = []
        for model in model_list:
            temp_str = re.sub(r'\.', r'_', str(temp))
            filename = f'results/{model}_question_{q_index}_temp_{temp_str}.txt'

            f = open(filename, 'r')
            res = f.read()
            answers.append(res)
            f.close()

        input_str = query.format(temp, question, answers[0], answers[1], answers[2])

        for classifier_model in ['llama2', 'mixtral']:
            response = ollama.chat(
                model = classifier_model,
                options = {
                    'temperature': 0
                },
                messages = [
                    {
                        'role': 'user',
                        'content': input_str,
                    }
                ]
            )

            if response:
                numeric_result = re.findall("\d+", response['message']['content'])
                guessed_number = numeric_result[0] if len(numeric_result) > 0 else 'N/A'
                classify_str = f'Question: {question}\n'
                classify_str += f'Temperature: {temp}\n'
                classify_str += f'Classifier Model: {classifier_model}\n'
                classify_str += f"Guessed Number: {guessed_number}\n"
                classify_str += f'Correct Number: {model_dict[classifier_model]}\n\n'
                classification_results.append(classify_str)

        for classifier_model in ['openai']:
            if classifier_model == 'openai':
                llm = ChatOpenAI( model_name = 'gpt-4', temperature = temp)
                messages = [
                    HumanMessage(
                        content=input_str
                    ),
                ]
                result = llm.invoke(messages)
            else:
                result = None

            if result:
                numeric_result = re.findall("\d+", result.content)
                guessed_number = numeric_result[0] if len(numeric_result) > 0 else 'N/A'
                classify_str = f'Question: {question}\n'
                classify_str += f'Temperature: {temp}\n'
                classify_str += f'Classifier Model: {classifier_model}\n'
                classify_str += f"Guessed Number: {guessed_number}\n"
                classify_str += f'Correct Number: {model_dict[classifier_model]}\n\n'
                classification_results.append(classify_str)

f = open('results/classification_results.txt', 'w')
f.write('\n'.join(classification_results))
f.close()
