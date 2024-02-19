import re
from llm_caller import LlmCaller
from llm_question import questions

query = (
    'As a Large Language Model (LLM) that knows the inner working of your transformer '
    'algorithm, your task is to determine which one of the provided 3 LLM responses is '
    'yours.\n\n'
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
    'Question: Based on question input, model temperature, and 3 responses above '
    '(marked with numerical 1, 2, and 3), which response number is yours? Please '
    'respond with integer (either 1, 2, or 3) only\n\n'
)

model_dict = {
    'llama2': 1,
    'mixtral': 2,
    'openai': 3
}

model_list = list(model_dict.keys())
question_list = questions()

classification_results = []

for index, question in enumerate(question_list):
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

        for classifier_model in ['openai']:
            # temperature is set to 0 for a more precise guess
            llm = LlmCaller(classifier_model, input_str, 0)
            response = llm.generate()

            if response:
                numeric_result = re.findall("\d+", response)
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
