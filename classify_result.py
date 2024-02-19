import re
from llm_caller import LlmCaller
from llm_models import LlmModels
from llm_questions import questions

query = (
    'You are a Large Language Model (LLM) that knows the inner workings of your own '
    'model algorithm and output language structure. I have an LLM classification '
    'challenge for you.\n\n'
    'There are only 3 models that are involved in this classification challenge: '
    'OpenAI GPT4, Mixtral 8x7B, and Llama 2.\n\n'
    'Your task is to determine which one of the provided 3 LLM outputs came from '
    'the same model as you.\n\n'
    'Let the challenge begin!\n\n'
    '#######\n\n'
    'LLM temperature: {}\n\n'
    'LLM input question: \n'
    '```\n'
    '{}\n'
    '```\n\n'
    'LLM outputs (marked with "Output 1", "Output 2", and "Output 3") based '
    'on the input question and model temperature above are:\n\n'
    'Output 1:\n'
    '```\n'
    '{}\n'
    '```\n\n'
    'Output 2:\n'
    '```\n'
    '{}\n'
    '```\n\n'
    'Output 3:\n'
    '```\n'
    '{}\n'
    '```\n\n'
    '#######\n\n'
    'Question: Based on the provided information above, which one of the LLM outputs '
    'most likely came from the same model as you? Please respond with integer only '
    '(either "1", "2", or "3")\n\n'
)

llm_models = LlmModels()
model_dict = llm_models.dict()
shuffled_model_list = llm_models.shuffled_list()
shuffled_model_dict = llm_models.shuffled_dict()

question_list = questions()

classification_results = []

for index, question in enumerate(question_list):
    q_index = index + 1

    for temp in [0, 0.3, 0.7]:
        print(f'Classifying question {q_index} with temperature {temp}')

        answers = []

        for model in shuffled_model_list:
            temp_str = re.sub(r'\.', r'_', str(temp))
            filename = f'results/{model}_question_{q_index}_temp_{temp_str}.txt'

            f = open(filename, 'r')
            res = f.read()
            answers.append(res)
            f.close()

        input_str = query.format(temp, question, answers[0], answers[1], answers[2])

        for classifier_model in ['llama2', 'mixtral', 'openai']:
            # temperature is set to 0 for a more precise guess
            llm = LlmCaller(classifier_model, input_str, 0)
            response = llm.generate()

            if response:
                numeric_result = re.findall("[123]", response)
                guessed_number = numeric_result[0] if len(numeric_result) > 0 else 'N/A'
                classify_str = f'Question: {question}\n'
                classify_str += f'Temperature: {temp}\n'
                classify_str += f'Classifier Model: {classifier_model}\n'
                classify_str += f"Guessed Number: {guessed_number}\n"
                classify_str += f'Correct Number: {shuffled_model_dict[classifier_model]}\n'
                classify_str += f'Guess Full Response: {response}\n\n'
                classification_results.append(classify_str)

f = open('results/classification_results.txt', 'w')
f.write('\n'.join(classification_results))
f.close()
