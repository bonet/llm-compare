import re
from llm_caller import LlmCaller

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

for model in ['llama2', 'mixtral', 'openai']:
    for index, question in enumerate(questions):
        q_index = index + 1

        for temp in [0, 0.3, 0.7]:
            print(f'{model} -- Processing question: {q_index} temperature: {temp}')

            llm = LlmCaller(model, question, temp)
            response = llm.generate()

            if response:
                temp_str = re.sub(r'\.', r'_', str(temp))
                filename = f'results/{model}_question_{q_index}_temp_{temp_str}.txt'
                f = open(filename, 'w')
                f.write(response.strip())
                f.close()
