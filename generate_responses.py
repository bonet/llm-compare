import ollama

questions = [
    'Give me a 5-paragraph summary of Moby-Dick',
    '''Write a step-by-step tutorial on how to create and launch a successful
Kickstarter campaign for your creative project''',
    '''Write a story about a group of friends who embark on a dangerous journey
#to find a legendary treasure hidden deep in the jungle. Along the way, they must
#face treacherous obstacles, deadly creatures, and their own personal demons'''
]

for model in ['llama2', 'mixtral']:
    for index, question in enumerate(questions):
        q_index = index + 1
        print(f'{model} -- Processing question: {q_index}')
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': question,
            },
        ])

        if response:
            filename = f'results/{model}_question_{q_index}.txt'
            f = open(filename, 'w')
            f.write(response['message']['content'].strip())
            f.close()
