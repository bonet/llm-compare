import ollama
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class LlmCaller:
    def __init__(self, model, input, temperature):
        self._model = model
        self._input = input
        self._temperature = temperature

    def generate(self):
        if self._model == 'openai':
            llm = ChatOpenAI(model_name='gpt-4', temperature=self._temperature)
            messages = [
                HumanMessage(
                    content=self._input
                ),
            ]
            result = llm.invoke(messages)

            return result.content.strip()
        elif self._model == 'gemini':
            llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=self._temperature)
            result = llm.invoke(self._input)

            return result.content.strip()
        else:
            result = ollama.chat(
                model = self._model,
                options = {
                    'temperature': self._temperature
                },
                messages = [
                    {
                        'role': 'user',
                        'content': self._input,
                    }
                ]
            )

            return result['message']['content']
