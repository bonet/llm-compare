import random, copy

class LlmModels:
    def __init__(self):
        self._model_dict = {
            'llama2': 1,
            'mixtral': 2,
            'openai': 3
        }
        self._model_list = list(self._model_dict.keys())
        self.__set_shuffled_list()
        self.__map_shuffled_list()

    def dict(self):
        return self._model_dict

    def list(self):
        return self._model_list

    def shuffled_dict(self):
        return self._shuffled_model_dict

    def shuffled_list(self):
        return self._shuffled_model_list

    def __set_shuffled_list(self):
        self._shuffled_model_list = copy.copy(self._model_list)

        while self._shuffled_model_list == self._model_list:
            random.shuffle(self._shuffled_model_list)

    def __map_shuffled_list(self):
        mapping = {}

        for index, model in enumerate(self._shuffled_model_list):
            mapping[model] = index + 1

        self._shuffled_model_dict = mapping
