from abc import abstractmethod, ABC
from openai import OpenAI
from typing import Dict, Any

class AbstractOpenAiAgent:
    def __init__(self):
        self._client = None
        self._model = None
        self._model_config = {}

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def client(self):
        return self._client

    @property
    def model_config(self):
        return self._model_config

    @model_config.setter
    def model_config(self, model_config):
        self._model_config = model_config

    def set_api_key(self, api_key: str, base_url: str = None) -> 'AbstractOpenAiAgent':
        if base_url is None:
            self._client = OpenAI(api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self

    def set_model(self, model: str) -> 'AbstractOpenAiAgent':
        self._model = model
        return self

    def set_model_config(self, config: Dict[str, Any]) -> 'AbstractOpenAiAgent':
        self._model_config.update(config)
        return self

    @abstractmethod
    def process_request_from_client(self):
        pass



class AbstractOpenAiChatAgent(AbstractOpenAiAgent, ABC):
    def __init__(self):
        super().__init__()
        self._messages = {}

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, messages):
        self._messages = messages

    def set_messages(self, messages: Dict[str, Any]) -> 'AbstractOpenAiChatAgent':
        self._messages = messages
        return self


class GenericOpenAiAgent(AbstractOpenAiAgent):
    def __init__(self, fn_process_request_from_client):
        super().__init__()
        self._fn_process_request_from_client = fn_process_request_from_client

    def process_request_from_client(self, ):
        return self._fn_process_request_from_client(self)


class BaseOpenAiCompletionsAgent(AbstractOpenAiAgent, ABC):
    def __init__(self):
        super().__init__()
        self._prompt = None

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str):
        self._prompt = prompt

    def process_request_from_client(self):
        response = self.client.completions.create(model=self.model, prompt = self._prompt,  **self._model_config)
        return response

class BaseOpenAiTextChatAgent(AbstractOpenAiChatAgent, ABC):
    def __init__(self):
        super().__init__()

    def process_request_from_client(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **self._model_config)
        return response


class BaseOpenAiStructuredChatAgent(AbstractOpenAiChatAgent, ABC):
    def __init__(self):
        super().__init__()

    def process_request_from_client(self):
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self.messages,
            **self._model_config)
        return response


class BaseOpenAiEmbeddingsAgent(AbstractOpenAiAgent):
    def __init__(self):
        super().__init__()
        self._encoding_format="float"
        self._input = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input: str):
        self._input = input

    @property
    def encoding_format(self):
        return self._encoding_format

    def process_request_from_client(self):
        response = self.client.embeddings.create(
            model=self.model,
            input = self._input,
            encodig_format=self._encoding_format,
            **self._model_config)
        return response


class AbstractBaseOpenAiAgentBuilder:
    EMBEDDING_OPTION=0
    COMPLETIONS_OPTION = 1
    CHAT_COMPLETIONS_CREATE_OPTION = 2
    CHAT_COMPLETIONS_PARSE_OPTION = 3

    def __init__(self, option: int):
        super().__init__()
        self._option = option



class OpenaiAgentBuilderByClass:
    def __init__(self):
        self._api_key = None
        self._model = None
        self._model_config = {}
        self._messages_config = {}
        self._class_name = None


class OpenaiAgentBuilderByParams:
    def __init__(self):
        self._api_key = None
        self._model = None
        self._model_config = {}
        self._messages_config = {}
        self._base_url = None

    def with_api_key(self, api_key: str):
        self._api_key = api_key
        return self

    def with_base_url(self, base_url: str):
        self._base_url = base_url
        return self

    def with_model(self, model: str):
        self._model = model
        return self

    def with_model_config(self, config: Dict[str, Any]):
        self._model_config = config
        return self

    def with_messages_config(self, messages_config: Dict[str, Any]):
        self._messages_config = messages_config
        return self

    def build(self, option):

        if option is None and self._base_url is not None:
            option = "GeminiInfoExtractor"
        elif option is not None and option.lower().startswith("gemini") and self._base_url is None:
            self._base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif option is not None and self._base_url is None:
            # raise ValueError("base_url mustn't be None if option is not none")
            option = None

        if option is None or option.startswith("openai"):
            extractor = InfoExtractor()
        else:
            extractor = GeminiInfoExtractor()

        extractor.set_api_key(self._api_key) \
            .set_model(self._model) \
            .set_json_schema(self._json_schema) \
            .set_model_config(self._model_config) \
            .set_field_definitions(self._field_definitions) \
            .set_messages_config(self._messages_config) \
            .set_json_template(self._json_template) \
            .set_examples(self._examples)
        return extractor
