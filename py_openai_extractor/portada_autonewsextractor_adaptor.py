from typing import Dict, Any
from.extractor import InfoExtractorBuilder


class AutonewsExtractorAdaptor:
    def __init__(self, api_key: str, config_json: Dict[str, Any]):
        self._config_json = config_json
        self._api_key = api_key
        api = config_json['api'] if "api" in config_json else None
        base_url = config_json['base_url'] if "base_url" in config_json else None
        self._extractor = InfoExtractorBuilder().with_api_key(api_key)\
            .with_model(config_json['model'])\
            .with_base_url(base_url)\
            .with_field_definitions(config_json['ai_instructions']['field_definitions'])\
            .with_json_template(config_json['ai_instructions']['json_template'])\
            .with_json_schema(config_json['ai_instructions']['json_schema'])\
            .with_model_config(config_json['model_config'])\
            .with_examples(config_json['ai_instructions']['examples'])\
            .with_messages_config(config_json['ai_instructions']['messages_config'])\
            .build(api)

    @property
    def config_json(self):
        return self._config_json

    @property
    def api_key(self):
        return self._api_key

    def extract_data(self, text):
        return self._extractor.extraer_informacion(text)


class AutonewsExtractorAdaptorBuilder:
    def __init__(self):
        self._config_json = None
        self._api_key = None

    def with_api_key(self, apikey: str) -> 'AutonewsExtractorAdaptorBuilder':
        self._api_key = apikey
        return self

    def with_config_json(self, config_json: Dict[str, Any]) -> 'AutonewsExtractorAdaptorBuilder':
        self._config_json = config_json
        return self

    # def with_field_definitions(self, field_definitions: Dict[str, str]):
    #     self._field_definitions = field_definitions
    #     return self

    def build(self) -> 'AutonewsExtractorAdaptor':
        adaptor = AutonewsExtractorAdaptor(self._api_key, self._config_json)
        return adaptor
