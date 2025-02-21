from openai import OpenAI
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from babel.dates import format_date

class InfoExtractor:
    def __init__(self):
        self._client = None
        self._model = None
        self._fallback_model = "gpt-4o"
        self._json_schema = None
        self._model_config = {}
        self._field_definitions = {}
        self._messages_config = {}
        self._json_template = {}
        self._examples = ""

    def set_api_key(self, api_key: str) -> 'InfoExtractor':
        self._client = OpenAI(api_key=api_key)
        return self

    def set_model(self, model: str) -> 'InfoExtractor':
        self._model = model
        return self

    def set_json_schema(self, json_schema: Dict[str, Any]) -> 'InfoExtractor':
        self._json_schema = json_schema
        return self

    def set_model_config(self, config: Dict[str, Any]) -> 'InfoExtractor':
        self._model_config.update(config)
        return self

    def set_field_definitions(self, field_definitions: Dict[str, str]) -> 'InfoExtractor':
        self._field_definitions = field_definitions
        return self

    def set_messages_config(self, messages_config: Dict[str, Any]) -> 'InfoExtractor':
        self._messages_config = messages_config
        return self

    def set_json_template(self, json_template: Dict[str, Any]) -> 'InfoExtractor':
        self._json_template = json_template
        return self

    def set_examples(self, examples: str) -> 'InfoExtractor':
        self._examples = examples
        return self

    def _create_messages(self, texto_entrada: str) -> List[Dict[str, str]]:
        field_definitions_text = '. '.join([
            f"'{key}': '{value}'" 
            for key, value in self._field_definitions.items()
        ])

        user_message = self._messages_config["template"]["content"].format(
            json_template=json.dumps(self._json_template, ensure_ascii=False),
            field_definitions=field_definitions_text,
            input_example=self._examples,
            input_text=texto_entrada
        )

        return [
            self._messages_config["system"],
            {"role": "user", "content": user_message}
        ]

    def extraer_informacion(self, texto: str) -> Union[Dict[str, Any], str, None]:
        if not all([self._client, self._model, self._json_schema]):
            raise ValueError("La configuración del extractor está incompleta.")

        models_to_try = [self._model, self._fallback_model]
        last_raw_content = None

        for model in models_to_try:
            try:
                respuesta = self._client.chat.completions.create(
                    model=model,
                    messages=self._create_messages(texto),
                    response_format={"type": "json_object"},
                    **self._model_config
                )

                contenido_respuesta = respuesta.choices[0].message.content
                last_raw_content = contenido_respuesta

                try:
                    return json.loads(contenido_respuesta)
                except json.JSONDecodeError:
                    print(f"No se pudo decodificar la respuesta como JSON usando el modelo {model}.")
                    if model == self._fallback_model:
                        print("Fallaron todos los intentos de extracción JSON. Devolviendo el contenido crudo.")
                        return contenido_respuesta
                    else:
                        print(f"Intentando con el modelo de respaldo: {self._fallback_model}")

            except Exception as e:
                print(f"Error al procesar la entrada con el modelo {model}: {str(e)}")
                if model == self._fallback_model:
                    print("Fallaron todos los intentos de extracción.")
                    if last_raw_content:
                        print("Devolviendo el último contenido crudo obtenido.")
                        return last_raw_content
                    else:
                        print("No se pudo obtener ningún contenido. Devolviendo None.")
                        return None
                else:
                    print(f"Intentando con el modelo de respaldo: {self._fallback_model}")

        return None  # Este return solo se alcanzará si hay un error inesperado en la lógica del bucle

    # def extraer_informacion(self, texto: str) -> Union[Dict[str, Any], str, None]:
    #     if not all([self._client, self._model, self._json_schema]):
    #         raise ValueError("La configuración del extractor está incompleta.")
    # 
    #     try:
    #         respuesta = self._client.chat.completions.create(
    #             model=self._model,
    #             messages=self._create_messages(texto),
    #             response_format={"type": "json_object"},
    #             **self._model_config
    #         )
    # 
    #         contenido_respuesta = respuesta.choices[0].message.content
    # 
    #         try:
    #             return json.loads(contenido_respuesta)
    #         except json.JSONDecodeError:
    #             print("No se pudo decodificar la respuesta como JSON. Devolviendo el contenido crudo.")
    #             return contenido_respuesta
    # 
    #     except Exception as e:
    #         print(f"Error al procesar la entrada: {str(e)}")
    #         return None

    # def extraer_informacion(self, texto: str) -> Optional[Dict[str, Any]]:
    #     if not all([self._client, self._model, self._json_schema]):
    #         raise ValueError("La configuración del extractor está incompleta.")
    # 
    #     try:
    #         respuesta = self._client.chat.completions.create(
    #             model=self._model,
    #             messages=self._create_messages(texto),
    #             response_format={"type": "json_object"},
    #             **self._model_config
    #         )
    # 
    #         return json.loads(respuesta.choices[0].message.content)
    # 
    #     except Exception as e:
    #         print(f"Error al procesar la entrada: {str(e)}")
    #         return None

class InfoExtractorBuilder:
    def __init__(self):
        self._extractor = InfoExtractor()

    def with_api_key(self, api_key: str) -> 'InfoExtractorBuilder':
        self._extractor.set_api_key(api_key)
        return self

    def with_model(self, model: str) -> 'InfoExtractorBuilder':
        self._extractor.set_model(model)
        return self

    def with_json_schema(self, json_schema: Dict[str, Any]) -> 'InfoExtractorBuilder':
        self._extractor.set_json_schema(json_schema)
        return self

    def with_model_config(self, config: Dict[str, Any]) -> 'InfoExtractorBuilder':
        self._extractor.set_model_config(config)
        return self

    def with_field_definitions(self, field_definitions: Dict[str, str]) -> 'InfoExtractorBuilder':
        self._extractor.set_field_definitions(field_definitions)
        return self

    def with_messages_config(self, messages_config: Dict[str, Any]) -> 'InfoExtractorBuilder':
        self._extractor.set_messages_config(messages_config)
        return self

    def with_json_template(self, json_template: Dict[str, Any]) -> 'InfoExtractorBuilder':
        self._extractor.set_json_template(json_template)
        return self

    def with_examples(self, examples: str) -> 'InfoExtractorBuilder':
        self._extractor.set_examples(examples)
        return self

    def build(self) -> InfoExtractor:
        return self._extractor

def calcular_fecha_entrada(fecha_nota: str, dia: str) -> Optional[datetime]:
    try:
        fecha_nota = datetime.strptime(fecha_nota, '%Y_%m_%d')
        dia = int(dia)
        
        if dia <= fecha_nota.day:
            return fecha_nota - timedelta(days=fecha_nota.day - dia)
        else:
            fecha_anterior = fecha_nota.replace(day=1) - timedelta(days=1)
            return fecha_anterior - timedelta(days=fecha_anterior.day - dia)
    except ValueError:
        return None

def formato_fecha_espanol(fecha: Optional[datetime]) -> str:
    if fecha:
        return format_date(fecha, format='d \'de\' MMMM \'de\' yyyy', locale='es')
    return "Fecha desconocida"
