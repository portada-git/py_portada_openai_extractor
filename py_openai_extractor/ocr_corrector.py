from .abstract_openai_agent import AbstractOpenAiChatAgent, BaseOpenAiTextChatAgent
from typing import Dict, Any, List
from pydoc import locate
import re

def remove_markdown(text):
    # Elimina encabezados (# Header)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Elimina negritas e itálicas (**texto** / *texto*)
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
    # Elimina listas (- item / 1. item)
    text = re.sub(r'^[\-\*\d+\.]\s+', '', text, flags=re.MULTILINE)
    # Elimina bloques de código (`code` o ```code```)
    text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Elimina enlaces [texto](url)
    text = re.sub(r'$$(.*?)$$$.*?$$', r'\1', text)
    return text.strip()

class QwenOcrProcessor(AbstractOpenAiChatAgent):
    def __init__(self):
        super().__init__()
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self._system_message = "Eres un experto en el proceso de conversion de imágenes a texto."
        self._user_message = (
            "Extrae el texto contenido en las imágenes proporcionadas, correspondientes a documentos de texto. "
            "Asegurate que el texto devuelto sea legible y fiel a los documentos originales escaneados.\n"
            "Sigue estas pautas para realizar la conversión de imagen a texto:\n\n"
            "  - Asegúrate de que el texto tenga sentido y fluya en el orden correcto según el diseño visual original. "
            "Procura no desordenar palabras ni líneas teniendo en cuenta que algunos documentos presentan ciertas "
            "distorsiones como líneas onduladas, líneas inclinadas, transparencia del reverso, poca tinta o exceso de "
            "tinta, etc. Se trata de obtener un texto fiel al original, pero legible y con sentido.\n "
            "  - Mantén la ortografía y la puntuación originales.\n"
            "  - Combina palabras cortadas por líneas, elimina saltos de línea dentro de los párrafos y conserva los "
            "saltos únicamente entre párrafos distintos.\n "
            "  - En el caso de los números (como fechas, cifras o referencias), presta especial atención a las imágenes"
            " originales ya que son datos importantes difíciles de verificar y corregir posteriormente.\n\n "
            "# Output Format\n\n"
            "El texto debe ser presentado como un bloque legible y ordenado, sin explicaciones ni "
            "anotaciones adicionales. No incluyas caracteres de formato de texto como por ejemplo del "
            "lenguaje Markdown ni de cualquier otro, para identificar títulos, ni cabeceras, "
            "ni listas, ni demás formatos de texto. Debes devolver únicamente, el texto plano con el "
            "contenido original.\n\n"
            "Asegúrate de que el contenido final sea una representación fiel de los documentos "
            "originales.\n\n"
            "Si en el documento original hay columnas de texto, al trasladarlas a texto hazlo poniendo una columna "
            "debajo de otra, siguiendo el orden lógico de lectura para que esta tenga sentido.\n\n"
            "# Notes\n\n"
            "A continuación, aportamos las imagenes del documento ordenadas secuencialmente. Utiliza todas "
            "las imágenes proporcionadas para extraer el texto:\n\n "
        )
        self._base64_images = []
        self._model = "qwen2.5-vl-32b-instruct"
        self._model_config = {
            "max_tokens": 8192,
            "top_p": 0.1,
            "frequency_penalty": 0,
            "presence_penalty": 2,
        }

    def set_api_key(self, api_key: str, base_url=None) -> 'QwenOcrProcessor':
        super().set_api_key(api_key=api_key, base_url=self.base_url)
        return self

    def set_messages_config(self, system_message: str = None, user_message: str = None) -> 'QwenOcrProcessor':
        if system_message is not None:
            self._system_message = system_message
        if user_message is not None:
            self._user_message = user_message
        return self

    def set_images(self, base64_images:List) -> 'QwenOcrProcessor':
        self._base64_images = base64_images
        return self

    def _create_messages(self) -> List[Dict[str, str]]:
        full_user_message = [
            {"type": "text", "text": self._user_message}
        ]

        for base64_image in self._base64_images:
            if isinstance(base64_image, dict):
                full_user_message.append({
                    "type": "image_url",
                    "image_url":{
                        "url": "data:"+base64_image["mime_type"]+";base64,"+base64_image["image"]
                    }
                })
            else:
                full_user_message.append({
                    "type": "image_url",
                    "image_url":{
                        "url": "data:image/jpeg;base64,"+base64_image
                    }
                })


        return [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": full_user_message}
        ]

    def process_request_from_client(self):
        if "temperature" not in self._model_config:
            self._model_config["temperature"]=0
        if "max_tokens" not in self._model_config or self._model_config["max_tokens"]>8192:
            self._model_config["max_tokens"]=8192
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **self._model_config
        )
        return response

    def getTextFromImage(self, images):
        self._base64_images = images
        self.messages = self._create_messages()
        response = self.process_request_from_client()
        text = response.choices[0].message.content
        return remove_markdown(text)


class QwenOcrCorrector(QwenOcrProcessor):
    def __init__(self):
        super().__init__()
        self.base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self._system_message="Eres un experto en corrección de textos OCR"
        self._user_message=("Corrige el texto extraído mediante OCR para que sea legible y fiel a los documentos "
                            "originales escaneados.\n "
                            "Debes inspeccionar tanto las imágenes proporcionadas como el texto extraído, y seguir "
                            "estas pautas para realizar las correcciones necesarias.\n\n "
                            "- Completa las omisiones del texto indicadas en las imágenes pero no capturadas por el "
                            "OCR.\n "
                            "- Corrige cualquier error de reconocimiento, como palabras mal escritas, signos de "
                            "puntuación incorrectos, o frases mal interpretadas.\n "
                            "  - Mantén intactos los fragmentos que no presentan errores evidentes ni omisiones.\n"
                            "- Asegúrate de que el texto fluya en el orden correcto según el diseño visual original, "
                            "ya que el OCR puede desordenar las palabras o las líneas.\n "
                            "- Ordena y estructura el texto según el formato visual correcto, especialmente si "
                            "aparece desordenado.\n "
                            "  - Mantén la ortografía y la puntuación originales.\n"
                            "- Combina palabras cortadas por líneas, elimina saltos de línea dentro de los párrafos y "
                            "conserva los saltos únicamente entre párrafos distintos.\n "
                            "- En el caso de los números (como fechas, cifras o referencias), presta especial "
                            "atención a las imágenes originales más que al texto extraído, ya que suelen ser mal "
                            "reconocidos por el OCR.\n\n "
                            "# Output Format\n\n"
                            "El texto debe ser presentado como un bloque legible y ordenado, sin explicaciones ni "
                            "anotaciones adicionales. Asegúrate de que el contenido final sea una representación fiel "
                            "de los documentos originales.\n\n "
                            "# Notes\n\n"
                            "A continuación, el texto extraído por OCR de archivo(s) relacionado(s) con la misma "
                            "fecha. Utiliza todas las imágenes proporcionadas para corregir este texto:\n\n "
                            )
        self._text=""

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    def set_text_and_images(self, texts:str, base64_images:List) -> 'QwenOcrCorrector':
        self._text = texts
        self._base64_images = base64_images
        return self

    def _create_messages(self) -> List[Dict[str, str]]:
        if "{full_text}" in self._user_message:
            user_message = self._user_message.format(
                full_text=self._text
            )
        elif self._user_message.endswith("\n"):
            user_message = self._user_message + "" + self._text
        else:
            user_message = self._user_message + "\n\n" + self._text

        full_user_message = [
            {"type": "text", "text": user_message}
        ]

        for base64_image in self._base64_images:
            if isinstance(base64_image, dict):
                full_user_message.append({
                    "type": "image_url",
                    "image_url":{
                        "url": "data:"+base64_image["mime_type"]+";base64,"+base64_image["image"]
                    }
                })
            else:
                full_user_message.append({
                    "type": "image_url",
                    "image_url":{
                        "url": "data:image/jpeg;base64,"+base64_image
                    }
                })


        return [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": full_user_message}
        ]

    def getFixedOcrText(self, text, images):
        self._text=text
        self._base64_images = images
        self.messages = self._create_messages()
        response = self.process_request_from_client()
        newText = response.choices[0].message.content
        return remove_markdown(newText)

