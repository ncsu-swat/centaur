from openai import OpenAI
import os, re, requests, logging
from bs4 import BeautifulSoup

class Response:
    def __init__(self, text):
        self.text = text

class OAChatWrapper:
    DEFAULT_MODEL = "gpt-5"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model=None, base_url=None, api_key=None):
        env_model = os.getenv("MODEL_NAME")
        env_url = os.getenv("MODEL_URL")

        # Prioritize explicit arguments, then env vars, then defaults
        self.model = model or env_model or self.DEFAULT_MODEL
        configured_url = base_url or env_url or self.DEFAULT_BASE_URL
        normalized_url = configured_url.rstrip("/")

        # Determine which OpenAI API surface to use based on the configured URL
        if normalized_url.endswith("/chat/completions"):
            self._api_surface = "chat_completions"
            client_base_url = normalized_url.rsplit("/chat/completions", 1)[0]
        elif normalized_url.endswith("/completions"):
            self._api_surface = "completions"
            client_base_url = normalized_url.rsplit("/completions", 1)[0]
        else:
            self._api_surface = "responses"
            client_base_url = normalized_url

        if not client_base_url:
            client_base_url = self.DEFAULT_BASE_URL

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=client_base_url,
        )
        self.previous_response_id = None
        self._messages = []  # Used when interacting via chat/completions
        self._logger = logging.getLogger(__name__)

    def send_message(self, prompt):
        self._logger.info("LLM prompt: %s", prompt)
        if self._api_surface == "responses":
            if self.previous_response_id is None:
                openai_response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}]
                )
            else:
                openai_response = self.client.responses.create(
                    model=self.model,
                    previous_response_id=self.previous_response_id,
                    input=[{"role": "user", "content": prompt}]
                )
            self.previous_response_id = openai_response.id
            response_text = openai_response.output_text
            self._log_usage(getattr(openai_response, "usage", None))
        elif self._api_surface == "chat_completions":
            self._messages.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
            )
            message_content = completion.choices[0].message.content
            if isinstance(message_content, list):
                response_text = "".join(part if isinstance(part, str) else part.get("text", "") for part in message_content)
            else:
                response_text = message_content or ""
            self._messages.append({"role": "assistant", "content": response_text})
            self._log_usage(getattr(completion, "usage", None))
        elif self._api_surface == "completions":
            # Fallback for legacy completion endpoints – maintain a running prompt.
            self._messages.append(prompt)
            compiled_prompt = "\n\n".join(self._messages)
            self._logger.info("LLM compiled prompt: %s", compiled_prompt)
            completion = self.client.completions.create(
                model=self.model,
                prompt=compiled_prompt,
            )
            response_text = completion.choices[0].text or ""
            self._messages.append(response_text)
            self._log_usage(getattr(completion, "usage", None))
        else:
            raise ValueError(f"Unsupported OpenAI API surface: {self._api_surface}")

        response_text = response_text or ""
        self._logger.info("LLM response: %s", response_text)
        return Response(text=response_text)

    def _log_usage(self, usage):
        if usage is None:
            self._logger.info("Token usage unavailable for the latest request.")
            return

        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", None)

        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            try:
                total_tokens = input_tokens + output_tokens
            except TypeError:
                total_tokens = None

        self._logger.info(
            "Token usage - input: %s, output: %s, total: %s",
            input_tokens if input_tokens is not None else "n/a",
            output_tokens if output_tokens is not None else "n/a",
            total_tokens if total_tokens is not None else "n/a",
        )
    
def extract_code_from_response(response, llm="gemini"):
    # Use regular expression to find the code block within the markdown
    try:
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
    except Exception as e:
        print(f"Error extracting code: {e}")

    if llm == "gemini":
        return ""
    elif llm == "openai":
        return response

def fetch_documentation(function_name):
    base_url = "https://pytorch.org/docs/stable/generated/"
    function_url = base_url + function_name + ".html"
    response = requests.get(function_url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def clean_text(text):
    #remove excessive newlines
    return ' '.join(text.split())

def extract_function_info(html_content, function_name):
    if not html_content:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try finding the section using different possible patterns
    section_id = function_name.replace('.', '-').lower()
    doc_content = soup.find('div', {'class': 'section', 'id': section_id})
    
    if not doc_content:
        section_id = function_name.replace('.', '-')
        doc_content = soup.find('div', {'class': 'section', 'id': section_id})
    
    if not doc_content:
        # If specific section ID is not found, use a more general approach
        doc_content = soup.find('dl', {'class': 'py class'})

    if doc_content:
        # Extract text while avoiding duplicates
        lines = []
        seen_lines = set()
        for element in doc_content.find_all(['p', 'pre', 'code', 'dd', 'dt', 'ul', 'li', 'h1', 'h2', 'h3']):
            cleaned_line = clean_text(element.get_text())
            if cleaned_line not in seen_lines:
                lines.append(cleaned_line)
                seen_lines.add(cleaned_line)
        return '\n'.join(lines)
    else:
        return None
