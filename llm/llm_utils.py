from openai import OpenAI
import os, re, requests
from bs4 import BeautifulSoup

class Response:
    def __init__(self, text):
        self.text = text

class OAChatWrapper:
    def __init__(self, model=None, base_url=None, api_key=None):
        # Allow overriding defaults through args or environment variables.
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b")

        default_base = os.getenv("OLLAMA_BASE_URL", "https://ollama.aaaab3n.moe")
        # Ensure the client sees a /v1 suffix as required by the OpenAI SDK.
        base = base_url or default_base
        if not base.rstrip("/").endswith("/v1"):
            base = base.rstrip("/") + "/v1"

        key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"

        self.client = OpenAI(api_key=key, base_url=base)
        self.messages = []

    def send_message(self, prompt):
        # Maintain conversation history for follow-up prompts.
        conversation = self.messages + [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
        )
        assistant_content = completion.choices[0].message.content
        if isinstance(assistant_content, list):
            # Some OpenAI-compatible servers return a list of content parts.
            assistant_message = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in assistant_content
            )
        else:
            assistant_message = assistant_content
        self.messages = conversation + [{"role": "assistant", "content": assistant_message}]
        return Response(text=assistant_message)
    
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
