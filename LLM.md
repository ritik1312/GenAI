# LangChain <svg xmlns="http://www.w3.org/2000/svg">
**LangChain** is an open-source framework which allows developers to build  **Large Language Models (LLMs)** applications easily by simplifying the integration of 
data sources, external tools, and workflows. 

It provides a structured way to chain together various components such as LLMs, prompt templates, memory, and external 
APIs (in the form of pipelines) to create robust and scalable AI applications.

> LLMs excel at responding to prompts in a general context, but struggle in a specific domain they were never trained on. To do that, ML engineers must integrate 
LLM with the organization’s internal data sources and apply **prompt engineering**.
LangChain streamlines the intermediate steps to develop such data-responsive applications, making prompt engineering more efficient


## Key Features of LangChain

### 1. LLM Wrappers
- LangChain supports many language models, like OpenAI, Anthropic, Hugging Face, Ollama, etc. You can integrate APIs or locally installed models to interact seamlessly.
- `LLM wrappers` encapsulate the interaction with underlying LLM. They provide standard interfaces so that all LLMs work in a consistent way with Langchain ecosystem.
- Wrappers also provide extra features like **error handling, logging, retry mechanisms, preprocessing and postprocessing** of prompts and responses.
```
from langchain.llms import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)
response = llm("I want to open an Italian restaurant. Suggest a fancy name for it.")
```

**For Custom models or local Ollama/Hugging Face models**

```
from langchain.llms.base import LLM
from typing import Optional
import requests

class OllamaLLM(LLM):
    """Custom LLM class for Ollama models."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        super().__init__()
        self.model = model
        self.base_url = base_url

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[str] = None) -> str:
        """Send a prompt to the Ollama model."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"max_tokens": max_tokens},
            "temperature": 0.1,    # Controls the randomness of the output. (deterministic )
            "top_p": 0.9,
            "stop": ["\n", "END"]
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        
        if response.status_code == 200:
            return response.json().get("output", "")
        else:
            raise Exception(f"Failed to connect to Ollama: {response.status_code}, {response.text}")

# Example usage:
ollama_llm = OllamaLLM(model="tiny-llama")
response = ollama_llm("Explain the benefits of DevOps.")
print(response)

```

> Note: Methods like `_call` or `_llm_type` are internal/private functions required by the LangChain framework. `_call` is required method for defining how to process a prompt with the model.

### 2. Prompts
When u need your prompts in a proper structure, or u need to reuse same prompts structure, or want to have parameterized prompts, use `PromptTemplate` instaed of directly passing prompt to llm
```
from langchain.prompts import PromptTemplate

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="What are the best practices for {topic}?"
)

# Format the prompt
ollama_llm = OllamaLLM(model="tiny-llama")
formatted_prompt = ollama_llm.format(topic="Docker")
response = llm(formatted_prompt)
```

### 3. 
###
###
###

<table>
  <tr>
    <th>LangChain Feature</th>
    <th>Benefit to LLM Models</th>
  </tr>
  <tr>
    <td>Prompt Optimization</td>
    <td>Helps structure and improve prompts for better outputs.</td>
  </tr>
  <tr>
    <td>Retrieval-Augmented Generation (RAG)</td>
    <td>Allows fetching real-time, accurate data from external sources</td>
  </tr>
  <tr>
    <td>Memory Management</td>
    <td>Maintains conversation state for better user experience.</td>
  </tr>
  <tr>
    <td>Multi-step Pipelines</td>
    <td>Automates complex workflows involving multiple tasks.</td>
  </tr>
  <tr>
    <td>Tool Integration</td>
    <td>Enables models to interact with APIs, databases, and documents.</td>
  </tr>
</table>


### LangChain Use Cases

1. Chatbots with Contextual Awareness: 
Using memory to track and personalize conversations over multiple interactions.

2. Automated Resume Matching System: 
Querying local document stores and comparing candidate resumes with job descriptions.

3. Financial or Legal Document Summarization: 
Retrieving important sections from lengthy documents and summarizing them for quick decision-making.

4. Customer Support Automation: 
Providing accurate responses by integrating with knowledge bases and CRM systems.

For more info, Medium link <a href="https://cobusgreyling.medium.com/the-growing-langchain-ecosystem-f3bcb688df7a">here</a>
