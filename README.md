# Wikipedia Agent with GROQ, Mixtral-8x7B, and LangChain
This general question answering agent was built using Mixtral-8x7B LLM through GROQ, a Wikipedia search tool, and LangChain.

# How to run

**Step 1**: Install dependencies using the following command

```
pip install -r requirements.txt
```

**Step 2**: Set your `GROQ_API_KEY` as an environment variable

```
export GROQ_API_KEY=XXXXXX
```

**Step 3**: Run the gradio app

```
python app.py
```

# Demo
The demo has been depolyed to the following HuggingFace space.

https://huggingface.co/spaces/rasyosef/Wikipedia-Agent-with-Groq-and-Mixtral