import gradio as gr
import os
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Wikipedia Search Tool
tools = [wiki_tool]

# Your GROQ API KEY
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
)

prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.pretty_print()

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


def generate(query):
    if query.strip() == "":
        return "Enter your question"
    output = agent_executor.invoke({"input": query})["output"]
    return output


with gr.Blocks() as demo:
    gr.Markdown(
        """
  ## Wikipedia Agent with GROQ, Mixtral-8x7B, and LangChain

  This is general question answering agent was created using Mixtral-8x7B LLM through GROQ, a Wikipedia search tool, and LangChain. 
  """
    )
    gr.Markdown("#### Enter your question")
    with gr.Row():
        with gr.Column():
            ques = gr.Textbox(label="Question", placeholder="Enter text here", lines=2)
        with gr.Column():
            ans = gr.Textbox(label="Answer", lines=4, interactive=False)
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Submit")
        with gr.Column():
            clear = gr.ClearButton([ques, ans])

    btn.click(fn=generate, inputs=[ques], outputs=[ans])
    examples = gr.Examples(
        examples=[
            "When is Leonhard Euler's birthday?",
            "Who were the 3 main characters in GTA V?",
            "Who was the voice actor for Kratos in God of War: Ragnarok?",
            "How much did 'Deadpool and Wolverine' make at the global box office?",
            "Who was the last monarch of Ethiopia?",
        ],
        inputs=[ques],
    )

demo.queue().launch(debug=True)
