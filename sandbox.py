from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain, LLMMathChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback

from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

from langchain.memory import ConversationBufferWindowMemory

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import json
import os
import re
from typing import List, Union
from pydantic import BaseModel, Field


# model_name = "gpt-4"
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, temperature=0)
class CalculatorInput(BaseModel):
    question: str = Field()

template_with_history = """You was developed by internet eshop... You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to give detailed, informative answers. IUse only provided data and not prior knowledge.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

import json
import re
from typing import Union

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "


        kwargs["agent_scratchpad"] = thoughts


        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        if "Final Answer:" in llm_output:
            return AgentFinish(
                # return_values={"output": combined_output},

                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},

                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)


        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(
        PyPDFLoader(pdf_data_file).load_and_split(),
        embeddings,
        persist_directory=Chroma_persist_directory)
retriever = db.as_retriever()

summarize_chain = load_summarize_chain(llm, chain_type="stuff")


def knowledge_base_to_summary(input):
    # Or it could be ChromaDB relevant documents
    documnets_summaty_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = documnets_summaty_retriever(input)
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce',
#                                      verbose=True # if you want to see the prompts being used
                                    )
    output = summary_chain.run(result['source_documents'])

    return output

retrieval_llm = OpenAI(temperature=0)

podcast_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=retriever)

# Якщо ми хочемо діставати контекст прямо з модельки
# podcast_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True, input_key="question")


llm_math_chain = LLMMathChain(llm=llm, verbose=True)
expanded_tools = [
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    Tool(
        name = 'Knowledge Base',
        func=podcast_retriever.run,
        # Якщо ми хочемо діставати контекст прямо з модельки
        # func=lambda query: podcast_retriever({"question": query}),
        description="Useful for specifi questions based on the provided documentation about how to do things."
    ),
    Tool(
        name = 'Summary based',
        func=knowledge_base_to_summary,
        description='use this tool ONLY! when user is mentioned word summary in input'
    ),
    Tool(
        name = 'Recomendation system',
        func=rec_sys,
        description='use this tool ONLY! when user is asking for recomendation or suggestion'
    )
]

# Re-initialize the agent with our new list of tools
prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=expanded_tools,
    input_variables=["input", "intermediate_steps", "history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
multi_tool_names = [tool.name for tool in expanded_tools]
multi_tool_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=multi_tool_names
)


multi_tool_memory = ConversationBufferWindowMemory(k=5)
multi_tool_executor = AgentExecutor.from_agent_and_tools(agent=multi_tool_agent, tools=expanded_tools, verbose=True, memory=multi_tool_memory)


# gpt 3.5
with get_openai_callback() as cb:
  question = 'Please provide a summary of Project toolbar'
  res = multi_tool_executor.run(question)
  print(res)
  print(f"Total Tokens: {cb.total_tokens}")
  print(f"Prompt Tokens: {cb.prompt_tokens}")
  print(f"Completion Tokens: {cb.completion_tokens}")
  print(f"Total Cost (USD): ${cb.total_cost}")
  question = 'What are those options?'
  res = multi_tool_executor.run(question)
  print(res)
  print(f"Total Tokens: {cb.total_tokens}")
  print(f"Prompt Tokens: {cb.prompt_tokens}")
  print(f"Completion Tokens: {cb.completion_tokens}")
  print(f"Total Cost (USD): ${cb.total_cost}")
  question = 'Where I can find this toolbar?'
  res = multi_tool_executor.run(question)
  print(res)
  print(f"Total Tokens: {cb.total_tokens}")
  print(f"Prompt Tokens: {cb.prompt_tokens}")
  print(f"Completion Tokens: {cb.completion_tokens}")
  print(f"Total Cost (USD): ${cb.total_cost}")