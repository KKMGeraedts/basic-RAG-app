import os 

import pandas as pd
from dotenv import load_dotenv
from llama_index.experimental import PandasQueryEngine  
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

from pdf import article_engine, materiality_assessment_engine
from note_engine import note_engine
from prompts import new_prompt, instruction_str, context

load_dotenv()

stock_path = os.path.join("data", "stockdata.csv")
stock_df = pd.read_csv(stock_path)

stock_query_engine = PandasQueryEngine(
    df=stock_df, verbose=True, instruction_str=instruction_str
)
stock_query_engine.update_prompts({"stock_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=stock_query_engine,
        metadata=ToolMetadata(
            name="stock_data",
            description="this gives information on intraday stock prices.",
        ),
    ),
    QueryEngineTool(
    query_engine=article_engine,
    metadata=ToolMetadata(
        name="academic_article",
        description="""this gives information on the PRG paper: "Coarse–graining and
        hints of scaling in a population of 1000+ neurons". Published in 2018 by
        Meshulam et al.""",
        ),
    ),
    QueryEngineTool(
    query_engine=materiality_assessment_engine,
    metadata=ToolMetadata(
        name="materiality_assessment_pdf",
        description="""this gives information on the ESRS Materiality Assessment.""",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
