from llama_index.core import PromptTemplate

instruction_str = """
    1. Convert the query to execurable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the 'eval()' function.
    3. Make sure that any relevant imports are included in your Python expression.
    4. The code should represent a solution to the query.
    5. PRINT ONLY THE EXPRESSION.
    6. Do not quote the expression.
    """

new_prompt = PromptTemplate(
    """
    You are working with a Pandas dataframe in Python.
    The name of the dataframe is 'df'.
    This is the result of 'print(df.head())':
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: 
    """
)

context = """
    Purpose: The primary role of this agent is to assist users by providing
    accurate information about stock pricing at specific times.
    """