from typing import Annotated
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool

tavily_tool = TavilySearchResults(max_results=5)
repl_tool = PythonREPL()
repl_tool.run("import matplotlib; matplotlib.use('Agg')")

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl_tool.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

ag_tavily_tool = LangChainToolAdapter(tavily_tool)
ag_repl_tool = LangChainToolAdapter(python_repl)