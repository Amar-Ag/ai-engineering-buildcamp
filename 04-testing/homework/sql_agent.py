from pydantic import BaseModel, Field
from pydantic_ai import Agent
import dotenv
dotenv.load_dotenv()

from sql_tools import SQLTools, con, setup_database

class SQLResult(BaseModel):
    sql_query: str = Field(description="The SQL query that was executed")
    result_text: str = Field(description="The text of the query result")
    row_count: int = Field(description="The number of rows returned by the query")

DEFAULT_INSTRUCTIONS = """
You are a SQL assistant. You will be given a question and access to a SQL database.
Always start by calling get_schema first before running any queries.
The database has a single table called 'trips'.
Only use the tools provided. Do not make up information.
"""

def create_agent():
    sql_tools = SQLTools(con)
    return Agent(
        name="sql_agent",
        model="google-gla:gemini-2.5-flash",
        tools=[sql_tools.get_schema, sql_tools.run_sql],
        instructions=DEFAULT_INSTRUCTIONS,
        output_type=SQLResult,
    )

async def main():
    setup_database()
    agent = create_agent()
    result = await agent.run("What's the average trip distance for rides with 2 passengers?")
    print(result.output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())