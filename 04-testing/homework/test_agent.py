import pytest
import dotenv
dotenv.load_dotenv()

from sql_agent import create_agent
from utils import collect_tools, ToolCall

@pytest.fixture(scope="module")
def agent():
    return create_agent()

@pytest.mark.asyncio
async def test_trips_more_than_5_passengers(agent):
    result = await agent.run("How many trips had more than 5 passengers?")
    output = result.output
    assert len(output.sql_query) > 0
    assert "22413" in output.result_text

@pytest.mark.asyncio
async def test_agent_uses_tools(agent):
    user_prompt = 'What is the most common payment type'
    result = await agent.run(user_prompt)

    messages = result.new_messages()

    tool_calls = collect_tools(messages)
    assert len(tool_calls) >= 2

    search_call = tool_calls[0]
    assert search_call.name == 'get_schema'

    get_file_call = tool_calls[1]
    assert get_file_call.name == 'run_sql'