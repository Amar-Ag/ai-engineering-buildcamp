import pytest
import dotenv
dotenv.load_dotenv()

from doc_agent import create_agent, run_agent_stream, DocumentationAgentConfig, DEFAULT_INSTRUCTIONS
from tools import create_documentation_tools_cached
from tests.utils import collect_tools, ToolCall


def create_test_agent():
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig(
        instructions=DEFAULT_INSTRUCTIONS
    )

    agent = create_agent(agent_config, tools)
    return agent

@pytest.mark.asyncio
async def test_agent_uses_tools():
    agent = create_test_agent()

    user_prompt = 'llm as a judge'
    result = await run_agent_stream(agent, user_prompt)

    messages = result.new_messages()

    tool_calls = collect_tools(messages)
    assert len(tool_calls) >= 2

    search_call = tool_calls[0]
    assert search_call.name == 'search'

    get_file_call = tool_calls[1]
    assert get_file_call.name == 'get_file'
    