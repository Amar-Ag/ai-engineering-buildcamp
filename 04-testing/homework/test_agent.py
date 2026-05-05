import pytest
import dotenv
dotenv.load_dotenv()

from sql_agent import create_agent, run_agent_test
from utils import collect_tools, ToolCall
from judge import assert_criteria
from sql_tools import setup_database

@pytest.fixture(scope="module")
def agent():
    setup_database()
    return create_agent()

@pytest.mark.asyncio
async def test_trips_more_than_5_passengers(agent):
    result = await run_agent_test(agent, "How many trips had more than 5 passengers?")
    output = result.output
    assert len(output.sql_query) > 0
    assert "22413" in output.result_text

@pytest.mark.asyncio
async def test_agent_uses_tools(agent):
    result = await run_agent_test(agent, "What is the most common payment type")
    messages = result.new_messages()
    tool_calls = collect_tools(messages)
    assert len(tool_calls) >= 2
    assert tool_calls[0].name == 'get_schema'
    assert tool_calls[1].name == 'run_sql'

@pytest.mark.asyncio
async def test_highest_fare_hour(agent):
    result = await run_agent_test(agent, "Which hour of the day has the highest average fare amount?")
    await assert_criteria(result, [
        "the SQL query correctly calculates average fare by hour of day",
        "the result identifies a specific hour as having the highest average fare",
        "the result includes the actual average fare amount",
    ])

@pytest.mark.asyncio
async def test_average_tip_credit_card(agent):
    result = await run_agent_test(agent, "What is the average tip amount for credit card payments?")
    output = result.output
    assert len(output.sql_query) > 0
    assert "4.1" in output.result_text

@pytest.mark.asyncio
async def test_busiest_pickup_location(agent):
    result = await run_agent_test(agent, "Which pickup location (PULocationID) has the most trips?")
    messages = result.new_messages()
    tool_calls = collect_tools(messages)
    assert tool_calls[0].name == 'get_schema'
    assert any(tc.name == 'run_sql' for tc in tool_calls)
    assert "132" in result.output.result_text

@pytest.mark.asyncio
async def test_average_fare_long_trips(agent):
    result = await run_agent_test(agent, "What is the average fare for trips longer than 10 miles?")
    output = result.output
    assert len(output.sql_query) > 0
    assert "62" in output.result_text

@pytest.mark.asyncio
async def test_zero_passenger_trips(agent):
    result = await run_agent_test(agent, "How many trips had zero passengers recorded?")
    output = result.output
    assert len(output.sql_query) > 0
    assert "31465" in output.result_text

@pytest.mark.asyncio
async def test_busiest_day_of_week(agent):
    result = await run_agent_test(agent, "What is the busiest day of the week for taxi trips?")
    await assert_criteria(result, [
        "the SQL query extracts the day of week from the pickup datetime",
        "the result identifies a specific day as the busiest",
        "the result includes the trip count for that day",
    ])