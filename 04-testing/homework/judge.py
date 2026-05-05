from pydantic import BaseModel, Field
from pydantic_ai import Agent
from utils import collect_tools

judge_instructions = """You are an expert judge evaluating the performance of a SQL agent.""".strip()

judge_user_prompt_template = """
Evaluate the agent's response against these criteria:
{criteria}

Agent output:
{output}

Tool calls made:
{tool_calls}
"""

class JudgeCriterion(BaseModel):
    criterion_description: str = Field(
        description="The specific requirement being evaluated."
    )
    passed: bool = Field(
        description="Whether the agent satisfied this requirement."
    )
    judgement: str = Field(
        description="Explanation of why the agent passed or failed."
    )

class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion] = Field(
        description="Individual evaluations for each criterion."
    )
    feedback: str = Field(
        description="Overall summary of the agent's performance."
    )

def create_judge_agent():
    return Agent(
        name="judge",
        model="google-gla:gemini-2.5-flash-lite",
        instructions=judge_instructions,
        output_type=JudgeFeedback,
    )

async def assert_criteria(result, criteria):
    messages = result.new_messages()
    tool_calls = collect_tools(messages)
    output = str(result.output)

    judge_agent = create_judge_agent()
    judge_prompt = judge_user_prompt_template.format(
        criteria='\n'.join(criteria),
        output=output,
        tool_calls='\n'.join([str(tc) for tc in tool_calls])
    )

    judge_result = await judge_agent.run(judge_prompt)

    print('judge feedback:')
    print(judge_result.output.feedback)

    for criterion in judge_result.output.criteria:
        print(f"{criterion.criterion_description}: {criterion.judgement}")
        assert criterion.passed, f"{criterion.criterion_description}: {criterion.judgement}"