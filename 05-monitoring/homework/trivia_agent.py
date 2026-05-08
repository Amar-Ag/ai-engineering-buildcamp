from pydantic import BaseModel, Field
from pydantic_ai import Agent
from trivia_tools import TriviaTools
import dotenv
import questionary
dotenv.load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()


instructions = """You are a trivia quizmaster. When asked to play trivia:
1. Use the available tools to fetch trivia questions
2. Ask the player one question at a time with multiple choice options
3. Wait for their answer before moving to the next question
4. When the player answers, explain why the correct answer is correct - add interesting context and facts
5. After all questions, give the final score
"""

trivia_tools = TriviaTools()

agent = Agent(
    model="google-gla:gemini-2.5-flash",
    tools=[trivia_tools.get_categories, trivia_tools.get_questions],
    instructions=instructions,
)

async def main():
    result = await agent.run("Let's play a trivia game! Ask me 3 medium difficulty questions about science.")
    print(result.output)
    
    for msg in result.all_messages():
        for part in msg.parts:
            if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                print(f"Tool called: {part.tool_name}")

def run(prompt):
    message_history = []

    while True:
        result = agent.run_sync(prompt, message_history=message_history)
        print(result.output)
        message_history = result.all_messages()

        prompt = input("You (write 'stop' to stop): ")
        if not prompt or prompt.lower().strip() == 'stop':
            break

def ask_feedback():
    result = questionary.select(
        "How was the trivia session?",
        choices=["👍 Good", "👎 Bad", "Skip"],
    ).ask()

    if result is None or result == "Skip":
        return None

    return 1 if "Good" in result else -1


if __name__ == "__main__":
    with logfire.span('trivia_session'):
        run("Let's play 5 easy questions from Science & Nature")
        
        feedback = ask_feedback()
        if feedback is not None:
            logfire.info("user_feedback", feedback=feedback)



