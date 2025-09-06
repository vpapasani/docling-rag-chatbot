from crewai import Agent, Task, Crew,LLM
import os
import json # Import the json module
from pydantic import BaseModel, Field

# Define a Pydantic model for the expected JSON output
class OptimizedPromptOutput(BaseModel):
    optimized_prompt: str = Field(description="The optimized version of the given prompt.")

def crew_optimize(prompt: str) -> str:
    print("Crew Optimizer Invoked")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    print(f"GROQ_API_KEY is loaded.")

    # Initialize the LLM and specify the Groq API base URL and model
    groq_llm = LLM(
        api_key=GROQ_API_KEY,
        model="groq/llama-3.1-8b-instant",  # Use the model name directly
        base_url="https://api.groq.com/openai/v1"
    )

    # Create an agent with the task
    optimizer_agent = Agent(
        role="Prompt Optimizer Agent",
        goal="Optimize prompts",
        backstory="A specialized AI agent for improving prompt quality",
        llm=groq_llm,
        verbose=True
    )

    """  -- text output 
    # Define the optimizer task
    optimization_task = Task(
        description=f"Optimize this prompt:\n\n{prompt}",
        expected_output="An optimized version of the given prompt.",
        agent=optimizer_agent,
        llm=groq_llm
    )"""
    # JSON output
    optimization_task = Task(
        description=f"Optimize this prompt:\n\n{prompt}",
        # Specify the expected output as a Pydantic model, which allows CrewAI to format as JSON
        expected_output="A JSON object containing the optimized prompt, structured as follows: {'optimized_prompt': 'your optimized prompt here'}",
        agent=optimizer_agent,
        llm=groq_llm,
        output_pydantic=OptimizedPromptOutput  # This tells CrewAI to output a Pydantic model
    )

    # Create a Crew to orchestrate
    crew = Crew(
        agents=[optimizer_agent],
        tasks=[optimization_task],
        verbose=True
    )


    # Execute the crew using kickoff()
    result = crew.kickoff()
    # Access the Pydantic model output and convert it to a JSON string or dictionary
    if result.pydantic:
        optimized_prompt_str = result.pydantic.optimized_prompt
    elif result.json_dict:
        # Fallback if somehow .pydantic is not set but .json_dict is
        optimized_prompt_str = result.json_dict.get("optimized_prompt",
                                                           "Error: Could not extract optimized prompt.")
    else:
        optimized_prompt_str = result.raw  # Fallback to raw output

    return optimized_prompt_str


if __name__ == "__main__":
    optimized = crew_optimize("You are very experienced in refining prompts. Optimize this for better results.")
    print("Optimized Prompt:", optimized)

