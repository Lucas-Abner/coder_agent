from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class Coder():
    """Coder crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm = LLM(
            api_key="ollama",
            base_url="http://localhost:11434",
            model="ollama/mistral:latest"
        )

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            code_execution_timeout=30,
            max_retry_limit=5
        )
    
    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config["coding_task"],
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

  