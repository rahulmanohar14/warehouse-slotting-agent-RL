"""CrewAI crew that coordinates warehouse slotting analysis across specialist agents."""

from __future__ import annotations

from crewai import Agent, Crew, Task


class OrchestratorAgent:
    """Runs a sequential CrewAI workflow that turns RL slotting metrics into an executive-ready narrative."""

    def __init__(self) -> None:
        # Create the shared Groq LLM handle plus three agents and three chained tasks used by run().
        self._llm = "groq/llama-3.1-8b-instant"

        # Agent 1 — owns the operational read on slotting performance and concrete floor-level recommendations.
        self.slotting_coordinator = Agent(
            role="Warehouse Slotting Coordinator",
            goal="Analyze warehouse performance metrics and recommend slotting strategy adjustments",
            backstory="Expert in warehouse operations with deep knowledge of demand patterns and space optimization",
            llm=self._llm,
            verbose=False,
        )

        # Agent 2 — focuses on what the RL curves imply about learning quality and remaining upside.
        self.performance_analyst = Agent(
            role="RL Performance Analyst",
            goal="Interpret reinforcement learning training results and identify improvement opportunities",
            backstory="Data scientist specializing in reinforcement learning systems for logistics optimization",
            llm=self._llm,
            verbose=False,
        )

        # Agent 3 — turns prior task outputs into a concise stakeholder-facing summary.
        self.report_writer = Agent(
            role="Logistics Report Writer",
            goal="Produce clear executive summaries of warehouse optimization results",
            backstory="Technical writer who translates complex RL metrics into actionable business insights",
            llm=self._llm,
            verbose=False,
        )

        # Task 1 — quantify the travel-distance lift versus random placement and return three practical warehouse actions.
        self.task_slotting_analysis = Task(
            description=(
                "Given baseline_avg_distance=180.21 and trained_avg_distance=95.86, analyze the 47% improvement "
                "in picker travel distance. Explain why this matters for throughput and labor cost. "
                "List exactly three numbered operational recommendations (slotting policy, replenishment, or "
                "picking workflow) that a warehouse manager could implement next quarter."
            ),
            expected_output=(
                "A short analysis section followed by a clearly numbered list of three operational recommendations."
            ),
            agent=self.slotting_coordinator,
        )

        # Task 2 — consume the coordinator’s narrative and connect training dynamics to convergence and bandit behavior.
        self.task_rl_interpretation = Task(
            description=(
                "Using the prior task output as grounding, interpret the learning curve for this system: "
                "episode reward improving over 500 training episodes, total picker distance falling from roughly "
                "180 toward roughly 96, and bandit-driven prime-zone promotions stabilizing across training. "
                "Summarize what this pattern suggests about DQN and LinUCB convergence, credit assignment, and "
                "whether exploitation has begun to dominate exploration."
            ),
            expected_output=(
                "A concise technical summary (several short paragraphs) aimed at an RL engineer or analytics lead."
            ),
            agent=self.performance_analyst,
            context=[self.task_slotting_analysis],
        )

        # Task 3 — merge operational and RL perspectives into a tight executive brief of about 150 words.
        self.task_executive_summary = Task(
            description=(
                "Building on the RL analyst’s interpretation, write an executive summary of approximately 150 words "
                "for a logistics director. Blend the operational recommendations and the convergence story into one "
                "coherent narrative with a clear headline takeaway, one paragraph on business impact, and one "
                "paragraph on risks or next monitoring steps."
            ),
            expected_output="A single executive summary of about 150 words suitable for email or a steering deck.",
            agent=self.report_writer,
            context=[self.task_rl_interpretation],
        )

    def run(self) -> str:
        # Instantiate the crew, execute all tasks in order, and return the final crew output as text for printing.
        crew = Crew(
            agents=[
                self.slotting_coordinator,
                self.performance_analyst,
                self.report_writer,
            ],
            tasks=[
                self.task_slotting_analysis,
                self.task_rl_interpretation,
                self.task_executive_summary,
            ],
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    orchestrator = OrchestratorAgent()
    final_output = orchestrator.run()
    print(final_output)
