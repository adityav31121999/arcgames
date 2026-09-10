"""Evidence review between perception and the next decision."""
from .brain import BrainChain
from .inference import invoke_stage
from ..core.context import context_part


class DebuggerChain(BrainChain):
    def analyse(self, before, after, action, prediction, observation, world, repeated=False):
        prompt = f"""Audit one completed experiment, then propose the next experiment.
Action actually executed: {action}
Prediction before action: {prediction or 'unknown'}
Measured/visual observation: {observation}
Repeated state or transition: {repeated}
Coordinates: x increases RIGHT, y increases DOWN. Only ACTION6 uses coordinates.
Do not infer goal progress from changed pixels or assign a player without evidence.
Compare two explanations; use before/after entity positions to test button effects.
If repeated, identify a different informative experiment. Do not invent verified rules.
Return concise labeled fields (no ACTION command):
Recent findings: <prediction supported, contradicted, or inconclusive; evidence>
Action model: <observed effects and remaining unknowns>
Hypotheses: <two alternatives, with evidence for/against each>
Open questions: <what remains unknown>
Plan: <next discriminating experiment and why>
Expected effect: <different observable outcomes under the alternatives>
"""
        self.last_result = invoke_stage(
            self.model, self.system_prompt, prompt, stage="Debugger", max_tokens=self.max_tokens,
            images=[("Before", before.get_pil_image()), ("After", after.get_pil_image())],
            context=[context_part("Working world model", world, 0, "head")],
            required_labels=("Recent findings", "Action model", "Hypotheses", "Plan"),
            temperature=0.0, enable_thinking=self.enable_thinking, raw_output=True,
        )
        return self.last_result.text
