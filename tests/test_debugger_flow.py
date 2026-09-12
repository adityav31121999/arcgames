from types import SimpleNamespace
from unittest.mock import Mock

from arc_agent.agent.runner import ARCRunner
from arc_agent.chains.debugger import DebuggerChain
from arc_agent.chains.inference import invoke_stage
from tests.test_agent_reliability import agent, Environment


AUDIT = """**Recent findings:** No motion; prediction contradicted.
**Action model:** ACTION1 has no measured displacement here.
**Hypotheses:** H1 blocked at boundary; H2 button selects a different object.
**Plan:** Test ACTION2 and compare both objects.
Expected effect: Movement supports H1; selection change supports H2.
"""


def test_debugger_receives_transition_and_updates_next_decision_memory(agent, monkeypatch):
    monkeypatch.setenv("FAST_STEP_EVAL", "true")
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content=AUDIT)
    agent.debugger = DebuggerChain(model, max_tokens=2048)
    agent.brain.decide_action = Mock(return_value="Plan: probe\nExpected effect: motion\nACTION=ACTION1")
    agent.review_failed_iteration = Mock()
    env = Environment()
    runner = ARCRunner(agent, fast_step_eval=False, speculative_plan_max_steps=0)
    runner.play_level("audit", 1, env, env.obs(), env.action_space, max_steps=1, max_iterations=1)
    assert agent.debugger.last_result.ok
    assert agent.world_model.current_plan == "Test ACTION2 and compare both objects."
    prompt = model.invoke.call_args.args[0][1].content
    assert sum(part['type'] == 'image' for part in prompt) == 2
    assert "motion" in prompt[0]['text']
    assert runner.last_stage_status['vision'] is not None
    journal = agent.cache.memory_root / 'audit' / 'experiments_level_1.md'
    assert "prediction contradicted" in journal.read_text()
    assert "prediction contradicted" not in agent.cache.actions_log('audit', 1)
    assert any("Experiment evidence" in str(s) for s in agent.cache.context_sections('audit', 1))


def test_invalid_decision_stops_without_fallback_moves_or_resets(agent):
    agent.halt_on_invalid_decision = True
    agent.brain.decide_action = Mock(return_value="No final action")
    agent.review_failed_iteration = Mock()
    env = Environment()
    runner = ARCRunner(agent, speculative_plan_max_steps=0)
    runner.play_level('invalid', 1, env, env.obs(), env.action_space, max_steps=20, max_iterations=3)
    assert env.ticks == env.resets == 0
    assert agent.brain.decide_action.call_count == 2
    agent.review_failed_iteration.assert_not_called()
    assert 'No final action' in (agent.cache.memory_root / 'invalid' / 'memory_history.md').read_text()


def test_thinking_exhaustion_retries_with_more_budget_and_validates_final():
    model = Mock()
    model.invoke.side_effect = [RuntimeError('Thinking budget exhausted'), SimpleNamespace(content=AUDIT)]
    result = invoke_stage(model, 'system', 'review', stage='Debugger', max_tokens=2048,
                          required_labels=('Hypotheses', 'Plan'))
    assert result.ok
    assert [c.kwargs['max_tokens'] for c in model.invoke.call_args_list] == [2048, 4096]


def test_invalid_stage_includes_final_preview_without_accepting_it():
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content='A coherent answer with missing fields.')
    result = invoke_stage(model, '', '', stage='Debugger', max_tokens=100, required_labels=('Plan',))
    assert not result.ok and not result.text
    assert 'coherent answer' in result.error


def test_set_action_space_when_debugger_none_or_deleted(agent):
    # Case 1: debugger attribute is explicitly None
    agent.debugger = None
    agent.set_action_space(["ACTION1", "ACTION2"])

    # Case 2: debugger attribute is deleted
    del agent.debugger
    agent.set_action_space(["ACTION1", "ACTION2"])

