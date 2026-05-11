import unittest

from ip3o.env.wrappers import SafetyGymnasiumWrapper, SafetyStep


class _FakeEnv:
    def __init__(self, step_result):
        self._step_result = step_result
        self.observation_space = object()
        self.action_space = object()

    def reset(self, **kwargs):
        return "reset_obs", {"seed": kwargs.get("seed")}

    def step(self, action):
        return self._step_result


class TestSafetyGymnasiumWrapper(unittest.TestCase):
    def test_step_handles_six_item_output(self):
        env = _FakeEnv(("obs", 1.5, 0.25, True, False, {"foo": "bar"}))
        wrapper = SafetyGymnasiumWrapper(env)

        step = wrapper.step("action")

        self.assertIsInstance(step, SafetyStep)
        self.assertEqual(step.observation, "obs")
        self.assertEqual(step.reward, 1.5)
        self.assertEqual(step.cost, 0.25)
        self.assertTrue(step.terminated)
        self.assertFalse(step.truncated)
        self.assertEqual(step.info, {"foo": "bar"})

    def test_step_handles_five_item_output_and_cost_fallback(self):
        env = _FakeEnv(("obs", 2.0, False, True, {"cost_sum": 0.75}))
        wrapper = SafetyGymnasiumWrapper(env)

        step = wrapper.step("action")

        self.assertEqual(step.reward, 2.0)
        self.assertEqual(step.cost, 0.75)
        self.assertFalse(step.terminated)
        self.assertTrue(step.truncated)
        self.assertEqual(step.info, {"cost_sum": 0.75})

    def test_step_rejects_unexpected_arity(self):
        env = _FakeEnv(("obs", 1.0, 0.0, False))
        wrapper = SafetyGymnasiumWrapper(env)

        with self.assertRaises(ValueError):
            wrapper.step("action")


if __name__ == "__main__":
    unittest.main()

