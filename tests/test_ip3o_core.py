import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

if torch is not None:
    from ip3o.algorithms.ip3o import IP3O, IP3OConfig
    from ip3o.models.actor_critic import ActorCritic


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TestIP3OCore(unittest.TestCase):
    def test_dual_critic_shapes(self):
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(16, 16))
        obs = torch.zeros((3, 4), dtype=torch.float32)
        action, logp, v_r, v_c = model.act(obs)

        self.assertEqual(action.shape, (3, 2))
        self.assertEqual(logp.shape, (3,))
        self.assertEqual(v_r.shape, (3,))
        self.assertEqual(v_c.shape, (3,))

    def test_incremental_penalty_increases_as_cost_approaches_limit(self):
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(16, 16))
        cfg = IP3OConfig(cost_limit=10.0, initial_penalty=1.0, penalty_increment=0.5, max_penalty=2.0)
        algo = IP3O(model, cfg, device=torch.device("cpu"))

        low_cost_penalty = algo.incremental_penalty(torch.tensor(2.0))
        high_cost_penalty = algo.incremental_penalty(torch.tensor(9.0))

        self.assertGreater(high_cost_penalty.item(), low_cost_penalty.item())

    def test_penalty_coefficient_anneals_and_caps(self):
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(16, 16))
        cfg = IP3OConfig(initial_penalty=0.1, penalty_increment=0.3, max_penalty=0.5)
        algo = IP3O(model, cfg, device=torch.device("cpu"))

        algo.anneal_penalty()
        algo.anneal_penalty()
        algo.anneal_penalty()

        self.assertAlmostEqual(algo.penalty_coeff.item(), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
