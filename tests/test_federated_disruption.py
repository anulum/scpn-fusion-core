# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Federated disruption prediction tests
"""Tests for federated learning framework (FedAvg / FedProx)."""

import json

import numpy as np
import pytest

from scpn_fusion.control.federated_disruption import (
    MACHINE_PROFILES,
    N_FEATURES,
    FederatedConfig,
    FederatedServer,
    _init_mlp_weights,
    create_machine_clients,
    differential_privacy_clip,
)


def _make_clients(machines=("DIII-D", "JET", "KSTAR"), seed=42):
    cfgs = [{"machine": m, "n_train": 120, "n_test": 40} for m in machines]
    return create_machine_clients(cfgs, seed=seed)


# ── Factory & client basics ──────────────────────────────────────────


class TestMachineClients:
    def test_create_three_clients(self):
        clients = _make_clients()
        assert len(clients) == 3
        assert {c.machine for c in clients} == {"DIII-D", "JET", "KSTAR"}

    def test_data_shapes(self):
        clients = _make_clients()
        for c in clients:
            assert c.X_train.shape == (120, N_FEATURES)
            assert c.y_train.shape == (120,)
            assert c.X_test.shape == (40, N_FEATURES)
            assert set(np.unique(c.y_train)).issubset({0.0, 1.0})

    def test_get_data_size(self):
        clients = _make_clients()
        assert clients[0].get_data_size() == 120

    def test_unknown_machine_rejected(self):
        with pytest.raises(ValueError, match="Unknown machine"):
            create_machine_clients([{"machine": "TOKAMAK-X"}])


# ── FedAvg aggregation ───────────────────────────────────────────────


class TestFedAvg:
    def test_weighted_average(self):
        rng = np.random.default_rng(0)
        w1 = _init_mlp_weights(rng)
        w2 = _init_mlp_weights(rng)
        updates = [
            {"weights": w1, "n_samples": 100},
            {"weights": w2, "n_samples": 300},
        ]
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        avg = server.aggregate(updates)
        for key in w1:
            expected = w1[key] * 0.25 + w2[key] * 0.75
            np.testing.assert_allclose(avg[key], expected, atol=1e-12)

    def test_single_client_passthrough(self):
        rng = np.random.default_rng(1)
        w = _init_mlp_weights(rng)
        cfg = FederatedConfig(min_clients=1, machines=["DIII-D"])
        server = FederatedServer(cfg)
        avg = server.aggregate([{"weights": w, "n_samples": 50}])
        for key in w:
            np.testing.assert_allclose(avg[key], w[key], atol=1e-12)

    def test_empty_raises(self):
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        with pytest.raises(ValueError, match="at least one"):
            server.aggregate([])


# ── FedProx ──────────────────────────────────────────────────────────


class TestFedProx:
    def test_proximal_differs_from_plain(self):
        clients = _make_clients(("DIII-D", "JET"), seed=7)
        cfg_avg = FederatedConfig(
            n_rounds=3, local_epochs=3, machines=["DIII-D", "JET"], aggregation="fedavg"
        )
        cfg_prox = FederatedConfig(
            n_rounds=3,
            local_epochs=3,
            machines=["DIII-D", "JET"],
            aggregation="fedprox",
            mu_proximal=0.5,
        )
        s_avg = FederatedServer(cfg_avg, seed=0)
        s_prox = FederatedServer(cfg_prox, seed=0)
        s_avg.train(clients, 3)
        s_prox.train(clients, 3)
        diffs = [
            float(np.max(np.abs(s_avg.global_weights[k] - s_prox.global_weights[k])))
            for k in s_avg.global_weights
        ]
        assert max(diffs) > 1e-6


# ── Single round ─────────────────────────────────────────────────────


class TestRunRound:
    def test_round_returns_finite_metrics(self):
        clients = _make_clients()
        cfg = FederatedConfig(machines=["DIII-D", "JET", "KSTAR"])
        server = FederatedServer(cfg)
        result = server.run_round(clients)
        for m in result["client_metrics"]:
            assert 0.0 <= m["accuracy"] <= 1.0
            assert np.isfinite(m["loss"])
            assert m["n_samples"] > 0
            assert m["machine"] in MACHINE_PROFILES

    def test_too_few_clients_rejected(self):
        clients = _make_clients(("DIII-D",))
        cfg = FederatedConfig(min_clients=2, machines=["DIII-D"])
        server = FederatedServer(cfg)
        with pytest.raises(ValueError, match="Need >="):
            server.run_round(clients)

    def test_round_metrics_contain_required_keys(self):
        clients = _make_clients(("DIII-D", "JET"))
        cfg = FederatedConfig(min_clients=2, machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        result = server.run_round(clients)
        for m in result["client_metrics"]:
            for key in ("accuracy", "precision", "recall", "f1", "loss", "n_samples"):
                assert key in m


# ── Full training ────────────────────────────────────────────────────


class TestFullTraining:
    def test_loss_decreases_over_rounds(self):
        clients = _make_clients(seed=99)
        cfg = FederatedConfig(
            n_rounds=8,
            local_epochs=5,
            learning_rate=0.02,
            machines=["DIII-D", "JET", "KSTAR"],
        )
        server = FederatedServer(cfg, seed=99)
        history = server.train(clients, 8)
        assert len(history) == 8
        first_loss = history[0]["mean_loss"]
        last_loss = history[-1]["mean_loss"]
        assert last_loss < first_loss

    def test_federation_helps_generalisation(self):
        """Global model accuracy >= mean of single-machine accuracies."""
        rng = np.random.default_rng(42)
        machines = ["DIII-D", "JET", "KSTAR"]
        clients = _make_clients(machines, seed=42)

        cfg = FederatedConfig(
            n_rounds=10,
            local_epochs=5,
            learning_rate=0.02,
            machines=machines,
        )
        server = FederatedServer(cfg, seed=42)
        server.train(clients, 10)

        # Evaluate global model on each client's test set
        global_accs = [c.local_evaluate(server.global_weights)["accuracy"] for c in clients]

        # Compare against locally-only trained models
        local_accs = []
        for c in clients:
            local_w = _init_mlp_weights(rng)
            local_w = c.local_train(local_w, 50)
            local_accs.append(c.local_evaluate(local_w)["accuracy"])

        assert np.mean(global_accs) >= np.mean(local_accs) - 0.05


# ── Differential privacy ─────────────────────────────────────────────


class TestDifferentialPrivacy:
    def test_clipping_bounds_norm(self):
        rng = np.random.default_rng(0)
        grads = _init_mlp_weights(rng)
        # Scale up to ensure clipping activates
        big = {k: v * 100.0 for k, v in grads.items()}
        clipped = differential_privacy_clip(big, max_norm=1.0, noise_sigma=0.0, rng=rng)
        total_norm = np.sqrt(sum(float(np.sum(g**2)) for g in clipped.values()))
        assert total_norm <= 1.0 + 1e-6

    def test_noise_adds_variance(self):
        rng = np.random.default_rng(1)
        grads = {"w": np.zeros((4, 4)), "b": np.zeros(4)}
        noised = differential_privacy_clip(grads, max_norm=10.0, noise_sigma=1.0, rng=rng)
        assert float(np.std(noised["w"])) > 0.1


# ── Data heterogeneity convergence ───────────────────────────────────


class TestHeterogeneity:
    def test_skewed_distributions_converge(self):
        """Clients with different disruption fractions still converge."""
        cfgs = [
            {"machine": "DIII-D", "n_train": 100, "disruption_fraction": 0.2},
            {"machine": "JET", "n_train": 100, "disruption_fraction": 0.7},
            {"machine": "KSTAR", "n_train": 100, "disruption_fraction": 0.5},
        ]
        clients = create_machine_clients(cfgs, seed=77)
        cfg = FederatedConfig(n_rounds=8, local_epochs=5, machines=["DIII-D", "JET", "KSTAR"])
        server = FederatedServer(cfg, seed=77)
        history = server.train(clients, 8)
        assert history[-1]["mean_loss"] < history[0]["mean_loss"]


# ── Serialisation ────────────────────────────────────────────────────


class TestSerialisation:
    def test_round_trip(self):
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg, seed=5)
        state = server.get_state()
        blob = json.dumps(state)
        restored = FederatedServer.from_state(json.loads(blob))
        for key in server.global_weights:
            np.testing.assert_allclose(
                restored.global_weights[key],
                server.global_weights[key],
                atol=1e-12,
            )
        assert restored.config.aggregation == "fedavg"
