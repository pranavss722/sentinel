"""Tests for ModelRegistry."""
import numpy as np


class FakeModel:
    """Stub that mimics a loaded ML model with a predict method."""
    def __init__(self, name: str = "fake"):
        self.name = name

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class FakeMlflowClient:
    """In-memory fake of mlflow.MlflowClient for testing."""
    def __init__(self):
        self._models = {}  # {(name, alias): version_number}
        self._artifacts = {}  # {uri: model_object}

    def get_model_version_by_alias(self, name: str, alias: str):
        key = (name, alias)
        if key not in self._models:
            raise Exception(f"No model version with alias '{alias}'")

        class _Version:
            def __init__(self, version, source):
                self.version = version
                self.source = source

        version = self._models[key]
        return _Version(version=str(version), source=f"models:/{name}/{version}")

    def register_model(self, name, alias, version, model):
        self._models[(name, alias)] = version
        self._artifacts[f"models:/{name}/{version}"] = model

    def set_registered_model_alias(self, name, alias, version):
        self._models[(name, alias)] = int(version)

    def delete_registered_model_alias(self, name, alias):
        self._models.pop((name, alias), None)


def make_registry_with_champion(champion_model=None):
    from app.model_registry import ModelRegistry

    fake_client = FakeMlflowClient()
    model = champion_model or FakeModel("champion")
    fake_client.register_model("ad-click-baseline", "champion", 1, model)

    def fake_load(model_uri):
        return fake_client._artifacts.get(model_uri)

    registry = ModelRegistry(
        client=fake_client,
        model_name="ad-click-baseline",
        loader_fn=fake_load,
    )
    return registry, model


def test_model_registry_loads_champion():
    registry, expected_model = make_registry_with_champion()
    champion = registry.get_champion()
    assert champion is expected_model


def test_model_registry_get_challenger_returns_none_when_absent():
    registry, _ = make_registry_with_champion()
    assert registry.get_challenger() is None


def test_model_registry_get_challenger_returns_model_when_present():
    from app.model_registry import ModelRegistry

    fake_client = FakeMlflowClient()
    champ = FakeModel("champion")
    chal = FakeModel("challenger")
    fake_client.register_model("ad-click-baseline", "champion", 1, champ)
    fake_client.register_model("ad-click-baseline", "challenger", 2, chal)

    def fake_load(model_uri):
        return fake_client._artifacts.get(model_uri)

    registry = ModelRegistry(
        client=fake_client,
        model_name="ad-click-baseline",
        loader_fn=fake_load,
    )
    assert registry.get_challenger() is chal


def test_model_registry_champion_predict():
    registry, _ = make_registry_with_champion()
    champion = registry.get_champion()
    preds = champion.predict(np.array([[1.0] * 10]))
    assert len(preds) == 1


def test_model_registry_promote_swaps_champion():
    from app.model_registry import ModelRegistry

    fake_client = FakeMlflowClient()
    champ = FakeModel("champion")
    chal = FakeModel("challenger")
    fake_client.register_model("ad-click-baseline", "champion", 1, champ)
    fake_client.register_model("ad-click-baseline", "challenger", 2, chal)

    def fake_load(model_uri):
        return fake_client._artifacts.get(model_uri)

    registry = ModelRegistry(
        client=fake_client,
        model_name="ad-click-baseline",
        loader_fn=fake_load,
    )
    # After promotion, champion should be the former challenger
    registry.promote_challenger()
    new_champion = registry.get_champion()
    assert new_champion is chal


def test_promote_challenger_uses_public_api_only():
    """promote_challenger must only use public MLflow client methods,
    never mutate private attributes like _models."""
    from unittest.mock import MagicMock
    from app.model_registry import ModelRegistry

    mock_client = MagicMock(spec=[
        "get_model_version_by_alias",
        "set_registered_model_alias",
        "delete_registered_model_alias",
    ])
    # Deliberately has NO _models attribute — will blow up if accessed

    challenger_version = MagicMock()
    challenger_version.version = "3"
    challenger_version.source = "models:/ad-click-baseline/3"
    mock_client.get_model_version_by_alias.return_value = challenger_version

    registry = ModelRegistry(
        client=mock_client,
        model_name="ad-click-baseline",
        loader_fn=lambda uri: None,
    )

    registry.promote_challenger()

    mock_client.get_model_version_by_alias.assert_called_once_with(
        "ad-click-baseline", "challenger"
    )
    mock_client.set_registered_model_alias.assert_called_once_with(
        "ad-click-baseline", "champion", "3"
    )
    mock_client.delete_registered_model_alias.assert_called_once_with(
        "ad-click-baseline", "challenger"
    )
