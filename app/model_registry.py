"""Model registry backed by MLflow."""
from __future__ import annotations

from typing import Any, Callable, Optional


class ModelRegistry:
    """Loads and caches champion/challenger models via an MLflow-compatible client."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        loader_fn: Callable[[str], Any],
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._loader_fn = loader_fn

    def get_champion(self) -> Any:
        version_info = self._client.get_model_version_by_alias(
            self._model_name, "champion"
        )
        return self._loader_fn(version_info.source)

    def get_challenger(self) -> Optional[Any]:
        try:
            version_info = self._client.get_model_version_by_alias(
                self._model_name, "challenger"
            )
            return self._loader_fn(version_info.source)
        except Exception:
            return None

    def promote_challenger(self) -> None:
        """Promote the current challenger to champion."""
        challenger_version = self._client.get_model_version_by_alias(
            self._model_name, "challenger"
        )
        self._client.set_registered_model_alias(
            self._model_name, "champion", challenger_version.version
        )
        self._client.delete_registered_model_alias(
            self._model_name, "challenger"
        )
