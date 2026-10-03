"""Prompt caching for cost reduction and performance.

Supports multiple caching backends:
- In-memory (default, fast, ephemeral)
- File-based (persistent across restarts)
- Redis (distributed, production-ready)
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """A cached prompt response.

    Attributes:
        content: The cached response content.
        created_at: Unix timestamp when entry was created.
        ttl: Time-to-live in seconds (None = never expires).
        metadata: Additional metadata (tokens, cost, etc.).
    """

    content: str
    created_at: float = field(default_factory=time.time)
    ttl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            content=data["content"],
            created_at=data["created_at"],
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {}),
        )


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        ...

    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set a cache entry."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Clear all cache entries. Returns count of deleted entries."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Get the number of entries in the cache."""
        ...


class MemoryCache(CacheBackend):
    """In-memory cache backend.

    Fast but ephemeral - data is lost when process exits.
    Suitable for development and short-running processes.

    Args:
        max_size: Maximum number of entries (None = unlimited).
    """

    def __init__(self, max_size: int | None = 1000) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size

    def get(self, key: str) -> CacheEntry | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[key]
            return None
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        # Evict oldest entries if at capacity
        if self._max_size and len(self._cache) >= self._max_size:
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]
        self._cache[key] = entry

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def size(self) -> int:
        # Clean expired entries first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
        return len(self._cache)


class FileCache(CacheBackend):
    """File-based cache backend.

    Persistent cache stored on disk. Suitable for development
    and single-machine deployments.

    Args:
        cache_dir: Directory to store cache files.
    """

    def __init__(self, cache_dir: str | Path = ".flowprompt_cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> CacheEntry | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            entry = CacheEntry.from_dict(data)
            if entry.is_expired:
                path.unlink()
                return None
            return entry
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, key: str, entry: CacheEntry) -> None:
        path = self._key_to_path(key)
        path.write_text(json.dumps(entry.to_dict()))

    def delete(self, key: str) -> bool:
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        count = 0
        for path in self._cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

    def size(self) -> int:
        return len(list(self._cache_dir.glob("*.json")))


class PromptCache:
    """High-level prompt caching interface.

    Provides semantic caching for prompt responses, with support
    for multiple backends and automatic cache key generation.

    Example:
        >>> cache = PromptCache()
        >>> result = cache.get_or_compute(
        ...     prompt=my_prompt,
        ...     model="gpt-4o",
        ...     compute_fn=lambda: provider.complete(prompt, model)
        ... )
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        default_ttl: float | None = 3600,  # 1 hour default
        enabled: bool = True,
    ) -> None:
        """Initialize the prompt cache.

        Args:
            backend: Cache backend to use (default: MemoryCache).
            default_ttl: Default TTL in seconds (None = never expires).
            enabled: Whether caching is enabled.
        """
        self._backend = backend or MemoryCache()
        self._default_ttl = default_ttl
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """Whether the cache stores and serves entries."""
        return self._enabled

    def _generate_key(
        self,
        prompt_hash: str,
        model: str,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Generate a cache key from prompt and parameters."""
        key_parts = [
            prompt_hash,
            model,
            f"temp={temperature}",
        ]
        # Add any other relevant kwargs
        for k, v in sorted(kwargs.items()):
            if k in ("max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
                key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def get(
        self,
        prompt_hash: str,
        model: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> CacheEntry | None:
        """Get a cached response if available."""
        if not self._enabled:
            return None
        key = self._generate_key(prompt_hash, model, temperature, **kwargs)
        entry = self._backend.get(key)
        if entry:
            self._hits += 1
        else:
            self._misses += 1
        return entry

    def set(
        self,
        prompt_hash: str,
        model: str,
        content: str,
        temperature: float = 0.0,
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Cache a response."""
        if not self._enabled:
            return
        key = self._generate_key(prompt_hash, model, temperature, **kwargs)
        entry = CacheEntry(
            content=content,
            ttl=ttl if ttl is not None else self._default_ttl,
            metadata=metadata or {},
        )
        self._backend.set(key, entry)

    def invalidate(
        self,
        prompt_hash: str,
        model: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> bool:
        """Invalidate a specific cache entry."""
        key = self._generate_key(prompt_hash, model, temperature, **kwargs)
        return self._backend.delete(key)

    def clear(self) -> int:
        """Clear all cached entries."""
        return self._backend.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": self._backend.size(),
            "enabled": self._enabled,
        }


# Global cache instance
_global_cache: PromptCache | None = None
# Prompt.run() only consults the cache after configure_cache() was called,
# so that merely inspecting get_cache() never changes model outputs.
_cache_configured: bool = False


def get_cache() -> PromptCache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PromptCache()
    return _global_cache


def get_active_cache() -> PromptCache | None:
    """Return the cache ``Prompt.run()`` should use, or None.

    The global cache is used once :func:`configure_cache` has been called
    with ``enabled=True``.
    """
    if not _cache_configured or _global_cache is None:
        return None
    return _global_cache if _global_cache.enabled else None


def configure_cache(
    backend: CacheBackend | None = None,
    default_ttl: float | None = 3600,
    enabled: bool = True,
) -> PromptCache:
    """Configure the global cache used by ``Prompt.run()`` and ``arun()``.

    After this call, identical requests (same messages, model, temperature,
    output schema and generation parameters) are answered from the cache
    without an LLM call. Streaming calls are not cached.

    Note:
        A cache returns the same answer for repeated identical requests. If
        you use ``compare(..., runs_per_input>1)`` to measure run-to-run
        variation at ``temperature > 0``, disable the cache for that run.
    """
    global _global_cache, _cache_configured
    _cache_configured = True
    _global_cache = PromptCache(
        backend=backend,
        default_ttl=default_ttl,
        enabled=enabled,
    )
    return _global_cache
