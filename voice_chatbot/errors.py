"""Project-specific exception types."""


class AudioDependencyError(RuntimeError):
    """Raised when the local audio backend is unavailable."""
