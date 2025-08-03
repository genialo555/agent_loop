#!/usr/bin/env python3
"""Example of properly typed Python code following modern best practices."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    NoReturn,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    overload,
)

import attrs
from pydantic import BaseModel, Field, field_validator


# Type aliases for clarity
UserId: TypeAlias = int
Score: TypeAlias = float
JsonDict: TypeAlias = dict[str, Any]

# Constants with type annotations
MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT: Final[float] = 30.0


class Status(Enum):
    """Status enumeration with auto values."""
    
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()


class UserDict(TypedDict):
    """TypedDict for user data validation."""
    
    id: UserId
    name: str
    email: str
    created_at: datetime
    metadata: NotRequired[JsonDict]


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable configuration with slots for memory efficiency."""
    
    host: str
    port: int
    debug: bool = False
    timeout: float = DEFAULT_TIMEOUT
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 < self.port <= 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, field: str, value: Any, message: str) -> None:
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error for {field}: {message}")


class Comparable(Protocol):
    """Protocol for comparable objects."""
    
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...


T = TypeVar("T")
ComparableT = TypeVar("ComparableT", bound=Comparable)


@attrs.define
class Result(Generic[T]):
    """Generic result container using attrs."""
    
    value: T
    timestamp: datetime = attrs.field(factory=lambda: datetime.now(timezone.utc))
    metadata: JsonDict = attrs.field(factory=dict)
    
    @property
    def age(self) -> float:
        """Calculate age of result in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


class User(BaseModel):
    """Pydantic model with validation."""
    
    id: UserId
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    score: Score = Field(default=0.0, ge=0.0, le=100.0)
    tags: set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate and normalize name."""
        return value.strip().title()
    
    class Config:
        """Pydantic configuration."""
        
        frozen = True
        validate_assignment = True


class DataProcessor:
    """Example class with type-safe methods."""
    
    __slots__ = ("_data", "_cache", "_logger")
    
    # Class variable with type annotation
    version: ClassVar[str] = "1.0.0"
    
    def __init__(self, data: list[User]) -> None:
        self._data: list[User] = data
        self._cache: dict[UserId, User] = {}
        self._logger: Callable[[str], None] = print
    
    def process(self) -> Iterator[Result[User]]:
        """Process users and yield results."""
        for user in self._data:
            self._cache[user.id] = user
            yield Result(value=user)
    
    @overload
    def get_user(self, user_id: UserId, *, default: None = None) -> User | None: ...
    
    @overload
    def get_user(self, user_id: UserId, *, default: User) -> User: ...
    
    def get_user(self, user_id: UserId, *, default: User | None = None) -> User | None:
        """Get user by ID with optional default."""
        return self._cache.get(user_id, default)
    
    def find_users(self, *, min_score: Score = 0.0) -> list[User]:
        """Find users with score above threshold."""
        return [user for user in self._data if user.score >= min_score]
    
    async def process_async(self) -> AsyncIterator[Result[User]]:
        """Async version of process method."""
        for user in self._data:
            await asyncio.sleep(0.1)  # Simulate async work
            self._cache[user.id] = user
            yield Result(value=user)


def validate_path(path: Path | str) -> Path:
    """Validate and return Path object."""
    path = Path(path) if isinstance(path, str) else path
    
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return path


def sort_items(items: Sequence[ComparableT]) -> list[ComparableT]:
    """Sort comparable items with type preservation."""
    return sorted(items)


def fail_with_error(message: str) -> NoReturn:
    """Function that never returns."""
    raise RuntimeError(message)


@asynccontextmanager
async def managed_resource(name: str) -> AsyncIterator[str]:
    """Async context manager example."""
    print(f"Acquiring resource: {name}")
    try:
        yield f"Resource_{name}"
    finally:
        print(f"Releasing resource: {name}")


async def main() -> None:
    """Main entry point with async operations."""
    # Create sample data
    users = [
        User(id=1, name="alice smith", email="alice@example.com", score=85.5),
        User(id=2, name="bob jones", email="bob@example.com", score=92.0),
        User(id=3, name="charlie brown", email="charlie@example.com", score=78.3),
    ]
    
    # Process data
    processor = DataProcessor(users)
    
    # Sync processing
    print("Sync processing:")
    for result in processor.process():
        print(f"  Processed: {result.value.name} (age: {result.age:.2f}s)")
    
    # Async processing
    print("\nAsync processing:")
    async for result in processor.process_async():
        print(f"  Async processed: {result.value.name}")
    
    # Find high scorers
    high_scorers = processor.find_users(min_score=80.0)
    print(f"\nHigh scorers: {[u.name for u in high_scorers]}")
    
    # Use async context manager
    async with managed_resource("database") as resource:
        print(f"Using {resource}")
    
    # Pattern matching example (Python 3.10+)
    status = Status.COMPLETED
    match status:
        case Status.PENDING | Status.PROCESSING:
            print("Still working...")
        case Status.COMPLETED:
            print("All done!")
        case Status.FAILED:
            fail_with_error("Process failed")
        case _:
            print("Unknown status")


if __name__ == "__main__":
    asyncio.run(main())