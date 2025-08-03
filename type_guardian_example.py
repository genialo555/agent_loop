#!/usr/bin/env python3
"""
Type Guardian Example: Strict typing with Pydantic v2
Demonstrates modern Python type annotations and data validation.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Protocol, TypeVar, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PaymentStatus(str, Enum):
    """Payment status enumeration with type safety."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes."""
    
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


class PaymentProcessor(Protocol):
    """Structural subtyping for payment processors (PY001)."""
    
    def process_payment(
        self,
        amount: Decimal,
        currency: CurrencyCode,
        customer_id: str,
    ) -> PaymentResult:
        """Process a payment transaction."""
        ...
    
    def refund_payment(
        self,
        transaction_id: str,
        amount: Decimal | None = None,
    ) -> PaymentResult:
        """Refund a payment transaction."""
        ...


@final
class PaymentResult(BaseModel):
    """Immutable payment result with validation (PY002, PY003)."""
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )
    
    transaction_id: Annotated[str, Field(min_length=10, max_length=50)]
    status: PaymentStatus
    amount: Annotated[Decimal, Field(gt=0, decimal_places=2)]
    currency: CurrencyCode
    timestamp: datetime
    message: str | None = None
    
    @field_validator("transaction_id")
    @classmethod
    def validate_transaction_id(cls, value: str) -> str:
        """Ensure transaction ID follows pattern TXN-YYYYMMDD-XXXXXX."""
        pattern = re.compile(r"^TXN-\d{8}-[A-Z0-9]{6}$")
        if not pattern.match(value):
            raise ValueError(
                f"Invalid transaction ID format: {value}. "
                "Expected: TXN-YYYYMMDD-XXXXXX"
            )
        return value
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return value


class Customer(BaseModel):
    """Customer model with comprehensive validation (PY002)."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )
    
    __slots__ = ("_loyalty_points",)  # Memory optimization (PY005)
    
    id: Annotated[str, Field(min_length=5, max_length=20)]
    email: Annotated[str, Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")]
    full_name: Annotated[str, Field(min_length=2, max_length=100)]
    phone: Annotated[str, Field(pattern=r"^\+?1?\d{9,15}$")]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._loyalty_points: int = 0  # Private attribute initialization
    
    @property
    def loyalty_points(self) -> int:
        """Get customer loyalty points."""
        return self._loyalty_points
    
    def add_loyalty_points(self, points: int) -> None:
        """Add loyalty points with validation."""
        if points < 0:
            raise ValueError("Cannot add negative loyalty points")
        self._loyalty_points += points


T = TypeVar("T", bound=BaseModel)


def process_transaction(
    customer: Customer,
    amount: Decimal,
    currency: CurrencyCode,
    processor: PaymentProcessor,
    metadata: dict[str, str] | None = None,
) -> tuple[PaymentResult, Path]:
    """
    Process a payment transaction with strict type safety (PY001).
    
    Args:
        customer: Validated customer model
        amount: Transaction amount (must be positive)
        currency: ISO 4217 currency code
        processor: Payment processor implementation
        metadata: Optional transaction metadata
    
    Returns:
        Tuple of payment result and transaction log path
    
    Raises:
        ValueError: If customer is inactive or amount is invalid
        RuntimeError: If payment processing fails
    """
    # Pattern matching for validation (PY006)
    match customer.is_active:
        case False:
            raise ValueError(f"Customer {customer.id} is inactive")
        case True:
            pass
    
    # Walrus operator for efficient validation (PY007)
    if (rounded_amount := amount.quantize(Decimal("0.01"))) != amount:
        raise ValueError(
            f"Amount must have at most 2 decimal places, got: {amount}"
        )
    
    # Process payment
    result = processor.process_payment(
        amount=rounded_amount,
        currency=currency,
        customer_id=customer.id,
    )
    
    # Pattern matching on result status (PY006)
    match result.status:
        case PaymentStatus.COMPLETED:
            customer.add_loyalty_points(int(amount * 10))
            status_emoji = "✅"
        case PaymentStatus.FAILED:
            status_emoji = "❌"
        case PaymentStatus.PENDING | PaymentStatus.PROCESSING:
            status_emoji = "⏳"
        case _:
            status_emoji = "❓"
    
    # Path operations with pathlib (PY004)
    log_dir = Path("/home/jerem/agent_loop/transaction_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create transaction log file
    timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{result.transaction_id}_{timestamp_str}.json"
    
    # Write transaction data
    transaction_data = {
        "result": result.model_dump(mode="json"),
        "customer_id": customer.id,
        "metadata": metadata or {},
    }
    
    import json
    log_file.write_text(
        json.dumps(transaction_data, indent=2, default=str),
        encoding="utf-8",
    )
    
    return result, log_file


def validate_bulk_transactions(
    transactions: list[dict[str, any]],
) -> list[PaymentResult]:
    """
    Validate and parse bulk transaction data (PY001, PY002).
    
    Args:
        transactions: Raw transaction data to validate
    
    Returns:
        List of validated PaymentResult models
    
    Raises:
        ValueError: If any transaction fails validation
    """
    validated_results: list[PaymentResult] = []
    
    for idx, transaction in enumerate(transactions):
        try:
            # Walrus operator in error handling (PY007)
            if not (tx_id := transaction.get("transaction_id")):
                raise ValueError(f"Transaction {idx}: Missing transaction_id")
            
            result = PaymentResult(**transaction)
            validated_results.append(result)
            
        except Exception as exc:
            raise ValueError(
                f"Transaction {idx} validation failed: {exc}"
            ) from exc
    
    return validated_results


# Example usage demonstrating the type-safe implementation
if __name__ == "__main__":
    # Create a sample customer
    customer = Customer(
        id="CUST-12345",
        email="john.doe@example.com",
        full_name="John Doe",
        phone="+1234567890",
    )
    
    # Mock payment processor implementation
    class MockProcessor:
        def process_payment(
            self,
            amount: Decimal,
            currency: CurrencyCode,
            customer_id: str,
        ) -> PaymentResult:
            return PaymentResult(
                transaction_id="TXN-20250128-ABC123",
                status=PaymentStatus.COMPLETED,
                amount=amount,
                currency=currency,
                timestamp=datetime.now(timezone.utc),
                message="Payment processed successfully",
            )
        
        def refund_payment(
            self,
            transaction_id: str,
            amount: Decimal | None = None,
        ) -> PaymentResult:
            return PaymentResult(
                transaction_id=f"REF-{transaction_id[4:]}",
                status=PaymentStatus.REFUNDED,
                amount=amount or Decimal("0.00"),
                currency=CurrencyCode.USD,
                timestamp=datetime.now(timezone.utc),
                message="Refund processed",
            )
    
    # Process a transaction
    processor: PaymentProcessor = MockProcessor()
    result, log_path = process_transaction(
        customer=customer,
        amount=Decimal("99.99"),
        currency=CurrencyCode.USD,
        processor=processor,
        metadata={"order_id": "ORD-2025-001"},
    )
    
    print(f"Transaction completed: {result.transaction_id}")
    print(f"Log saved to: {log_path}")