import pytest
from fastapi import HTTPException
from ml_portfolio_churn.service import PredictRequest


def test_predict_request_accepts_valid_records():
    """Test that valid records are accepted."""
    req = PredictRequest(records=[{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    req.validate_records()  # Should not raise


def test_predict_request_rejects_empty_records():
    """Test that empty records list raises error."""
    req = PredictRequest(records=[])

    with pytest.raises(HTTPException) as exc_info:
        req.validate_records()

    assert exc_info.value.status_code == 400
    assert "Empty records list" in exc_info.value.detail


def test_predict_request_rejects_too_many_records():
    """Test that exceeding max_records raises error."""
    # Create 11 records but set max to 10
    records = [{"a": i} for i in range(11)]
    req = PredictRequest(records=records)

    with pytest.raises(HTTPException) as exc_info:
        req.validate_records(max_records=10)

    assert exc_info.value.status_code == 400
    assert "Too many records" in exc_info.value.detail
    assert "11" in exc_info.value.detail
    assert "10" in exc_info.value.detail


def test_predict_request_accepts_records_at_limit():
    """Test that exactly max_records is accepted."""
    records = [{"a": i} for i in range(10)]
    req = PredictRequest(records=records)

    req.validate_records(max_records=10)  # Should not raise


def test_predict_request_num_records_property():
    """Test the num_records property."""
    req = PredictRequest(records=[{"a": 1}, {"a": 2}, {"a": 3}])

    assert req.num_records == 3


def test_predict_request_num_records_empty():
    """Test num_records for empty list."""
    req = PredictRequest(records=[])

    assert req.num_records == 0


def test_predict_request_default_max_records():
    """Test that default max_records (1000) works."""
    # Create exactly 1000 records
    records = [{"a": i} for i in range(1000)]
    req = PredictRequest(records=records)

    req.validate_records()  # Should not raise with default max


def test_predict_request_exceeds_default_max():
    """Test that exceeding default max (1000) raises error."""
    # Create 1001 records
    records = [{"a": i} for i in range(1001)]
    req = PredictRequest(records=records)

    with pytest.raises(HTTPException) as exc_info:
        req.validate_records()

    assert exc_info.value.status_code == 400
    assert "1001" in exc_info.value.detail
