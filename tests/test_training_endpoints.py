"""
Tests for training endpoints - Sprint 2 validation.

These tests validate the training API endpoints for async fine-tuning jobs:
- POST /training/train - Submit training job
- GET /training/train/{job_id}/status - Get job status  
- GET /training/jobs - List training jobs
- DELETE /training/train/{job_id} - Cancel training job
- GET /training/train/{job_id}/logs - Get training logs
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from agent_loop.models.inference.models.schemas import (
    TrainingRequest, 
    TrainingResponse, 
    TrainingStatusResponse,
    TrainingJobsListResponse
)
from agent_loop.models.inference.routers.training import router as training_router
from agent_loop.models.inference.services.training import TrainingService, JobStatus


@pytest.fixture
def training_app():
    """Create FastAPI app with training router for testing."""
    app = FastAPI()
    app.include_router(training_router)
    return app


@pytest.fixture
def mock_training_service():
    """Mock training service for testing."""
    service = Mock(spec=TrainingService)
    
    # Mock successful job submission
    service.submit_training_job = AsyncMock(return_value=TrainingResponse(
        job_id="test_job_123",
        status="queued",
        message="Training job submitted successfully", 
        estimated_duration_minutes=45,
        experiment_name="test_experiment",
        created_at=datetime.now(timezone.utc),
        websocket_url="ws://localhost:8000/ws/training/test_job_123"
    ))
    
    # Mock job status
    service.get_job_status = AsyncMock(return_value=TrainingStatusResponse(
        job_id="test_job_123",
        status="running",
        progress={
            "current_epoch": 2,
            "total_epochs": 3,
            "current_step": 150,
            "total_steps": 300,
            "progress_percentage": 66.7
        },
        metrics={
            "train_loss": 0.65,
            "learning_rate": 1.8e-4,
            "tokens_per_second": 245.3,
            "gpu_utilization": 85.2
        },
        timing={
            "started_at": "2025-01-28T10:30:00Z",
            "elapsed_seconds": 1800,
            "estimated_remaining_seconds": 900
        },
        logs_url="/training/test_job_123/logs",
        checkpoint_path="/checkpoints/test_job_123/epoch_2"
    ))
    
    # Mock job listing
    service.list_jobs = AsyncMock(return_value=TrainingJobsListResponse(
        jobs=[{
            "job_id": "test_job_123",
            "status": "running",
            "experiment_name": "test_experiment",
            "created_at": "2025-01-28T10:30:00Z",
            "base_model": "gemma:3n-e2b"
        }],
        total=1,
        page=1,
        per_page=10
    ))
    
    # Mock job cancellation
    service.cancel_job = AsyncMock(return_value=True)
    
    return service


class TestTrainingJobSubmission:
    """Test training job submission endpoint."""
    
    @pytest.mark.asyncio
    async def test_submit_training_job_success(self, training_app, mock_training_service):
        """Test successful training job submission."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                request_data = {
                    "base_model": "gemma:3n-e2b",
                    "dataset_path": "/data/test_dataset.jsonl",
                    "training_config": {
                        "epochs": 3,
                        "batch_size": 4,
                        "learning_rate": 2e-4
                    },
                    "experiment_name": "test_experiment"
                }
                
                response = await client.post("/training/train", json=request_data)
                
                assert response.status_code == 202
                data = response.json()
                assert data["job_id"] == "test_job_123"
                assert data["status"] == "queued"
                assert data["message"] == "Training job submitted successfully"
                assert data["estimated_duration_minutes"] == 45
                assert data["experiment_name"] == "test_experiment"
                assert "websocket_url" in data
                
                # Verify service was called
                mock_training_service.submit_training_job.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_training_job_validation_error(self, training_app, mock_training_service):
        """Test training job submission with validation errors."""
        mock_training_service.submit_training_job.side_effect = ValueError("Invalid dataset path")
        
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                request_data = {
                    "base_model": "gemma:3n-e2b",
                    "dataset_path": "/invalid/path.jsonl",
                    "training_config": {"epochs": 3}
                }
                
                response = await client.post("/training/train", json=request_data)
                
                assert response.status_code == 400
                data = response.json()
                assert "Invalid training request" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_submit_training_job_missing_fields(self, training_app):
        """Test training job submission with missing required fields."""
        async with AsyncClient(app=training_app, base_url="http://test") as client:
            request_data = {
                "base_model": "gemma:3n-e2b"
                # Missing dataset_path
            }
            
            response = await client.post("/training/train", json=request_data)
            
            assert response.status_code == 422  # Validation error


class TestTrainingJobStatus:
    """Test training job status endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_training_status_success(self, training_app, mock_training_service):
        """Test successful training status retrieval."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/train/test_job_123/status")
                
                assert response.status_code == 200
                data = response.json()
                assert data["job_id"] == "test_job_123"
                assert data["status"] == "running"
                assert "progress" in data
                assert "metrics" in data
                assert "timing" in data
                assert data["progress"]["progress_percentage"] == 66.7
                
                mock_training_service.get_job_status.assert_called_once_with("test_job_123")
    
    @pytest.mark.asyncio
    async def test_get_training_status_not_found(self, training_app, mock_training_service):
        """Test training status for non-existent job."""
        mock_training_service.get_job_status.return_value = None
        
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/train/nonexistent_job/status")
                
                assert response.status_code == 404
                data = response.json()
                assert "not found" in data["detail"]


class TestTrainingJobsList:
    """Test training jobs listing endpoint."""
    
    @pytest.mark.asyncio
    async def test_list_training_jobs_success(self, training_app, mock_training_service):
        """Test successful training jobs listing."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/jobs")
                
                assert response.status_code == 200
                data = response.json()
                assert "jobs" in data
                assert "total" in data
                assert "page" in data
                assert "per_page" in data
                assert len(data["jobs"]) == 1
                assert data["jobs"][0]["job_id"] == "test_job_123"
                assert data["total"] == 1
                
                mock_training_service.list_jobs.assert_called_once_with(
                    page=1, per_page=10, status_filter=None
                )
    
    @pytest.mark.asyncio
    async def test_list_training_jobs_with_pagination(self, training_app, mock_training_service):
        """Test training jobs listing with pagination parameters."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/jobs?page=2&per_page=5&status=running")
                
                assert response.status_code == 200
                mock_training_service.list_jobs.assert_called_once_with(
                    page=2, per_page=5, status_filter="running"
                )


class TestTrainingJobCancellation:
    """Test training job cancellation endpoint."""
    
    @pytest.mark.asyncio
    async def test_cancel_training_job_success(self, training_app, mock_training_service):
        """Test successful training job cancellation."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.delete("/training/train/test_job_123")
                
                assert response.status_code == 200
                data = response.json()
                assert "cancelled successfully" in data["message"]
                assert data["job_id"] == "test_job_123"
                assert data["status"] == "cancelled"
                
                mock_training_service.cancel_job.assert_called_once_with("test_job_123")
    
    @pytest.mark.asyncio
    async def test_cancel_training_job_not_found(self, training_app, mock_training_service):
        """Test cancellation of non-existent job."""
        mock_training_service.cancel_job.return_value = False
        mock_training_service.get_job_status.return_value = None
        
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.delete("/training/train/nonexistent_job")
                
                assert response.status_code == 404


class TestTrainingJobLogs:
    """Test training job logs endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_training_logs_success(self, training_app, mock_training_service):
        """Test successful training logs retrieval."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/train/test_job_123/logs")
                
                assert response.status_code == 200
                data = response.json()
                assert data["job_id"] == "test_job_123"
                assert "logs" in data
                assert "total_entries" in data
                assert "limit" in data
                assert "note" in data  # WebSocket note
    
    @pytest.mark.asyncio  
    async def test_get_training_logs_with_limit(self, training_app, mock_training_service):
        """Test training logs with custom limit."""
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/train/test_job_123/logs?limit=50")
                
                assert response.status_code == 200
                data = response.json()
                assert data["limit"] == 50
    
    @pytest.mark.asyncio
    async def test_get_training_logs_job_not_found(self, training_app, mock_training_service):
        """Test logs retrieval for non-existent job."""
        mock_training_service.get_job_status.return_value = None
        
        with patch('inference.routers.training.get_training_service', return_value=mock_training_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                response = await client.get("/training/train/nonexistent_job/logs")
                
                assert response.status_code == 404


class TestTrainingServiceIntegration:
    """Integration tests for training service functionality."""
    
    @pytest.mark.asyncio
    async def test_training_workflow_end_to_end(self, training_app):
        """Test complete training workflow: submit -> status -> cancel."""
        # This would be an actual integration test with real TrainingService
        # For now, we'll simulate the workflow
        
        mock_service = Mock(spec=TrainingService)
        job_id = "integration_test_job"
        
        # 1. Submit job
        mock_service.submit_training_job = AsyncMock(return_value=TrainingResponse(
            job_id=job_id,
            status="queued",
            message="Job submitted",
            created_at=datetime.now(timezone.utc)
        ))
        
        # 2. Check status progression: queued -> running -> completed
        status_progression = [
            TrainingStatusResponse(job_id=job_id, status="queued"),
            TrainingStatusResponse(job_id=job_id, status="running", 
                                 progress={"progress_percentage": 50.0}),
            TrainingStatusResponse(job_id=job_id, status="completed",
                                 progress={"progress_percentage": 100.0})
        ]
        
        mock_service.get_job_status = AsyncMock(side_effect=status_progression)
        mock_service.cancel_job = AsyncMock(return_value=True)
        
        with patch('inference.routers.training.get_training_service', return_value=mock_service):
            async with AsyncClient(app=training_app, base_url="http://test") as client:
                # Submit job
                submit_response = await client.post("/training/train", json={
                    "base_model": "gemma:3n-e2b",
                    "dataset_path": "/data/test.jsonl",
                    "training_config": {"epochs": 1}
                })
                assert submit_response.status_code == 202
                
                # Check status multiple times
                for expected_status in ["queued", "running", "completed"]:
                    status_response = await client.get(f"/training/train/{job_id}/status")
                    assert status_response.status_code == 200
                    assert status_response.json()["status"] == expected_status


class TestTrainingServiceDependency:
    """Test training service dependency injection."""
    
    def test_training_service_creation(self):
        """Test that training service can be created with proper dependencies."""
        # This would test the actual get_training_service dependency
        from inference.services.dependencies import get_training_service
        
        # For now, we'll just check that the function exists
        # In a full implementation, this would test actual service instantiation
        assert callable(get_training_service)


# Additional utility tests

class TestTrainingRequestValidation:
    """Test training request validation logic."""
    
    def test_training_request_valid(self):
        """Test valid training request."""
        request = TrainingRequest(
            base_model="gemma:3n-e2b",
            dataset_path="/data/test.jsonl",
            training_config={
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-4
            },
            experiment_name="test_exp"
        )
        
        assert request.base_model == "gemma:3n-e2b"
        assert request.dataset_path == "/data/test.jsonl"
        assert request.training_config["epochs"] == 3
        assert request.experiment_name == "test_exp"
    
    def test_training_request_defaults(self):
        """Test training request with default values."""
        request = TrainingRequest(
            base_model="gemma:3n-e2b",
            dataset_path="/data/test.jsonl"
        )
        
        assert request.training_config["epochs"] == 3
        assert request.training_config["batch_size"] == 4
        assert request.training_config["learning_rate"] == 2e-4
        assert request.experiment_name is None
        assert request.resume_from_checkpoint is None


@pytest.mark.integration
class TestDockerTrainingIntegration:
    """Integration tests for Docker training setup."""
    
    @pytest.mark.asyncio
    async def test_docker_compose_training_config(self):
        """Test that docker-compose.training.yml is properly configured."""
        # This would test actual Docker integration
        # For Sprint 2, we'll simulate the test structure
        
        # In a full implementation, this would:
        # 1. Start docker-compose.training.yml services
        # 2. Wait for services to be healthy
        # 3. Submit a test training job
        # 4. Monitor job progression
        # 5. Verify output artifacts
        # 6. Clean up containers
        
        pass  # Placeholder for actual Docker integration test


if __name__ == "__main__":
    # Run specific test classes for development
    pytest.main([__file__, "-v", "--tb=short"])