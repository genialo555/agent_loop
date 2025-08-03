#!/usr/bin/env python3
"""
Model Registry Management System
================================

Production-grade model versioning, metadata management, and registry operations
for MLOps fine-tuning pipeline. Provides centralized model lifecycle management
with automatic artifact tracking and deployment coordination.
"""

import json
import hashlib
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status enumeration."""
    TRAINING = "training"
    COMPLETED = "completed"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata structure."""
    model_version: str
    training_config: str
    environment: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    model_path: Path
    checksum: str
    file_size: int
    training_duration: Optional[int] = None
    final_loss: Optional[float] = None
    accuracy: Optional[float] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    parent_model: Optional[str] = None
    tags: List[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if isinstance(self.status, str):
            self.status = ModelStatus(self.status)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)


class ModelRegistry:
    """Production model registry with SQLite backend."""
    
    def __init__(self, registry_path: Path = None, models_root: Path = None):
        """Initialize model registry.
        
        Args:
            registry_path: Path to SQLite registry database
            models_root: Root directory for model storage
        """
        project_root = Path(__file__).parent.parent
        self.registry_path = registry_path or project_root / "models" / "registry.db"
        self.models_root = models_root or project_root / "models" / "finetuned"
        
        # Ensure directories exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"Model registry initialized: {self.registry_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_version TEXT PRIMARY KEY,
                    training_config TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    training_duration INTEGER,
                    final_loss REAL,
                    accuracy REAL,
                    git_commit TEXT,
                    git_branch TEXT,
                    parent_model TEXT,
                    tags TEXT,  -- JSON array
                    notes TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON models(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_config ON models(training_config)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON models(created_at)")
            
            conn.commit()
    
    def _calculate_checksum(self, model_path: Path) -> Tuple[str, int]:
        """Calculate model directory checksum and size.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Tuple of (checksum, total_size)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        hasher = hashlib.sha256()
        total_size = 0
        
        if model_path.is_file():
            # Single file
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
                    total_size += len(chunk)
        else:
            # Directory - hash all files recursively
            for file_path in sorted(model_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                            total_size += len(chunk)
        
        return hasher.hexdigest(), total_size
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model in the registry.
        
        Args:
            metadata: Model metadata object
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Calculate checksum and size
            if not metadata.checksum or not metadata.file_size:
                checksum, file_size = self._calculate_checksum(metadata.model_path)
                metadata.checksum = checksum
                metadata.file_size = file_size
            
            # Prepare data for database
            tags_json = json.dumps(metadata.tags) if metadata.tags else "[]"
            
            with sqlite3.connect(self.registry_path) as conn:
                conn.execute("""
                    INSERT INTO models (
                        model_version, training_config, environment, status,
                        created_at, updated_at, model_path, checksum, file_size,
                        training_duration, final_loss, accuracy, git_commit,
                        git_branch, parent_model, tags, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_version,
                    metadata.training_config,
                    metadata.environment,
                    metadata.status.value,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    str(metadata.model_path),
                    metadata.checksum,
                    metadata.file_size,
                    metadata.training_duration,
                    metadata.final_loss,
                    metadata.accuracy,
                    metadata.git_commit,
                    metadata.git_branch,
                    metadata.parent_model,
                    tags_json,
                    metadata.notes
                ))
                conn.commit()
            
            logger.info(f"Model registered: {metadata.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.model_version}: {e}")
            return False
    
    def update_model_status(self, model_version: str, status: ModelStatus, 
                           metadata_updates: Dict[str, Any] = None) -> bool:
        """Update model status and optional metadata.
        
        Args:
            model_version: Model version identifier
            status: New model status
            metadata_updates: Optional metadata fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            updates = {"status": status.value, "updated_at": datetime.now(timezone.utc).isoformat()}
            
            if metadata_updates:
                updates.update(metadata_updates)
            
            # Build dynamic query
            set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
            values = list(updates.values()) + [model_version]
            
            with sqlite3.connect(self.registry_path) as conn:
                cursor = conn.execute(
                    f"UPDATE models SET {set_clause} WHERE model_version = ?",
                    values
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"Model not found for update: {model_version}")
                    return False
                
                conn.commit()
            
            logger.info(f"Model updated: {model_version} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model {model_version}: {e}")
            return False
    
    def get_model(self, model_version: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by version.
        
        Args:
            model_version: Model version identifier
            
        Returns:
            ModelMetadata object or None if not found
        """
        try:
            with sqlite3.connect(self.registry_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM models WHERE model_version = ?",
                    (model_version,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Convert row to ModelMetadata
                data = dict(row)
                data['tags'] = json.loads(data['tags']) if data['tags'] else []
                data['status'] = ModelStatus(data['status'])
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                data['model_path'] = Path(data['model_path'])
                
                return ModelMetadata(**data)
                
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_version}: {e}")
            return None
    
    def list_models(self, status: Optional[ModelStatus] = None,
                   training_config: Optional[str] = None,
                   environment: Optional[str] = None,
                   limit: int = 100) -> List[ModelMetadata]:
        """List models with optional filtering.
        
        Args:
            status: Filter by model status
            training_config: Filter by training configuration
            environment: Filter by environment
            limit: Maximum number of results
            
        Returns:
            List of ModelMetadata objects
        """
        try:
            query = "SELECT * FROM models WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if training_config:
                query += " AND training_config = ?"
                params.append(training_config)
            
            if environment:
                query += " AND environment = ?"
                params.append(environment)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.registry_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                models = []
                for row in rows:
                    data = dict(row)
                    data['tags'] = json.loads(data['tags']) if data['tags'] else []
                    data['status'] = ModelStatus(data['status'])
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    data['model_path'] = Path(data['model_path'])
                    
                    models.append(ModelMetadata(**data))
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_latest_model(self, training_config: str, 
                        status: ModelStatus = ModelStatus.COMPLETED) -> Optional[ModelMetadata]:
        """Get the latest model for a training configuration.
        
        Args:
            training_config: Training configuration name
            status: Model status filter
            
        Returns:
            Latest ModelMetadata object or None
        """
        models = self.list_models(
            status=status,
            training_config=training_config,
            limit=1
        )
        return models[0] if models else None
    
    def archive_model(self, model_version: str, archive_path: Path = None) -> bool:
        """Archive a model by moving it to archive location.
        
        Args:
            model_version: Model version to archive
            archive_path: Optional custom archive path
            
        Returns:
            True if archival successful, False otherwise
        """
        try:
            model = self.get_model(model_version)
            if not model:
                logger.error(f"Model not found for archival: {model_version}")
                return False
            
            if not archive_path:
                archive_root = self.models_root.parent / "archived"
                archive_root.mkdir(exist_ok=True)
                archive_path = archive_root / model_version
            
            # Move model files
            if model.model_path.exists():
                shutil.move(str(model.model_path), str(archive_path))
                logger.info(f"Model files moved to: {archive_path}")
            
            # Update registry
            self.update_model_status(
                model_version,
                ModelStatus.ARCHIVED,
                {"model_path": str(archive_path)}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive model {model_version}: {e}")
            return False
    
    def delete_model(self, model_version: str, force: bool = False) -> bool:
        """Delete a model from registry and filesystem.
        
        Args:
            model_version: Model version to delete
            force: Force deletion without confirmation
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            model = self.get_model(model_version)
            if not model:
                logger.error(f"Model not found for deletion: {model_version}")
                return False
            
            # Safety check - don't delete deployed models without force
            if model.status == ModelStatus.DEPLOYED and not force:
                logger.error(f"Cannot delete deployed model without force flag: {model_version}")
                return False
            
            # Remove from filesystem
            if model.model_path.exists():
                if model.model_path.is_dir():
                    shutil.rmtree(model.model_path)
                else:
                    model.model_path.unlink()
                logger.info(f"Model files deleted: {model.model_path}")
            
            # Remove from registry
            with sqlite3.connect(self.registry_path) as conn:
                conn.execute("DELETE FROM models WHERE model_version = ?", (model_version,))
                conn.commit()
            
            logger.info(f"Model deleted from registry: {model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_version}: {e}")
            return False
    
    def export_metadata(self, output_path: Path) -> bool:
        """Export all model metadata to JSON file.
        
        Args:
            output_path: Path for exported metadata file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            models = self.list_models(limit=1000)  # Export all models
            
            export_data = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_models": len(models),
                "models": []
            }
            
            for model in models:
                model_dict = asdict(model)
                # Convert non-JSON serializable types
                model_dict['status'] = model.status.value
                model_dict['created_at'] = model.created_at.isoformat()
                model_dict['updated_at'] = model.updated_at.isoformat()
                model_dict['model_path'] = str(model.model_path)
                
                export_data["models"].append(model_dict)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metadata exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        try:
            with sqlite3.connect(self.registry_path) as conn:
                stats = {}
                
                # Total models
                cursor = conn.execute("SELECT COUNT(*) FROM models")
                stats['total_models'] = cursor.fetchone()[0]
                
                # Models by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) 
                    FROM models 
                    GROUP BY status
                """)
                stats['by_status'] = dict(cursor.fetchall())
                
                # Models by config
                cursor = conn.execute("""
                    SELECT training_config, COUNT(*) 
                    FROM models 
                    GROUP BY training_config
                """)
                stats['by_config'] = dict(cursor.fetchall())
                
                # Storage usage
                cursor = conn.execute("SELECT SUM(file_size) FROM models")
                total_size = cursor.fetchone()[0] or 0
                stats['total_storage_bytes'] = total_size
                stats['total_storage_gb'] = round(total_size / (1024**3), 2)
                
                # Recent models (last 7 days)
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM models 
                    WHERE created_at > datetime('now', '-7 days')
                """)
                stats['recent_models_7d'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            return {}


def main():
    """CLI interface for model registry operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry Management")
    parser.add_argument("--action", required=True,
                       choices=["list", "get", "update", "archive", "delete", "stats", "export"],
                       help="Registry action to perform")
    parser.add_argument("--model-version", help="Model version identifier")
    parser.add_argument("--status", choices=[s.value for s in ModelStatus],
                       help="Model status filter or update value")
    parser.add_argument("--config", help="Training configuration filter")
    parser.add_argument("--environment", help="Environment filter")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--force", action="store_true", help="Force operation")
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = ModelRegistry()
    
    if args.action == "list":
        status = ModelStatus(args.status) if args.status else None
        models = registry.list_models(
            status=status,
            training_config=args.config,
            environment=args.environment
        )
        
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  {model.model_version} [{model.status.value}] "
                  f"({model.training_config}, {model.environment})")
    
    elif args.action == "get":
        if not args.model_version:
            print("--model-version required for get action")
            return
        
        model = registry.get_model(args.model_version)
        if model:
            print(json.dumps(asdict(model), indent=2, default=str))
        else:
            print(f"Model not found: {args.model_version}")
    
    elif args.action == "update":
        if not args.model_version or not args.status:
            print("--model-version and --status required for update action")
            return
        
        success = registry.update_model_status(
            args.model_version,
            ModelStatus(args.status)
        )
        print("Update successful" if success else "Update failed")
    
    elif args.action == "archive":
        if not args.model_version:
            print("--model-version required for archive action")
            return
        
        success = registry.archive_model(args.model_version)
        print("Archive successful" if success else "Archive failed")
    
    elif args.action == "delete":
        if not args.model_version:
            print("--model-version required for delete action")
            return
        
        success = registry.delete_model(args.model_version, force=args.force)
        print("Delete successful" if success else "Delete failed")
    
    elif args.action == "stats":
        stats = registry.get_registry_stats()
        print(json.dumps(stats, indent=2))
    
    elif args.action == "export":
        if not args.output:
            print("--output required for export action")
            return
        
        success = registry.export_metadata(args.output)
        print("Export successful" if success else "Export failed")


if __name__ == "__main__":
    main()