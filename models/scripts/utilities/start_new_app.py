#!/usr/bin/env python3
"""
Startup script for the new modular FastAPI architecture.

This script provides easy ways to start the application with different configurations.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def start_application(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    environment: str = "production",
    workers: int = 1
):
    """Start the FastAPI application with specified configuration."""
    
    # Set environment variables
    os.environ["ENVIRONMENT"] = environment
    
    if environment == "development":
        os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000"
    else:
        # In production, set your actual allowed origins
        os.environ["ALLOWED_ORIGINS"] = os.getenv("ALLOWED_ORIGINS", "https://yourdomain.com")
    
    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "inference.app:app",  # New modular app
        "--host", host,
        "--port", str(port),
    ]
    
    if reload and environment == "development":
        cmd.append("--reload")
    
    if workers > 1 and not reload:
        cmd.extend(["--workers", str(workers)])
    
    # Set log level based on environment
    log_level = "debug" if environment == "development" else "info"
    cmd.extend(["--log-level", log_level])
    
    print(f"🚀 Starting FastAPI application...")
    print(f"📊 Environment: {environment}")
    print(f"🌐 Host: {host}:{port}")
    print(f"🔄 Reload: {reload}")
    print(f"👥 Workers: {workers}")
    print(f"📝 Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✋ Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Start the new modular FastAPI application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development mode with auto-reload
  python scripts/start_new_app.py --dev
  
  # Production mode with multiple workers
  python scripts/start_new_app.py --prod --workers 4
  
  # Custom host and port
  python scripts/start_new_app.py --host 127.0.0.1 --port 8080
        """
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with auto-reload"
    )
    
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Run in production mode"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (ignored in dev mode)"
    )
    
    parser.add_argument(
        "--environment",
        choices=["development", "production"],
        help="Environment mode (overrides --dev/--prod)"
    )
    
    args = parser.parse_args()
    
    # Determine environment
    if args.environment:
        environment = args.environment
    elif args.dev:
        environment = "development"
    elif args.prod:
        environment = "production"
    else:
        environment = "production"  # Default to production
    
    # Auto-reload only in development
    reload = environment == "development"
    
    # Validate workers
    if args.workers > 1 and reload:
        print("⚠️  Warning: Multiple workers disabled in development/reload mode")
        workers = 1
    else:
        workers = args.workers
    
    start_application(
        host=args.host,
        port=args.port,
        reload=reload,
        environment=environment,
        workers=workers
    )


if __name__ == "__main__":
    main()