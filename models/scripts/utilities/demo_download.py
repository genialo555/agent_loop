#!/usr/bin/env python3
"""
Demo script to test dataset downloading with the smallest dataset (ReAct).
"""

import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.secure_dataset_downloader import SecureDatasetDownloaderManager

def demo_download():
    """Demo download of the smallest dataset (ReAct) for testing."""
    
    print("🚀 Demo: Testing Dataset Download")
    print("=" * 40)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"📁 Using temporary directory: {temp_path}")
        
        # Initialize downloader manager
        print("🔧 Initializing downloader manager...")
        try:
            manager = SecureDatasetDownloaderManager(
                base_directory=temp_path / "datasets",
                security_policy="minimal"  # Use minimal policy for demo
            )
            print("✅ Manager initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize manager: {e}")
            return False
        
        # Download ReAct dataset (smallest one)
        print("\n📊 Downloading ReAct dataset (demo)...")
        try:
            success, report = manager.download_dataset("react")
            
            if success and report:
                print("✅ Download successful!")
                print(f"   📊 Files analyzed: {report.files_analyzed}")
                print(f"   🛡️  Files blocked: {report.files_blocked}")
                print(f"   🔒 Risk score: {report.overall_risk_score:.1f}")
                print(f"   ⏱️  Processing time: {report.processing_time:.2f}s")
                
                # List some downloaded files
                dataset_dir = temp_path / "datasets" / "react"
                if dataset_dir.exists():
                    files = list(dataset_dir.rglob("*.py"))[:5]  # First 5 Python files
                    if files:
                        print(f"   📄 Sample files downloaded:")
                        for file in files:
                            rel_path = file.relative_to(dataset_dir)
                            print(f"      - {rel_path}")
                
                return True
            else:
                print("❌ Download failed")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

if __name__ == "__main__":
    try:
        success = demo_download()
        if success:
            print("\n🎉 Demo completed successfully!")
            print("   The secure dataset downloader is working correctly.")
        else:
            print("\n⚠️  Demo failed - check configuration and network.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")