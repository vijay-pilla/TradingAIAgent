#!/usr/bin/env python3
"""
Trade Genius Agent - Startup Script
Simple script to start the application with proper setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ('fastapi', 'fastapi'), ('uvicorn', 'uvicorn'), ('yfinance', 'yfinance'), ('pandas', 'pandas'), ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'), ('seaborn', 'seaborn'), ('plotly', 'plotly'), ('scikit-learn', 'sklearn'),
        ('statsmodels', 'statsmodels'), ('tensorflow', 'tensorflow'), ('beautifulsoup4', 'bs4'), ('textblob', 'textblob'),
        ('sqlalchemy', 'sqlalchemy'), ('apscheduler', 'apscheduler'), ('python-dotenv', 'dotenv'), ('loguru', 'loguru')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All required packages are installed")

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Creating .env file from template...")
        
        # Copy from example
        example_file = Path("env_example.txt")
        if example_file.exists():
            with open(example_file, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ .env file created from template")
            print("⚠️  Please edit .env file with your API keys and settings")
        else:
            print("❌ env_example.txt not found")
            sys.exit(1)
    else:
        print("✅ .env file found")

def create_directories():
    """Create necessary directories"""
    directories = ['static', 'static/css', 'static/js', 'static/images', 'charts', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def run_demo():
    """Run the demo script"""
    print("\n🧪 Running demo to test core functionality...")
    try:
        # Add current directory to Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()

        result = subprocess.run([sys.executable, "-m", "src.demo"],
                              capture_output=True, text=True, timeout=120, env=env)
        if result.returncode == 0:
            print("✅ Demo completed successfully")
            return True
        else:
            print("❌ Demo failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out")
        return False
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def start_application():
    """Start the FastAPI application"""
    print("\n🚀 Starting Trade Genius Agent...")
    print("="*50)
    print("Web Interface: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    print("="*50)

    try:
        # Add current directory to Python path and start the application as module
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()

        # Start the application as module
        subprocess.run([sys.executable, "-m", "src.main"], env=env)
    except KeyboardInterrupt:
        print("\n👋 Trade Genius Agent stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main startup function"""
    print("🤖 Trade Genius Agent - Startup")
    print("="*50)
    
    # Pre-flight checks
    check_python_version()
    check_dependencies()
    check_env_file()
    create_directories()
    
    # Ask user what to do
    # print("\nWhat would you like to do?")
    # print("1. Run demo (test functionality)")
    # print("2. Start web application")
    # print("3. Both (demo then start)")
    # print("4. Exit")

    # while True:
    #     try:
    #         choice = input("\nEnter your choice (1-4): ").strip()
    #
    #         if choice == "1":
    #             run_demo()
    #             break
    #         elif choice == "2":
    #             start_application()
    #             break
    #         elif choice == "3":
    #             if run_demo():
    #                 start_application()
    #             break
    #         elif choice == "4":
    #             print("👋 Goodbye!")
    #             break
    #         else:
    #             print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    #     except KeyboardInterrupt:
    #         print("\n👋 Goodbye!")
    #         break
    #     except Exception as e:
    #         print(f"❌ Error: {e}")

    # Directly start the web application
    start_application()

if __name__ == "__main__":
    main()
