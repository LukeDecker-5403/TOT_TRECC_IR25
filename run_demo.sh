#!/bin/bash

# TOT Retrieval System - Demo Setup and Run Script
# This script sets up everything needed for the Flask demo

# Don't exit on error - we want to check everything and report issues

echo "=========================================="
echo "TOT Retrieval System - Demo Setup"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo "✓ Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.8+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "   Python version: $PYTHON_VERSION"
echo ""

# Check Java (required for Lucene/Pyserini)
echo "✓ Checking Java installation..."
if ! command -v java &> /dev/null; then
    echo "❌ Error: Java not found. Please install Java 11+"
    echo "   Ubuntu/Debian: sudo apt-get install openjdk-11-jdk"
    exit 1
fi
JAVA_VERSION=$(java -version 2>&1 | head -1)
echo "   $JAVA_VERSION"
echo ""

# Check/install Python dependencies
echo "✓ Checking Python dependencies..."
REQUIRED_PACKAGES=("flask" "openai" "pyserini" "numpy" "pandas" "python-dotenv")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    # Handle package name differences (python-dotenv -> dotenv)
    import_name="${package//-/_}"
    if [ "$package" = "python-dotenv" ]; then
        import_name="dotenv"
    fi
    if ! python3 -c "import $import_name" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "   Installing missing packages: ${MISSING_PACKAGES[*]}"
    python3 -m pip install --quiet "${MISSING_PACKAGES[@]}"
    echo "   ✓ Dependencies installed"
else
    echo "   ✓ All dependencies installed"
fi
echo ""

# Check for dataset
echo "✓ Checking dataset..."
DATASET_PATH="tot_retrieval/gutenberg_data/gutenberg_subset.json"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    echo "   Please ensure the Gutenberg dataset is available"
    exit 1
fi
DATASET_SIZE=$(du -h "$DATASET_PATH" | cut -f1)
echo "   Dataset found: $DATASET_PATH ($DATASET_SIZE)"
echo ""

# Check for Lucene indices
echo "✓ Checking Lucene indices..."
REQUIRED_FIELDS=("plot" "title" "author" "genre" "date" "cover")
MISSING_INDICES=()

for field in "${REQUIRED_FIELDS[@]}"; do
    INDEX_DIR="tot_retrieval/lucene_indices/${field}_index"
    if [ ! -d "$INDEX_DIR" ]; then
        MISSING_INDICES+=("$field")
    fi
done

if [ ${#MISSING_INDICES[@]} -gt 0 ]; then
    echo "   ⚠ Warning: Missing indices for fields: ${MISSING_INDICES[*]}"
    echo "   The app will rebuild indices on first run (this may take a few minutes)"
    echo "   To rebuild indices manually, edit app.py and set rebuild_index=True"
else
    echo "   ✓ All Lucene indices found"
fi
echo ""

# Check for .env file with API key
echo "✓ Checking OpenAI API key..."
ENV_FILE="tot_retrieval/.env"
if [ -f "$ENV_FILE" ]; then
    if grep -qi "openai.*api.*key" "$ENV_FILE" 2>/dev/null; then
        echo "   ✓ API key found in .env file"
    else
        echo "   ⚠ Warning: .env file exists but no API key found"
        echo "   The app may not work without an OpenAI API key"
    fi
else
    echo "   ⚠ Warning: .env file not found"
    echo "   Create tot_retrieval/.env with: OPENAI_API_KEY=your-key-here"
fi
echo ""

# Check if Flask app is already running and kill it
if pgrep -f "python.*app.py" > /dev/null; then
    echo "⚠ Found existing Flask app instance"
    echo "   Stopping existing Flask app..."
    pkill -f "python.*app.py"
    sleep 2
    # Verify it's stopped
    if pgrep -f "python.*app.py" > /dev/null; then
        echo "   ⚠ Warning: Could not stop existing Flask app. Trying force kill..."
        pkill -9 -f "python.*app.py"
        sleep 1
    fi
    echo "   ✓ Stopped existing Flask app"
    echo ""
fi

echo "=========================================="
echo "Starting Flask Application"
echo "=========================================="
echo ""
echo "🌐 The app will be available at: http://localhost:5000"
echo "📝 Press Ctrl+C to stop the server"
echo ""
echo "Starting server..."
echo ""

# Run the Flask app
python3 app.py

