# TOT Retrieval System - Demo Guide

## Quick Start

Run the demo setup script:

```bash
./run_demo.sh
```

This script will:
- ✅ Check Python and Java installations
- ✅ Install missing dependencies
- ✅ Verify dataset and indices exist
- ✅ Check for OpenAI API key
- ✅ Start the Flask application

## Manual Setup

If you prefer to set up manually:

### 1. Install Dependencies

```bash
pip install flask openai pyserini numpy pandas python-dotenv
```

### 2. Verify Requirements

- **Python 3.8+**: `python3 --version`
- **Java 11+**: `java -version`
- **Dataset**: `tot_retrieval/gutenberg_data/gutenberg_subset.json`
- **OpenAI API Key**: Set in `tot_retrieval/.env` as `OPENAI_API_KEY=your-key-here`

### 3. Run the App

```bash
python3 app.py
```

The app will be available at: **http://localhost:5000**

## Demo Queries

Try these example queries:

1. **Detective Story:**
   ```
   A detective story about a man who solves mysteries in London
   ```

2. **Wizard of Oz (with all fields):**
   ```
   I'm looking for a children's fantasy book from around 1900 by an author whose last name starts with B. It's about a young girl from Kansas who gets swept away to a magical land with a yellow brick road. She meets a scarecrow looking for a brain, a tin man wanting a heart, and a cowardly lion. The cover probably shows the yellow brick road leading to an emerald city. The title has something about a wizard.
   ```

3. **Horror Novel:**
   ```
   A classic horror novel about a vampire from Transylvania
   ```

## Features

- **Query Decomposition**: Shows how the system breaks down your query into different fields (plot, title, author, genre, date, cover)
- **Multi-field Search**: Searches across all metadata fields simultaneously
- **Ranked Results**: Shows top results with scores and contributing fields
- **Clean UI**: Matches console output format for easy comparison

## Troubleshooting

### Indices Not Found
If you see errors about missing indices:
- Edit `app.py` and change `rebuild_index=False` to `rebuild_index=True`
- Run the app once to build indices (may take a few minutes)
- Change back to `rebuild_index=False` for faster startup

### API Key Issues
- Ensure `tot_retrieval/.env` exists with: `OPENAI_API_KEY=your-key-here`
- Or set environment variable: `export OPENAI_API_KEY=your-key-here`

### Port Already in Use
If port 5000 is already in use:
- Kill existing process: `pkill -f "python.*app.py"`
- Or change port in `app.py`: `app.run(debug=True, port=5001)`

## File Structure

```
.
├── app.py                 # Flask application
├── run_demo.sh            # Demo setup script
├── templates/
│   ├── index.html         # Search interface
│   └── result.html        # Results page
└── tot_retrieval/
    ├── main_lucene.py     # Main retrieval system
    ├── gutenberg_data/    # Dataset directory
    └── lucene_indices/    # Search indices
```

