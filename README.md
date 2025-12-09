# TOT_TRECC_IR25
Luke Decker, Freya Downey, Dylan Iddings, Michael Leonhard, Xhorxhi Olldashi 

Home Repository for TOT Project IR25

## Quick Start

1. **Download the pre-formatted Gutenberg dataset attached on canvas** (`gutenberg_subset.json`) and place it in `data/`

2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

3. **Run the application:**
   ```bash
   ./run.sh
   ```
   
   The script will install dependencies and start the Flask app. Access it at **http://localhost:5000**
   
   On first run, the system will build Lucene indices (takes a few minutes). After that, set `rebuild_index=False` in `app.py` for faster startup.
