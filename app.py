from flask import Flask, render_template, request
from tot_retrieval.main_lucene import TOTRetrievalSystem
from tot_retrieval.data_loader import DataLoader
from tot_retrieval.config import Config

app = Flask(__name__)

# ---------------------------
# Load Dataset
# ---------------------------

loader = DataLoader(Config.DATA_DIR)
documents = loader.load_dataset("small_sample.json")  # Replace with real dataset on ting precision tower

# ---------------------------------------------------------------------------
# INDEX BUILD INSTRUCTIONS
#
# Use rebuild_index=True ONLY when:
#   - running for the first time
#   - switching to a new dataset
#
# After the Lucene index is built once, switch to rebuild_index=False
# for instant loading on all future runs.
# ---------------------------------------------------------------------------

system = TOTRetrievalSystem()
system.setup(documents, rebuild_index=False)   # <-- set to True on first run

# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        results = system.search(query, mode="extractive", top_k=5)
        return render_template("result.html", query=query, results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
