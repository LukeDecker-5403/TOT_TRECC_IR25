from flask import Flask, render_template, request
from tot_retrieval.main_lucene import TOTRetrievalSystem
from tot_retrieval.data_loader import DataLoader
from tot_retrieval.config import Config

app = Flask(__name__)

# ---------------------------
# Load Dataset
# ---------------------------

loader = DataLoader(Config.DATA_DIR)
documents = loader.load_dataset(str(Config.BASE_DIR / "gutenberg_data" / "gutenberg_subset.json"))

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
        
        # Get decomposed query breakdown for display
        decomposed = system.query_decomposer.decompose(query, mode="extractive")
        decomposed_dict = decomposed.to_dict()
        # Filter out N/A fields for display
        decomposed_breakdown = {k: v for k, v in decomposed_dict.items() if v and v != "N/A"}
        
        results = system.search(query, mode="extractive", top_k=5)
        
        # Format field scores for display (top 3 non-zero fields)
        for result in results:
            if result.get('field_scores'):
                field_scores = result['field_scores']
                # Filter and sort top fields
                non_zero_fields = [(k, v) for k, v in field_scores.items() if v > 0.001]
                sorted_fields = sorted(non_zero_fields, key=lambda x: x[1], reverse=True)
                result['top_fields'] = sorted_fields[:3]
        
        return render_template("result.html", query=query, results=results, decomposed_breakdown=decomposed_breakdown)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
