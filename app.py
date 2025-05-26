from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc
import numpy as np
import os

app = Flask(__name__)

STATIC_FOLDER = "static"
PDB_FILE = os.path.join(STATIC_FOLDER, "predicted.pdb")
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form.get("sequence", "").strip().upper()

    # Validate sequence
    valid_chars = "ABCDEFGHIKLMNPQRSTVWYZX*"  # Standard amino acids + unknown + stop
    if not sequence or not all(c in valid_chars for c in sequence):
        return jsonify({"error": "Invalid sequence! Use standard amino acids only."}), 400
    if len(sequence) > 1000:
        return jsonify({"error": "Sequence too long! Max 1000 residues."}), 400

    try:
        # Call ESMFold API
        response = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence)
        response.raise_for_status()
        pdb_string = response.text

        # Save PDB file
        with open(PDB_FILE, "w") as f:
            f.write(pdb_string)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch prediction: {e}"}), 500

    # Ensure PDB file exists
    if not os.path.exists(PDB_FILE):
        return jsonify({"error": "PDB file not found after saving."}), 500

    try:
        # Load and process the PDB file
        pdb_file = pdbio.PDBFile.read(PDB_FILE)
        struct_data = pdbio.get_structure(pdb_file)

        if isinstance(struct_data, struc.AtomArrayStack):
            struct_data = struct_data[0]  # Use first model if multiple exist

        # Extract b-factors (confidence scores)
        if hasattr(struct_data, "b_factor") and struct_data.b_factor is not None:
            avg_confidence = round(np.mean(struct_data.b_factor), 2)
        else:
            avg_confidence = "N/A"

        # Compute molecular weight (excluding hydrogen atoms)
        protein_atoms = struct_data[struct_data.element != "H"]
        try:
            molecular_weight = round(struc.mass(protein_atoms), 2)
        except Exception:
            molecular_weight = "Unknown"

        num_residues = len(sequence)  # Count of residues in the sequence

    except Exception as e:
        return jsonify({"error": f"Error processing PDB file: {e}"}), 500

    return jsonify({
        "pdb_url": f"/static/{os.path.basename(PDB_FILE)}",
        "confidence": avg_confidence,
        "molecular_weight": molecular_weight,
        "sequence_length": num_residues
    })

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
