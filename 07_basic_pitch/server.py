from flask import Flask, render_template, request, send_file
import os
from pathlib import Path
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return "No file part", 400
        file = request.files["audio"]
        if file.filename == "":
            return "No selected file", 400

        # Read sliders (0.0–1.0 for merge & thres, seconds for minlen)
        merge = float(request.form.get("merge", 0.25))   # Merge short notes
        thres = float(request.form.get("thres", 0.5))    # Confidence threshold
        min_len = float(request.form.get("min", 0.05))   # Minimum note length in seconds

        # Save uploaded file
        input_path = Path(UPLOAD_FOLDER) / file.filename
        file.save(input_path)

        # Format slider values for filename
        merge_str = f"{int(merge*100):03d}"
        thres_str = f"{int(thres*100):03d}"
        minlen_str = f"{int(min_len*1000):04d}"  # milliseconds

        # Output MIDI filename
        base_name = input_path.stem
        ext_name = input_path.suffix.replace(".", "_")
        midi_filename = f"{base_name}{ext_name}_merge{merge_str}_thres{thres_str}_minlen{minlen_str}.mid"
        midi_path = Path(OUTPUT_FOLDER) / midi_filename

        # Run Basic Pitch
        model_output, midi_data, note_events = predict(
            audio_path=str(input_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            minimum_note_length=int(min_len*1000),  # convert sec → ms
            onset_threshold=thres,
            # merge parameter may need conversion depending on library version
        )

        # Write MIDI
        midi_data.write(str(midi_path))

        return send_file(midi_path, as_attachment=True)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
