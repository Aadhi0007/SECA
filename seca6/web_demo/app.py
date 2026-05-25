import sys
import os
import json
import threading
import csv
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from flask import Flask, render_template, jsonify, request, redirect, flash, send_file

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seca_core.seca_engine import SECAEngine
from data.dataset_loader import load_dataset

app = Flask(__name__)

LOG_FILE = "logs/live_results.json"
UPLOAD_FOLDER = os.path.join("data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RUNNING = False
ACTIVE_THREAD = None
RESULTS = []
SECA_INSTANCE = None

ACTIVE_DATASET_TYPE = "mnist"
ACTIVE_DATASET_PATH = None

PROGRESS_STATE = {
    "generation": 0,
    "individual": 0,
    "total_individuals": 0,
    "total_generations": 0,
    "pop_size": 0,
    "estimated_remaining_seconds": 0,
    "results": []
}


# -------------------------------
# Dashboard
# -------------------------------

@app.route("/")
def dashboard():

    dataset_display_map = {
        "mnist": "MNIST",
        "fashion_mnist": "Fashion MNIST",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "imdb": "IMDB Reviews",
        "reuters": "Reuters Newswire",
        "custom": f"Custom ({os.path.basename(ACTIVE_DATASET_PATH) if ACTIVE_DATASET_PATH else 'Unknown'})"
    }
    
    active_dataset = dataset_display_map.get(ACTIVE_DATASET_TYPE, "Unknown")

    if not RESULTS:
        return render_template(
            "index.html",
            generations=[],
            accuracy=[],
            fitness=[],
            params=[],
            summary={
                "generations": 0,
                "best_accuracy": 0,
                "best_fitness": 0,
                "min_parameters": 0
            },
            best_model={"genome": "Run SECA to start evolution"},
            active_dataset=active_dataset
        )

    generations = [r["generation"] for r in RESULTS]
    accuracy = [r["accuracy"] for r in RESULTS]
    fitness = [r["fitness"] for r in RESULTS]
    params = [r["params"] for r in RESULTS]

    summary = {
        "generations": len(RESULTS),
        "best_accuracy": max(accuracy),
        "best_fitness": max(fitness),
        "min_parameters": min(params)
    }

    best_model = RESULTS[-1]

    return render_template(
        "index.html",
        generations=generations,
        accuracy=accuracy,
        fitness=fitness,
        params=params,
        summary=summary,
        best_model=best_model,
        progress_state=PROGRESS_STATE,
        active_dataset=active_dataset
    )

# -------------------------------
# Select Dataset
# -------------------------------

@app.route("/select_dataset", methods=['GET', 'POST'])
def select_dataset():
    global ACTIVE_DATASET_TYPE, ACTIVE_DATASET_PATH
    
    if request.method == 'POST':
        dataset_choice = request.form.get('dataset_choice')
        
        if dataset_choice == 'custom':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and (file.filename.endswith('.npz') or file.filename.endswith('.csv')):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                ACTIVE_DATASET_TYPE = 'custom'
                ACTIVE_DATASET_PATH = filepath
                return redirect("/")
        elif dataset_choice in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters']:
            ACTIVE_DATASET_TYPE = dataset_choice
            return redirect("/")
            
    return render_template("datasets.html", current_dataset=ACTIVE_DATASET_TYPE)


# -------------------------------
# Start Evolution
# -------------------------------

@app.route("/run_seca")
def run_seca():

    global RUNNING, ACTIVE_THREAD

    if RUNNING:
        return jsonify({"status": "already running"})

    model_type = request.args.get('model_type', 'cnn')

    RUNNING = True

    thread = threading.Thread(target=start_evolution, args=(model_type,))
    ACTIVE_THREAD = thread
    thread.start()

    return jsonify({"status": "started"})

@app.route("/stop_seca")
def stop_seca():
    global SECA_INSTANCE, RUNNING
    if SECA_INSTANCE:
        SECA_INSTANCE.stop()
    return jsonify({"status": "stopping"})


# -------------------------------
# Evolution Process
# -------------------------------

def start_evolution(model_type='cnn'):

    global RUNNING, RESULTS, PROGRESS_STATE, ACTIVE_DATASET_TYPE, ACTIVE_DATASET_PATH, SECA_INSTANCE, ACTIVE_THREAD

    print(f"Starting evolution with dataset: {ACTIVE_DATASET_TYPE} and model_type: {model_type}")

    if ACTIVE_DATASET_TYPE == 'custom' and ACTIVE_DATASET_PATH and os.path.exists(ACTIVE_DATASET_PATH):
        try:
            if ACTIVE_DATASET_PATH.endswith('.npz'):
                with np.load(ACTIVE_DATASET_PATH) as data:
                    x_train = data['x_train']
                    y_train = data['y_train']
                    x_test = data['x_test']
                    y_test = data['y_test']
            elif ACTIVE_DATASET_PATH.endswith('.csv'):
                # Load CSV using pandas for speed
                df = pd.read_csv(ACTIVE_DATASET_PATH)
                data = df.values
                
                # Assume last column is target y, first columns are feature x
                x = data[:, :-1]
                y = data[:, -1]
                
                # model_builder.py handles 1D automatically, so leave x as (samples, features)
                x = x.astype('float32')
                y = y.astype('int32')
                
                # Simple Manual 80/20 train/test Split
                split_idx = int(0.8 * len(x))
                indices = np.random.permutation(len(x))
                x_train, x_test = x[indices[:split_idx]], x[indices[split_idx:]]
                y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
        except Exception as e:
            print(f"Error loading custom dataset {ACTIVE_DATASET_PATH}, falling back to MNIST: {e}")
            x_train, x_test, y_train, y_test = load_dataset(name="mnist")
    else:
        dataset_name = ACTIVE_DATASET_TYPE if ACTIVE_DATASET_TYPE != 'custom' else 'mnist'
        
        # Auto-switch model_type to 'llm' if the user clicked Start Evolution (default 'cnn') 
        # while a language dataset was selected.
        if dataset_name in ['imdb', 'reuters'] and model_type == 'cnn':
            print(f"Auto-switching model_type to 'llm' since {dataset_name} is active.")
            model_type = 'llm'
        
        if model_type in ['llm', 'gpt', 'llama']:
            if dataset_name not in ['imdb', 'reuters']:
                print(f"Routing dataset to IMDB for {model_type} sequence generation...")
                dataset_name = 'imdb'
                ACTIVE_DATASET_TYPE = 'imdb'
        elif model_type == 'bert':
            if dataset_name not in ['imdb', 'reuters']:
                print("Routing dataset to Reuters for BERT text classification...")
                dataset_name = 'reuters'
                ACTIVE_DATASET_TYPE = 'reuters'
            
        x_train, x_test, y_train, y_test = load_dataset(name=dataset_name)

    # For sequence data (IMDB), shape is (seq_len,)
    # For images, shape is (H, W, C)
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    seca = SECAEngine(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_shape=input_shape,
        num_classes=num_classes,
        network_type=model_type
    )
    SECA_INSTANCE = seca

    RESULTS = []
    
    import time
    start_time = time.time()
    
    total_individuals = seca.generations * seca.population_size
    
    PROGRESS_STATE = {
        "generation": 0,
        "individual": 0,
        "total_individuals": total_individuals,
        "total_generations": seca.generations,
        "pop_size": seca.population_size,
        "estimated_remaining_seconds": 0,
        "results": []
    }
    state = {
        "individuals_processed": 0,
        "start_time": time.time()
    }

    def on_individual(gen, idx, result):
        state["individuals_processed"] += 1
        
        elapsed = time.time() - state["start_time"]
        avg_time_per_ind = elapsed / state["individuals_processed"]
        remaining_inds = total_individuals - state["individuals_processed"]
        
        PROGRESS_STATE["generation"] = gen + 1
        PROGRESS_STATE["individual"] = idx + 1
        PROGRESS_STATE["estimated_remaining_seconds"] = int(avg_time_per_ind * remaining_inds)
        
        # Save training log
        log_path = os.path.join("logs", "training_logs.csv")
        file_exists = os.path.isfile(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generation", "individual", "accuracy", "fitness", "params"])
            writer.writerow([gen, idx, result["accuracy"], result["fitness"], result["params"]])

    def on_generation(gen, best_stats, best_genome):
        entry = {
            "generation": gen,
            "accuracy": best_stats["accuracy"],
            "fitness": best_stats["fitness"],
            "params": best_stats["params"],
            "genome": best_genome
        }
        RESULTS.append(entry)
        PROGRESS_STATE["results"] = RESULTS
        
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump(PROGRESS_STATE, f, indent=4)
            
        # Save performance log
        perf_path = os.path.join("logs", "performance_logs.csv")
        file_exists = os.path.isfile(perf_path)
        with open(perf_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generation", "best_accuracy", "best_fitness", "best_params", "best_genome"])
            writer.writerow([gen, best_stats["accuracy"], best_stats["fitness"], best_stats["params"], json.dumps(best_genome)])

    best_genome, best_stats, history, best_model = seca.run_evolution(
        on_individual=on_individual,
        on_generation=on_generation
    )

    import datetime

    if best_model is not None:
        # Save the usual latest best model
        best_model.save(os.path.join("logs", "best_model.keras"))
        
        # Save a unique persistent copy with date and time
        os.makedirs("saved_models", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_filename = f"best_{ACTIVE_DATASET_TYPE}_{model_type}_{timestamp}.keras"
        best_model.save(os.path.join("saved_models", unique_filename))
        
        # Save the historical training data associated with this model persistently
        history_filename = f"history_{ACTIVE_DATASET_TYPE}_{model_type}_{timestamp}.json"
        with open(os.path.join("saved_models", history_filename), "w") as f:
            json.dump(history, f, indent=4)

    if ACTIVE_THREAD == threading.current_thread():
        PROGRESS_STATE["estimated_remaining_seconds"] = 0
        PROGRESS_STATE["generation"] = seca.generations
        PROGRESS_STATE["individual"] = seca.population_size

        with open(LOG_FILE, "w") as f:
            json.dump(PROGRESS_STATE, f, indent=4)

        RUNNING = False


# -------------------------------
# Live Results API
# -------------------------------

@app.route("/get_progress")
def get_progress():

    if RUNNING:
        return jsonify(PROGRESS_STATE)

    if not os.path.exists(LOG_FILE):
        return jsonify({
            "generation": 0,
            "individual": 0,
            "total_individuals": 0,
            "total_generations": 0,
            "pop_size": 0,
            "estimated_remaining_seconds": 0,
            "results": []
        })

    with open(LOG_FILE) as f:
        data = json.load(f)

    # Convert old format list to new format dictionary for backward compatibility
    if isinstance(data, list):
        data = {
            "generation": len(data),
            "individual": 0,
            "total_individuals": len(data) * 6, # Assuming default pop size 6
            "total_generations": len(data),
            "pop_size": 6,
            "estimated_remaining_seconds": 0,
            "results": data
        }

    return jsonify(data)


# -------------------------------
# Status API
# -------------------------------

@app.route("/status")
def status():
    stopping = False
    if SECA_INSTANCE and getattr(SECA_INSTANCE, 'stop_requested', False):
        stopping = True
    return jsonify({"running": RUNNING, "stopping": stopping})


# -------------------------------
# Download History
# -------------------------------

@app.route("/download_history")
def download_history():
    log_path = os.path.abspath(os.path.join("logs", "training_logs.csv"))
    if os.path.exists(log_path):
        return send_file(log_path, as_attachment=True, download_name="training_logs.csv")
    return "No training logs found.", 404


# -------------------------------
# Compare LLMs
# -------------------------------

@app.route("/compare")
def compare_dashboard():
    return render_template("compare_dashboard.html")

@app.route("/compare/gpt")
def compare_gpt():
    return render_template("compare_gpt.html", active_comparison="gpt")

@app.route("/compare/llama")
def compare_llama():
    return render_template("compare_llama.html", active_comparison="llama")

@app.route("/compare/bert")
def compare_bert():
    return render_template("compare_bert.html", active_comparison="bert")


# -------------------------------
# Run Server
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)