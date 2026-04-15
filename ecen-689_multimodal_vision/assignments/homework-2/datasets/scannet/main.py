import os
import subprocess
import json
import base64
import requests
import re

# --- CONFIGURATION ---
MODEL = "qwen3.5:9b"
FRAME_STEP = 10  # Extracts and evaluates only 1 out of every 10 frames
SCENES_DIR = "scans/"
OLLAMA_URL = "http://localhost:11434/api/generate"
BENCHMARK_FILE = "vsti_tasks.json" # Downloaded from VSTI-Bench

def run_extraction(scene_id):
    """Calls the updated Python 3 reader script if data is missing."""
    scene_path = os.path.join(SCENES_DIR, scene_id)
    # Check if the color directory exists and has files in it
    color_dir = os.path.join(scene_path, "color")
    
    if not os.path.exists(color_dir) or not os.listdir(color_dir):
        print(f"[*] Extracting {scene_id} at a frame step of {FRAME_STEP}...")
        sens_file = os.path.join(scene_path, f"{scene_id}.sens")
        
        # Now using python3 and passing the native frame_skip flag!
        cmd = [
            "python3", "reader.py",
            "--filename", sens_file,
            "--output_path", scene_path,
            "--export_color_images", 
            "--export_poses", 
            "--export_intrinsics", 
            "--export_depth_images",
            "--frame_skip", str(FRAME_STEP)
        ]
        subprocess.run(cmd, check=True)

def get_vlm_prediction(question, image_paths):
    """Sends images + prompt to the local Ollama Qwen model."""
    encoded_images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                encoded_images.append(base64.b64encode(f.read()).decode('utf-8'))
        else:
            print(f"[!] Warning: Missing frame {img_path}")
            
    payload = {
        "model": MODEL,
        "prompt": f"{question}\nAnswer as concisely as possible.",
        "images": encoded_images,
        "stream": False,
        "options": {"temperature": 0.0} # Deterministic output for benchmarking
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", "")
    except Exception as e:
        print(f"[!] API Error: {e}")
        return ""

def calculate_mra(pred, gt):
    """Calculates Mean Relative Accuracy: 1 - |pred - gt| / gt"""
    try:
        # Extract the first floating point or integer number from the model's text response
        pred_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", pred)[0])
        return max(0, 1 - abs(pred_val - gt) / gt)
    except (IndexError, ValueError):
        # If the model fails to output a number at all
        return 0.0

# --- MAIN LOOP ---
def main():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"[!] Benchmark file {BENCHMARK_FILE} not found. Please download it first.")
        return

    with open(BENCHMARK_FILE, 'r') as f:
        tasks = json.load(f)

    results = []
    print(f"Starting evaluation of {len(tasks)} tasks...")
    
    for task in tasks:
        scene_id = task['scene_id']
        run_extraction(scene_id)
        
        print(f"\nEvaluating Task ID: {task.get('question_id', 'Unknown')}")
        
        # Collect the specifically extracted frames
        frames = [os.path.join(SCENES_DIR, scene_id, "color", f"{i}.jpg") 
                  for i in range(task['start_frame'], task['end_frame'] + 1, FRAME_STEP)]
        
        prediction = get_vlm_prediction(task['question'], frames)
        ground_truth = task['ground_truth']
        
        # Store for analysis
        task['prediction'] = prediction
        
        # Score based on metric type
        if isinstance(ground_truth, (int, float)): # Absolute Distance (MRA)
            task['score'] = calculate_mra(prediction, ground_truth)
            metric_label = "MRA"
        else: # Multiple Choice Direction (ACC)
            task['score'] = 1.0 if str(ground_truth).lower() in prediction.lower() else 0.0
            metric_label = "ACC"
            
        results.append(task)
        print(f"Qwen: '{prediction}' | GT: '{ground_truth}' | {metric_label}: {task['score']:.2f}")

    # Save final results for your 1-page report
    with open("final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[*] Evaluation complete. Results saved to final_results.json")

if __name__ == "__main__":
    main()