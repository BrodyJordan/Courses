import os
import json
import re
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct" # Or 3B / 72B depending on VRAM
DATASET_DIR = "/content/vstibench"
BENCHMARK_FILE = os.path.join(DATASET_DIR, "test.json")
ALLOWED_TASK_TYPES = ["camera_movement_direction", "camera_obj_abs_dist"]

# 1. Initialize vLLM with Native Video Support
print(f"[*] Initializing {MODEL}...")
llm = LLM(
    model=MODEL,
    max_model_len=16384,          # High limit for long video sequences
    gpu_memory_utilization=0.85, 
    limit_mm_per_prompt={"video": 1}, 
    dtype="bfloat16",
    trust_remote_code=True,
    allowed_local_media_path="/content"
)

sampling_params = SamplingParams(
    temperature=0.0, 
    max_tokens=20, 
    stop=["\n", "<|im_end|>", "<|endoftext|>"]
)

# --- SCORING UTILS ---

def calculate_mra(pred, gt):
    try:
        # Regex to find numbers, handling 'm' or 'meters'
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred.replace('m', ''))
        if not nums: return 0.0
        pred_val = float(nums[0])
        return max(0, 1 - abs(pred_val - gt) / gt)
    except:
        return 0.0

# --- EVALUATION LOOP ---

with open(BENCHMARK_FILE, 'r') as f:
    tasks = json.load(f)

results = []

for task in tasks:
    if task.get('question_type') not in ALLOWED_TASK_TYPES:
        continue

    video_full_path = os.path.join(DATASET_DIR, task['video_path'])
    if not os.path.exists(video_full_path):
        continue

    # 1. Clean Prompt Construction
    user_prompt = task['question']
    if task.get('options'):
        user_prompt += "\nOptions: " + ", ".join(task['options'])
    
    # Anchor the assistant to prevent echoing "The user wants to..."
    # Qwen2.5-VL works best with standard ChatML formatting
    formatted_prompt = (
        f"<|im_start|>system\n"
        f"You are a spatial reasoning assistant. Answer with ONLY the correct letter or number.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\nAnswer:"
    )

    # 2. Native vLLM Video Content Structure
    # Use 'file://' prefix for local paths
    message_content = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url", 
                    "video_url": {
                        "url": f"file://{video_full_path}",
                        "fps": 1.0,           # Lower FPS = fewer tokens = more stability
                        "max_pixels": 360*360 # Reduces VRAM pressure
                    }
                },
                {"type": "text", "text": formatted_prompt}
            ]
        }
    ]
    
    # 3. Inference
    try:
        # Note: we pass message_content directly as the conversation
        outputs = llm.chat(message_content, sampling_params=sampling_params)
        prediction = outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"[!] Error on task {task['id']}: {e}")
        prediction = ""

    # 4. Hardened Scoring (Regex Word Boundary)
    task['prediction'] = prediction
    if task.get('mc_answer'):
        gt = task['mc_answer'].upper()
        # \b prevents matching letters inside other words
        if re.search(rf"\b{gt}\b", prediction.upper()):
            task['score'] = 1.0
        else:
            # Check for the literal choice word (e.g., 'Right')
            gt_word = ""
            for opt in task['options']:
                if opt.startswith(gt): gt_word = opt.split('.')[-1].strip().upper()
            task['score'] = 1.0 if (gt_word and re.search(rf"\b{gt_word}\b", prediction.upper())) else 0.0
        metric = "ACC"
    else:
        gt_val = float(task['ground_truth'])
        task['score'] = calculate_mra(prediction, gt_val)
        metric = "MRA"
        
    results.append(task)
    print(f"ID: {task['id']} | Type: {task['question_type']} | Pred: '{prediction}' | {metric}: {task['score']:.2f}")

# --- SAVE ---
output_file = "vstibench_qwen2_5_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[*] Evaluation Complete. Saved to {output_file}")