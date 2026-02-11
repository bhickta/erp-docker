import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer


# --- 1. HEALTH CHECK ---
if not torch.cuda.is_available():
    print("="*50)
    print("FATAL ERROR: ML-WORKER CANNOT DETECT GPU!")
    print("="*50)
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# --- 2. LOAD MODEL (ON STARTUP) ---
# Load the model into GPU memory *once* when the server starts.
model_name = "Qwen/Qwen3-Embedding-4B"
print(f"Loading {model_name} onto GPU...")
model = SentenceTransformer(
    model_name,
    trust_remote_code=True,
    device="cuda"  # Move model to GPU
)
print("Model loaded. Worker is ready.")


# --- 3. CREATE FLASK SERVER ---
app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def embed_texts():
    try:
        data = request.get_json()
        if 'texts' not in data:
            return jsonify({"error": "No 'texts' key found in JSON"}), 400

        texts_to_embed = data['texts']
        
        print(f"Received {len(texts_to_embed)} texts. Embedding on GPU...")

        # Run the actual embedding on the GPU
        embeddings = model.encode(
            texts_to_embed,
            show_progress_bar=False,
            batch_size=32 
        )
        
        print("Embedding complete. Sending results back.")
        
        # Convert numpy array to a standard list to send as JSON
        return jsonify(embeddings.tolist())

    except Exception as e:
        print(f"Error during embedding: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server and make it accessible from other containers
    app.run(host='0.0.0.0', port=5001)