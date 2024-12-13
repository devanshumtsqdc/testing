from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Optional
from config import DATA_TYPES, PARAMETERS, OPTIMIZERS
from utils import calculate_inference_memory, calculate_training_memory

# Initialize FastAPI app
app = FastAPI(title="LLM Memory Requirements API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or replace with a list of specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ----------------- Request Models ----------------- #
class InferenceRequest(BaseModel):
    model_size: float  # in billions
    precision: str  # data type
    batch_size: int
    sequence_length: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int

class TrainingRequest(InferenceRequest):
    optimizer: str
    trainable_parameters: int  # percentage

# ----------------- Endpoints ----------------- #
@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM Memory Requirements API"}

@app.post("/upload-model/")
def upload_model(file: UploadFile):
    """Endpoint to upload a model configuration JSON file."""
    try:
        content = json.load(file.file)
        return {"message": f"Model '{file.filename}' uploaded successfully!", "content": content}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

@app.post("/calculate-inference-memory/")
def calculate_inference(req: InferenceRequest):
    """Calculate the inference memory requirements based on the provided model parameters."""
    if req.precision not in DATA_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid precision type. Valid options are: {DATA_TYPES}")
    
    try:
        inference_memory = calculate_inference_memory(
            req.model_size,
            req.precision,
            req.batch_size,
            req.sequence_length,
            req.hidden_size,
            req.num_hidden_layers,
            req.num_attention_heads,
        )
        return {
            "Total Inference Memory": inference_memory["inference_memory"],
            "Model Weights": inference_memory["model_weights"],
            "KV Cache": inference_memory["kv_cache"],
            "Activation Memory": inference_memory["activation_memory"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-training-memory/")
def calculate_training(req: TrainingRequest):
    """Calculate the training memory requirements based on the provided model parameters."""
    if req.precision not in DATA_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid precision type. Valid options are: {DATA_TYPES}")
    if req.optimizer not in OPTIMIZERS:
        raise HTTPException(status_code=400, detail=f"Invalid optimizer. Valid options are: {list(OPTIMIZERS.keys())}")

    try:
        training_memory = calculate_training_memory(
            req.model_size,
            req.precision,
            req.batch_size,
            req.sequence_length,
            req.hidden_size,
            req.num_hidden_layers,
            req.num_attention_heads,
            req.optimizer,
            req.trainable_parameters,
        )
        return {
            "Total Training Memory": training_memory["training_memory"],
            "Model Weights": training_memory["model_weights"],
            "KV Cache": training_memory["kv_cache"],
            "Activation Memory": training_memory["activation_memory"],
            "Optimizer Memory": training_memory["optimizer_memory"],
            "Gradients Memory": training_memory["gradients_memory"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Running the Server ----------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
