from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="LLM Memory Requirements API",
    description="API to calculate memory requirements for large language models during inference and training.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas and helper functions remain the same
class InferenceMemoryRequest(BaseModel):
    model_size: float
    precision: str
    batch_size: int
    sequence_length: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int


class InferenceMemoryResponse(BaseModel):
    inference_memory: float
    model_weights: float
    kv_cache: float
    activation_memory: float


class TrainingMemoryRequest(BaseModel):
    model_size: float
    precision: str
    batch_size: int
    sequence_length: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    optimizer: str
    trainable_parameters: int


class TrainingMemoryResponse(BaseModel):
    training_memory: float
    model_weights: float
    kv_cache: float
    activation_memory: float
    optimizer_memory: float
    gradients_memory: float


class Error(BaseModel):
    code: int
    message: str


def calculate_inference_memory(data: InferenceMemoryRequest) -> InferenceMemoryResponse:
    model_weights = data.model_size * 1024
    kv_cache = data.sequence_length * data.hidden_size * data.num_attention_heads * 2 / 1e6
    activation_memory = data.batch_size * data.sequence_length * data.hidden_size / 1e6
    total_memory = model_weights + kv_cache + activation_memory
    return InferenceMemoryResponse(
        inference_memory=total_memory,
        model_weights=model_weights,
        kv_cache=kv_cache,
        activation_memory=activation_memory,
    )


def calculate_training_memory(data: TrainingMemoryRequest) -> TrainingMemoryResponse:
    model_weights = data.model_size * 1024
    kv_cache = data.sequence_length * data.hidden_size * data.num_attention_heads * 2 / 1e6
    activation_memory = data.batch_size * data.sequence_length * data.hidden_size / 1e6
    optimizer_memory = model_weights * 0.5
    gradients_memory = model_weights * 0.3
    total_memory = model_weights + kv_cache + activation_memory + optimizer_memory + gradients_memory
    return TrainingMemoryResponse(
        training_memory=total_memory,
        model_weights=model_weights,
        kv_cache=kv_cache,
        activation_memory=activation_memory,
        optimizer_memory=optimizer_memory,
        gradients_memory=gradients_memory,
    )


@app.post("/memory/inference", response_model=InferenceMemoryResponse, responses={400: {"model": Error}})
async def calculate_inference(data: InferenceMemoryRequest):
    try:
        return calculate_inference_memory(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/memory/training", response_model=TrainingMemoryResponse, responses={400: {"model": Error}})
async def calculate_training(data: TrainingMemoryRequest):
    try:
        return calculate_training_memory(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, log_level="info")