from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

from ..ml.services.model_service import ModelService
from ..ml.models.behavior_model import BehaviorModel
from ..ml.models.communication_model import CommunicationModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    responses={404: {"description": "Not found"}},
)

# Model service instance
model_service = ModelService()

# Request and response models
class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    created_at: str
    updated_at: str
    training_history: int

class BehaviorPredictionRequest(BaseModel):
    sequences: List[List[int]]
    top_k: Optional[int] = 3

class BehaviorPredictionResponse(BaseModel):
    predictions: List[List[Dict[str, Any]]]

class BehaviorTrainingRequest(BaseModel):
    sequences: List[List[int]]
    targets: List[int]
    validation_split: Optional[float] = 0.2
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 64

class CommunicationGenerationRequest(BaseModel):
    seed_texts: List[str]
    max_length: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 3

class CommunicationGenerationResponse(BaseModel):
    generated_texts: List[str]

class CommunicationTrainingRequest(BaseModel):
    texts: List[str]
    validation_split: Optional[float] = 0.2
    epochs: Optional[int] = 30
    batch_size: Optional[int] = 32

class TrainingResponse(BaseModel):
    success: bool
    model_name: str
    metrics: Dict[str, Any]
    training_completed: str

# Dependency for getting the model service
def get_model_service():
    return model_service

# Routes
@router.get("/models", response_model=List[ModelInfo])
async def list_models(model_service: ModelService = Depends(get_model_service)):
    """List all available ML models."""
    return model_service.list_models()

@router.post("/behavior/predict", response_model=BehaviorPredictionResponse)
async def predict_behavior(
    request: BehaviorPredictionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Predict next activities based on sequences of past activities.
    """
    # Check if behavior model exists, create if it doesn't
    model_name = "behavior_model"
    if model_name not in [m["name"] for m in model_service.list_models()]:
        model_service.create_behavior_model()
    
    # Make predictions
    predictions = model_service.predict_behavior(
        model_name=model_name,
        sequences=request.sequences,
        top_k=request.top_k
    )
    
    # Format predictions as dictionaries
    formatted_predictions = []
    for sequence_preds in predictions:
        formatted_sequence = []
        for activity_id, probability in sequence_preds:
            formatted_sequence.append({
                "activity_id": activity_id,
                "probability": probability
            })
        formatted_predictions.append(formatted_sequence)
    
    return {"predictions": formatted_predictions}

@router.post("/communication/generate", response_model=CommunicationGenerationResponse)
async def generate_communication(
    request: CommunicationGenerationRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Generate communication text based on seed texts in the user's style.
    """
    # Check if communication model exists, create if it doesn't
    model_name = "communication_model"
    if model_name not in [m["name"] for m in model_service.list_models()]:
        model_service.create_communication_model()
    
    # Generate text
    generated_texts = model_service.generate_communication(
        model_name=model_name,
        seed_texts=request.seed_texts,
        max_length=request.max_length,
        temperature=request.temperature,
        top_k=request.top_k
    )
    
    return {"generated_texts": generated_texts}

@router.post("/behavior/train", response_model=TrainingResponse)
async def train_behavior_model(
    request: BehaviorTrainingRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Train the behavior model with sequences of user activities.
    
    This is an asynchronous operation that runs in the background.
    """
    model_name = "behavior_model"
    
    # Check if model exists, create if not
    if model_name not in [m["name"] for m in model_service.list_models()]:
        model_service.create_behavior_model()
    
    # Validate data
    if len(request.sequences) == 0 or len(request.targets) == 0:
        raise HTTPException(status_code=400, detail="Empty training data")
    
    if len(request.sequences) != len(request.targets):
        raise HTTPException(
            status_code=400, 
            detail=f"Mismatch between sequences ({len(request.sequences)}) and targets ({len(request.targets)})"
        )
    
    # Define background training task
    def train_model_task():
        try:
            metrics = model_service.train_behavior_model(
                model_name=model_name,
                X=request.sequences,
                y=request.targets,
                validation_split=request.validation_split,
                epochs=request.epochs,
                batch_size=request.batch_size
            )
            logger.info(f"Behavior model training completed: {metrics}")
        except Exception as e:
            logger.error(f"Error training behavior model: {e}")
    
    # Add training task to background tasks
    background_tasks.add_task(train_model_task)
    
    return {
        "success": True,
        "model_name": model_name,
        "metrics": {"status": "training_started"},
        "training_completed": "Training job started in background"
    }

@router.post("/communication/train", response_model=TrainingResponse)
async def train_communication_model(
    request: CommunicationTrainingRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Train the communication model with text samples in the user's style.
    
    This is an asynchronous operation that runs in the background.
    """
    model_name = "communication_model"
    
    # Check if model exists, create if not
    if model_name not in [m["name"] for m in model_service.list_models()]:
        model_service.create_communication_model()
    
    # Validate data
    if len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Empty training data")
    
    # Define background training task
    def train_model_task():
        try:
            metrics = model_service.train_communication_model(
                model_name=model_name,
                texts=request.texts,
                validation_split=request.validation_split,
                epochs=request.epochs,
                batch_size=request.batch_size
            )
            logger.info(f"Communication model training completed: {metrics}")
        except Exception as e:
            logger.error(f"Error training communication model: {e}")
    
    # Add training task to background tasks
    background_tasks.add_task(train_model_task)
    
    return {
        "success": True,
        "model_name": model_name,
        "metrics": {"status": "training_started"},
        "training_completed": "Training job started in background"
    }

@router.post("/behavior/create")
async def create_behavior_model(
    model_service: ModelService = Depends(get_model_service)
):
    """Create a new behavior model with default parameters."""
    model = model_service.create_behavior_model()
    return {"success": True, "model_name": model.name, "version": model.version}

@router.post("/communication/create")
async def create_communication_model(
    model_service: ModelService = Depends(get_model_service)
):
    """Create a new communication model with default parameters."""
    model = model_service.create_communication_model()
    return {"success": True, "model_name": model.name, "version": model.version}
