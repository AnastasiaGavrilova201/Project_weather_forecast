from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from API_backend import API_Backend
import uvicorn
import asyncio
import os
import csv
import io

# Инициализация бэкенда
api_backend = API_Backend()

class ModelDesc(BaseModel):
    """Pydantic модель для описания модели."""
    name: str
    table_nm: str
    n_epochs: int
    fitted: bool

class SetActiveModelRequest(BaseModel):
    """Pydantic модель для запроса на установку активной модели."""
    model_name: str

class SetActiveModelResponse(BaseModel):
    """Pydantic модель для ответа на установку активной модели."""
    message: str

class FitModelResponse(BaseModel):
    """Pydantic модель для ответа после запуска обучения модели."""
    message: str

class PredictRequest(BaseModel):
    """Pydantic модель для запроса на предсказание."""
    start_time: str

class PredictResponse(BaseModel):
    """Pydantic модель для ответа с предсказаниями."""
    predictions: str

class LoadNewModelRequest(BaseModel):
    """Pydantic модель для запроса на загрузку новой модели."""
    model_name: str

class LoadNewModelResponse(BaseModel):
    """Pydantic модель для ответа после загрузки новой модели."""
    message: str

class CSVContent(BaseModel):
    """Pydantic модель для представления содержимого CSV."""
    file_name: str
    content: str

class LoadNewModelRequest(BaseModel):
    """Pydantic модель для запроса на загрузку новой модели."""
    csv_path: Optional[str]
    table_nm: Optional[str]
    model_name: str
    n_epochs: int

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)

@app.get("/models", response_model=List[ModelDesc])
async def get_models() -> List[ModelDesc]:
    """Получает список загруженных моделей."""
    try:
        models = api_backend.get_loaded_models()
        if not models:
            raise HTTPException(status_code=404, detail="No models loaded")
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model", response_model=SetActiveModelResponse)
async def set_active_model(request: SetActiveModelRequest) -> SetActiveModelResponse:
    """Устанавливает активную модель по имени."""
    try:
        api_backend.set_active(request.model_name)
        return {"message": f"Active model set to {request.model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fit", response_model=FitModelResponse)
async def fit_model() -> FitModelResponse:
    """Запускает обучение активной модели."""
    try:
        async def train():
            api_backend.fit()

        # Устанавливаем тайм-аут на 10 секунд
        await asyncio.wait_for(train(), timeout=10.0)
        return {"message": "Model training completed"}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Model training took too long and was aborted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Возвращает предсказания от активной модели."""
    try:
        predictions = api_backend.predict(request.start_time)
        if predictions is None:
            raise HTTPException(status_code=400, detail="Model is not fitted")
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_new_model", response_model=LoadNewModelResponse)
async def load_new_model(request: LoadNewModelRequest) -> LoadNewModelResponse:
    """
    Загружает вручную заданное содержимое CSV в базу данных и создает модель.
    """
    try:
        api_backend.load_new_model('csv_uploads/'+request.csv_path, request.table_nm, request.model_name, request.n_epochs)
        return {"message": "New model loaded successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid data. Uploaded data should be a valid dataset for this model")


# Создаем директорию для сохранения загруженных файлов, если она не существует
os.makedirs("csv_uploads", exist_ok=True)

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)) -> dict:
    """
    Загружает CSV-файл и сохраняет его на сервере.

    Args:
        file (UploadFile): Загружаемый файл CSV.

    Возвращает:
        dict: Ответ с подтверждением успешной загрузки и именем файла.
    
    Raises:
        HTTPException: При ошибке сохранения файла.
    """
    try:
        file_location = f"csv_uploads/{file.filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(await file.read())
        return {"message": f"CSV file '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload the CSV file: {str(e)}")


# Запуск сервера
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)