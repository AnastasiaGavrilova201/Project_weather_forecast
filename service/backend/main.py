from typing import List, Optional
import asyncio
import os
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

from API_backend import API_Backend
from realtime_parser import TaskManager
from log import Logger

logger = Logger(__name__).get_logger()


main_db_table_name = 'test_realtime_10'
# Инициализация бэкенда
api_backend = API_Backend(main_db_table_name=main_db_table_name)


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


class LoadNewModelResponse(BaseModel):
    """Pydantic модель для ответа после загрузки новой модели."""
    message: str


class CSVContent(BaseModel):
    """Pydantic модель для представления содержимого CSV."""
    file_name: str
    content: str


class LoadNewModelRequest(BaseModel):
    """Pydantic модель для запроса на загрузку новой модели."""
    csv_path: Optional[str] = None
    table_nm: Optional[str] = 'test_realtime_6'
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/set_model", response_model=SetActiveModelResponse)
async def set_active_model(request: SetActiveModelRequest) -> SetActiveModelResponse:
    """Устанавливает активную модель по имени."""
    try:
        api_backend.set_active(request.model_name)
        return {"message": f"Active model set to {request.model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/fit", response_model=FitModelResponse)
async def fit_model() -> FitModelResponse:
    """Запускает обучение активной модели."""
    try:
        async def train():
            api_backend.fit()

        await asyncio.wait_for(train(), timeout=10.0)
        return {"message": "Model training completed"}
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail="Model training took too long and was aborted") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Возвращает предсказания от активной модели."""
    try:
        predictions = api_backend.predict(request.start_time)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/load_new_model", response_model=LoadNewModelResponse)
async def load_new_model(request: LoadNewModelRequest) -> LoadNewModelResponse:
    """
    Загружает вручную заданное содержимое CSV в базу данных и создает модель.
    """
    try:
        if request.csv_path is not None:
            csv_path = 'csv_uploads/' + request.csv_path
        else:
            csv_path = None
        api_backend.load_new_model(
            csv_path,
            request.table_nm,
            request.model_name,
            request.n_epochs)
        return {"message": "New model loaded successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Invalid data. Error: {str(e)}") from e


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
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are accepted.")
        file_location = f"csv_uploads/{file.filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(await file.read())
        return {"message": f"CSV file '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload the CSV file: {str(e)}") from e


# Запуск сервера
if __name__ == "__main__":
    realtime = TaskManager(main_db_table_name)
    realtime.start()
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except RuntimeError:
        logger.error('RuntimeError %s', traceback.format_exc())
    except KeyboardInterrupt:
        logger.error('KeyboardInterrupt %s', traceback.format_exc())
    finally:
        realtime.finish(timeout=5)
