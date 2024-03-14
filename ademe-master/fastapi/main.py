"""
run with
> uvicorn script/main:app --reload
"""
import typing as t
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import os
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8088")
mlflow.set_experiment("dpe_forever")

import pandas as pd

# start fast API app
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/run_id")
def get_run_id():
    runs = mlflow.search_runs(
        experiment_ids=[2], order_by=["metrics.best_cv_score desc"]
    )
    best_run = runs.head(1).to_dict(orient="records")[0]
    print(best_run)
    return best_run["run_id"]


@app.get("/model")
def get_model():
    run_id = get_run_id()
    model_uri = f"runs:/{run_id}/best_estimator"
    print(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    return model


@app.get("/sample_data")
def load_data(filename: str = "sample.json") -> t.Dict:
    print("loading sample data from ", filename)
    data = pd.read_json(filename)
    return data.to_dict(orient="records")[0]


class PredictionInput(BaseModel):
    # Example with a single feature for simplicity.
    conso_kwhep_m2_an: float


# post query to endpoint
@app.post("/predict/")
def predict(input: PredictionInput):
    model = get_model()

    data = load_data()
    data.update(input)
    print(data)
    prediction = model.predict(data)
    return {"prediction": int(prediction)}
