import logging
from io import StringIO

import pandas as pd
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas import Prediction, Text


router = APIRouter(prefix="/items", tags=["items"])


@router.post("/predict_text", response_model=Prediction)
def predict_text(text: Text, request: Request):
    """
    Function predicts author of the text.

    :param text: Text
    :param request: request to access app state
    :return: predicted label and corresponding author name
    """
    df = pd.Series(text.text)
    label, name = request.app.state.model.predict(df, return_names=True).flatten()
    logging.info("Model made prediction")
    response = Prediction(label=label, name=name)
    return response


@router.post("/predict_texts", response_class=StreamingResponse)
def predict_texts(file: UploadFile, request: Request):
    """
    Function predicts authors of texts in csv-file.

    :param file: csv-file with texts
    :param request: request to access app state
    :return: csv-file with texts and predictions
    """
    df = pd.read_csv(file.file).squeeze("columns")
    prediction = request.app.state.model.predict(
        df, return_labels=False, return_names=True
    )
    logging.info("Model made prediction")
    df = df.to_frame()
    df["author"] = prediction
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    response = StreamingResponse(
        content=iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file.filename}"},
    )
    return response
