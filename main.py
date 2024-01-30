from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pipeline import FakeNewsPipeline
app = FastAPI()

# Serve static files (CSS and JS)
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Initialize the FakeNewsPipeline
pipeline = FakeNewsPipeline(model_path='model.pkl', tfidf_path='tfidf_vectorizer.pkl')

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Define a Pydantic model for the input data
class NewsInput(BaseModel):
    news: str

# Define the HTML endpoint
@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("templates/index.html")

# Define an endpoint for predicting fake news
@app.post("/predict")
async def predict_fake_news(news_input: NewsInput):
    try:
        # Make a prediction using the pipeline
        prediction_result = pipeline.predict_fake_news(news_input.news)

        # Get and return the result
        result_message = "FAKE News" if prediction_result == 0 else "REAL News"
        return {"prediction": result_message}
    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))
