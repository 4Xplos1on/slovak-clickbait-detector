from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# We load the model from local folder
tokenizer = AutoTokenizer.from_pretrained("./best_model")
model = AutoModelForSequenceClassification.from_pretrained("./best_model")
model.eval() # Tell it to only use, not train the model

# What we recieve is going to be in that exact structure
class HeadlineRequest(BaseModel):
    headline: str

# Same strucure definition but for our response
class PredictionResponse(BaseModel):
    headline: str
    label: str
    confidence: float



# Now the predict function wrapped in a route
@app.post("/predict", response_model=PredictionResponse)
def predict(request: HeadlineRequest):
    headline = request.headline

    # Tokenize
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    # Run the model, and unpack the dictionary with **
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits to normal probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    # Label
    if predicted_class == 1:
        label = "CLICKBAIT"
    else:
        label = "LEGIT"

    # Return the Labeled evaluation
    return PredictionResponse(
        headline=headline,
        label=label,
        confidence=round(confidence, 4)
    )

# When someone opens "http://127.0.0.1:8000/" run this function
# So the brower gets our html file
@app.get("/")
def serve_frontend():
    file = open("index.html", "r", encoding="utf-8")
    html_content = file.read()
    file.close()
    return HTMLResponse(content=html_content)
