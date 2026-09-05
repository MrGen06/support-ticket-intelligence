# AI Support Ticket Intelligence

An AI-assisted support ticket triage and response system. It predicts a ticket's queue and priority, estimates the SLA deadline, and drafts a response for a support agent to review.

The application combines deterministic machine-learning inference with a generative assistant. Final responses remain human-approved.

## Features

- Support queue classification
- Ticket priority prediction
- SLA deadline estimation
- Ticket summary and response drafting
- Machine-learning inference latency tracking
- FastAPI backend
- Streamlit dashboard

## Architecture

1. **Classification layer:** A DistilBERT model provides the primary queue prediction. A scikit-learn TF-IDF baseline is also included for comparison.
2. **SLA layer:** The predicted priority is converted into an SLA deadline.
3. **Generative layer:** Qwen is called through the Groq API to generate a concise summary and response draft.
4. **User layer:** The FastAPI service exposes the prediction endpoint, and the Streamlit dashboard consumes it.

## Models

- **Baseline:** TF-IDF vectorizer with scikit-learn classifiers
- **Primary classifier:** DistilBERT with PyTorch and Transformers
- **LLM assistant:** Qwen through the Groq API

## Requirements

- Python 3.10 or later
- A Groq API key for response generation

## Local Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
```

Start the API in one terminal:

```powershell
uvicorn api.main:app --reload
```

Start the dashboard in a second terminal:

```powershell
streamlit run app/app.py
```

The API is available at `http://localhost:8000`. The dashboard URL is printed by Streamlit when it starts.

## API

### Health check

```http
GET /health
```

Example response:

```json
{
	"status": "healthy",
	"model": "distilbert"
}
```

### Predict and draft a response

```http
POST /predict
Content-Type: application/json
```

Request body:

```json
{
	"subject": "Unable to sign in",
	"body": "The password reset link returns an error."
}
```

The response includes the predicted queue, confidence, priority, SLA information, response draft, and model latency.

## Project Structure

```text
api/                    FastAPI application
app/                    Streamlit dashboard
configs/                Configuration files
data/                   Raw and processed ticket data
models/baseline/       Scikit-learn artifacts
models/distilbert/     DistilBERT artifacts
src/                    Inference, SLA, LLM, and training code
tests/                  Test suite
```

## Training and Deployment

The models were trained using the scripts in `src/models/` and the training notebook in `notebooks/training/`. GPU training was performed in a Kaggle environment.

The intended deployment target is Azure Container Apps. Docker deployment files are present in the repository but are not configured yet.

## Evaluation

| Task | Model | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Queue classification | scikit-learn baseline | 48.71% | 36.94% |
| Queue classification | DistilBERT | 59.66% | 54.92% |
| Priority classification | scikit-learn baseline | 52.13% | 46.98% |

## Limitations

- The dataset is synthetic.
- Support queues have significant semantic overlap.
- LLM-generated responses require human review.
- Groq API access is required for response generation.