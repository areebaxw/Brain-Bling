# Brain Bling - AI Reading Comprehension System

An intelligent multiple-choice question answering system built on the RACE dataset with ML-powered answer verification, distractor generation, and hint generation.

## Project Overview

**QuizMind AI** (Brain Bling) is a reading comprehension system with two main components:

- **Model A**: Traditional ML classifiers (Logistic Regression, SVM, Naive Bayes, Random Forest) for answer verification
- **Model B**: Distractor and hint generation for question enhancement

## Architecture

- **Frontend**: React + Vite (neobrutalist design)
- **Backend**: FastAPI (Python) serving ML models
- **ML Models**: scikit-learn based classifiers
- **Features**: TF-IDF, One-Hot Encoding, handcrafted lexical features

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install API dependencies:
```bash
pip install -r api/requirements.txt
```

3. Start the FastAPI backend:
```bash
cd api
python app.py
```
The API will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```
The React app will run on `http://localhost:3000`

## Running the Project

### Option 1: Full Stack (Recommended)

Run both backend and frontend:
```bash
# Terminal 1 - Backend
cd api
python app.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Option 2: Frontend Only (Demo Mode)

The React frontend includes sample data and works in demo mode without the backend.

## Project Structure

```
Brain-Bling/
├── api/                    # FastAPI backend
│   ├── app.py             # API endpoints
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   └── main.jsx      # Entry point
│   ├── index.html        # HTML template
│   ├── package.json      # Node dependencies
│   └── vite.config.js    # Vite configuration
├── src/                   # ML model training scripts
│   ├── preprocessing.py   # Data preprocessing
│   ├── model_a_train.py   # Model A training
│   ├── model_b_generator.py # Model B (distractors/hints)
│   └── main.py           # Baseline model
├── models/                # Trained model checkpoints
├── data/                  # Dataset files
│   ├── raw/              # RACE dataset (train.csv, dev.csv, test.csv)
│   └── processed/        # Preprocessed features
└── ui/                    # Deprecated Streamlit UI
    └── app.py.deprecated # Old Streamlit interface
```

## API Endpoints

- `GET /` - API status
- `GET /health` - Health check and model loading status
- `GET /metrics` - Model performance metrics
- `POST /api/generate-question` - Generate question from article
- `POST /api/generate-distractors` - Generate distractors
- `POST /api/generate-hints` - Generate hints
- `POST /api/check-answer` - Verify answer
- `GET /api/sample-article` - Get sample RACE article

## Model Performance

Based on trained model metrics:

| Model | Accuracy | Precision | Recall | Macro F1 |
|-------|----------|-----------|--------|----------|
| SVM | 67.41% | 0.30 | 0.22 | 0.52 |
| Logistic Regression | 67.25% | 0.29 | 0.22 | 0.52 |
| Naive Bayes | 62.18% | 0.25 | 0.26 | 0.50 |
| Random Forest | 59.32% | 0.28 | 0.38 | 0.52 |

## Data Requirements

The project expects RACE dataset CSV files in `data/raw/`:
- `train.csv`
- `dev.csv`
- `test.csv`

Each CSV should have columns: `id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`

## Training Models

To train the ML models:

```bash
# Preprocess data
python src/preprocessing.py

# Train Model A
python src/model_a_train.py

# Train Model B
python src/model_b_generator.py
```

## Migration Notes

- **Streamlit UI has been deprecated** (moved to `ui/app.py.deprecated`)
- New React frontend provides better UX and modern design
- FastAPI backend enables API-first architecture
- Frontend can work independently in demo mode

## License

This project uses the RACE dataset for educational purposes.
