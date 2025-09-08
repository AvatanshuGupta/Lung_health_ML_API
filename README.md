
#  Lung Health Prediction - End-to-End MLOps Pipeline with FastAPI, GitHub Actions & Render Deployment

##  Project Overview

This project is a **fully automated MLOps pipeline** for predicting Lung health status based on patient data. It includes:

-  **Automated data ingestion** from a PostgreSQL database  
-  **Model training and hyperparameter tuning** using multiple classification algorithms  
-  **Best model serialization and tracking**
-  **FastAPI** for real-time prediction serving
-  **CI/CD with GitHub Actions** for automatic retraining and deployment
-  **Deployed on Render**

---

## Checkout the live API on Render

[RENDER_API_LINK](https://lung-health-ml-api.onrender.com)


---

##  Features

###  Data Ingestion

- Connects to PostgreSQL using SQLAlchemy
- Reads the `diabetes` table
- Splits data into train/test
- Saves CSVs to `artifacts/`

###  Model Training

- Trains and tunes multiple models using `GridSearchCV`:
  - Logistic Regression
  - Random Forest
  - Naive Bayes
  - KNN, SVM, MLP, etc.
- Selects best model based on test accuracy
- Saves best model as `artifacts/best_model.pkl`
- Logs metrics and model parameters
- Saves best model info in `best_model_info.csv`

###  FastAPI Server

- `/` → Welcome message  
- `/predict` → Accepts patient input via POST request and returns prediction

```json
POST /predict
{
  "GENDER": 0,
  "AGE": 0,
  "SMOKING": 1,
  "YELLOW_FINGERS": 1,
  "ANXIETY": 1,
  "FATIGUE": 1,
  "ALLERGY": 1,
  "WHEEZING": 1,
  "ALCOHOL_CONSUMING": 1,
  "COUGHING": 1,
  "SHORTNESS_OF_BREATH": 1,
  "SWALLOWING_DIFFICULTY": 1,
  "CHEST_PAIN": 1
}
```

Returns:

```json
{
  "prediction": 1
}
```

---

##  CI/CD: GitHub Actions

### `Retrain Model` Workflow

A custom GitHub Actions workflow is triggered manually (`workflow_dispatch`) that:

1. Sets up Python and installs dependencies
2. Loads `DATABASE_URL` from secrets
3. Runs:
   - `data_ingestion_pipeline.py`
   - `training_pipeline.py`
4. Commits the updated `best_model.pkl` (if changed)
5. Triggers a **Render redeployment hook** using `curl`

```yaml
name: Retrain Model

on:
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Set PYTHONPATH
      run: echo "PYTHONPATH=$PYTHONPATH:." >> $GITHUB_ENV

    - name: Export DATABASE_URL
      run: echo "DATABASE_URL=${{ secrets.DATABASE_URL }}" >> $GITHUB_ENV

    - name: Run data ingestion pipeline
      run: python -m src.pipeline.data_ingestion_pipeline

    - name: Run training pipeline
      run: python -m src.pipeline.training_pipeline

    - name: Commit new model
      run: |
        git config user.name github-actions
        git config user.email github-actions@github.com
        git add artifacts/best_model.pkl
        git commit -m "Retrained model via GitHub Actions" || echo "No changes to commit"
        git push https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }} HEAD:main

    - name: Trigger Render deployment
      run: |
        RENDER_URL="${{ secrets.RENDER_DEPLOY_HOOK }}"
        CLEAN_URL=$(echo "$RENDER_URL" | tr -d '[:space:]')
        echo "Deploying to Render at $CLEAN_URL"
        curl -X POST -H "Content-Type: application/json" -d '{}' "$CLEAN_URL"
```

---

##  Environment Variables

Use a `.env` file at the root level:

```
DATABASE_URL=postgresql://username:password@host:port/dbname
```

Load using:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

##  Requirements

```bash
pip install -r requirements.txt
```

**Sample `requirements.txt`**:

```
pandas
sqlalchemy
psycopg2-binary
scikit-learn
fastapi
uvicorn
python-dotenv
```

---

##  Run Locally

###  Start FastAPI Server

```bash
uvicorn main:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI

---

##  Example Output

```text
Training and tuning LogisticRegression...
Best parameters for LogisticRegression: {'C': 1, 'solver': 'liblinear'}
Test accuracy: 0.85

Best model 'LogisticRegression' saved to artifacts/best_model.pkl
Saved best model info to artifacts/best_model_info.csv
```

---

##  Tech Stack

- **Python**
- **Scikit-learn**
- **FastAPI**
- **SQLAlchemy**
- **PostgreSQL**
- **Render (Cloud Hosting)**
- **GitHub Actions (CI/CD)**

---


