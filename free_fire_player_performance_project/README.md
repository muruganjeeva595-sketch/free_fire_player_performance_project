# Game Player Performance Prediction Using Machine Learning
## Free Fire / Battle Royale themed MCA Data Science project

### Files
- `app.py` — Streamlit dashboard and predictor.
- `data/battle_royale_synthetic_matches.csv` — 3,000 clearly labelled synthetic match records.
- `requirements.txt` — Python dependencies.
- `Game_Player_Performance_Colab.ipynb` — step-by-step notebook for training and evaluation.
- `MCA_Project_Report.docx` — editable academic report.
- `MCA_Project_Presentation.pptx` — presentation slides.

### Run locally
1. Install Python 3.10 or newer.
2. Extract the ZIP file and open a terminal in the project folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

### Run in Google Colab
Open `Game_Player_Performance_Colab.ipynb` in Google Colab and run cells from top to bottom. The notebook installs requirements, loads the CSV, trains a Random Forest classifier, reports metrics, and writes `app.py`. The last cell provides an optional localtunnel command to preview Streamlit. Public tunnel services may change or rate-limit access.

### Data and research integrity
The bundled records and labels are synthetic. The performance tier is created from a transparent composite score for demonstration. Test accuracy therefore only evaluates this synthetic setup and must not be presented as real-world Free Fire accuracy. For a stronger research project, use an appropriately licensed real match dataset, document its source, define a target independent of the input features where possible, and evaluate on a held-out set.

### Suggested target for real data
Predict a measurable future outcome such as top-10 finish, match win, or next-match performance using prior-match features. Avoid using post-match information to predict an outcome that would not yet be known at prediction time.
