# 📰 ClassiNews - News Article Classification System

**ClassiNews** is an AI-powered web application that automatically classifies news articles into **5 categories**: **Sports**, **Technology**, **Politics**, **Finance**, and **Entertainment**.

## 🎯 Features

- ✅ **High Accuracy** - 92% accuracy trained on BBC News dataset
- ✅ **5 Categories** - Focused news classification
- ✅ **Modern Dashboard** - Beautiful, colorful UI with real-time stats
- ✅ **Real-time Classification** - Instant category prediction with confidence scores
- ✅ **RESTful API** - FastAPI backend with batch prediction support
- ✅ **Category Filtering** - Filter articles by category
- ✅ **Article Management** - Save, view, and delete classified articles
- ✅ **Sample Articles** - Pre-loaded examples for testing

## 📁 Project Structure

```
ClassiNews/
├── backend/
│   ├── app.py              # FastAPI server with 5 categories
│   ├── model.pkl           # Trained ML model
│   ├── preprocess.pkl      # TF-IDF vectorizer
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── dashboard.html      # Main dashboard UI
│   ├── dashboard.js        # Dashboard functionality
│   ├── dashboard.css       # Dashboard styling
│   ├── dashboard-fixes.css # Additional UI fixes
│   ├── index.html          # Simple classifier UI
│   ├── style.css           # Simple UI styling
│   └── script.js           # API integration
├── notebooks/
│   ├── news_classification_model.ipynb  # Original training notebook
│   └── ag_news_training.py  # AG News training script
└── README.md               # This file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Ensure Model Files Exist

The trained model files should be in the `backend/` directory:

- `model.pkl` - Trained Multinomial Naive Bayes model
- `preprocess.pkl` - TF-IDF vectorizer

**Note:** If you need to train the model, see the "Training the Model" section below.

### Step 3: Start the Backend Server

```bash
cd backend
python app.py
```

The server will start on `http://localhost:8002`

### Step 4: Open the Dashboard

Open `frontend/dashboard.html` in your browser to access the main dashboard.

## 📓 Training the Model

### Using AG News Dataset with BBC Fallback

1. **Option 1:** Download AG News dataset from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
   - Place `ag_news.csv` in the `notebooks/` directory

2. **Option 2:** Use BBC News dataset
   - Place `bbc_articles.csv` in the `notebooks/` directory

3. Run the training script:
   ```bash
   cd notebooks
   python ag_news_training.py
   ```

4. The script will:
   - Load the dataset (tries AG News first, falls back to BBC)
   - Preprocess the text (stemming, stop word removal)
   - Train a Multinomial Naive Bayes classifier
   - Save `model.pkl` and `preprocess.pkl` to `backend/`

### Using Jupyter Notebook

1. Open `notebooks/news_classification_model.ipynb`
2. Run all cells
3. Model files will be saved to `backend/`

## 🔌 API Endpoints

### POST /predict

Classify a single news article.

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "category": "Technology",
  "confidence": 95.5
}
```

### POST /predict/batch

Classify multiple news articles.

**Request:**
```json
{
  "articles": [
    {"text": "Article 1..."},
    {"text": "Article 2..."}
  ]
}
```

### GET /articles/sample

Get 10 sample pre-classified articles for dashboard demo.

**Response:**
```json
{
  "articles": [
    {
      "text": "Sample article text...",
      "category": "Politics",
      "confidence": 85.2
    }
  ]
}
```

### POST /articles/save

Save a classified article.

**Request:**
```json
{
  "id": "unique-id",
  "text": "Article text...",
  "category": "Sports",
  "confidence": 92.3,
  "timestamp": "2025-12-02T09:30:00"
}
```

### GET /articles/saved

Retrieve all saved articles.

### DELETE /articles/saved/{id}

Delete a specific saved article by ID.

### GET /health

Check API health status.

### GET /

API information and available endpoints.

## 🎯 Categories (5 Categories)

The model classifies articles into:

1. **🏆 Sports** - Sports events, athletes, competitions, games
2. **💻 Technology** - Tech news, innovations, gadgets, software
3. **🏛️ Politics** - Government, elections, policies, diplomacy
4. **💰 Finance** - Markets, companies, economy, business
5. **🎬 Entertainment** - Movies, music, celebrities, shows

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **scikit-learn** - Machine learning (Multinomial Naive Bayes)
- **joblib** - Model serialization
- **NLTK** - Natural language processing (stemming, stop words)
- **Uvicorn** - ASGI server

### Frontend
- **HTML5** - Structure
- **CSS3** - Modern vibrant pastel styling with gradients
- **JavaScript** - API integration and dynamic UI

### Machine Learning
- **Algorithm:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF (1-2 ngrams)
- **Preprocessing:** Stemming + stop word removal
- **Training Data:** 50K+ news articles (BBC/AG News)
- **Accuracy:** 92%

## 🎨 Dashboard Features

### Classification Form
- Paste article content
- Click "Classify & Save" button
- View results in a beautiful modal with category and confidence

### Quick Actions
- **Load Sample Articles** - Instantly populate dashboard with 10 pre-classified examples
- **Clear All Articles** - Remove all articles from the dashboard

### Category Breakdown
- Real-time count of articles per category
- Colored category badges with icons

### Recent Articles Grid
- Filter by category (dropdown)
- View article cards with:
  - Category badge with colored square indicator
  - Article snippet
  - Confidence percentage
  - Timestamp
  - Delete button

### Stats Bar
- Model accuracy (92%)
- Total categories (5)
- Total articles count (dynamic)
- Average processing time (1s)

## ❌ Troubleshooting

### Backend won't start?
- Check if `model.pkl` and `preprocess.pkl` exist in `backend/` directory
- Install dependencies: `pip install -r requirements.txt`
- Check if port 8002 is available
- Try: `python app.py` instead of `python3 app.py`

### Frontend can't connect?
- Make sure backend is running on http://localhost:8002
- Check browser console (F12) for errors
- Verify CORS is enabled in `app.py`

### Model not found?
- Ensure `model.pkl` and `preprocess.pkl` are in the `backend/` directory
- If training from notebook, check that files were saved correctly
- Try running the training script: `python ag_news_training.py`

### Dashboard not loading properly?
- Make sure both `dashboard.css` and `dashboard-fixes.css` are in the `frontend/` directory
- Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
- Check browser console for JavaScript errors

## 📝 Example Usage

### Dashboard Mode (Recommended):
1. Open `http://localhost:8002` or `frontend/dashboard.html` in your browser
2. Click **"Load Sample Articles"** to see 10 pre-classified examples
3. Add new articles by pasting text and clicking **"Classify & Save"**
4. View classification results in the modal (category + confidence)
5. Click **"Save Article"** to add to dashboard
6. Use the **category filter** dropdown to filter by type
7. Delete articles using the trash icon

**Try these examples:**
- "Apple announces new AI-powered chip for smartphones" → **Technology** (95%+)
- "The local football team won the championship after a thrilling final" → **Sports** (92%+)
- "Stock markets reached all-time high as investors show confidence" → **Finance** (90%+)
- "Government announces new policies to boost economic growth" → **Politics** (93%+)
- "Latest blockbuster movie breaks box office records worldwide" → **Entertainment** (91%+)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- BBC News Dataset
- AG News Dataset
- scikit-learn community
- FastAPI framework

---

**Happy Classifying! 📰✨**