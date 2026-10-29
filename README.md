# Book Recommender

An intelligent book recommendation system built with Streamlit, combining popularity-based filtering and item-based collaborative filtering using cosine similarity. Features an interactive dashboard, hybrid scoring, and beautiful UI.

## 📚 Features

- **Smart Recommendations** - Hybrid scoring combining collaborative filtering and popularity
- **Interactive Dashboard** - KPIs, trends, and analytics at a glance
- **Book Explorer** - Deep metadata view for any title
- **Discover For You** - Randomized suggestions to break out of your bubble
- **My Library** - Build and manage your reading list
- **Advanced Filtering & Search** - Find exactly what you're looking for
- **Responsive UI** - Modern design with dark mode support
- **Pickle Caching** - Fast model loading

## 🛠️ Tech Stack

- **Frontend:** Streamlit with custom CSS/HTML
- **Data Processing:** Pandas & NumPy
- **Recommendation Engine:** Scikit-learn Cosine Similarity
- **Visualization:** Plotly
- **Caching:** Pickle-based model caching

## 📊 Dataset

Sourced from the Book-Crossing dataset containing:
- 271,360 books
- 278,858 users
- 1,149,780 explicit/implicit ratings

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/Bookrecommender.git
cd Bookrecommender
```

2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

3. Run the app
```bash
streamlit run main.py
```

4. Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
Bookrecommender-main/
├── main.py              # Main application
├── Books.csv            # Book metadata
├── Ratings.csv          # User ratings
├── Users.csv            # User information
├── model_cache.pkl      # Precomputed model cache
└── README.md
```

## 🧠 Recommendation Algorithm

The system uses **item-based collaborative filtering**:
1. Creates a user-item pivot table
2. Computes cosine similarity between books
3. Combines similarity scores with popularity weighting
4. Returns top N recommendations

## 📝 License

This project is for educational purposes.

## 👤 Author

Created with ❤️ by Book Recommender Team