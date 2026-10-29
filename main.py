import os
import zipfile
import pickle
import random
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= CUSTOM CSS (PREMIUM AESTHETICS) =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-banner {
        background: radial-gradient(circle at top left, #4f46e5 0%, #312e81 45%, #1e1b4b 100%);
        border-radius: 24px;
        padding: 2.6rem 3rem;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 40px 60px -40px rgba(79, 70, 229, 0.9);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.16);
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(129, 140, 248, 0.25), transparent 55%);
        pointer-events: none;
    }
    .hero-banner::after {
        content: "📚";
        font-size: 9rem;
        position: absolute;
        right: -10px;
        bottom: -40px;
        opacity: 0.1;
        pointer-events: none;
    }
    .hero-content {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.2rem, 4vw, 3.6rem);
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: clamp(1rem, 1.4vw, 1.15rem);
        color: #e0e7ff;
        max-width: 780px;
        line-height: 1.8;
        margin: 0;
    }
    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-top: 1.2rem;
    }
    .hero-pill {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(90deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.12) 100%);
        backdrop-filter: blur(12px);
        padding: 0.55rem 1.1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid rgba(255, 255, 255, 0.28);
        box-shadow: 0 20px 40px -30px rgba(255, 255, 255, 0.9);
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin: -1rem 0 2rem 0;
    }
    .kpi-card {
        position: relative;
        background: radial-gradient(circle at top, rgba(99, 102, 241, 0.12), transparent 70%), #ffffff;
        border-radius: 18px;
        border: 1px solid rgba(99, 102, 241, 0.18);
        padding: 1.6rem;
        box-shadow: 0 30px 50px -40px rgba(79, 70, 229, 0.8);
        overflow: hidden;
    }
    .kpi-card::after {
        content: "";
        position: absolute;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(129, 140, 248, 0.18) 0%, transparent 70%);
        right: -60px;
        top: -80px;
    }
    .kpi-label {
        position: relative;
        z-index: 2;
        font-size: 0.95rem;
        color: #475569;
        font-weight: 600;
    }
    .kpi-value {
        position: relative;
        z-index: 2;
        font-size: clamp(2.2rem, 4vw, 3.2rem);
        font-weight: 800;
        margin-top: 0.4rem;
        color: #1e1b4b;
    }
    .kpi-footnote {
        position: relative;
        z-index: 2;
        margin-top: 0.6rem;
        color: #6366f1;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Book Card */
    .book-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15, 23, 42, 0.06);
        border-radius: 18px;
        padding: 1rem 1rem 1.2rem 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
        transition: all 0.25s ease;
        box-shadow: 0 30px 60px -40px rgba(15, 23, 42, 0.9);
        position: relative;
        overflow: hidden;
    }
    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 40px 70px -40px rgba(79, 70, 229, 0.6);
        border-color: rgba(99, 102, 241, 0.8);
    }
    .book-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top, rgba(129, 140, 248, 0.08), transparent 60%);
        pointer-events: none;
    }
    .book-card-content {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
        height: 100%;
    }
    .rank-badge {
        position: absolute;
        top: 14px;
        left: 14px;
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        color: #ffffff;
        font-weight: 800;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 9999px;
        box-shadow: 0 15px 30px -20px rgba(249, 115, 22, 0.9);
        z-index: 3;
    }
    .book-img-wrapper {
        position: relative;
        z-index: 2;
        width: 100%;
        height: 250px;
        border-radius: 14px;
        overflow: hidden;
        background: radial-gradient(circle, #e2e8f0 0%, #cbd5f5 100%);
    }
    .book-img-wrapper img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .book-card:hover .book-img-wrapper img {
        transform: scale(1.05);
    }
    .book-text {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }
    .book-title {
        font-weight: 700;
        font-size: 1rem;
        color: #0f172a;
        line-height: 1.6;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 3.2rem;
    }
    .book-author {
        font-size: 0.92rem;
        color: #475569;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .book-meta {
        margin-top: auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
    }
    .rating-pill {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #b45309;
        font-weight: 700;
    }
    .count-pill {
        color: #475569;
        font-size: 0.9rem;
    }
    .match-pill {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #059669;
        font-weight: 700;
    }

    /* Target Banner */
    .target-banner {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 1.4rem;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.16) 0%, rgba(59, 130, 246, 0.12) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 1.6rem 2rem;
        margin: 1.6rem 0;
        box-shadow: 0 40px 60px -40px rgba(16, 185, 129, 0.6);
    }
    .target-banner img {
        width: 78px;
        height: 110px;
        object-fit: cover;
        border-radius: 12px;
        box-shadow: 0 30px 40px -30px rgba(15, 23, 42, 0.9);
    }
    .target-text {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }
    .target-kicker {
        text-transform: uppercase;
        letter-spacing: 0.5rem;
        font-size: 0.8rem;
        font-weight: 700;
        color: #047857;
    }
    .target-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(1.6rem, 3vw, 2.6rem);
        font-weight: 800;
        color: #064e3b;
    }
    .target-author {
        font-size: 1rem;
        color: #065f46;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: radial-gradient(circle at top, #eef2ff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.18);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #312e81;
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background: radial-gradient(circle at top, rgba(129, 140, 248, 0.24), transparent 70%), #1e293b;
            border-color: rgba(129, 140, 248, 0.4);
        }
        .kpi-label { color: #cbd5f5; }
        .kpi-value { color: #f8fafc; }
        .kpi-footnote { color: #a5b4fc; }
        .book-card {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-color: rgba(148, 163, 184, 0.22);
        }
        .book-title { color: #f8fafc; }
        .book-author { color: #cbd5f5; }
        .book-meta { border-top-color: rgba(148, 163, 184, 0.32); }
        .count-pill { color: #cbd5f5; }
        .target-banner {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.28) 0%, rgba(59, 130, 246, 0.22) 100%);
            border-color: rgba(52, 211, 153, 0.6);
        }
        .target-kicker, .target-title, .target-author { color: #ecfdf5; }
        [data-testid="stSidebar"] {
            background: radial-gradient(circle at top, #1e1b4b 0%, #0f172a 100%);
            border-right-color: rgba(129, 140, 248, 0.4);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown p {
            color: #e0e7ff;
        }
    }
</style>
""", unsafe_allow_html=True)


# ================= DATA LOADER & CACHING =================
def resolve_dataset_files():
    """Ensure Books.csv and Ratings.csv exist, extracting from zip if needed."""

    def find_file(names):
        for n in names:
            if os.path.exists(n):
                return n
        return None

    books_path = find_file(['Books.csv', 'books.csv'])
    ratings_path = find_file(['Ratings.csv', 'ratings.csv'])

    if not books_path or not ratings_path:
        zip_candidates = [f for f in os.listdir('.') if f.endswith('.zip')]
        for z in zip_candidates:
            try:
                with zipfile.ZipFile(z, 'r') as zip_ref:
                    zip_names = zip_ref.namelist()
                    for target in ['Books.csv', 'Ratings.csv', 'Users.csv']:
                        matching = [n for n in zip_names if n.lower() == target.lower()]
                        if matching and not os.path.exists(matching[0].split('/')[-1]):
                            zip_ref.extract(matching[0], '.')
            except Exception:
                pass
        books_path = find_file(['Books.csv', 'books.csv'])
        ratings_path = find_file(['Ratings.csv', 'ratings.csv'])

    return books_path, ratings_path


@st.cache_resource(show_spinner="⚡ Crafting your intelligent recommendation engine...")
def load_and_prepare_data():
    cache_file = "model_cache.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                return (
                    data['books_lookup'],
                    data['final_df'],
                    data['pt'],
                    data['similarity_score'],
                    data.get('ratings_with_name'),
                    data.get('full_books'),
                )
        except Exception:
            pass

    books_path, ratings_path = resolve_dataset_files()
    if not books_path or not ratings_path:
        st.error("⚠️ Dataset files (Books.csv / Ratings.csv) not found in the workspace.")
        st.stop()

    books = pd.read_csv(books_path, low_memory=False)
    ratings = pd.read_csv(ratings_path)

    full_books = books.copy()

    # Clean and merge
    ratings_with_name = ratings.merge(books, on='ISBN')
    ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
    ratings_with_name = ratings_with_name.dropna(subset=['Book-Rating'])

    # Popularity based
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(75)

    final_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
        ['Book-Title', 'Book-Author', 'Image-URL-M', 'Image-URL-L', 'num_ratings', 'avg_rating', 'ISBN', 'Year-Of-Publication', 'Publisher']
    ]

    # Collaborative based
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    padhe_likhe_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    similarity_score = cosine_similarity(pt)

    books_lookup = books.drop_duplicates('Book-Title').set_index('Book-Title')[
        ['Book-Author', 'Image-URL-M', 'Image-URL-L', 'ISBN', 'Year-Of-Publication', 'Publisher']
    ]

    try:
        with open(cache_file, "wb") as f:
            pickle.dump({
                'books_lookup': books_lookup,
                'final_df': final_df,
                'pt': pt,
                'similarity_score': similarity_score,
                'ratings_with_name': ratings_with_name,
                'full_books': full_books,
            }, f)
    except Exception:
        pass

    return books_lookup, final_df, pt, similarity_score, ratings_with_name, full_books


books_lookup, final_df, pt, similarity_score, ratings_with_name, full_books = load_and_prepare_data()

# ================= SESSION STATE =================
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_target' not in st.session_state:
    st.session_state['current_target'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []


# ================= HELPER FUNCTIONS =================
def get_book_cover_url(isbn, fallback_url=None, high_res=False):
    clean_isbn = str(isbn).strip() if pd.notna(isbn) else ""
    # Try dataset image URL first (convert http to https)
    if fallback_url and isinstance(fallback_url, str) and fallback_url.strip():
        url = fallback_url.strip()
        if url.startswith('http://'):
            url = url.replace('http://', 'https://', 1)
        elif url.startswith('//'):
            url = 'https:' + url
        if url.startswith('https://') and 'no-image' not in url.lower():
            return url
    # Try Google Books by ISBN as fallback
    if clean_isbn and clean_isbn.lower() != 'nan' and clean_isbn != '':
        zoom = 2 if high_res else 1
        return f"https://books.google.com/books/content?vid=isbn{clean_isbn}&printsec=frontcover&img=1&zoom={zoom}"
    return "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80"


def render_book_card(title, author, image_url, rank=None, avg_rating=None, num_ratings=None, sim_score=None):
    rank_html = f'<div class="rank-badge">#{rank}</div>' if rank is not None else ''
    if avg_rating is not None and num_ratings is not None:
        meta_html = (
            '<div class="book-meta">'
            f'<div class="rating-pill">★ {avg_rating:.2f}</div>'
            f'<div class="count-pill">{num_ratings:,} ratings</div>'
            '</div>'
        )
    elif sim_score is not None:
        match_pct = max(0, min(100, sim_score * 100))
        meta_html = (
            '<div class="book-meta">'
            f'<div class="match-pill">⚡ {match_pct:.0f}% Match</div>'
            '<div class="count-pill">Collaborative</div>'
            '</div>'
        )
    else:
        meta_html = '<div class="book-meta" style="border-top:none; padding-top:0;"></div>'

    safe_title = str(title).replace('"', '&quot;')
    safe_author = str(author).replace('"', '&quot;')

    card_html = (
        '<div class="book-card">'
        f'{rank_html}'
        '<div class="book-card-content">'
        '<div class="book-img-wrapper">'
        f'<img src="{image_url}" alt="{safe_title}" loading="lazy" onerror="this.onerror=null;this.src=\'https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80\';">'
        '</div>'
        '<div class="book-text">'
        f'<div class="book-title" title="{safe_title}">{title}</div>'
        f'<div class="book-author">by {author}</div>'
        '</div>'
        f'{meta_html}'
        '</div>'
        '</div>'
    )
    return card_html


def recommend(book_name, top_n=6):
    if book_name not in pt.index:
        return []
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_score[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    suggestions = []
    for i in similar_items:
        similar_title = pt.index[i[0]]
        score = float(i[1])
        if similar_title in books_lookup.index:
            info = books_lookup.loc[similar_title]
            if isinstance(info, pd.DataFrame):
                info = info.iloc[0]
            isbn = info.get('ISBN', '')
            author = info.get('Book-Author', 'Unknown')
            fallback_img = info.get('Image-URL-M', '')
            image_url = get_book_cover_url(isbn, fallback_img)
            year = info.get('Year-Of-Publication', 'N/A')
            publisher = info.get('Publisher', 'N/A')
        else:
            author = "Unknown"
            image_url = get_book_cover_url("", None)
            year = "N/A"
            publisher = "N/A"
        suggestions.append({
            "title": similar_title,
            "author": author,
            "image": image_url,
            "score": score,
            "year": year,
            "publisher": publisher,
        })
    return suggestions


def hybrid_recommend(book_name, top_n=6, alpha=0.7):
    cf = recommend(book_name, top_n=top_n * 2)
    if not cf:
        return []
    for item in cf:
        avg_r = 0
        if item['title'] in final_df['Book-Title'].values:
            avg_r = final_df[final_df['Book-Title'] == item['title']]['avg_rating'].values[0]
        norm_cf = (item['score'] - 0) / 1.0
        norm_pop = (avg_r - 3.5) / 2.5
        item['hybrid_score'] = alpha * norm_cf + (1 - alpha) * norm_pop
    cf = sorted(cf, key=lambda x: x['hybrid_score'], reverse=True)[:top_n]
    return cf


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## Book Recommender")
    st.caption("Find books you might like")
    st.markdown("---")

    menu = [
        "Home",
        "Popular Books",
        "Recommendations",
        "Discover",
        "Browse",
        "Insights",
        "My Library",
        "About",
    ]
    if '_menu_choice' in st.session_state and st.session_state['_menu_choice'] in menu:
        default_idx = menu.index(st.session_state['_menu_choice'])
        del st.session_state['_menu_choice']
    else:
        default_idx = 0
    choice = st.radio("Navigate", menu, index=default_idx)

    st.markdown("---")
    st.markdown("### Engine Snapshot")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Catalog", "271K+")
    with col_b:
        st.metric("Ratings", "1.15M+")
    st.metric("Curated for Display", len(final_df))
    st.metric("Collaborative Space", f"{pt.shape[0]} × {pt.shape[1]}")

    st.markdown("---")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Built with Streamlit • Cosine Similarity • Hybrid Scoring")


# ================= HERO =================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-content">
        <h1 class="hero-title">Book Recommender</h1>
        <p class="hero-subtitle">
            Find your next read based on popularity and reader behavior.
        </p>
        <div class="hero-stats">
            <div class="hero-pill">Popularity based</div>
            <div class="hero-pill">Collaborative filtering</div>
            <div class="hero-pill">Simple and fast</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= DASHBOARD =================
if choice == "Home":
    st.markdown("## Overview")
    st.caption("Quick stats and top picks")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Books in Catalog</div>
            <div class="kpi-value">271,360</div>
            <div class="kpi-footnote">Comprehensive Goodreads-style dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">User Ratings Captured</div>
            <div class="kpi-value">1,149,780</div>
            <div class="kpi-footnote">Authentic reader feedback</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Vetted for Recommendations</div>
            <div class="kpi-value">{len(final_df)}</div>
            <div class="kpi-footnote">≥250 ratings, high signal</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Similarity Pairs Analyzed</div>
            <div class="kpi-value">{pt.shape[0]*pt.shape[0]:,}</div>
            <div class="kpi-footnote">Cosine vector space</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("Browse Top Books", use_container_width=True):
            st.session_state['jump'] = "Popular Books"
            st.rerun()
    with qa2:
        if st.button("Get Recommendations", use_container_width=True):
            st.session_state['jump'] = "Recommendations"
            st.rerun()
    with qa3:
        if st.button("Surprise Me", use_container_width=True):
            st.session_state['jump'] = "Discover"
            st.rerun()
    with qa4:
        if st.button("View Insights", use_container_width=True):
            st.session_state['jump'] = "Insights"
            st.rerun()

    st.markdown("### Spotlight: Top Rated Gems")
    spotlight = final_df.sort_values(['avg_rating', 'num_ratings'], ascending=[False, False]).head(8)
    cols = st.columns(4)
    for i, (_, row) in enumerate(spotlight.iterrows()):
        with cols[i % 4]:
            orig_rank = list(final_df['Book-Title']).index(row['Book-Title']) + 1
            img_url = get_book_cover_url(row.get('ISBN', ''), row['Image-URL-M'])
            st.markdown(render_book_card(row['Book-Title'], row['Book-Author'], img_url, rank=orig_rank, avg_rating=row['avg_rating'], num_ratings=row['num_ratings']), unsafe_allow_html=True)
            if st.button("➕ Add to Library", key=f"dash_add_{i}", use_container_width=True):
                if row['Book-Title'] not in st.session_state['favorites']:
                    st.session_state['favorites'].append(row['Book-Title'])
                    st.toast("Added to My Library!", icon="❤️")

    st.markdown("### Reading Trends")
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        top_authors = final_df['Book-Author'].value_counts().head(10)
        fig_auth = px.bar(top_authors, x=top_authors.values, y=top_authors.index, orientation='h', labels={'x': 'Number of Top Titles', 'y': 'Author'}, title="Most Featured Authors in Top List")
        fig_auth.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig_auth, use_container_width=True)
    with trend_col2:
        fig_dist = px.histogram(final_df, x='avg_rating', nbins=20, labels={'avg_rating': 'Average Rating'}, title="Distribution of Average Ratings")
        fig_dist.update_layout(height=420)
        st.plotly_chart(fig_dist, use_container_width=True)


# ================= POPULAR BOOKS =================
elif choice == "Popular Books":
    st.markdown("## Top Books")
    st.caption("Ranked by average rating with at least 250 reviews.")

    f1, f2, f3 = st.columns([2.5, 1, 1])
    with f1:
        search_query = st.text_input("Search titles or authors", placeholder="e.g. Tolkien, Stephen King, Austen...")
    with f2:
        min_star = st.slider("Minimum Rating", 3.5, 5.0, 4.0, 0.1)
    with f3:
        sort_by = st.selectbox("Sort By", ["Rank (Default)", "Highest Rated", "Most Reviews", "Title (A-Z)"])

    displayed_df = final_df[final_df['avg_rating'] >= min_star].copy()
    if search_query.strip():
        q = search_query.lower()
        displayed_df = displayed_df[
            displayed_df['Book-Title'].str.lower().str.contains(q, na=False) |
            displayed_df['Book-Author'].str.lower().str.contains(q, na=False)
        ]

    if sort_by == "Highest Rated":
        displayed_df = displayed_df.sort_values('avg_rating', ascending=False)
    elif sort_by == "Most Reviews":
        displayed_df = displayed_df.sort_values('num_ratings', ascending=False)
    elif sort_by == "Title (A-Z)":
        displayed_df = displayed_df.sort_values('Book-Title')

    st.info(f"Showing **{len(displayed_df)}** results")
    cols_per_row = 4
    for i in range(0, len(displayed_df), cols_per_row):
        grid = st.columns(cols_per_row)
        for j, col in enumerate(grid):
            if i + j < len(displayed_df):
                row = displayed_df.iloc[i + j]
                with col:
                    orig_rank = list(final_df['Book-Title']).index(row['Book-Title']) + 1
                    img_url = get_book_cover_url(row.get('ISBN', ''), row['Image-URL-M'])
                    st.markdown(render_book_card(row['Book-Title'], row['Book-Author'], img_url, rank=orig_rank, avg_rating=row['avg_rating'], num_ratings=row['num_ratings']), unsafe_allow_html=True)
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("Save", key=f"pop_save_{i}_{j}", use_container_width=True):
                            if row['Book-Title'] not in st.session_state['favorites']:
                                st.session_state['favorites'].append(row['Book-Title'])
                                st.toast("Saved to library")
                    with bc2:
                        if st.button("Recommend Similar", key=f"pop_rec_{i}_{j}", use_container_width=True):
                            st.session_state['current_target'] = row['Book-Title']
                            st.session_state['recommendations'] = hybrid_recommend(row['Book-Title'], top_n=6)
                            st.session_state['jump'] = "Recommendations"
                            st.rerun()


# ================= SMART RECOMMENDATIONS =================
elif choice == "Recommendations":
    st.markdown("## Recommendations")
    st.caption("Find books similar to one you like.")

    book_list = sorted(pt.index.tolist())
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        preferred_defaults = ["Harry Potter and the Chamber of Secrets (Book 2)", "1984", "The Fellowship of the Ring (The Lord of the Rings, Part 1)", "To Kill a Mockingbird"]
        default_idx = 0
        for p in preferred_defaults:
            if p in book_list:
                default_idx = book_list.index(p)
                break
        selected_book = st.selectbox("Select a book you enjoyed", book_list, index=default_idx)
    with c2:
        top_n = st.select_slider("Results", options=[4, 6, 8, 10], value=6)
    with c3:
        mode = st.selectbox("Strategy", ["Hybrid (Recommended)", "Pure Collaborative"])

    if selected_book in books_lookup.index:
        info = books_lookup.loc[selected_book]
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        sel_img = get_book_cover_url(info.get('ISBN', ''), info.get('Image-URL-M', ''), high_res=True)
        sel_author = info.get('Book-Author', 'Unknown')
        sel_year = info.get('Year-Of-Publication', 'N/A')
    else:
        sel_img = get_book_cover_url("", None)
        sel_author = "Unknown"
        sel_year = "N/A"

    st.markdown(f"""
    <div class="target-banner">
        <img src="{sel_img}" onerror="this.onerror=null;this.src='https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80';">
        <div class="target-text">
            <div class="target-kicker">Because you liked</div>
            <div class="target-title">{selected_book}</div>
            <div class="target-author">by {sel_author} • {sel_year}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        if mode == "Hybrid (Recommended)":
            st.session_state['recommendations'] = hybrid_recommend(selected_book, top_n=top_n)
        else:
            st.session_state['recommendations'] = recommend(selected_book, top_n=top_n)
        st.session_state['current_target'] = selected_book
        if selected_book not in st.session_state['history']:
            st.session_state['history'].insert(0, selected_book)
            st.session_state['history'] = st.session_state['history'][:10]

    recs = st.session_state.get('recommendations', [])
    if recs:
        st.markdown(f"### Top picks for fans of *{selected_book}*")
        for i in range(0, len(recs), 4):
            grid = st.columns(4)
            for j, col in enumerate(grid):
                if i + j < len(recs):
                    rec = recs[i + j]
                    with col:
                        st.markdown(render_book_card(rec['title'], rec['author'], rec['image'], sim_score=rec['score']), unsafe_allow_html=True)
                        if st.button("Add", key=f"rec_add_{i}_{j}", use_container_width=True):
                            if rec['title'] not in st.session_state['favorites']:
                                st.session_state['favorites'].append(rec['title'])
                                st.toast(f"Added {rec['title']} to library")
    elif st.session_state.get('current_target'):
        st.info("Click 'Generate Recommendations' to see results.")


# ================= DISCOVER FOR YOU =================
elif choice == "Discover":
    st.markdown("## Discover")
    st.caption("Get a random pick with suggestions.")

    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown("### Ready for your next read?")
            st.write("Click the button below to get a randomized pick from our curated collection, along with smart recommendations based on similar reader tastes.")
        with c2:
            if st.button("Surprise Me", type="primary", use_container_width=True, key="surprise_btn"):
                candidates = final_df.sample(1).iloc[0]
                target = candidates['Book-Title']
                st.session_state['current_target'] = target
                st.session_state['recommendations'] = hybrid_recommend(target, top_n=6)
                st.session_state['last_surprise'] = {
                    'title': target,
                    'author': candidates['Book-Author'],
                    'img': get_book_cover_url(candidates.get('ISBN', ''), candidates['Image-URL-M'])
                }
                st.rerun()
        with c3:
            if st.button("Try Another", use_container_width=True, key="surprise_again"):
                candidates = final_df.sample(1).iloc[0]
                target = candidates['Book-Title']
                st.session_state['current_target'] = target
                st.session_state['recommendations'] = hybrid_recommend(target, top_n=6)
                st.session_state['last_surprise'] = {
                    'title': target,
                    'author': candidates['Book-Author'],
                    'img': get_book_cover_url(candidates.get('ISBN', ''), candidates['Image-URL-M'])
                }
                st.rerun()

    if st.session_state.get('last_surprise'):
        surprise = st.session_state['last_surprise']
        st.markdown(f"""
        <div class="target-banner">
            <img src="{surprise['img']}" onerror="this.onerror=null;this.src='https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80';">
            <div class="target-text">
                <div class="target-kicker">Your Surprise Pick</div>
                <div class="target-title">{surprise['title']}</div>
                <div class="target-author">by {surprise['author']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Based on this pick, here are books readers with similar taste also loved:")

    if st.session_state.get('current_target') and st.session_state.get('recommendations'):
        recs = st.session_state['recommendations']
        for i in range(0, len(recs), 4):
            grid = st.columns(4)
            for j, col in enumerate(grid):
                if i + j < len(recs):
                    rec = recs[i + j]
                    with col:
                        st.markdown(render_book_card(rec['title'], rec['author'], rec['image'], sim_score=rec['score']), unsafe_allow_html=True)
                        if st.button("Add to Library", key=f"surp_add_{i}_{j}", use_container_width=True):
                            if rec['title'] not in st.session_state['favorites']:
                                st.session_state['favorites'].append(rec['title'])
                                st.toast("Added to library")
    elif not st.session_state.get('last_surprise'):
        st.info("Click 'Surprise Me' to get your first recommendation!")


# ================= BOOK EXPLORER =================
elif choice == "Browse":
    st.markdown("## Browse")
    st.caption("Look up any book in the catalog.")

    if full_books is None:
        st.error("Book catalog not available. Please ensure Books.csv is present.")
        st.stop()
    all_titles = sorted(full_books['Book-Title'].dropna().unique().tolist())
    selected = st.selectbox("Search for a book", all_titles)
    info = full_books[full_books['Book-Title'] == selected].drop_duplicates('Book-Title').iloc[0]
    img = get_book_cover_url(info.get('ISBN', ''), info.get('Image-URL-L') if pd.notna(info.get('Image-URL-L')) and str(info.get('Image-URL-L')).strip() != 'nan' else (info.get('Image-URL-M') if pd.notna(info.get('Image-URL-M')) and str(info.get('Image-URL-M')).strip() != 'nan' else None), high_res=True)

    ex1, ex2 = st.columns([1, 2.2])
    with ex1:
        try:
            st.image(img, use_container_width=True)
        except Exception:
            st.image("https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80", use_container_width=True)
        if st.button("Add to Library", use_container_width=True, key="expl_add"):
            if selected not in st.session_state['favorites']:
                st.session_state['favorites'].append(selected)
                st.toast("Added to library")
        if st.button("Get Recommendations", use_container_width=True, key="expl_rec"):
            st.session_state['current_target'] = selected
            if selected in pt.index:
                st.session_state['recommendations'] = hybrid_recommend(selected, top_n=8)
            st.session_state['jump'] = "Recommendations"
            st.rerun()
    with ex2:
        st.markdown(f"### {selected}")
        st.markdown(f"**Author:** {info.get('Book-Author', 'Unknown')}")
        st.markdown(f"**Publisher:** {info.get('Publisher', 'Unknown')}")
        st.markdown(f"**Year of Publication:** {info.get('Year-Of-Publication', 'Unknown')}")
        st.markdown(f"**ISBN:** {info.get('ISBN', 'Unknown')}")
        st.markdown("---")
        st.markdown("#### Reader Engagement")
        if selected in final_df['Book-Title'].values:
            row_f = final_df[final_df['Book-Title'] == selected].iloc[0]
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Average Rating", f"{row_f['avg_rating']:.2f} / 5.0")
            with m2:
                st.metric("Number of Ratings", f"{row_f['num_ratings']:,}")
        elif selected in ratings_with_name['Book-Title'].values if ratings_with_name is not None else False:
            tmp = ratings_with_name[ratings_with_name['Book-Title'] == selected]
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Average Rating", f"{tmp['Book-Rating'].mean():.2f} / 10.0" if tmp['Book-Rating'].max() > 5 else f"{tmp['Book-Rating'].mean():.2f} / 5.0")
            with m2:
                st.metric("Number of Ratings", f"{len(tmp):,}")
        else:
            st.info("This title is in the full catalog but no rating data found.")

        if selected in pt.index:
            st.markdown("#### Because others read this, they also read")
            recs = hybrid_recommend(selected, top_n=6)
            for i in range(0, len(recs), 3):
                grid = st.columns(3)
                for j, col in enumerate(grid):
                    if i + j < len(recs):
                        rec = recs[i + j]
                        with col:
                            st.markdown(render_book_card(rec['title'], rec['author'], rec['image'], sim_score=rec['score']), unsafe_allow_html=True)


# ================= INSIGHTS =================
elif choice == "Insights":
    st.markdown("## Insights")
    st.caption("Trends from the data.")

    t1, t2, t3 = st.columns(3)
    with t1:
        if full_books is not None:
            st.metric("Unique Books (Catalog)", f"{full_books['Book-Title'].nunique():,}")
        else:
            st.metric("Unique Books (Catalog)", "N/A")
    with t2:
        if ratings_with_name is not None:
            st.metric("Unique Users", f"{ratings_with_name['User-ID'].nunique():,}")
        else:
            st.metric("Unique Users", "N/A")
    with t3:
        if ratings_with_name is not None:
            active_count = sum(1 for u,v in ratings_with_name.groupby('User-ID').count()['Book-Rating'].items() if v>200)
            st.metric("Active Users (>200 ratings)", f"{active_count:,}")
        else:
            st.metric("Active Users (>200 ratings)", "N/A")

    c1, c2 = st.columns(2)
    with c1:
        if 'Publisher' in final_df.columns:
            top_pub = final_df['Publisher'].value_counts().head(10)
            fig_pub = px.bar(top_pub, x=top_pub.values, y=top_pub.index, orientation='h', title="Top Publishers by Featured Titles")
            fig_pub.update_layout(yaxis=dict(autorange="reversed"), height=450)
            st.plotly_chart(fig_pub, use_container_width=True)
        else:
            st.info("Publisher data not available")
    with c2:
        fig_scatter = px.scatter(final_df, x='num_ratings', y='avg_rating', size='num_ratings', hover_name='Book-Title', color='avg_rating', color_continuous_scale='Viridis', title="Rating vs. Popularity (Top List)")
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Rating Distribution Across All Reviews")
    rating_counts = ratings_with_name['Book-Rating'].value_counts().sort_index()
    fig_r = go.Figure(data=[go.Bar(x=rating_counts.index, y=rating_counts.values, marker_color='#6366f1')])
    fig_r.update_layout(title="Overall Book-Rating Distribution (1-10)", xaxis_title="Rating", yaxis_title="Count", height=400)
    st.plotly_chart(fig_r, use_container_width=True)


# ================= MY LIBRARY =================
elif choice == "My Library":
    st.markdown("## My Library")
    st.caption("Books you've saved.")

    if not st.session_state['favorites']:
        st.info("Your library is empty. Browse Popular Books or get recommendations and click 'Add to Library'!")
    else:
        st.success(f"You have **{len(st.session_state['favorites'])}** book(s) in your library")
        if st.button("Clear Library"):
            st.session_state['favorites'] = []
            st.rerun()
        cols = st.columns(4)
        for i, title in enumerate(st.session_state['favorites']):
            with cols[i % 4]:
                if title in books_lookup.index:
                    info = books_lookup.loc[title]
                    if isinstance(info, pd.DataFrame):
                        info = info.iloc[0]
                    img = get_book_cover_url(info.get('ISBN', ''), info.get('Image-URL-M', ''))
                    author = info.get('Book-Author', 'Unknown')
                else:
                    img = get_book_cover_url("", None)
                    author = "Unknown"
                st.markdown(render_book_card(title, author, img), unsafe_allow_html=True)
                if st.button("Get Similar", key=f"lib_rec_{i}", use_container_width=True):
                    st.session_state['current_target'] = title
                    if title in pt.index:
                        st.session_state['recommendations'] = hybrid_recommend(title, top_n=6)
                    st.session_state['jump'] = "Recommendations"
                    st.rerun()

    st.markdown("### Recently Viewed For Recommendations")
    if st.session_state['history']:
        for h in st.session_state['history']:
            st.write(f"• {h}")
    else:
        st.caption("No history yet.")


# ================= ABOUT =================
elif choice == "About":
    st.markdown("## About")
    st.markdown("""
    ### Project Overview
    LitMatch is an intelligent book recommender system that helps readers discover their next great read using real-world rating data.
    It combines **Popularity-Based Filtering** and **Item-Based Collaborative Filtering** with a **Hybrid Scoring Mechanism** for more balanced suggestions.

    ### Key Features
    - **Interactive Dashboard** - At-a-glance KPIs and trends
    - **Smart Recommendations** - Switch between pure collaborative and hybrid strategies
    - **Discover For You** - Randomized suggestions to break out of your bubble
    - **Book Explorer** - Deep metadata view for any title
    - **Insights & Analytics** - Visualize ratings, authors, and publishers
    - **My Library** - Build and manage your reading list
    - **Advanced Filtering & Search** - Find exactly what you're looking for
    - **Responsive UI** - Modern, clean design with dark mode support

    ### Technical Architecture
    - **Data Processing:** Pandas & NumPy for cleaning and aggregation
    - **Recommendation Engine:** Scikit-learn Cosine Similarity on user-item pivot table
    - **Visualization:** Plotly for interactive charts
    - **Frontend:** Streamlit with custom CSS/HTML components
    - **Caching:** Pickle-based model caching for instant loads
    - **Image Resolution:** Google Books API fallback with graceful errors

    ### Dataset
    Sourced from the Book-Crossing dataset containing 271,360 books, 278,858 users, and 1,149,780 explicit/implicit ratings.

    ### Future Enhancements
    Content-based filtering using genres/descriptions, SVD/Matrix Factorization, user login, book reviews, and integration with external book APIs.
    """)

# Handle jump from buttons
if 'jump' in st.session_state:
    choice_jump = st.session_state.pop('jump')
    if choice_jump in menu:
        st.session_state['_menu_choice'] = choice_jump
        st.rerun()

st.markdown("---")
st.caption("© 2025 Book Recommender")
