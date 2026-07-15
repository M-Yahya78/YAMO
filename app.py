import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def configure_page():
    """
    Configures the main settings of the Streamlit page.
    """
    st.set_page_config(
        page_title="YAMO | Movie Recommender",
        page_icon="📼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """
    Injects custom CSS to create dark-mode animations and glowing effects.
    """
    custom_css = """
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .fade-in-section { animation: fadeIn 0.8s ease-out forwards; }
        
        div[data-baseweb="input"] {
            transition: box-shadow 0.3s ease-in-out;
            border-radius: 8px;
        }
        div[data-baseweb="input"]:focus-within {
            box-shadow: 0 0 15px rgba(138, 43, 226, 0.6) !important;
            border: 1px solid #8a2be2 !important;
        }

        .movie-card {
            background-color: #1e1e2e;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #2d2d3f;
            transition: transform 0.2s, border-color 0.2s;
            margin-bottom: 1.5rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .movie-card:hover {
            transform: scale(1.02);
            border-color: #8a2be2;
        }
        .genre-tag {
            background-color: #8a2be2;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
            display: inline-block;
        }
        .match-badge {
            background-color: #27ae60;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def fetch_tmdb_data(api_key, query):
    """
    Calls the live TMDB API to fetch movies and TV shows based on the user's keywords.
    Why we do this: We need a dynamic, live dataset. The 'multi' search endpoint 
    allows us to grab both movies and TV shows in a single request.
    """
    if not api_key:
        return pd.DataFrame() # Return empty if no key is provided

    # The TMDB endpoint for searching multiple media types simultaneously
    url = f"https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": api_key,
        "query": query,
        "language": "en-US",
        "page": 1,
        "include_adult": False
    }

    try:
        # We use requests.get() to ask the TMDB servers for the data
        response = requests.get(url, params=params)
        response.raise_for_status() # Automatically catches connection errors (like 404 or 401)
        data = response.json()      # Converts the string response into a Python dictionary
    except Exception as e:
        st.error(f"Error fetching data from TMDB: {e}")
        return pd.DataFrame()

    # TMDB returns genres as raw numbers (IDs). We need a map to convert them back to text words 
    # so our TF-IDF text vectorizer can read them.
    genre_map = {
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 
        99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 
        27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 
        10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 
        10759: "Action & Adventure", 10765: "Sci-Fi & Fantasy", 10768: "War & Politics"
    }

    movies_list = []
    
    # Loop through the raw JSON results and format them into our standardized dictionary
    for item in data.get("results", []):
        # Skip people or entries without a description, as our AI needs text to work
        if item.get("media_type") == "person" or not item.get("overview"):
            continue

        # TV shows use 'name' and 'first_air_date', movies use 'title' and 'release_date'
        title = item.get("title") or item.get("name", "Unknown Title")
        date_str = item.get("release_date") or item.get("first_air_date", "0000")
        year = date_str.split("-")[0] if date_str else "N/A"
        
        # Build the full image URL. 'w500' means we are requesting an image 500 pixels wide
        poster_path = item.get("poster_path")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            # Fallback placeholder if TMDB doesn't have a poster
            poster_url = "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=400"

        # Convert the numeric genre IDs into a comma-separated string of words
        genre_ids = item.get("genre_ids", [])
        genres_text = ", ".join([genre_map.get(gid, "Unknown") for gid in genre_ids if gid in genre_map])

        movies_list.append({
            "Title": title,
            "Year": year,
            "IMDb_Rating": round(item.get("vote_average", 0.0), 1), # TMDB uses vote_average out of 10
            "Genres": genres_text,
            "Poster_URL": poster_url,
            "Description": item.get("overview", "")
        })

    return pd.DataFrame(movies_list)

def get_ai_recommendations(user_query, df):
    """
    Computes semantic similarity between the user input and the fetched TMDB dataset using TF-IDF.
    """
    if df.empty or not user_query:
        if not df.empty:
            df['Similarity'] = 0.0
        return df

    # Combine Plot and Genres for maximum context
    combined_metadata = df['Description'] + " " + df['Genres']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_metadata)
    query_vector = vectorizer.transform([user_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    df['Similarity'] = similarity_scores
    
    return df

def sort_movies(df, sort_method):
    """
    Handles sorting logic based on the user's chosen UI drop-down preference.
    """
    if df.empty:
        return df
        
    if sort_method == "Most Related (AI Score)":
        return df.sort_values(by=["Similarity", "IMDb_Rating"], ascending=False)
    elif sort_method == "Top Rated (TMDB)":
        return df.sort_values(by="IMDb_Rating", ascending=False)
    return df

def render_sidebar():
    """
    Builds the sidebar branding layout and grabs the secure API key.
    """
    with st.sidebar:
        st.title("📼 YAMO")
        st.markdown("### Yet Another Movie Organizer")
        
        st.divider()
        st.markdown(
            "Type a detailed sentence describing what you are in the mood for. "
            "Our TF-IDF math matches your vibe against live TMDB data."
        )
        st.markdown("**Version:** 3.0.0 (Live API Edition)")
        
        # This securely pulls the key from the Streamlit Cloud vault!
        return st.secrets["TMDB_API_KEY"]
        
        return api_key

def render_header():
    """
    Renders the title section.
    """
    st.markdown(
        """
        <div class="fade-in-section" style="text-align: center; padding-top: 1.5rem; padding-bottom: 1.5rem;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0;">Find Your Next Watch 🍿</h1>
            <p style="font-size: 1.1rem; color: #a0a0b0;">Type a mood, vibe, plot element, or keyword below.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def render_search_and_controls_section():
    """
    Creates both the main text input box and the sorting option control layout.
    """
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    
    with center_col:
        user_mood = st.text_input(
            "Search by plot, vibe, or keywords:",
            placeholder="e.g., An intricate political fantasy series with dragons...",
            label_visibility="collapsed"
        )
        
        lbl_col, choice_col = st.columns([1, 2])
        with lbl_col:
            st.markdown("<p style='padding-top:25px; text-align:right; color:#a0a0b0;'>Sort results by:</p>", unsafe_allow_html=True)
        with choice_col:
            sort_method = st.selectbox(
                "Sort Results By",
                options=["Most Related (AI Score)", "Top Rated (TMDB)"],
                label_visibility="collapsed"
            )
            
    return user_mood, sort_method

def render_movie_card(row):
    """
    Generates a single structural card column for a movie given a pandas row object.
    """
    genres_list = [g.strip() for g in row['Genres'].split(',') if g.strip()]
    tags_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in genres_list])
    
    match_html = ""
    if 'Similarity' in row and row['Similarity'] > 0:
        match_percentage = int(row['Similarity'] * 100)
        match_html = f'<div class="match-badge">🎯 {match_percentage}% Match</div>'
        
    st.markdown(
        f"""
        <div class="movie-card fade-in-section">
            <div>
                <img src="{row['Poster_URL']}" style="width:100%; height:350px; object-fit:cover; border-radius:6px; margin-bottom:12px;" />
                {match_html}
                <h3 style="margin: 0 0 5px 0; font-size:1.3rem;">{row['Title']}</h3>
                <p style="color: #a0a0b0; font-size:0.85rem; margin-bottom:8px;">Year: {row['Year']}</p>
                <div style="margin-bottom: 12px;">{tags_html}</div>
            </div>
            <div>
                <p style="color: #f39c12; font-weight: bold; margin-bottom:8px;">⭐ {row['IMDb_Rating']} / 10</p>
                <p style="color: #d1d1d1; font-size: 0.9rem; line-height:1.4;">
                    {row['Description'][:250]}...
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def render_movie_grid(movies_df):
    """
    Dynamically maps our sorted DataFrame columns into rows of structural visual cards.
    """
    if movies_df.empty:
        st.warning("No movies matched your vibe or you need to enter an API key. Try different keywords!")
        return

    # Filter out absolute zero matches to keep the grid clean
    if 'Similarity' in movies_df:
        movies_df = movies_df[movies_df['Similarity'] > 0.0]
        
    if movies_df.empty:
        st.warning("No semantic matches found for that specific vibe. Try broadening your terms!")
        return

    chunk_size = 3
    for i in range(0, len(movies_df), chunk_size):
        chunk = movies_df.iloc[i:i+chunk_size]
        cols = st.columns(3)
        
        for idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[idx]:
                render_movie_card(row)

def main():
    """
    Main pipeline controller pulling Step 3 tasks together in sequence.
    """
    configure_page()
    load_custom_css()
    
    # Render sidebar and capture the API key string
    tmdb_api_key = render_sidebar()
    render_header()
    
    user_query, sort_method = render_search_and_controls_section()
    st.divider()

    # The Logic Pipeline
    if user_query:
        if not tmdb_api_key:
            st.error("⚠️ Please enter your TMDB API Key in the sidebar to search.")
        else:
            with st.spinner("Fetching live data from TMDB..."):
                # 1. Fetch raw data broadly matching the query
                movie_dataframe = fetch_tmdb_data(tmdb_api_key, user_query)
                
                # 2. Run TF-IDF against the fetched results
                processed_df = get_ai_recommendations(user_query, movie_dataframe)
                
                # 3. Sort based on user preference
                sorted_df = sort_movies(processed_df, sort_method)
                
                st.markdown(f"### 🔍 AI Results matching: *\"{user_query}\"*")
                
                # 4. Render the UI
                render_movie_grid(sorted_df)
    else:
        st.markdown("### ✨ Waiting for your movie mood...")

if __name__ == "__main__":
    main()
