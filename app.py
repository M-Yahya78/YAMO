import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def configure_page():
    st.set_page_config(
        page_title="YAMO | Movie Recommender",
        page_icon="📼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css(language):
    """
    Injects custom CSS. Dynamically switches between LTR (English) and RTL (Arabic) layouts.
    """
    direction = "rtl" if language == "العربية" else "ltr"
    text_align = "right" if language == "العربية" else "left"
    
    custom_css = f"""
    <style>
        /* Flip the main container layout based on language */
        .block-container {{
            direction: {direction};
            text-align: {text_align};
        }}
        
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in-section {{ animation: fadeIn 0.8s ease-out forwards; }}
        
        div[data-baseweb="input"] {{
            transition: box-shadow 0.3s ease-in-out;
            border-radius: 8px;
        }}
        div[data-baseweb="input"]:focus-within {{
            box-shadow: 0 0 15px rgba(138, 43, 226, 0.6) !important;
            border: 1px solid #8a2be2 !important;
        }}

        .movie-card {{
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
        }}
        .movie-card:hover {{
            transform: scale(1.02);
            border-color: #8a2be2;
        }}
        .genre-tag {{
            background-color: #8a2be2;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 0.2rem;
            display: inline-block;
        }}
        .match-badge {{
            background-color: #27ae60;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 0.5rem;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def fetch_tmdb_data(api_key, query, language):
    """
    Builds a large 'Discovery Pool' of movies. 
    It checks the user's direct search, but ALSO pulls in trending and top-rated 
    movies so the AI always has a large batch of plots to read and rank.
    """
    if not api_key:
        return pd.DataFrame()

    api_lang = "ar-SA" if language == "العربية" else "en-US"
    base_params = {"api_key": api_key, "language": api_lang, "include_adult": False}
    
    # We will hit 3 different TMDB endpoints at once to build a massive pool of options
    urls_to_fetch = [
        # 1. The exact search (in case they typed a specific title like "Batman")
        ("https://api.themoviedb.org/3/search/multi", {**base_params, "query": query, "page": 1}),
        # 2. Trending this week (captures current popular culture)
        ("https://api.themoviedb.org/3/trending/all/week", base_params),
        # 3. Top Rated of all time (captures classics)
        ("https://api.themoviedb.org/3/movie/top_rated", base_params)
    ]

    # Dictionaries mapping TMDB's numeric genre IDs to actual text words
    genre_map_en = {
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 
        99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 
        27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 
        10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 
        10759: "Action & Adventure", 10765: "Sci-Fi & Fantasy", 10768: "War & Politics"
    }
    
    genre_map_ar = {
        28: "أكشن", 12: "مغامرة", 16: "رسوم متحركة", 35: "كوميديا", 80: "جريمة", 
        99: "وثائقي", 18: "دراما", 10751: "عائلي", 14: "فانتازيا", 36: "تاريخ", 
        27: "رعب", 10402: "موسيقى", 9648: "غموض", 10749: "رومانسية", 878: "خيال علمي", 
        10770: "فيلم تلفزيوني", 53: "إثارة", 10752: "حرب", 37: "غربي", 
        10759: "أكشن ومغامرة", 10765: "خيال علمي وفانتازيا", 10768: "حرب وسياسة"
    }

    active_genre_map = genre_map_ar if language == "العربية" else genre_map_en
    movies_list = []
    
    # Loop through our 3 endpoints and gather all the movies into one giant list
    for url, params in urls_to_fetch:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get("results", []):
                    if item.get("media_type") == "person" or not item.get("overview"):
                        continue

                    title = item.get("title") or item.get("name", "Unknown Title")
                    date_str = item.get("release_date") or item.get("first_air_date", "0000")
                    year = date_str.split("-")[0] if date_str else "N/A"
                    
                    poster_path = item.get("poster_path")
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=400"

                    genre_ids = item.get("genre_ids", [])
                    genres_text = ", ".join([active_genre_map.get(gid, "") for gid in genre_ids if gid in active_genre_map])

                    movies_list.append({
                        "Title": title,
                        "Year": year,
                        "IMDb_Rating": round(item.get("vote_average", 0.0), 1),
                        "Genres": genres_text,
                        "Poster_URL": poster_url,
                        "Description": item.get("overview", "")
                    })
        except Exception as e:
            pass # If one endpoint fails, silently skip it and keep building the pool

    # Convert to DataFrame and remove duplicates (so we don't show the same movie twice)
    df = pd.DataFrame(movies_list)
    if not df.empty:
        df = df.drop_duplicates(subset=['Title'])
        
    return df

def get_ai_recommendations(user_query, df):
    """
    Computes semantic similarity using TF-IDF. 
    Now utilizes a Natural Language Processing (NLP) technique called "Stop Words" 
    to filter out conversational fluff.
    """
    if df.empty or not user_query:
        if not df.empty:
            df['Similarity'] = 0.0
        return df

    # A custom list of conversational filler words in both English and Arabic.
    # The AI will completely ignore these words, focusing only on the important nouns/verbs.
    nlp_stop_words = [
        "i", "want", "to", "watch", "a", "an", "the", "that", "contains", "contain", 
        "both", "and", "or", "movie", "series", "show", "me", "about", "like", "with", 
        "some", "really", "good", "looking", "for",
        "أريد", "فيلم", "مسلسل", "عن", "يحتوي", "على", "ابحث", "لي", "يعرض", "مسلسلات", 
        "افلام", "فيلمًا", "مشاهدة", "ان", "أشاهد", "اريد", "ابغى"
    ]

    combined_metadata = df['Description'] + " " + df['Genres']
    
    # We pass our custom stop words into the Vectorizer so it knows what to ignore
    vectorizer = TfidfVectorizer(stop_words=nlp_stop_words) 
    
    tfidf_matrix = vectorizer.fit_transform(combined_metadata)
    query_vector = vectorizer.transform([user_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    df['Similarity'] = similarity_scores
    
    return df
def sort_movies(df, sort_method, language):
    """
    Handles sorting logic.
    """
    if df.empty:
        return df
        
    if sort_method in ["Most Related (AI Score)", "الأكثر صلة (بالذكاء الاصطناعي)"]:
        return df.sort_values(by=["Similarity", "IMDb_Rating"], ascending=False)
    else:
        return df.sort_values(by="IMDb_Rating", ascending=False)

def render_sidebar():
    """
    Builds the sidebar, handles language selection, and supplies the default API key.
    """
    with st.sidebar:
        st.title("📼 YAMO")
        
        # Add the language toggle
        language = st.radio("Language / اللغة", ["English", "العربية"])
        
        st.divider()
        if language == "العربية":
            st.markdown("### يــامــو: منظم الأفلام الخاص بك")
            st.markdown("اكتب جملة تفصيلية تصف فيها مزاجك أو نوع الفيلم الذي تبحث عنه. سيقوم محرك الذكاء الاصطناعي الخاص بنا بمطابقة طلبك مع قاعدة بيانات الأفلام الحية.")
            st.markdown("**الإصدار:** 4.0.0 (النسخة العربية)")
        else:
            st.markdown("### Yet Another Movie Organizer")
            st.markdown("Type a detailed sentence describing what you are in the mood for. Our AI math matches your vibe against live TMDB data.")
            st.markdown("**Version:** 4.0.0 (Bilingual Edition)")
        
        # Returning your hardcoded key directly
        api_key = "36adbb04844b21191fa84ba357b5e64a"
        
        return api_key, language

def render_header(language):
    """
    Renders the title section based on language.
    """
    title_text = "ابحث عن فيلمك القادم 🍿" if language == "العربية" else "Find Your Next Watch 🍿"
    sub_text = "اكتب مزاجك، نوع الفيلم، أو تفاصيل القصة بالأسفل." if language == "العربية" else "Type a mood, vibe, plot element, or keyword below."
    
    st.markdown(
        f"""
        <div class="fade-in-section" style="text-align: center; padding-top: 1.5rem; padding-bottom: 1.5rem;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0;">{title_text}</h1>
            <p style="font-size: 1.1rem; color: #a0a0b0;">{sub_text}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def render_search_and_controls_section(language):
    """
    Creates the main text input and sorting controls translated to the active language.
    """
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    
    if language == "العربية":
        placeholder = "مثال: مسلسل خيال علمي مع قصة سياسية معقدة..."
        search_label = "ابحث بالقصة، أو المزاج، أو الكلمات المفتاحية:"
        sort_label = "ترتيب النتائج حسب:"
        sort_options = ["الأكثر صلة (بالذكاء الاصطناعي)", "الأعلى تقييماً"]
    else:
        placeholder = "e.g., An intricate political fantasy series with dragons..."
        search_label = "Search by plot, vibe, or keywords:"
        sort_label = "Sort results by:"
        sort_options = ["Most Related (AI Score)", "Top Rated (TMDB)"]

    with center_col:
        user_mood = st.text_input(
            search_label,
            placeholder=placeholder,
            label_visibility="collapsed"
        )
        
        lbl_col, choice_col = st.columns([1, 2])
        with lbl_col:
            st.markdown(f"<p style='padding-top:25px; color:#a0a0b0;'>{sort_label}</p>", unsafe_allow_html=True)
        with choice_col:
            sort_method = st.selectbox(
                "Sort",
                options=sort_options,
                label_visibility="collapsed"
            )
            
    return user_mood, sort_method

def render_movie_card(row, language):
    """
    Generates a single structural card column for a movie.
    """
    genres_list = [g.strip() for g in row['Genres'].split(',') if g.strip()]
    tags_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in genres_list])
    
    match_html = ""
    if 'Similarity' in row and row['Similarity'] > 0:
        match_percentage = int(row['Similarity'] * 100)
        match_text = f"نسبة التطابق {match_percentage}% 🎯" if language == "العربية" else f"🎯 {match_percentage}% Match"
        match_html = f'<div class="match-badge">{match_text}</div>'
    
    year_text = "السنة:" if language == "العربية" else "Year:"
    
    st.markdown(
        f"""
        <div class="movie-card fade-in-section">
            <div>
                <img src="{row['Poster_URL']}" style="width:100%; height:350px; object-fit:cover; border-radius:6px; margin-bottom:12px;" />
                {match_html}
                <h3 style="margin: 0 0 5px 0; font-size:1.3rem;">{row['Title']}</h3>
                <p style="color: #a0a0b0; font-size:0.85rem; margin-bottom:8px;">{year_text} {row['Year']}</p>
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

def render_movie_grid(movies_df, language):
    """
    Dynamically maps our sorted DataFrame columns into rows of structural visual cards.
    """
    if movies_df.empty:
        msg = "لم يتم العثور على نتائج. جرب كلمات مفتاحية أخرى!" if language == "العربية" else "No movies matched your vibe. Try different keywords!"
        st.warning(msg)
        return

    if 'Similarity' in movies_df:
        movies_df = movies_df[movies_df['Similarity'] > 0.0]
        
    if movies_df.empty:
        msg = "لا توجد تطابقات دقيقة. جرب توسيع نطاق البحث!" if language == "العربية" else "No semantic matches found. Try broadening your terms!"
        st.warning(msg)
        return

    chunk_size = 3
    for i in range(0, len(movies_df), chunk_size):
        chunk = movies_df.iloc[i:i+chunk_size]
        cols = st.columns(3)
        
        for idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[idx]:
                render_movie_card(row, language)

def main():
    configure_page()
    
    # Render sidebar and capture the API key and active language
    tmdb_api_key, language = render_sidebar()
    
    # Load CSS with the correct text direction (LTR or RTL)
    load_custom_css(language)
    render_header(language)
    
    user_query, sort_method = render_search_and_controls_section(language)
    st.divider()

    if user_query:
        msg = "جاري جلب البيانات من TMDB..." if language == "العربية" else "Fetching live data from TMDB..."
        with st.spinner(msg):
            # 1. Fetch raw data matching the query (returns Arabic descriptions if Arabic is selected)
            movie_dataframe = fetch_tmdb_data(tmdb_api_key, user_query, language)
            
            # 2. Run TF-IDF against the fetched results
            processed_df = get_ai_recommendations(user_query, movie_dataframe)
            
            # 3. Sort based on user preference
            sorted_df = sort_movies(processed_df, sort_method, language)
            
            result_header = f"### 🔍 نتائج الذكاء الاصطناعي لـ: *\"{user_query}\"*" if language == "العربية" else f"### 🔍 AI Results matching: *\"{user_query}\"*"
            st.markdown(result_header)
            
            # 4. Render the UI
            render_movie_grid(sorted_df, language)
    else:
        wait_msg = "### ✨ بانتظار مزاجك السينمائي..." if language == "العربية" else "### ✨ Waiting for your movie mood..."
        st.markdown(wait_msg)

if __name__ == "__main__":
    main()
