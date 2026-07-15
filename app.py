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
