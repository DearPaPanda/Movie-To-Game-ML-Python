import streamlit as st
from utils import get_game_suggestions, save_feedback

st.set_page_config(page_title="Movie to Game Recommender 🎬🎮")
st.title("🎬 Movie to Game Recommender")
st.markdown("Enter a **movie title**, and get **game suggestions** based on plot similarity!")

movie_input = st.text_input("Enter a movie name:")

if st.button("Suggest Games"):
    with st.spinner("Fetching real data and computing similarities..."):
        suggestions = get_game_suggestions(movie_input)

    st.subheader("🎮 Game Suggestions:")
    for title, score in suggestions:
        st.write(f"• **{title}** (Similarity: {score:.2f})")

    feedback = st.radio("Was this helpful?", ["👍 Yes", "👎 No"])
    if st.button("Submit Feedback"):
        for title, _ in suggestions:
            save_feedback(movie_input, title, feedback)
        st.success("Thanks for your feedback! ✅")