import requests
import streamlit as st

# DuckDuckGo API endpoint
DUCKDUCKGO_URL = "https://api.duckduckgo.com/"

# Function to fetch results from DuckDuckGo
def fetch_duckduckgo_results(query):
    params = {
        "q": query,
        "format": "json",
        "pretty": 1
    }
    response = requests.get(DUCKDUCKGO_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data from DuckDuckGo"}

# Streamlit UI
def main():
    st.title("DuckDuckGo-Powered GPT App")
    st.write("Ask anything and I'll try to help!")

    # Input field for the user's question
    user_input = st.text_input("Enter your query:")

    if user_input:
        # Fetch data from DuckDuckGo
        results = fetch_duckduckgo_results(user_input)

        # Parse and display the response
        if "error" in results:
            st.error("Something went wrong. Please try again later.")
        else:
            abstract = results.get("AbstractText", "")
            if abstract:
                st.success(f"**Here's what I found:**\n\n{abstract}")
            else:
                st.warning("No results found. Try refining your query.")
            
            # Display related topics (optional)
            related_topics = results.get("RelatedTopics", [])
            if related_topics:
                st.write("### Related Topics:")
                for topic in related_topics[:5]:  # Show up to 5 related topics
                    if "Text" in topic:
                        st.write(f"- {topic['Text']}")

if __name__ == "__main__":
    main()
