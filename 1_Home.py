import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.switch_page_button import switch_page


# --- Page Config ---
st.set_page_config(
    page_title="AutoBI.AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #f0f4ff, #ffffff);
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin: 20px;
            width: 100%;
            height: 420px;
            overflow: hidden;
            position: relative;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
        }
        .card img {
            width: 100%;
            height: 170px;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        .card h2 {
            color: #3b82f6;
            font-size: 18px;
            margin: 0 0 10px 0;
        }
        .card p {
            font-size: 15px;
            color: #4b5563;
            margin: 0;
        }
        .card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url('https://media.istockphoto.com/id/1434617519/photo/hrm-or-human-resource-management-magnifier-glass-focus-to-manager-business-icon-which-is.webp?a=1&b=1&s=612x612&w=0&k=20&c=1lY9_cpu8uVKWIYB8WV9Fb54FkPu9S7RO-J3TyXNOZA=');
            background-size: cover;
            background-position: center;
            opacity: 0.08;
            border-radius: 16px;
            z-index: 0;
        }
        .card > * {
            position: relative;
            z-index: 1;
        }
        .hero-section {
            background-image: url('https://img.freepik.com/free-vector/ai-technology-brain-background-vector-digital-transformation-concept_53876-117820.jpg?w=2000');
            background-size: cover;
            background-position: center;
            height: 60vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 2rem;
            border-radius: 32px 32px 32px 32px;
            color: white;
        }
        .hero-section h1 {
            font-size: 48px;
            margin-bottom: 0.5rem;
        }
        .hero-section p {
            font-size: 20px;
            max-width: 700px;
        }
    </style>

    <div class="hero-section">
        <h1 style='text-align: center; color: #1f2937;'>🚀 Welcome to <span style="color:#3b82f6;">AutoBI.AI</span></h1>
        <p style='text-align: center; font-size: 18px; color: #6b7280;'>
            Your AI-powered Business Intelligence suite for instant data insights, predictions, and decisions.
        </p>
    </div>
    </style>
""", unsafe_allow_html=True)


# --- Feature Cards ---
def feature_card(title, description, image_url):
    return f"""
        <div class="card">
            <img src="{image_url}" alt="{title}">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
    """


# Create 3 columns for cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(feature_card(
        "📁 Upload Any CSV & Auto-Clean",
        "Upload your raw business data. AutoBI.AI detects column types, cleans, preprocesses, and transforms it automatically.",
        "https://plus.unsplash.com/premium_photo-1677397576132-5d7412825d0e?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8ZmlsZXxlbnwwfHwwfHx8MA%3D%3D"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(feature_card(
        "🤖 AutoML Predictions & Forecasting",
        "Our backend auto-trains ML models and gives accurate predictions and future forecasting on any selected column.",
        "https://plus.unsplash.com/premium_photo-1681810994162-43dbe0919d3f?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8bWFjaGluZSUyMGxlYXJuaW5nJTIwbW9kZWx8ZW58MHx8MHx8fDA%3D"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(feature_card(
        "💬 AI Assistant with Insights",
        "Ask business questions like “Top-selling products?” or “Ways to improve profit?” and get instant insights.",
        "https://plus.unsplash.com/premium_photo-1726079246917-46f2b37f7e9e?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YWklMjBhc2lzdGFudHxlbnwwfHwwfHx8MA%3D%3D"
    ), unsafe_allow_html=True)

# Create another row of 3 columns for the rest of the cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(feature_card(
        "🔍 Text-to-SQL Querying",
        "Ask data questions in plain English and get real-time SQL results instantly.",
        "https://images.unsplash.com/photo-1662026911591-335639b11db6?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8c3FsfGVufDB8fDB8fHww"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(feature_card(
        "📊 Dashboards & KPIs Automatically",
        "Get instant dashboards, KPIs, and visual insights generated with zero configuration required.",
        "https://images.unsplash.com/photo-1666875753105-c63a6f3bdc86?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YW5hbHl0aWNzfGVufDB8fDB8fHww"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(feature_card(
            title="📈 Easy Decision-Making",
            description="Take confident, data-driven decisions powered by AI insights.",
            image_url="https://plus.unsplash.com/premium_photo-1670213989458-a05faa974cc4?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8bWFjaGluZSUyMGxlYXJuaW5nJTIwbW9kZWx8ZW58MHx8MHx8fDA%3D"   
    ), unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px; color: gray;'>Built with ❤️ using Streamlit + Shadcn UI</p>
""", unsafe_allow_html=True)

# Add the "Get Started" Button
# if st.button("Get Started"):
# switch_page("upload")