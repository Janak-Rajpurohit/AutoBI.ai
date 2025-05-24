# import streamlit as st
# import pandas as pd
# import google.generativeai as genai

# # Configure Gemini API
# genai.configure(api_key="AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY")  # Replace with your API key

# # Streamlit UI
# st.title("🤖 Gemini AI - Data Chatbot")
# uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("Preview of Uploaded Data")
#     st.dataframe(df.head())

#     prompt = st.text_area("Ask a question about the data:")

#     if prompt:
#         with st.spinner("Gemini is thinking..."):
#             # Convert DataFrame to string (you can truncate if too large)
#             data_str = df.head(100).to_csv(index=False)
#             full_prompt = f"""
#             You are a data analyst. Analyze the following data and answer the user's question.

#             Data:
#             {data_str}

#             Question:
#             {prompt}
#             """

#         model = genai.GenerativeModel('gemini-1.5-flash')
#         response = model.generate_content(full_prompt)
#         st.success("Answer:")
#         st.markdown(response.text)


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandasql as ps
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.caches import InMemoryCache
from sqlparse import format
import re
import os

# Set up Google API key
API_KEY = "AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY"
os.environ['GOOGLE_API_KEY'] = API_KEY
genai.configure(api_key=API_KEY)

# Page Config
st.set_page_config(
    page_title="Agentic BI Assistant", 
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: white;
        color: #333;
        padding: 0.8rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .data-preview {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stExpander > div:first-child {
        background-color: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic BI Assistant</h1>
    <p>Chat with your data - Ask questions, get SQL queries & insights</p>
</div>
""", unsafe_allow_html=True)

# Configure Cache
cache = InMemoryCache()

# Set up the models
@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def get_langchain_gemini():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    llm.cache = cache
    return llm

# System Prompts
SQL_SYSTEM_PROMPT = """
You are an intelligent SQL assistant. Your task is to generate SQL queries based on the provided schema description and user questions.
You should:
1. Understand the schema context provided.
2. Generate valid SQL queries that adhere to the schema and use the table name as 'df'.
3. Use column names as they are from schema with no quotes or special characters. Don't change column names - use them exactly from schema context.
4. If a column name in the user query is not an exact match but is close, suggest the closest matching column name in the schema and adjust the query accordingly.
5. If there is any ambiguity, explicitly ask the user for clarification before generating the query.
6. Provide clear and accurate SQL queries without unnecessary information and don't add special characters.
7. Avoid including null values in results when data contains many values, filter out nulls where applicable.
8. ONLY RETURN SQL QUERY
"""

ROUTER_PROMPT = """
You are an AI agent router that determines if a user's question requires:
1. SQL query generation (for specific data questions)
2. General data insights (for broader analytical questions)

Examples:
- "Show me sales by region" -> SQL
- "What are the top 10 customers?" -> SQL  
- "Find outliers in the data" -> Insights
- "What interesting patterns do you see?" -> Insights
- "Can you analyze this dataset?" -> Insights
- "Give me insights about this data" -> Insights

Analyze the following question and respond ONLY with either "SQL" or "INSIGHTS":
"""

INSIGHTS_PROMPT = """
You are a data analyst providing insights on CSV data. Analyze the data and answer thoughtfully.
- Identify key trends, patterns, and outliers
- Provide relevant statistical observations  
- Suggest potential business implications
- Be specific and reference actual values from the data
- Format your response in clear markdown with bullet points and sections
- Keep insights actionable and meaningful
"""

# Check for dataframe in session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("📄 No data found! Please upload a CSV file from the main page first.")
    st.stop()

df = st.session_state.df
df.columns = [col.replace(" ", "_") for col in df.columns]

# Data preview and schema in expandable section
with st.expander("📊 Data Preview & Schema", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Dataset Info")
        
        # Display metrics in cards
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">{df.shape[0]:,}</h3>
                <p style="margin: 0;">Rows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #764ba2; margin: 0;">{df.shape[1]}</h3>
                <p style="margin: 0;">Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🏗️ Schema")
        schema_info = []
        for col, dtype in zip(df.columns, df.dtypes):
            schema_info.append(f"• **{col}** ({dtype})")
        st.markdown("\n".join(schema_info))

# Auto-generate schema context
schema_context = "Tables: df("
schema_context += ", ".join([f"{col}({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
schema_context += ")"

# How to use section
with st.expander("❓ How to Use This Chatbot", expanded=False):
    st.markdown("""
    ### 🎯 Types of Questions You Can Ask:
    
    **📊 For SQL-like Queries:**
    - *"Show me the top 10 products by sales"*
    - *"What is the average revenue by region?"*
    - *"Count customers by city"*
    - *"Find all orders above $1000"*
    
    **🔍 For Data Insights:**
    - *"What insights can you give me about this data?"*
    - *"Analyze the relationships between variables"*
    - *"What trends do you notice?"*
    - *"Find outliers and anomalies"*
    - *"What patterns exist in the data?"*
    
    **💡 Tips:**
    - Be specific in your questions for better results
    - The assistant automatically determines if you need SQL or insights
    - Charts and visualizations are generated automatically
    - You can download query results as CSV files
    """)

# Initialize session state for chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chart_counter" not in st.session_state:
    st.session_state.chart_counter = 0

# Helper functions
def determine_question_type(question, schema):
    model = get_langchain_gemini()
    response = model.predict(f"{ROUTER_PROMPT}\n\nSchema: {schema}\nQuestion: {question}")
    return "SQL" if "SQL" in response.upper() else "INSIGHTS"

def generate_sql_query(prompt, schema_context):
    model = get_langchain_gemini()
    query_context = f"{schema_context}\nQuestion: {prompt}"
    sql_query = model.predict(f"{SQL_SYSTEM_PROMPT}\n{query_context}")
    
    # Clean up the query
    sql_query = re.sub(r"```.*?\n", "", sql_query.strip())
    sql_query = re.sub(r"\n```", "", sql_query.strip())
    
    return sql_query

def generate_insights(prompt, data_sample):
    model = get_gemini_model()
    
    # Convert sample data to string
    data_str = data_sample.head(50).to_string(index=False)
    
    # Create dataset summary
    data_summary = f"""
    Dataset Summary:
    - Total rows: {data_sample.shape[0]:,}
    - Total columns: {data_sample.shape[1]}
    - Column names and types: {dict(data_sample.dtypes.astype(str))}
    - Numeric columns summary: {data_sample.describe().to_string()}
    """
    
    full_prompt = f"""
    {INSIGHTS_PROMPT}
    
    {data_summary}
    
    Data Sample:
    {data_str}
    
    Question: {prompt}
    """
    
    response = model.generate_content(full_prompt)
    return response.text

def create_enhanced_visualizations(query_result, chart_title="Data Visualization"):
    """Create enhanced visualizations based on data characteristics"""
    st.session_state.chart_counter += 1
    
    if len(query_result) == 0:
        return
    
    numeric_cols = query_result.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = query_result.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create multiple visualizations based on data types
    viz_cols = st.columns(2)
    
    with viz_cols[0]:
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Bar chart for categorical vs numeric
            fig = px.bar(
                query_result.head(20), 
                x=categorical_cols[0], 
                y=numeric_cols[0],
                title=f"{categorical_cols[0]} vs {numeric_cols[0]}",
                color=numeric_cols[0],
                color_continuous_scale="viridis"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        elif len(numeric_cols) >= 1:
            # Histogram for single numeric column
            fig = px.histogram(
                query_result, 
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_cols[1]:
        if len(numeric_cols) >= 2:
            # Scatter plot for two numeric columns
            color_col = categorical_cols[0] if categorical_cols else None
            fig = px.scatter(
                query_result.head(100),
                x=numeric_cols[0],
                y=numeric_cols[1],
                color=color_col,
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif len(categorical_cols) >= 1:
            # Pie chart for categorical data
            value_counts = query_result[categorical_cols[0]].value_counts().head(10)
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {categorical_cols[0]}"
            )
            st.plotly_chart(fig, use_container_width=True)

# Chat Interface
st.markdown("### 💬 Chat with Your Data")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display charts if they exist in the message
            if "chart_data" in message and message["chart_data"] is not None:
                create_enhanced_visualizations(message["chart_data"], message.get("chart_title", "Query Results"))
            
            # Display dataframe if it exists
            if "dataframe" in message and message["dataframe"] is not None:
                st.dataframe(message["dataframe"], use_container_width=True)
                
                # Download button
                csv = message["dataframe"].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f'query_results_{st.session_state.chart_counter}.csv',
                    mime='text/csv',
                )

# Chat input
user_input = st.chat_input("Ask me anything about your data... 💭")

if user_input:
    # Add user message to history
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🧠 Analyzing your question..."):
        try:
            # Determine question type
            question_type = determine_question_type(user_input, schema_context)
            
            if question_type == "SQL":
                # Generate and execute SQL query
                sql_query = generate_sql_query(user_input, schema_context)
                
                # Execute SQL query
                query_result = ps.sqldf(sql_query, locals())
                
                # Prepare response
                if not query_result.empty:
                    response = f"""**🔍 SQL Query Generated:**
```sql
{sql_query}
```

**📊 Results:** Found {len(query_result)} rows"""
                    
                    # Add to chat history with chart data
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": response,
                        "dataframe": query_result,
                        "chart_data": query_result,
                        "chart_title": f"Results for: {user_input[:50]}..."
                    })
                else:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": f"""**🔍 SQL Query Generated:**
```sql
{sql_query}
```

**❌ No results found.** The query executed successfully but returned no data."""
                    })
            
            else:  # Insights mode
                insights = generate_insights(user_input, df)
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"**📈 Data Insights:**\n\n{insights}"
                })
        
        except Exception as e:
            error_msg = f"❌ **Error occurred:** {str(e)}\n\n💡 *Try rephrasing your question or check if column names match the schema.*"
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": error_msg
            })
    
    # Rerun to display new messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚀 Powered by Google Gemini AI | Built with Streamlit | 
    <a href="#" style="color: #667eea;">Need Help?</a>
</div>
""", unsafe_allow_html=True)