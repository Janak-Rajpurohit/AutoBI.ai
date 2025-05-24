import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import json
import time
from datetime import datetime
import numpy as np

# --- Enhanced Configuration & Setup ---
st.set_page_config(
    page_title="AI Dashboard Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI/UX
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff9800;
        --error-color: #d62728;
        --background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide sidebar completely */
    .css-1d391kg {display: none;}
    .css-1rs6os {display: none;}
    .css-17eq0hr {display: none;}
    
    /* Custom styling for metrics */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #fff;
    }
    
    /* Warning boxes */
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #fff;
    }
    
    /* Success boxes */
    .success-box {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #fff;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_KEY = "AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY"
os.environ['GEMINI_API_KEY'] = API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🔑 GEMINI_API_KEY not found. Please set it in Streamlit Secrets or as an environment variable.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {e}")
    st.stop()

# --- Enhanced Helper Functions ---
def create_loading_animation():
    """Create a custom loading animation"""
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown("🤖 Analyzing your data" + "." * (i + 1))
        time.sleep(0.5)
    placeholder.empty()

def display_custom_metric(label, value, prefix="", suffix="", color_gradient="135deg, #667eea 0%, #764ba2 100%"):
    """Display a custom styled metric"""
    st.markdown(f"""
    <div class="metric-container" style="background: linear-gradient({color_gradient});">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)

def profile_dataframe(df):
    """Enhanced DataFrame profiling with better error handling"""
    if not isinstance(df, pd.DataFrame):
        return None
    
    try:
        cols = df.columns.tolist()
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        sample_data = df.head().to_string()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Enhanced date column detection
        date_cols = []
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif any(date_word in col.lower() for date_word in ['date', 'time', 'created', 'updated', 'timestamp']):
                try:
                    pd.to_datetime(df[col].head(), errors='raise', infer_datetime_format=True)
                    date_cols.append(col)
                except (ValueError, TypeError):
                    pass

        # Data quality metrics
        missing_data = df.isnull().sum().to_dict()
        data_quality = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": sum(missing_data.values()),
            "completeness": (1 - sum(missing_data.values()) / (len(df) * len(df.columns))) * 100
        }

        return {
            "columns": cols,
            "dtypes": dtypes,
            "sample_data": sample_data,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "date_cols": list(set(date_cols)),
            "description": df.describe(include='all').to_string(),
            "missing_data": missing_data,
            "data_quality": data_quality
        }
    except Exception as e:
        st.warning(f"⚠️ Error profiling DataFrame: {e}")
        return None

def get_gemini_suggestions(df_profile, user_query=""):
    """Enhanced Gemini suggestions with better prompting"""
    if not df_profile:
        st.error("📊 DataFrame profile is missing, cannot get Gemini suggestions.")
        return None

    # Enhanced prompt with more specific instructions
    prompt = f"""
    You are an expert data analyst and dashboard designer. Create comprehensive dashboard suggestions for the following dataset:

    DATASET PROFILE:
    - Columns: {df_profile['columns']}
    - Data Types: {df_profile['dtypes']}
    - Numeric Columns: {df_profile['numeric_cols']}
    - Categorical Columns: {df_profile['categorical_cols']}
    - Date Columns: {df_profile['date_cols']}
    - Data Quality: {df_profile['data_quality']}
    
    Sample Data:
    {df_profile['sample_data']}

    {f"USER REQUIREMENTS: {user_query}" if user_query else ""}

    Provide suggestions in VALID JSON format with these exact keys:

    {{
        "kpis": [
            {{
                "name": "Clear KPI Name",
                "column": "column_name_or_array",
                "calculation": "sum|mean|count|nunique|median|std|min|max|correlation",
                "prefix": "$",
                "suffix": "%",
                "description": "What this KPI measures"
            }}
        ],
        "charts": [
            {{
                "chart_type": "bar|line|scatter|pie|histogram|boxplot|heatmap",
                "title": "Descriptive Chart Title",
                "x_column": "column_name",
                "y_column": "column_name_or_array",
                "color_by_column": "optional_column",
                "aggregation_on_y": "sum|mean|count",
                "time_series_column": "date_column_if_applicable",
                "names_column": "for_pie_charts",
                "values_column": "for_pie_charts",
                "rationale": "Why this chart is valuable",
                "interactivity": "hover|zoom|filter|drill_down"
            }}
        ],
        "insights": [
            "Specific business question or insight that can be answered",
            "Another actionable insight"
        ]
    }}

    REQUIREMENTS:
    - All column names must exist in the provided columns list
    - Prioritize business-relevant KPIs and charts
    - Include interactive elements where possible
    - Focus on actionable insights
    - Ensure calculations match data types
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        
        text_response = response.text.strip()
        json_start = text_response.find('{')
        json_end = text_response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text_response[json_start:json_end+1]
            return json.loads(json_str)
        else:
            st.error("❌ Could not extract valid JSON from Gemini's response.")
            with st.expander("🔍 View Raw Response"):
                st.text(text_response)
            return None

    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parsing error: {e}")
        with st.expander("🔍 View Raw Response"):
            st.text(response.text if 'response' in locals() else 'No response')
        return None
    except Exception as e:
        st.error(f"❌ API call error: {e}")
        return None

def calculate_kpi(df, kpi_info):
    """Enhanced KPI calculation with better formatting"""
    try:
        kpi_name = kpi_info.get('name', 'Unnamed KPI')
        column_spec = kpi_info.get('column')
        calculation = kpi_info.get('calculation', '').lower()
        
        if not column_spec or not calculation:
            return None, kpi_name, "Missing column or calculation"

        # Validate columns
        if isinstance(column_spec, list):
            for col in column_spec:
                if col not in df.columns:
                    return None, kpi_name, f"Column '{col}' not found"
        elif column_spec not in df.columns:
            return None, kpi_name, f"Column '{column_spec}' not found"

        # Perform calculation
        value = None
        if calculation == 'sum': value = df[column_spec].sum()
        elif calculation == 'mean': value = df[column_spec].mean()
        elif calculation == 'median': value = df[column_spec].median()
        elif calculation == 'nunique': value = df[column_spec].nunique()
        elif calculation == 'count': value = df[column_spec].count()
        elif calculation == 'std': value = df[column_spec].std()
        elif calculation == 'min': value = df[column_spec].min()
        elif calculation == 'max': value = df[column_spec].max()
        elif calculation == 'correlation':
            if isinstance(column_spec, list) and len(column_spec) == 2:
                if all(df[col].dtype.kind in 'ifc' for col in column_spec):
                    value = df[column_spec[0]].corr(df[column_spec[1]])
                else:
                    return None, kpi_name, "Correlation requires numeric columns"
            else:
                return None, kpi_name, "Correlation requires two columns"
        else:
            return None, kpi_name, f"Unsupported calculation: {calculation}"

        if pd.isna(value):
            return "N/A", kpi_name, "Calculation resulted in NaN"

        # Enhanced formatting
        if isinstance(value, (int, float)):
            if abs(value) >= 1e6:
                formatted_value = f"{value/1e6:.1f}M"
            elif abs(value) >= 1e3:
                formatted_value = f"{value/1e3:.1f}K"
            else:
                formatted_value = f"{value:,.2f}"
        else:
            formatted_value = str(value)

        prefix = kpi_info.get('prefix', '')
        suffix = kpi_info.get('suffix', '')
        final_value = f"{prefix}{formatted_value}{suffix}"
        
        return final_value, kpi_name, kpi_info.get('description', '')

    except Exception as e:
        return None, kpi_name, f"Error: {e}"

def create_enhanced_chart(df, chart_info):
    """Create enhanced interactive charts with better styling"""
    try:
        chart_type = chart_info.get('chart_type', '').lower()
        title = chart_info.get('title', 'Chart')
        x_col = chart_info.get('x_column')
        y_col = chart_info.get('y_column')
        color_col = chart_info.get('color_by_column')
        
        # Enhanced color palette - best theme colors
        color_sequence = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
                         '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
        
        fig = None
        
        # Data preparation
        df_chart = df.copy()
        
        # Apply time series conversion if needed
        if chart_info.get('time_series_column') and x_col:
            try:
                df_chart[x_col] = pd.to_datetime(df_chart[x_col], errors='coerce')
                df_chart = df_chart.sort_values(by=x_col)
            except:
                pass
        
        # Create charts with enhanced styling
        if chart_type == 'bar':
            fig = px.bar(df_chart, x=x_col, y=y_col, color=color_col, title=title,
                        color_discrete_sequence=color_sequence)
            fig.update_traces(hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>')
            
        elif chart_type == 'line':
            fig = px.line(df_chart, x=x_col, y=y_col, color=color_col, title=title,
                         markers=True, color_discrete_sequence=color_sequence)
            fig.update_traces(hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>')
            
        elif chart_type == 'scatter':
            fig = px.scatter(df_chart, x=x_col, y=y_col, color=color_col, title=title,
                           color_discrete_sequence=color_sequence)
            fig.update_traces(hovertemplate='<b>(%{x}, %{y})</b><extra></extra>')
            
        elif chart_type == 'pie':
            names_col = chart_info.get('names_column')
            values_col = chart_info.get('values_column')
            if names_col and values_col:
                df_pie = df_chart.groupby(names_col)[values_col].sum().reset_index()
                fig = px.pie(df_pie, names=names_col, values=values_col, title=title,
                           color_discrete_sequence=color_sequence)
                fig.update_traces(hovertemplate='<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>')
                
        elif chart_type == 'histogram':
            fig = px.histogram(df_chart, x=x_col, color=color_col, title=title,
                             color_discrete_sequence=color_sequence)
            
        elif chart_type == 'boxplot':
            fig = px.box(df_chart, x=x_col, y=y_col, color=color_col, title=title,
                        color_discrete_sequence=color_sequence)
            
        elif chart_type == 'heatmap':
            if len(df_chart.select_dtypes(include=[np.number]).columns) >= 2:
                corr_matrix = df_chart.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix, title=title, aspect="auto",
                              color_continuous_scale='RdBu_r')
        
        if fig:
            # Enhanced styling for all charts
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", size=12),
                title=dict(font=dict(size=16, color='#2c3e50'), x=0.5),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Add interactivity
            fig.update_layout(
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating chart '{title}': {e}")
        return None

# --- Enhanced Streamlit App ---
def main():
    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Dashboard Builder</h1>
        <p>Transform your data into stunning interactive dashboards with Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = None
    if 'df_profile' not in st.session_state:
        st.session_state.df_profile = None

    # Check if dataframe exists in session state
    if 'df' not in st.session_state or st.session_state.df is None:
        # No data available message
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2>📊 No Data Available</h2>
                <p style="font-size: 1.2rem; color: #666;">
                    Please upload your CSV file from the data upload page first to begin creating 
                    your AI-powered dashboard
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature highlights
            st.markdown("""
            ### ✨ Features
            - 🤖 **AI-Powered Analysis**: Gemini AI automatically suggests relevant KPIs and charts
            - 📊 **Interactive Charts**: Hover, zoom, and explore your data
            - 🎨 **Beautiful Design**: Modern, responsive dashboard layouts
            - ⚡ **Real-time Updates**: Instant dashboard generation
            - 📱 **Mobile Friendly**: Responsive design that works everywhere
            """)
    else:
        df = st.session_state.df
        
        # Generate df_profile if not exists
        if st.session_state.df_profile is None:
            st.session_state.df_profile = profile_dataframe(df)
        
        # Data overview section
        with st.expander("🔍 Data Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=['number']).columns)
                st.metric("🔢 Numeric", numeric_cols)
            with col4:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("❓ Missing %", f"{missing_pct:.1f}%")
            
            # Data preview with styling
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

        # Query input section with better UX
        st.markdown("### 💬 Tell AI What You Want to Explore")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_area(
                "Describe your analysis goals:",
                placeholder="e.g., 'Show me sales trends over time, top performing products, and customer segments by region'",
                height=100,
                help="Be specific about what insights you're looking for"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("✨ Generate Dashboard", type="primary", use_container_width=True):
                if st.session_state.df_profile:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("🧠 Analyzing your data...")
                        elif i < 60:
                            status_text.text("🤖 Generating AI suggestions...")
                        elif i < 90:
                            status_text.text("📊 Creating visualizations...")
                        else:
                            status_text.text("✨ Finalizing dashboard...")
                        time.sleep(0.02)
                    
                    suggestions = get_gemini_suggestions(st.session_state.df_profile, user_query)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if suggestions:
                        st.session_state.suggestions = suggestions
                        st.success("🎉 Dashboard generated successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to generate suggestions. Please try again.")
            
            if st.session_state.suggestions:
                if st.button("🔄 Reset Dashboard", use_container_width=True):
                    st.session_state.suggestions = None
                    st.rerun()

        # Display dashboard if suggestions exist
        if st.session_state.suggestions:
            suggestions = st.session_state.suggestions
            
            # KPIs Section with enhanced styling
            if 'kpis' in suggestions and suggestions['kpis']:
                st.markdown("### 📈 Key Performance Indicators")
                
                kpis = suggestions['kpis']
                cols = st.columns(min(len(kpis), 4))
                
                for i, kpi_info in enumerate(kpis):
                    with cols[i % 4]:
                        value, name, description = calculate_kpi(df, kpi_info)
                        if value is not None:
                            # Color gradients for different KPIs
                            gradients = [
                                "135deg, #667eea 0%, #764ba2 100%",
                                "135deg, #f093fb 0%, #f5576c 100%", 
                                "135deg, #4facfe 0%, #00f2fe 100%",
                                "135deg, #43e97b 0%, #38f9d7 100%"
                            ]
                            gradient = gradients[i % len(gradients)]
                            display_custom_metric(name, value, color_gradient=gradient)
                            if description:
                                st.caption(description)
                        else:
                            st.warning(f"⚠️ Could not calculate: {name}")

            # Charts Section with enhanced interactivity
            if 'charts' in suggestions and suggestions['charts']:
                st.markdown("### 📊 Interactive Visualizations")
                
                for i, chart_info in enumerate(suggestions['charts']):
                    with st.container():
                        st.markdown(f"""
                        <div class="chart-container">
                            <h4>{chart_info.get('title', f'Chart {i+1}')}</h4>
                            <p style="color: #666; margin-bottom: 1rem;">
                                {chart_info.get('rationale', 'Interactive chart for data exploration')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig = create_enhanced_chart(df, chart_info)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                            })
                        else:
                            st.warning(f"⚠️ Could not generate chart: {chart_info.get('title')}")
                        
                        st.markdown("---")

            # Insights Section
            if 'insights' in suggestions and suggestions['insights']:
                st.markdown("### 💡 AI-Generated Insights")
                
                for i, insight in enumerate(suggestions['insights']):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); 
                                color: white; padding: 1rem; border-radius: 10px; 
                                margin: 0.5rem 0; border-left: 4px solid #fff;">
                        <strong>Insight {i+1}:</strong> {insight}
                    </div>
                    """, unsafe_allow_html=True)

            # Raw data section for transparency
            with st.expander("🔧 Technical Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Generated Suggestions:**")
                    st.json(suggestions)
                with col2:
                    st.markdown("**Data Profile:**")
                    if st.session_state.df_profile:
                        st.json({
                            "columns": st.session_state.df_profile['columns'],
                            "data_types": st.session_state.df_profile['dtypes'],
                            "data_quality": st.session_state.df_profile['data_quality']
                        })

if __name__ == "__main__":
    main()