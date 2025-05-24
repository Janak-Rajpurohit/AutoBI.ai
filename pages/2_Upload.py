import streamlit as st
import pandas as pd
# from pandas_profiling import ProfileReport
import time
from autoviz.AutoViz_Class import AutoViz_Class
import matplotlib.pyplot as plt
from datetime import datetime



from fpdf import FPDF
import os

def create_pdf_from_images(image_folder, pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(0)
    
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.endswith(".png"):
            pdf.add_page()
            pdf.image(os.path.join(image_folder, img_file), x=10, y=10, w=190)  # Adjust size

    pdf.output(pdf_path)


# Page configuration with improved title and favicon
st.set_page_config(
    page_title="AI Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the UI
st.markdown("""
<style>
    /* Main Styling */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* General styles */
    .stApp {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        padding-bottom: 1.5rem;
    }
    
    .main-header h1 {
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .main-header p {
        color: #666;
        font-size: 1.1rem;
        margin-top: 0;
    }
    
    /* Card containers */
    .card-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Progress container */
    .progress-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-pending {
        background-color: #FFC107;
    }
    
    .status-complete {
        background-color: #4CAF50;
    }
    
    /* File uploader styling */
    .upload-container {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        color: #666;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .upload-container:hover {
        border-color: #0D47A1;
        background-color: #E3F2FD;
    }
    
    .upload-icon {
        font-size: 3rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    
    /* Feature cards */
    .feature-card {
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        color: #1E88E5;
    }
    
    /* Button styling */
    .css-1x8cf1d {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
    }
    
    .css-1x8cf1d:hover {
        background-color: #1565C0;
    }
    
    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .step {
        flex: 1;
        text-align: center;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0 0.5rem;
        background-color: #f0f0f0;
    }
    
    .step.active {
        background-color: #E3F2FD;
        color: #1565C0;
        font-weight: 500;
    }
    
    .step.complete {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    
    /* Info bubbles */
    .info-bubble {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    
    /* Stats box */
    .stats-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .stats-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.3rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Type indicators */
    .type-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    
    .type-datetime {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    
    .type-numerical {
        background-color: #E3F2FD;
        color: #1565C0;
    }
    
    .type-categorical {
        background-color: #FFF3E0;
        color: #E65100;
    }
    
    .type-boolean {
        background-color: #F3E5F5;
        color: #7B1FA2;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        text-align: center;
        color: #999;
        font-size: 0.8rem;
    }
    
    /* Divider */
    .custom-divider {
        margin: 2rem 0;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Header with improved styling
st.markdown("""
<div class="main-header">
    <h1>📊 AI Business Intelligence Platform</h1>
    <p>Automated data analysis, insights, and reporting for your business data</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'col_types' not in st.session_state:
    st.session_state.col_types = {}
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []
if 'upload_time' not in st.session_state:
    st.session_state.upload_time = None

# Improved datetime detection function
def detect_column_type(series):
    # First try to detect datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"
    
    # Try explicit datetime conversion for object columns
    if pd.api.types.is_object_dtype(series):
        try:
            # Sample some values for datetime detection
            sample = series.dropna().sample(min(10, len(series)))
            if any(pd.to_datetime(sample, errors='coerce', infer_datetime_format=True).notna()):
                return "Datetime"
        except:
            pass
    
    # Check other types
    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    return "Categorical"

# Show progress steps
# def show_progress_steps():
#     steps = ["Upload Data", "Confirm Types", "Generate Report", "Explore Insights"]
    
#     if st.session_state.df is None:
#         active_step = 0
#     elif not st.session_state.report_generated:
#         active_step = 1
#     else:
#         active_step = 2
    
#     st.markdown('<div class="step-container">', unsafe_allow_html=True)
    
#     for i, step in enumerate(steps):
#         if i < active_step:
#             st.markdown(f'<div class="step complete">✓ {step}</div>', unsafe_allow_html=True)
#         elif i == active_step:
#             st.markdown(f'<div class="step active">● {step}</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="step">{step}</div>', unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # Display the progress steps
# show_progress_steps()

# File upload section with improved UI
if st.session_state.df is None:
    st.markdown("### 📤 Upload Your Data")
    st.caption("""Upload your CSV file to get started with automated analysis. The system will intelligently detect column types and provide insights.""")
    
    # Create three columns for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧹</div>
            <h4>Automatic Cleaning</h4>
            <p>Your data will be automatically cleaned and prepared for analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h4>Smart Detection</h4>
            <p>Column types are intelligently detected, including dates and categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h4>Complete Analysis</h4>
            <p>Get comprehensive statistics, correlations, and visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Improved file uploader
    st.markdown("""
    <div class="upload-container">
        <div class="upload-icon">📁</div>
        <h3>Drag and drop your CSV file here</h3>
        <p>or click to browse files</p>
    </div>
    """, unsafe_allow_html=True)
    
    f = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum file size: 200MB",
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
        <b>Supported format:</b> CSV files (.csv)<br>
        <b>Requirements:</b> Headers in first row, clean data preferred
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle file upload
    if f:
        try:
            with st.spinner("Reading CSV file..."):
                # Read with explicit datetime parsing
                df = pd.read_csv(f, parse_dates=True)
                df.reset_index()
                date_columns = []  # Temporary list to store detected date columns
                
                # Show progress animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(101):
                    progress_bar.progress(i)
                    if i < 30:
                        status_text.text("Reading file...")
                    elif i < 60:
                        status_text.text("Detecting column types...")
                    elif i < 90:
                        status_text.text("Processing data...")
                    else:
                        status_text.text("Finalizing...")
                    time.sleep(0.01)
                
                # Process columns
                for col in df.columns:
                    if detect_column_type(df[col]) == "Datetime":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        date_columns.append(col)
                
                # Store in session state
                st.session_state.df = df
                st.session_state.date_columns = date_columns
                st.session_state.upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Initialize column types with improved detection
                st.session_state.col_types = {
                    col: detect_column_type(df[col]) 
                    for col in df.columns
                }
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ File uploaded successfully!")
                time.sleep(0.5)
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please check if the file is a valid CSV and try again")

else:
    # Display file information with better styling
    df = st.session_state.df
    
    # File info card
    
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📄 Uploaded File")
        st.success(f"Your data has been successfully uploaded and is ready for analysis.")
    
    with col2:
        if st.button("Upload New File", type="primary", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Display file stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-value">{:,}</div>
            <div class="stats-label">Rows</div>
        </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-value">{}</div>
            <div class="stats-label">Columns</div>
        </div>
        """.format(df.shape[1]), unsafe_allow_html=True)
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum()
        memory_display = f"{memory_usage / 1048576:.2f} MB" if memory_usage > 1048576 else f"{memory_usage / 1024:.2f} KB"
        
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-value">{memory_display}</div>
            <div class="stats-label">Memory Usage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-value">{st.session_state.upload_time or "N/A"}</div>
            <div class="stats-label">Upload Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview with improved styling
    st.markdown("### 👁️ Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Column type summary
    col_type_counts = {"Datetime": 0, "Numerical": 0, "Categorical": 0, "Boolean": 0}
    for col_type in st.session_state.col_types.values():
        if col_type in col_type_counts:
            col_type_counts[col_type] += 1
    
    st.markdown("#### Column Type Summary")
    cols = st.columns(4)
    for i, (type_name, count) in enumerate(col_type_counts.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-value">{count}</div>
                <div class="stats-label">{type_name} Columns</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('---')
    
    # Column type confirmation with improved UI
    
    st.markdown("### 🔄 Column Type Configuration")
    st.caption("Review and adjust detected column types below. Accurate column type detection improves the quality of analysis.")
# Organize columns by type
col_by_type = {"Datetime": [], "Numerical": [], "Categorical": [], "Boolean": []}
for col, col_type in st.session_state.col_types.items():
    if col_type in col_by_type:
        col_by_type[col_type].append(col)

# Use native Streamlit tabs
tab_labels = ["Datetime Columns", "Numerical Columns", "Categorical Columns", "Boolean Columns"]
tabs = st.tabs(tab_labels)

new_types = {}

# Datetime Columns Tab
with tabs[0]:
    st.header("Datetime Columns")
    if col_by_type["Datetime"]:
        for col in col_by_type["Datetime"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{col}**")
                st.markdown(f"Sample: `{df[col].iloc[0] if not df[col].empty else 'N/A'}`")
            with col2:
                new_type = st.selectbox(
                    "Type",
                    options=["Datetime", "Numerical", "Categorical", "Boolean"],
                    index=0,
                    key=f"type_select_{col}"
                )
            st.divider()
            new_types[col] = new_type
    else:
        st.info("No datetime columns detected.")

# Numerical Columns Tab
with tabs[1]:
    st.header("Numerical Columns")
    if col_by_type["Numerical"]:
        for col in col_by_type["Numerical"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{col}**")
                st.markdown(f"Range: `{df[col].min()} - {df[col].max()}`")
            with col2:
                new_type = st.selectbox(
                    "Type",
                    options=["Datetime", "Numerical", "Categorical", "Boolean"],
                    index=1,
                    key=f"type_select_{col}"
                )
            st.divider()
            new_types[col] = new_type
    else:
        st.info("No numerical columns detected.")

# Categorical Columns Tab
with tabs[2]:
    st.header("Categorical Columns")
    if col_by_type["Categorical"]:
        for col in col_by_type["Categorical"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{col}**")
                unique_count = df[col].nunique()
                st.markdown(f"Unique values: `{unique_count}`")
            with col2:
                new_type = st.selectbox(
                    "Type",
                    options=["Datetime", "Numerical", "Categorical", "Boolean"],
                    index=2,
                    key=f"type_select_{col}"
                )
            st.divider()
            new_types[col] = new_type
    else:
        st.info("No categorical columns detected.")

# Boolean Columns Tab
with tabs[3]:
    st.header("Boolean Columns")
    if col_by_type["Boolean"]:
        for col in col_by_type["Boolean"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{col}**")
                st.markdown(f"Values: `{df[col].unique().tolist()}`")
            with col2:
                new_type = st.selectbox(
                    "Type",
                    options=["Datetime", "Numerical", "Categorical", "Boolean"],
                    index=3,
                    key=f"type_select_{col}"
                )
            st.divider()
            new_types[col] = new_type
    else:
        st.info("No boolean columns detected.")


    # Save button for column types
    if st.button("Save Column Type Changes", use_container_width=True):
        st.session_state.col_types = new_types
        st.success("Column types updated successfully!")

    st.markdown('---')
    
    # Report Configuration with improved UI
    
    st.markdown("### 📊 Report Configuration")
    st.caption("Configure your analysis options below before generating the report.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Settings")
        minimal_mode = st.checkbox("Minimal Mode (faster analysis)", True)
        dark_mode = st.checkbox("Dark Mode Theme", False)
    
    with col2:
        st.markdown("#### Analysis Options")
        explorative = st.checkbox("Include Exploratory Analysis", True)
        correlation_analysis = st.checkbox("Include Correlation Analysis", True)
    
    st.markdown('---')
    
    # Generate Report Section
    
    st.markdown("### 🚀 Generate Report")
    
    if not st.session_state.report_generated:
        st.caption("Click the button below to generate a comprehensive data analysis report with statistics, visualizations, and insights.")
        
        generate_col1, generate_col2 = st.columns([1, 3])
        with generate_col1:
            generate_button = st.button("Generate Report", type="primary", use_container_width=True)
        with generate_col2:
            st.markdown("""
            <div style="padding-top: 5px;">
                This may take a few minutes depending on the size of your dataset.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("""Your report has been generated! You can view it below or regenerate if you've made changes.""")
        
        generate_col1, generate_col2 = st.columns([1, 3])
        with generate_col1:
            generate_button = st.button("Regenerate Report", type="primary", use_container_width=True)
        with generate_col2:
            st.markdown("""
            <div style="padding-top: 5px;">
                Regenerate if you've updated column types or configuration settings.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('---')
    
    if not st.session_state.report_generated or generate_button:
        with st.spinner("Generating comprehensive EDA report..."):
            # st.markdown('<div class="progress-card">', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Convert columns to their selected types
            df_typed = df.copy()
            
            # Phase 1: Converting column types
            status_text.text("Phase 1/3: Converting column types...")
            for i in range(33):
                progress_bar.progress(i)
                time.sleep(0.02)
                
            for col, col_type in st.session_state.col_types.items():
                try:
                    if col_type == "Datetime":
                        df_typed[col] = pd.to_datetime(df_typed[col], errors='coerce')
                    elif col_type == "Numerical":
                        df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
                    elif col_type == "Boolean":
                        df_typed[col] = df_typed[col].astype(bool)
                except Exception as e:
                    st.warning(f"Could not convert column '{col}' to {col_type}: {str(e)}")
            
            # Phase 2: Generating profile
            status_text.text("Phase 2/3: Analyzing data patterns...")
            for i in range(33, 66):
                progress_bar.progress(i)
                time.sleep(0.02)
                
            # Phase 3: Creating visualizations
            status_text.text("Phase 3/3: Creating visualizations and insights...")
            for i in range(66, 100):
                progress_bar.progress(i)
                time.sleep(0.02)
            
            # Generate the profile report
            # st.session_state.profile = ProfileReport(
            #     df_typed,
            #     title="Data Analysis Report",
            #     minimal=minimal_mode,
            #     explorative=explorative,
            #     dark_mode=dark_mode,
            #     progress_bar=True
            # )
            
            av = AutoViz_Class()
            plt.close('all')

            df_av = av.AutoViz(
                filename="",
                sep=",",
                save_plot_dir="pages/autoviz",
                depVar="",
                dfte=df,
                header=0,
                verbose=2,
                lowess=False,
                chart_format="png",
                max_rows_analyzed=150000,
                max_cols_analyzed=30,
            )
            st.session_state.profile = df_av
            status_text.text("Report generation completed!")
            # Grab all current figures from matplotlib
            figs = [plt.figure(n) for n in plt.get_fignums()]
            if len(figs) == 0:
                st.warning("No plots generated.")
            else:
                st.write(f"{len(figs)} plots generated.")
                for fig in figs:
                    st.pyplot(fig)            

            # Finishing
            status_text.text("Finalizing report...")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            st.session_state.report_generated = True
            status_text.empty()
            progress_bar.empty()
            st.success("Report generated successfully!")
            time.sleep(0.5)
            st.rerun()

# Display the report
if st.session_state.report_generated and st.session_state.profile is not None:
    st.caption("Below is your comprehensive data analysis report. You can download it for offline viewing and sharing.")
    

    figs = [plt.figure(n) for n in plt.get_fignums()]
    if len(figs) == 0:
        st.warning("No plots generated.")
    else:
        st.write(f"{len(figs)} plots generated.")
        for fig in figs:
            st.pyplot(fig)         
    # Download button
    # report_html = st.session_state.profile.to_html()
    pdf_path = create_pdf_from_images("./pages/autoviz/AutoViz","./pages/autoviz/report.pdf")
    with open("./pages/autoviz/report.pdf", "rb") as f:
        pdf_bytes = f.read()
    st.download_button(
        label="📥 Download Full Report",
        data=pdf_bytes,
        file_name="data_analysis_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    st.markdown('---')
    
    # Next steps section
    st.markdown("### 🔜 Next Steps")
    
    next_col1, next_col2, next_col3 = st.columns(3)
    
    with next_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h4>Train ML Models</h4>
            <p>Use your prepared data to train predictive models</p>
            <a href="#" target="_self">Continue to ML Models →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with next_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h4>Generate Forecasts</h4>
            <p>Create time series forecasts for your business metrics</p>
            <a href="#" target="_self">Continue to Forecasting →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with next_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h4>Ask AI Assistant</h4>
            <p>Get insights and recommendations from our AI</p>
            <a href="#" target="_self">Continue to AI Insights →</a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('---')

# Footer with improved styling
st.markdown("""
<div class="footer">
<p>© 2025 AI Business Intelligence Platform | Powered by Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)