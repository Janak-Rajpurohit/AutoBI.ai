import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import numpy as np
from datetime import datetime, timedelta
import json
import re
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Gemini API
API_KEY = "AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY"
genai.configure(api_key=API_KEY)

# Page Config
st.set_page_config(
    page_title="Auto Dashboard Builder", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dashboard styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .kpi-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 1rem;
        color: #666;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .trend-up {
        color: #28a745;
        font-weight: bold;
    }
    
    .trend-down {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DashboardBuilder:
    def __init__(self, df):
        self.df = df
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = self._detect_date_columns()
        
    def _detect_date_columns(self):
        """Detect potential date columns"""
        date_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Try to parse as date
                try:
                    pd.to_datetime(self.df[col].head(10))
                    date_cols.append(col)
                except:
                    continue
        return date_cols
    
    def analyze_data_with_ai(self):
        """Use Gemini to analyze data and suggest KPIs and insights"""
        
        # Prepare data summary for AI
        data_summary = f"""
            Dataset Analysis Request:

            Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns

            Numeric Columns: {self.numeric_cols}
            Categorical Columns: {self.categorical_cols}
            Date Columns: {self.date_cols}

            Sample Data:
            {self.df.head(10).to_string()}

            Statistical Summary:
            {self.df.describe().to_string()}

            Please analyze this data and provide:
            1. Key Business KPIs (5-8 important metrics)
            2. Business insights and patterns
            3. Recommended chart types for visualization
            4. Data quality observations
            5. Potential business questions this data could answer

            Format your response as JSON with the following structure:
            {{
                "kpis": [
                    {{"name": "KPI Name", "value": "calculation_method", "description": "what it means"}},
                    ...
                ],
                "insights": [
                    "insight 1",
                    "insight 2",
                    ...
                ],
                "chart_recommendations": [
                    {{"type": "chart_type", "columns": ["col1", "col2"], "purpose": "what it shows"}},
                    ...
                ],
                "business_questions": [
                    "question 1",
                    "question 2",
                    ...
                ]
            }}
            """

        
        try:
            response = self.model.generate_content(data_summary)
            # Clean and parse JSON response
            json_text = response.text
            json_text = re.sub(r'```json\n?', '', json_text)
            json_text = re.sub(r'\n?```', '', json_text)
            
            return json.loads(json_text)
        except Exception as e:
            st.error(f"AI Analysis Error: {e}")
            return self._default_analysis()
    
    def _default_analysis(self):
        """Fallback analysis if AI fails"""
        return {
            "kpis": [
                {"name": "Total Records", "value": len(self.df), "description": "Total number of records"},
                {"name": "Completeness", "value": f"{(1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))*100:.1f}%", "description": "Data completeness percentage"}
            ],
            "insights": [
                f"Dataset contains {self.df.shape[0]} records with {self.df.shape[1]} features",
                f"Found {len(self.numeric_cols)} numeric and {len(self.categorical_cols)} categorical columns"
            ],
            "chart_recommendations": [],
            "business_questions": ["What are the main trends in this data?"]
        }
    
    def calculate_advanced_kpis(self):
        """Calculate various KPIs based on data type"""
        kpis = {}
        
        # Basic KPIs
        kpis['total_records'] = len(self.df)
        kpis['total_columns'] = len(self.df.columns)
        kpis['data_completeness'] = (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        
        # Numeric column KPIs
        if self.numeric_cols:
            for col in self.numeric_cols[:3]:  # Top 3 numeric columns
                kpis[f'{col}_mean'] = self.df[col].mean()
                kpis[f'{col}_median'] = self.df[col].median()
                kpis[f'{col}_std'] = self.df[col].std()
                kpis[f'{col}_max'] = self.df[col].max()
                kpis[f'{col}_min'] = self.df[col].min()
        
        # Categorical KPIs
        if self.categorical_cols:
            for col in self.categorical_cols[:2]:
                kpis[f'{col}_unique'] = self.df[col].nunique()
                kpis[f'{col}_mode'] = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "N/A"
        
        return kpis
    
    def create_correlation_analysis(self):
        """Create correlation matrix and analysis"""
        if len(self.numeric_cols) >= 2:
            corr_matrix = self.df[self.numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            fig.update_layout(height=500)
            
            return fig, corr_matrix
        return None, None
    
    def create_distribution_analysis(self):
        """Create distribution plots for numeric columns"""
        if not self.numeric_cols:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"Distribution of {col}" for col in self.numeric_cols[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(self.numeric_cols[:4]):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, nbinsx=30),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=600, title_text="Distribution Analysis")
        return fig
    
    def create_categorical_analysis(self):
        """Create analysis for categorical columns"""
        if not self.categorical_cols:
            return None
            
        charts = []
        for col in self.categorical_cols[:4]:
            value_counts = self.df[col].value_counts().head(10)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Top Categories in {col}",
                labels={'x': col, 'y': 'Count'}
            )
            fig.update_layout(height=400)
            charts.append(fig)
        
        return charts
    
    def create_outlier_analysis(self):
        """Detect and visualize outliers"""
        if not self.numeric_cols:
            return None, []
            
        outliers_info = []
        fig = make_subplots(
            rows=1, cols=min(3, len(self.numeric_cols)),
            subplot_titles=[f"Outliers in {col}" for col in self.numeric_cols[:3]]
        )
        
        for i, col in enumerate(self.numeric_cols[:3]):
            # Calculate outliers using IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_info.append({
                'column': col,
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100
            })
            
            fig.add_trace(
                go.Box(y=self.df[col], name=col),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400, title_text="Outlier Detection")
        return fig, outliers_info
    
    def create_time_series_analysis(self):
        """Create time series analysis if date columns exist"""
        if not self.date_cols or not self.numeric_cols:
            return None
            
        # Convert first date column to datetime
        date_col = self.date_cols[0]
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # Group by date and aggregate numeric columns
        numeric_col = self.numeric_cols[0]
        time_series = self.df.groupby(self.df[date_col].dt.date)[numeric_col].agg(['sum', 'mean', 'count']).reset_index()
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f'{numeric_col} - Sum', f'{numeric_col} - Average', 'Record Count'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=time_series[date_col], y=time_series['sum'], mode='lines+markers', name='Sum'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_series[date_col], y=time_series['mean'], mode='lines+markers', name='Average'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_series[date_col], y=time_series['count'], mode='lines+markers', name='Count'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title_text="Time Series Analysis")
        return fig
    
    def generate_business_insights(self, ai_analysis):
        """Generate business insights using AI analysis"""
        insights_prompt = f"""
        Based on this data analysis, provide 5-7 specific business insights:
        
        Data Summary:
        - Records: {len(self.df)}
        - Numeric Columns: {self.numeric_cols}
        - Categorical Columns: {self.categorical_cols}
        
        Key Statistics:
        {self.df.describe().to_string()}
        
        Previous AI Analysis: {ai_analysis}
        
        Provide actionable business insights in bullet points:
        """
        
        try:
            response = self.model.generate_content(insights_prompt)
            return response.text
        except:
            return "• Data contains multiple dimensions for analysis\n• Consider focusing on key performance indicators\n• Look for trends and patterns in the data"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Auto Dashboard Builder</h1>
        <p>AI-Powered Comprehensive Data Dashboard Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for dataframe in session state
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("📄 No data found! Please upload a CSV file from the main page first.")
        st.stop()
    
    df = st.session_state.df
    
    # Initialize dashboard builder
    with st.spinner("🔍 Analyzing your data with AI..."):
        dashboard = DashboardBuilder(df)
        ai_analysis = dashboard.analyze_data_with_ai()
        kpis = dashboard.calculate_advanced_kpis()
    
    # Dashboard Title and Overview
    st.markdown(f"""
    ## 📈 Dashboard Overview
    **Dataset:** {df.shape[0]:,} rows × {df.shape[1]} columns  
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    # KPI Section
    st.markdown('<div class="section-header">🎯 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    kpi_cols = st.columns(4)
    
    # Display main KPIs
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{kpis['total_records']:,}</p>
            <p class="kpi-label">Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{kpis['data_completeness']:.1f}%</p>
            <p class="kpi-label">Data Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{len(dashboard.numeric_cols)}</p>
            <p class="kpi-label">Numeric Fields</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[3]:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{len(dashboard.categorical_cols)}</p>
            <p class="kpi-label">Categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional KPIs from numeric columns
    if dashboard.numeric_cols:
        st.markdown("### 📊 Numeric Column Insights")
        metric_cols = st.columns(len(dashboard.numeric_cols[:4]))
        
        for i, col in enumerate(dashboard.numeric_cols[:4]):
            with metric_cols[i]:
                st.metric(
                    label=f"{col} (Avg)",
                    value=f"{kpis.get(f'{col}_mean', 0):.2f}",
                    delta=f"Max: {kpis.get(f'{col}_max', 0):.2f}"
                )
    
    # AI Insights Section
    st.markdown('<div class="section-header">🧠 AI-Generated Insights</div>', unsafe_allow_html=True)
    
    if 'insights' in ai_analysis:
        for insight in ai_analysis['insights']:
            st.markdown(f"""
            <div class="insight-box">
                💡 {insight}
            </div>
            """, unsafe_allow_html=True)
    
    # Business Insights
    with st.spinner("🔍 Generating business insights..."):
        business_insights = dashboard.generate_business_insights(ai_analysis)
    
    st.markdown("### 💼 Business Intelligence")
    st.markdown(business_insights)
    
    # Visualization Section
    st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tabs = st.tabs(["📊 Overview", "🔗 Correlations", "📈 Distributions", "📋 Categories", "⚠️ Outliers", "⏰ Time Series"])
    
    with tabs[0]:  # Overview
        col1, col2 = st.columns(2)
        
        with col1:
            if dashboard.numeric_cols:
                # Summary statistics chart
                stats_df = df[dashboard.numeric_cols].describe().T
                fig = px.bar(
                    stats_df, 
                    x=stats_df.index, 
                    y='mean',
                    title="Average Values by Column",
                    labels={'x': 'Columns', 'y': 'Average Value'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if dashboard.categorical_cols:
                # Category distribution
                cat_col = dashboard.categorical_cols[0]
                value_counts = df[cat_col].value_counts().head(8)
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Correlations
        corr_fig, corr_matrix = dashboard.create_correlation_analysis()
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Show strongest correlations
            if corr_matrix is not None:
                st.markdown("### 🔗 Strongest Correlations")
                # Get correlation pairs
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
                st.dataframe(corr_df.head(10))
        else:
            st.info("No numeric columns available for correlation analysis.")
    
    with tabs[2]:  # Distributions
        dist_fig = dashboard.create_distribution_analysis()
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
        else:
            st.info("No numeric columns available for distribution analysis.")
    
    with tabs[3]:  # Categories
        cat_charts = dashboard.create_categorical_analysis()
        if cat_charts:
            for i, chart in enumerate(cat_charts):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                
                if i % 2 == 0:
                    with col1:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("No categorical columns available for analysis.")
    
    with tabs[4]:  # Outliers
        outlier_fig, outlier_info = dashboard.create_outlier_analysis()
        if outlier_fig:
            st.plotly_chart(outlier_fig, use_container_width=True)
            
            st.markdown("### ⚠️ Outlier Summary")
            outlier_df = pd.DataFrame(outlier_info)
            st.dataframe(outlier_df)
        else:
            st.info("No numeric columns available for outlier analysis.")
    
    with tabs[5]:  # Time Series
        ts_fig = dashboard.create_time_series_analysis()
        if ts_fig:
            st.plotly_chart(ts_fig, use_container_width=True)
        else:
            st.info("No date columns detected for time series analysis.")
    
    # Data Quality Section
    st.markdown('<div class="section-header">🔍 Data Quality Assessment</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Missing Values")
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df)) * 100
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if not missing_df.empty:
            fig = px.bar(missing_df, x='Column', y='Missing %', title="Missing Data by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
    
    with col2:
        st.markdown("### Data Types")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=[str(dtype) for dtype in dtype_counts.index],
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary Report
    st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Characteristics")
        st.write(f"• **Size**: {df.shape[0]:,} records, {df.shape[1]} features")
        st.write(f"• **Data Quality**: {kpis['data_completeness']:.1f}% complete")
        st.write(f"• **Numeric Features**: {len(dashboard.numeric_cols)}")
        st.write(f"• **Categorical Features**: {len(dashboard.categorical_cols)}")
        st.write(f"• **Date Features**: {len(dashboard.date_cols)}")
    
    with col2:
        st.markdown("### 🎯 Key Recommendations")
        if 'business_questions' in ai_analysis:
            for question in ai_analysis['business_questions'][:5]:
                st.write(f"• {question}")
        else:
            st.write("• Focus on data completeness and quality")
            st.write("• Investigate correlations between variables")
            st.write("• Monitor outliers and anomalies")
    
    # Export Dashboard Data
    st.markdown('<div class="section-header">💾 Export Options</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export summary statistics
        summary_stats = df.describe()
        csv_stats = summary_stats.to_csv()
        st.download_button(
            label="📊 Download Summary Statistics",
            data=csv_stats,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export correlation matrix
        if len(dashboard.numeric_cols) >= 2:
            corr_csv = df[dashboard.numeric_cols].corr().to_csv()
            st.download_button(
                label="🔗 Download Correlation Matrix",
                data=corr_csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
    
    with col3:
        # Export insights as text
        insights_text = f"""
Dashboard Insights Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AI Insights:
{chr(10).join(ai_analysis.get('insights', []))}

Business Insights:
{business_insights}

KPIs:
Total Records: {kpis['total_records']:,}
Data Quality: {kpis['data_completeness']:.1f}%
        """
        
        st.download_button(
            label="📝 Download Insights Report",
            data=insights_text,
            file_name="dashboard_insights.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()