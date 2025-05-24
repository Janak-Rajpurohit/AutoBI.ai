import streamlit as st
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
import time
import pickle

st.set_page_config(page_title="🔮 ML Prediction", layout="wide")
st.title("🔮 AutoML")
st.markdown("Upload a CSV on the **Upload** page first. Then train and test ML models here automatically.")

# Load data from session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("❗ Please upload a CSV file on the 'Upload' page first.")
    st.stop()

df = st.session_state.df
df=df.drop(st.session_state.date_columns,axis=1)
print("date dropped")
# Target column selection
st.subheader("🎯 Select the Target Column")
target = st.selectbox("Choose the column you want to predict:", df.columns)

# Show data preview
with st.expander("📂 Preview Uploaded Data"):
    st.dataframe(df.head())
    st.write(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Detect task type
if df[target].nunique() <= 20 and df[target].dtype in ['object', 'int', 'category']:
    task = 'classification'
else:
    task = 'regression'

st.info(f"📌 Detected ML Task: **{task.title()}**")

# Start training
if st.button("🚀 Start Training"):
    with st.spinner("Setting up experiment..."):
        progress_bar = st.progress(0)
        time.sleep(1)

        # Manually specify categorical columns before training
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if task == 'classification':
            exp = ClassificationExperiment()
        else:
            exp = RegressionExperiment()

        # Use a subset of the data for faster comparisons
        sample_df = df.sample(frac=0.2, random_state=123)  # Use 20% of the data for comparison

        # Setup with optimizations
        exp.setup(data=sample_df, 
          target=target, 
          session_id=123, 
          preprocess=True, 
          train_size=0.8, 
          categorical_features=categorical_cols, 
          use_gpu=False, 
          n_jobs=-1)

        progress_bar.progress(30)

        st.write("🔍 Comparing models...")
        # Use different models for classification and regression
        if task == 'classification':
            top_models = exp.compare_models(include=['rf', 'xgboost'], n_select=3)
        else:  # Regression
            top_models = exp.compare_models(
                include=['rf', 'xgboost', 'lasso', 'ridge', 'en', 'gbr', 'et'], 
                n_select=3
            )
        best_model = top_models[0]
        results_df = exp.pull()
        progress_bar.progress(60)

        with st.expander("📊 Model Comparison Results"):
            st.dataframe(results_df)

        progress_bar.progress(70)
        exp.evaluate_model(best_model)
        progress_bar.progress(85)

        st.success("✅ Model training completed!")
        progress_bar.progress(100)

        # Retrain best model on the full dataset
        st.write("🔄 Retraining best model on the full dataset...")
        exp.setup(data=df, target=target, session_id=123, preprocess=True, train_size=0.8, 
                  categorical_features=categorical_cols, use_gpu=False, n_jobs=-1)
        best_model = exp.finalize_model(best_model)

        # Highlight best model
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px">
        <h4>🏆 Best Model: <code>{str(best_model).split('(')[0]}</code></h4>
        <p style="color:green;font-weight:bold;font-size:18px;">
        ✔️ Accuracy/Error Metrics shown above.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # Save model in session
        st.session_state.trained_model = best_model
        st.session_state.task_type = task
        st.session_state.exp_obj = exp


# Prediction section
if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
    st.subheader("🧪 Test the Model with New Inputs")
    st.markdown("Enter input values below to make predictions:")

    model = st.session_state.trained_model
    exp = st.session_state.exp_obj

    input_data = {}
    input_form_cols = st.columns(2)
    i = 0

    # Iterate over feature columns (Ensure feature columns match the training dataset)
    for col in exp.get_config("X_train").columns:
        col_type = df[col].dtype
        with input_form_cols[i % 2]:
            if col_type == 'object' or df[col].nunique() < 10:
                # For categorical columns
                value = st.selectbox(f"{col} (categorical)", df[col].unique())
            elif str(col_type).startswith('int') or str(col_type).startswith('float'):
                # For numerical columns
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                value = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # For date columns
                value = st.date_input(f"{col}", value=pd.to_datetime(df[col].mean()))
            else:
                # For all other types, use text input
                value = st.text_input(f"{col}")
            input_data[col] = value
        i += 1

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict using the model
    if st.button("🔍 Predict"):
        prediction = exp.predict_model(model, data=input_df)
        st.subheader("📈 Prediction Output")
        st.dataframe(prediction)
        prediction_value = prediction["prediction_label"].iloc[0]
        st.success(f"🔮 Predicted {target}: **{prediction_value}**")


# Download model
if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
    with st.expander("📥 Download Trained Model"):
        model_bytes = pickle.dumps(st.session_state.trained_model)
        st.download_button(
            label="💾 Download .pkl File",
            data=model_bytes,
            file_name="trained_model.pkl"
        )