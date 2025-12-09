"""
Feature Importance Page - Model Interpretability
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from apps.utils import (
    load_model, get_available_models, plot_feature_importance,
    get_predictions
)

st.set_page_config(page_title="Feature Importance", page_icon="", layout="wide")

def show():
    st.title("Feature Importance & Model Interpretability")
    st.markdown("Understand which features drive model predictions using SHAP values and feature importance")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("Select Model")
    
    models = get_available_models()
    
    if not models:
        st.error("No models found. Please train a model first.")
        return
    
    model_names = [Path(m).name for m in models]
    selected_model_name = st.selectbox("Choose Model", model_names)
    selected_model_path = models[model_names.index(selected_model_name)]
    
    model = load_model(selected_model_path)
    
    if model is None:
        st.error("Failed to load model")
        return

    st.success(f" Model loaded: {selected_model_name}")

    st.markdown("---")
    
    # Tabs for different interpretability views
    tab1, tab2, tab3 = st.tabs(["Feature Importance", " SHAP Analysis", " Feature Details"])
    
    with tab1:
        global_importance(model)
    
    with tab2:
        shap_analysis(model)
    
    with tab3:
        feature_details(model)

def global_importance(model):
    """Display global feature importance"""
    st.markdown("### Feature Importance")
    st.markdown("Shows which features have the most impact on predictions across all samples")
    
    # Show info about ensemble features
    if hasattr(model, 'meta_model'):
        meta_model = model.meta_model
        trad_count = len(model.traditional_features) if hasattr(model, 'traditional_features') else 0
        behav_count = len(model.behavioral_features) if hasattr(model, 'behavioral_features') else 0
        total_count = trad_count + behav_count + 7  # 7 meta-features
        
        st.info(f"""
        **Ensemble Model Features:** This model uses all {total_count} combined features:
        - Traditional features: {trad_count}
        - Behavioral features: {behav_count}
        - Meta-features (predictions from base models): 7
        
        The meta-learner (CatBoost) learns from the complete feature set of both models.
        """)
    
    top_n = st.slider("Number of top features to display", 10, 50, 20)
    
    fig = plot_feature_importance(model, top_n=top_n)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        #st.markdown("---")
        # st.markdown("#### Interpretation Guide")
        
       # col1, col2 = st.columns(2)
        
        # with col1:
           # st.markdown("""
           # **What is Feature Importance?**
            
           # Feature importance measures how much each feature contributes to the model's predictions.
            
           # - **Higher values** = More influential features
           # - **Lower values** = Less influential features
            
           # The model uses these features to split decision trees.
          #   """)
        
        # with col2:
          #  st.success("""
           # **How to Use This Information:**
            
            #- Focus data quality efforts on top features
            #- Investigate why certain features rank high/low
            #- Consider feature engineering for high-impact variables
            #- Remove low-importance features to simplify model
            #""")
    else:
        st.warning("Could not generate feature importance plot. Model may not support feature_importances_.")

def shap_analysis(model):
    """SHAP-based model interpretability"""
    st.markdown("### SHAP")
    st.markdown("SHAP values explain individual predictions by showing each feature's contribution")
    
    # st.info("""
    # **About SHAP:**
    
    #SHAP provides both global and local interpretability:
    #- **Global**: Overall feature impact across dataset
    #- **Local**: How features affect a single prediction
    #""")
    
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Model SHAP", "Local (Upload / Manual Entry)"],
        horizontal=True
    )
    
    if analysis_type == "Model SHAP":
        global_shap(model)
    else:
        local_shap(model)

def global_shap(model):
    """Global SHAP summary plot"""
    st.markdown("#### SHAP Analysis")
    
    # Automatically load appropriate test data based on model type
    model_type = None
    test_data_path = None
    
    # Detect model type from model attributes or name
    if hasattr(model, 'meta_model'):
        model_type = 'ensemble'
        test_data_path = "data/smoke_hybrid_features.csv"
    elif hasattr(model, 'n_features_in_'):
        if model.n_features_in_ > 100:  # Traditional has 487 features
            model_type = 'traditional'
            test_data_path = "data/traditional_test_data.csv"
        else:  # Behavioral has 44 features
            model_type = 'behavioral'
            test_data_path = "data/behavioral_test_data.csv"
    
    # Try to load the test data
    df = None
    if test_data_path and Path(test_data_path).exists():
        try:
            df = pd.read_csv(test_data_path)
            st.success(f"Loaded test data: {test_data_path}")
        except Exception as e:
            st.error(f"Error loading test data: {e}")
    
    if df is None:
        st.warning("Could not automatically load test data. Please upload a CSV file.")
        uploaded_file = st.file_uploader("Upload sample data (CSV) for SHAP analysis", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            return
    
    # Sample for performance
    max_samples = st.slider("Max samples for SHAP computation", 100, 1000, 500)
    
    if len(df) > max_samples:
        df_sample = df.sample(max_samples, random_state=42)
        st.info(f"Sampled {max_samples} rows from {len(df)} for SHAP analysis")
    else:
        df_sample = df
    
    if st.button("Plot SHAP Values", type="primary"):
        try:
            import shap
            
            with st.spinner("Computing SHAP values... This may take a few minutes"):
                # Prepare data
                X = df_sample.copy()
                
                # Remove target column if present
                target_cols = ['TARGET', 'target', 'default.payment.next.month']
                for col in target_cols:
                    if col in X.columns:
                        X = X.drop(col, axis=1)
                
                # Handle ensemble model - use 538 features (7 meta + 531 raw)
                if hasattr(model, 'meta_model'):
                    from sklearn.preprocessing import LabelEncoder
                    from src.data_cleaning import impute_categorical_columns, impute_numeric_columns
                    
                    # Prepare features for each base model
                    X_trad = X[model.traditional_features].copy()
                    X_behav = X[model.behavioral_features].copy()
                    
                    # Clean data
                    X_trad = impute_categorical_columns(X_trad, fill_value='MISSING')
                    X_trad = impute_numeric_columns(X_trad, strategy='median')
                    for col in X_trad.columns:
                        if X_trad[col].dtype in ['object', 'category']:
                            le = LabelEncoder()
                            X_trad[col] = le.fit_transform(X_trad[col].astype(str))
                    
                    X_behav = impute_categorical_columns(X_behav, fill_value='MISSING')
                    X_behav = impute_numeric_columns(X_behav, strategy='median')
                    for col in X_behav.columns:
                        if X_behav[col].dtype in ['object', 'category']:
                            le = LabelEncoder()
                            X_behav[col] = le.fit_transform(X_behav[col].astype(str))
                    
                    # Generate meta-features from base model predictions
                    pred_trad = model.model_traditional.predict_proba(X_trad)[:, 1]
                    pred_behav = model.model_behavioral.predict_proba(X_behav)[:, 1]
                    
                    meta_features = pd.DataFrame({
                        'pred_traditional': pred_trad,
                        'pred_behavioral': pred_behav,
                        'pred_avg': (pred_trad + pred_behav) / 2,
                        'pred_max': np.maximum(pred_trad, pred_behav),
                        'pred_min': np.minimum(pred_trad, pred_behav),
                        'pred_diff': np.abs(pred_trad - pred_behav),
                        'pred_ratio': pred_trad / (pred_behav + 0.001)
                    })
                    
                    # Reset indices for proper concatenation
                    meta_features.reset_index(drop=True, inplace=True)
                    X_trad.reset_index(drop=True, inplace=True)
                    X_behav.reset_index(drop=True, inplace=True)
                    
                    # Combine with ORIGINAL feature names for CatBoost compatibility
                    X_for_shap = pd.concat([meta_features, X_trad, X_behav], axis=1)
                    
                    # Create feature name mapping with prefixes for display
                    feature_names_display = (
                        list(meta_features.columns) + 
                        [f'trad_{col}' for col in X_trad.columns] + 
                        [f'behav_{col}' for col in X_behav.columns]
                    )
                    
                    final_estimator = model.meta_model
                    
                # Handle regular models
                else:
                    # Get final estimator from pipeline
                    if hasattr(model, 'named_steps'):
                        final_estimator = list(model.named_steps.values())[-1]
                    else:
                        final_estimator = model
                    
                    # Align features
                    from apps.utils import align_features
                    X_for_shap = align_features(X, model)
                    feature_names_display = None
                
                # Compute SHAP
                explainer = shap.TreeExplainer(final_estimator)
                shap_values = explainer.shap_values(X_for_shap)
                
                # Create a copy with renamed columns for display
                if hasattr(model, 'meta_model') and feature_names_display:
                    X_for_display = X_for_shap.copy()
                    X_for_display.columns = feature_names_display
                else:
                    X_for_display = X_for_shap
                
                # Plot
                # st.success("SHAP values computed successfully!")
                
               # st.markdown("#### Feature Importance (Mean Absolute SHAP)")
               # fig, ax = plt.subplots(figsize=(10, 8))
               # shap.summary_plot(shap_values, X_for_display, plot_type="bar", show=False)
                # st.pyplot(fig)
                
                st.markdown("---")
                
                # Detailed summary
                st.markdown("#### SHAP Summary Plot (Feature Impact)")
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_for_display, show=False)
                st.pyplot(fig2)
                
                st.markdown("""
                **How to read this plot:**
                - Each dot is a sample
                - Red = High feature value
                - Blue = Low feature value
                - X-axis = SHAP value (impact on prediction)
                - Positive SHAP = Increases default probability
                - Negative SHAP = Decreases default probability
                """)
                
        except ImportError:
            st.error("SHAP library not installed. Install with: pip install shap")
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            st.exception(e)

def local_shap(model):
    """Local SHAP explanation for single prediction"""
    st.markdown("#### Local Explanation - Single Applicant")
    st.markdown("Upload a single-row CSV or enter data manually to see how each feature contributes to the prediction")
    
    input_method = st.radio("Input method:", ["Upload CSV", "Manual Entry"], horizontal=True)
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload single-row CSV", type=["csv"], key="local_shap")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if len(df) > 1:
                st.warning(f"File contains {len(df)} rows. Using only the first row.")
                df = df.head(1)
            
            st.dataframe(df, use_container_width=True)
            
            if st.button("Generate SHAP Explanation", type="primary"):
                generate_local_shap(model, df)
    else:
        st.info("Manual entry for local SHAP - Use the Prediction page for detailed manual input, then return here with results")

def generate_local_shap(model, X):
    """Generate local SHAP explanation"""
    try:
        import shap
        
        with st.spinner("Computing SHAP explanation..."):
            # Handle ensemble model differently
            if hasattr(model, 'meta_model'):
                # For ensemble, generate 538 features (7 meta + 531 raw)
                from sklearn.preprocessing import LabelEncoder
                from src.data_cleaning import impute_categorical_columns, impute_numeric_columns
                
                # Prepare features
                X_trad = X[model.traditional_features].copy()
                X_behav = X[model.behavioral_features].copy()
                
                # Clean data
                X_trad = impute_categorical_columns(X_trad, fill_value='MISSING')
                X_trad = impute_numeric_columns(X_trad, strategy='median')
                for col in X_trad.columns:
                    if X_trad[col].dtype in ['object', 'category']:
                        le = LabelEncoder()
                        X_trad[col] = le.fit_transform(X_trad[col].astype(str))
                
                X_behav = impute_categorical_columns(X_behav, fill_value='MISSING')
                X_behav = impute_numeric_columns(X_behav, strategy='median')
                for col in X_behav.columns:
                    if X_behav[col].dtype in ['object', 'category']:
                        le = LabelEncoder()
                        X_behav[col] = le.fit_transform(X_behav[col].astype(str))
                
                # Generate meta-features from base model predictions
                pred_trad = model.model_traditional.predict_proba(X_trad)[:, 1]
                pred_behav = model.model_behavioral.predict_proba(X_behav)[:, 1]
                
                meta_features = pd.DataFrame({
                    'pred_traditional': pred_trad,
                    'pred_behavioral': pred_behav,
                    'pred_avg': (pred_trad + pred_behav) / 2,
                    'pred_max': np.maximum(pred_trad, pred_behav),
                    'pred_min': np.minimum(pred_trad, pred_behav),
                    'pred_diff': np.abs(pred_trad - pred_behav),
                    'pred_ratio': pred_trad / (pred_behav + 0.001)
                })
                
                # Reset indices for proper concatenation
                meta_features.reset_index(drop=True, inplace=True)
                X_trad.reset_index(drop=True, inplace=True)
                X_behav.reset_index(drop=True, inplace=True)
                
                # Combine with ORIGINAL feature names for CatBoost compatibility
                X_aligned = pd.concat([meta_features, X_trad, X_behav], axis=1)
                
                # Create feature name mapping with prefixes for display
                feature_names_display = (
                    list(meta_features.columns) + 
                    [f'trad_{col}' for col in X_trad.columns] + 
                    [f'behav_{col}' for col in X_behav.columns]
                )
                
                final_estimator = model.meta_model
            else:
                # Get final estimator from pipeline
                if hasattr(model, 'named_steps'):
                    final_estimator = list(model.named_steps.values())[-1]
                else:
                    final_estimator = model
                
                # Align features
                from apps.utils import align_features
                X_aligned = align_features(X, model)
                feature_names_display = None
            
            # Prediction
            pred, prob = get_predictions(model, X)
            
            if pred is not None:
                st.markdown("### Prediction")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", "Default" if pred[0] == 1 else "No Default")
                if prob is not None:
                    col2.metric("Probability", f"{prob[0] * 100:.2f}%")
            
            # SHAP
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(X_aligned)
            
            # Create a copy with renamed columns for display
            if hasattr(model, 'meta_model') and feature_names_display:
                X_for_display = X_aligned.copy()
                X_for_display.columns = feature_names_display
            else:
                X_for_display = X_aligned
            
            st.markdown("### Feature Contributions")
            
            # Waterfall plot
            st.markdown("#### Waterfall Plot")
            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_for_display.iloc[0],
                feature_names=X_for_display.columns.tolist()
            ), show=False)
            st.pyplot(fig)
            
            st.markdown("""
            **How to read:**
            - Starting point (E[f(x)]): Model's base prediction
            - Red bars: Features pushing prediction towards default
            - Blue bars: Features pushing prediction away from default
            - Final value (f(x)): Actual prediction for this applicant
            """)
            
            st.success("SHAP explanation generated successfully!")
            
    except ImportError:
        st.error("SHAP library not installed. Install with: pip install shap")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

def feature_details(model):
    """Display feature details and statistics"""
    st.markdown("### Feature Details")
    
    try:
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            
            st.write(f"**Total Features:** {len(features)}")
            
            # Feature list
            features_df = pd.DataFrame({
                'Feature Name': features,
                'Index': range(len(features))
            })
            
            st.dataframe(features_df, use_container_width=True)
            
            # Search
            search = st.text_input("Search features:")
            if search:
                filtered = features_df[features_df['Feature Name'].str.contains(search, case=False)]
                st.dataframe(filtered, use_container_width=True)
        else:
            st.warning("Feature names not available in this model")
    except Exception as e:
        st.error(f"Could not retrieve feature details: {e}")

if __name__ == "__main__":
    show()
else:
    show()
