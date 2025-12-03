"""
Model Metrics Page - Display Training Metrics from Pickle Files
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from apps.utils import (
    load_model, get_available_models, get_model_type, load_data, get_predictions, compute_metrics
)

st.set_page_config(page_title="Model Metrics", page_icon="", layout="wide")


def evaluate_ensemble_on_test_data(model, model_type):
    """Evaluate ensemble model on test data"""
    
    st.markdown("---")
    st.subheader(" Evaluation on Training Dataset")
    
    # For ensemble: use smoke_hybrid_features.csv (the training data)
    if model_type == 'ensemble':
        # Load the wrapper model instead of raw booster
        wrapper_path = "models/model_ensemble_wrapper.pkl"
        if Path(wrapper_path).exists():
            import sys
            import joblib
            # Add src to path so ensemble_model can be imported
            sys.path.insert(0, 'src')
            model = joblib.load(wrapper_path)
        else:
            return
        
        test_file = "data/smoke_hybrid_features.csv"
     
    elif model_type == 'behavioral':
        test_file = "data/test_behavioral_high_risk.csv"
    elif model_type == 'traditional':
        test_file = "data/test_traditional_high_risk.csv"
    else:
        return
    
    if not Path(test_file).exists():
        return
    
    df_test = load_data(test_file)
    
    if df_test is None:
        return
    
    # Determine target column
    if 'TARGET' in df_test.columns:
        target_col = 'TARGET'
    elif 'target' in df_test.columns:
        target_col = 'target'
    elif 'default.payment.next.month' in df_test.columns:
        target_col = 'default.payment.next.month'
    else:
        return
    
    # Separate features and target
    X_test = df_test.drop(target_col, axis=1)
    y_test = df_test[target_col].values
    
    # Remove NaN values from target
    valid_mask = ~pd.isna(y_test)
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]
    
    # Get predictions
    y_pred, y_proba = get_predictions(model, X_test)
    
    if y_pred is None:
        return
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Display metrics
    st.markdown("###  Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['Precision']:.4f}")
    col3.metric("Recall", f"{metrics['Recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    
    if 'AUC-ROC' in metrics:
        col5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
    
    # Confusion matrix
    st.markdown("---")
    st.markdown("###  Prediction Distribution")
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction distribution
        pred_dist = pd.DataFrame({
            'Prediction': ['No Default', 'Default'],
            'Count': [(y_pred == 0).sum(), (y_pred == 1).sum()]
        })
        
        fig2 = go.Figure(data=[
            go.Bar(x=pred_dist['Prediction'], y=pred_dist['Count'],
                   text=pred_dist['Count'], textposition='auto')
        ])
        
        fig2.update_layout(
            title="Prediction Distribution",
            xaxis_title="Prediction",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # ROC Curve
    st.markdown("---")
    st.markdown("###  ROC Curve")
    
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='blue', width=2)
    ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature Importance for Ensemble
    st.markdown("---")
    st.markdown("###  Feature Importance")
    
    # Get the meta-model from wrapper
    if hasattr(model, 'meta_model'):
        meta_model = model.meta_model
        
        # LightGBM Booster has feature_importance method
        if hasattr(meta_model, 'feature_importance'):
            importance_scores = meta_model.feature_importance(importance_type='gain')
            feature_names = meta_model.feature_name()
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False)
            
            # Plot all features
            fig_imp = go.Figure()
            
            fig_imp.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig_imp.update_layout(
                title="Meta-Learner Feature Importance",
                xaxis_title="Importance Score (Gain)",
                yaxis_title="Meta-Feature",
                height=max(400, len(feature_names) * 25),
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Show features table
            with st.expander("View All Meta-Features"):
                st.dataframe(
                    importance_df.reset_index(drop=True),
                    width='stretch'
                )
    
def display_stored_metrics(model, model_name, model_type):
    """Display metrics stored in the model from training"""
    
    st.markdown("---")
    st.subheader(" Model Performance")
    
    try:
        # Check if model has stored metrics
        if not hasattr(model, 'best_score_'):
            evaluate_ensemble_on_test_data(model, model_type)
            return
        
        # ROC Curve for training metrics
        st.markdown("---")
        st.markdown("### ROC Curve")
        
        # Try to evaluate on appropriate test data to generate ROC curve
        roc_curve_generated = False
        try:
            # Check if we have evaluation results with validation data stored
            if hasattr(model, 'evals_result_') and hasattr(model, 'X_valid_') and hasattr(model, 'y_valid_'):
                # Use stored validation data
                X_test = model.X_valid_
                y_test = model.y_valid_
                
                # Get predictions
                _, y_proba = get_predictions(model, X_test)
                
                if y_proba is not None:
                    from sklearn.metrics import roc_curve, auc
                    
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = go.Figure()
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.4f})',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_roc.update_layout(
                        title=f"ROC Curve (AUC = {roc_auc:.4f})",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    roc_curve_generated = True
            
            # If we haven't generated ROC curve yet, fall back to loading data files
            if not roc_curve_generated:
                # First, try to load the saved test data that matches the model
                test_data_file = None
                target_col = None
                
                # Check for saved test data files (created during training)
                if model_type == 'behavioral':
                    # Try behavioral test data first
                    test_data_file = "models/Behaviorial_model_test_data.csv"
                    target_col = 'default.payment.next.month'
                    fallback_file = "data/behavioral_test_data.csv"
                elif model_type == 'traditional':
                    # Try traditional test data first
                    test_data_file = "models/Traditional_model_test_data.csv"
                    target_col = 'TARGET'
                    fallback_file = "data/traditional_test_data.csv"
                else:
                    test_data_file = None
                    fallback_file = None
                
                # Try to load saved test data first
                if test_data_file and Path(test_data_file).exists():
                    df_test = load_data(test_data_file)
                    use_split = False  # Already the test set
                elif fallback_file and Path(fallback_file).exists():
                    df_test = load_data(fallback_file)
                    use_split = True  # Need to split
                else:
                    df_test = None
                    use_split = False
                
                if df_test is not None:
                    # Use predetermined target column or try to detect it
                    if target_col is None:
                        if 'TARGET' in df_test.columns:
                            target_col = 'TARGET'
                        elif 'target' in df_test.columns:
                            target_col = 'target'
                        elif 'default.payment.next.month' in df_test.columns:
                            target_col = 'default.payment.next.month'
                        else:
                            target_col = None
                    
                    if target_col and target_col in df_test.columns:
                        # Separate features and target
                        X = df_test.drop(target_col, axis=1)
                        y = df_test[target_col].values
                        
                        # Remove NaN values
                        valid_mask = ~pd.isna(y)
                        X = X[valid_mask]
                        y = y[valid_mask]
                        
                        # If using saved test data, use it directly; otherwise split
                        if use_split:
                            from sklearn.model_selection import train_test_split
                            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        else:
                            X_test = X
                            y_test = y
                        
                        # Get predictions
                        _, y_proba = get_predictions(model, X_test)
                        
                        if y_proba is not None:
                            from sklearn.metrics import roc_curve, auc
                            
                            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            fig_roc = go.Figure()
                            
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig_roc.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Classifier',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig_roc.update_layout(
                                title=f"ROC Curve (AUC = {roc_auc:.4f})",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_roc, use_container_width=True)
                            
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        except Exception as e:
            st.warning(f"Could not generate ROC curve: {str(e)}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.markdown("### Feature Importance")
            
            feature_importance = model.feature_importances_
            
            # Get feature names if available
            if hasattr(model, 'feature_name_'):
                feature_names = model.feature_name_
            elif hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(20)
            
            # Plot top 20 features
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Top 20 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=600,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features table
            with st.expander(" View Top Features Table"):
                st.dataframe(
                    importance_df.reset_index(drop=True),
                    width='stretch'
                )
        
    except Exception as e:
        pass

def show_confusion_matrix_for_model(model_type):
    """Show confusion matrix for the selected model type"""
    st.markdown("---")
    st.subheader("Model Confusion Matrix")
    
    if model_type == 'traditional':
        st.markdown("### Traditional Model")
        model_path = "models/Traditional_model.pkl"
        test_path = "data/traditional_test_data.csv"
        target_col = 'TARGET'
        color_scheme = 'Blues'
        
    elif model_type == 'behavioral':
        st.markdown("### Behavioral Model")
        model_path = "models/Behaviorial_model.pkl"
        test_path = "data/behavioral_test_data.csv"
        target_col = 'default.payment.next.month'
        color_scheme = 'Greens'
    else:
        return  # Don't show confusion matrix for ensemble
    
    if Path(model_path).exists() and Path(test_path).exists():
        try:
            model = load_model(model_path)
            df_test = load_data(test_path)
            
            if df_test is not None and model is not None:
                if target_col not in df_test.columns:
                    target_col = 'TARGET' if 'TARGET' in df_test.columns else 'target'
                
                if target_col in df_test.columns:
                    X_test = df_test.drop(target_col, axis=1)
                    y_test = df_test[target_col].values
                    
                    # Remove NaN values
                    valid_mask = ~pd.isna(y_test)
                    X_test = X_test[valid_mask]
                    y_test = y_test[valid_mask]
                    
                    # Split for testing
                    from sklearn.model_selection import train_test_split
                    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)
                    
                    # Get predictions
                    y_pred, y_proba = get_predictions(model, X_test)
                    
                    if y_pred is not None:
                        # Compute metrics
                        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                        
                        cm = confusion_matrix(y_test, y_pred)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Plot confusion matrix
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predicted 0', 'Predicted 1'],
                                y=['Actual 0', 'Actual 1'],
                                colorscale=color_scheme,
                                text=cm,
                                texttemplate='%{text}',
                                textfont={"size": 20}
                            ))
                            
                            fig.update_layout(
                                title="Confusion Matrix",
                                xaxis_title="Predicted",
                                yaxis_title="Actual",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Show metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                            
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            st.metric("Precision", f"{precision:.4f}")
                            st.metric("Recall", f"{recall:.4f}")
                            st.metric("F1 Score", f"{f1:.4f}")
                            st.metric("AUC-ROC", f"{auc:.4f}")
                    else:
                        st.warning("Could not generate predictions")
                else:
                    st.warning(f"Target column '{target_col}' not found")
            else:
                st.warning("Could not load model or data")
        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")
    else:
        st.warning(f"{model_type.capitalize()} model or test data not found")


def show():
    st.title(" Model Performance Metrics")
    st.markdown("View training metrics and performance stored in model files")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader(" Select Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        models = get_available_models()
        
        if not models:
            st.error(" No models found in models/ directory")
            return
        
        model_names = [Path(m).name for m in models]
        selected_model_name = st.selectbox("Select Model", model_names, key="model_select")
        selected_model_path = models[model_names.index(selected_model_name)]
    
    with col2:
        # Get model type
        model_type = get_model_type(selected_model_name)
        
        st.info(f"**Selected:** {selected_model_name}")
        
        # Model type indicators
        if model_type == 'ensemble':
            st.markdown(" **Ensemble Hybrid Model**")
        elif model_type == 'traditional':
            st.markdown(" **Traditional Model**")
        elif model_type == 'behavioral':
            st.markdown(" **Behavioral Model**")
    
    # Show confusion matrix for Traditional and Behavioral models only
    if model_type in ['traditional', 'behavioral']:
        show_confusion_matrix_for_model(model_type)
    
    # Load model
    with st.spinner("Loading model..."):
        # For ensemble, use the wrapper instead of raw booster
        if model_type == 'ensemble' and 'ensemble' in selected_model_name.lower():
            wrapper_path = "models/model_ensemble_wrapper.pkl"
            if Path(wrapper_path).exists():
                import joblib
                model = joblib.load(wrapper_path)
                st.info("ℹ Loaded ensemble wrapper (handles meta-feature generation)")
            else:
                model = load_model(selected_model_path)
                st.warning(" Using raw booster (wrapper not found)")
        else:
            model = load_model(selected_model_path)
    
    if model is None:
        st.error(" Failed to load model")
        return
    
    st.success(" Model loaded successfully")
    
    # Display stored metrics
    display_stored_metrics(model, selected_model_name, model_type)
    
    # Info box
    st.markdown("---")
    
    # Check model type to show appropriate info
    if model_type == 'ensemble' and not hasattr(model, 'best_score_'):
        pass
    else:
        pass

if __name__ == "__main__":
    show()
else:
    show()

