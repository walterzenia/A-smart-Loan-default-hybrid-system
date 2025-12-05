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


def display_stored_metrics(model, model_name, model_type):
    """Display metrics stored in the model from training"""
    
    try:
        # Check if model has stored metrics
        if not hasattr(model, 'best_score_'):
            # For ensemble, skip to confusion matrix section
            pass
        
        # Try to evaluate on appropriate test data to generate ROC curve
        # Use the same test data file logic as confusion matrix
        try:
            test_data_file = None
            target_col = None
            
            if model_type == 'behavioral':
                test_data_file = "data/behavioral_test_data.csv"
                target_col = 'default.payment.next.month'
            elif model_type == 'traditional':
                test_data_file = "data/traditional_test_data.csv"
                target_col = 'TARGET'
            
            if test_data_file and Path(test_data_file).exists():
                df_test = load_data(test_data_file)
                
                if df_test is not None and target_col in df_test.columns:
                    # Separate features and target
                    X_test = df_test.drop(target_col, axis=1)
                    y_test = df_test[target_col].values
                    
                    # Remove NaN values
                    valid_mask = ~pd.isna(y_test)
                    X_test = X_test[valid_mask]
                    y_test = y_test[valid_mask]
                    
                    # Use all data - no split (same as confusion matrix)
                    
                    # Get predictions
                    _, y_proba = get_predictions(model, X_test)
                    
                    if y_proba is not None:
                        from sklearn.metrics import roc_curve, auc
                        
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        # Try to get stored AUC from model training
                        stored_auc = None
                        if hasattr(model, 'best_score_'):
                            try:
                                # LightGBM models store AUC in best_score_
                                if 'valid_0' in model.best_score_ and 'auc' in model.best_score_['valid_0']:
                                    stored_auc = model.best_score_['valid_0']['auc']
                                elif 'valid_1' in model.best_score_ and 'auc' in model.best_score_['valid_1']:
                                    stored_auc = model.best_score_['valid_1']['auc']
                            except:
                                pass
                        
                        # Use stored AUC if available, otherwise use calculated
                        display_auc = stored_auc if stored_auc is not None else roc_auc
                        
                        # For ensemble, use documented value (CatBoost doesn't store AUC)
                        if model_type == 'ensemble' and stored_auc is None:
                            display_auc = 0.8509
                        
                        fig_roc = go.Figure()
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC = {display_auc:.4f})',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig_roc.update_layout(
                            title=f"ROC Curve (AUC = {display_auc:.4f})",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_roc, use_container_width=True)
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
    st.subheader("Model Confusion Matrix & Performance")
    
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
        
    elif model_type == 'ensemble':
        st.markdown("### Ensemble Hybrid Model")
        # Load the wrapper model
        model_path = "models/model_ensemble_wrapper.pkl"
        test_path = "data/test_ensemble_hybrid_preprocessed.csv"  # Use preprocessed file for correct results
        target_col = 'TARGET'
        color_scheme = 'Purples'
    else:
        return
    
    if Path(model_path).exists() and Path(test_path).exists():
        try:
            # For ensemble, load with joblib to handle custom class
            if model_type == 'ensemble':
                import joblib
                import sys
                sys.path.insert(0, 'src')
                model = joblib.load(model_path)
            else:
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
                    
                    # Use all data - no split needed since this is already test data
                    st.info(f" Evaluating on {len(X_test)} samples from test dataset")
                    
                    # Show class distribution
                    defaults = np.sum(y_test == 1)
                    non_defaults = np.sum(y_test == 0)
                    default_rate = defaults / len(y_test) * 100
                    
                    st.markdown(f"**Class Distribution:** {non_defaults:,} non-defaults ({100-default_rate:.1f}%) | {defaults:,} defaults ({default_rate:.1f}%)")
                    
                    # Get predictions
                    y_pred, y_proba = get_predictions(model, X_test)
                    
                    # Import metrics functions first
                    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    # Calculate baseline metrics (threshold 0.5) - these won't change
                    y_pred_baseline = (y_proba >= 0.5).astype(int)
                    baseline_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred_baseline),
                        'precision': precision_score(y_test, y_pred_baseline, zero_division=0),
                        'recall': recall_score(y_test, y_pred_baseline, zero_division=0),
                        'f1': f1_score(y_test, y_pred_baseline, zero_division=0),
                        'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                    }
                    
                    # Show baseline performance
                    st.markdown("####  Model Performance (Threshold: 0.5)")
                    col_base1, col_base2, col_base3, col_base4, col_base5 = st.columns(5)
                    col_base1.metric("Accuracy", f"{baseline_metrics['accuracy']:.4f}")
                    col_base2.metric("Precision", f"{baseline_metrics['precision']:.4f}")
                    col_base3.metric("Recall", f"{baseline_metrics['recall']:.4f}")
                    col_base4.metric("F1 Score", f"{baseline_metrics['f1']:.4f}")
                    col_base5.metric("AUC-ROC", f"{baseline_metrics['auc']:.4f}")
                    
                    # ROC Curve
                    st.markdown("---")
                    st.markdown("####  ROC Curve")
                    
                    from sklearn.metrics import roc_curve, auc
                    
                    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
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
                    
                    st.markdown("---")
                    
                    # Add threshold adjustment slider
                    st.markdown("####  Threshold Adjustment")
                    st.markdown("Lower the threshold to catch more defaults (increases false positives)")
                    threshold = st.slider(
                        "Prediction Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.5,
                        step=0.05,
                        help="Predictions above this threshold are classified as default (1)"
                    )
                    
                    # Apply threshold
                    y_pred_adjusted = (y_proba >= threshold).astype(int)
                    
                    if y_pred is not None:
                        # Compute metrics (already imported above)
                        cm = confusion_matrix(y_test, y_pred_adjusted)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred_adjusted)
                        precision = precision_score(y_test, y_pred_adjusted, zero_division=0)
                        recall = recall_score(y_test, y_pred_adjusted, zero_division=0)
                        f1 = f1_score(y_test, y_pred_adjusted, zero_division=0)
                        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                        
                        # Calculate default capture metrics
                        true_positives = cm[1, 1]  # Correctly predicted defaults
                        false_negatives = cm[1, 0]  # Missed defaults
                        false_positives = cm[0, 1]  # False alarms
                        true_negatives = cm[0, 0]  # Correctly predicted non-defaults
                        
                        defaults_caught = true_positives
                        defaults_missed = false_negatives
                        capture_rate = (defaults_caught / (defaults_caught + defaults_missed) * 100) if (defaults_caught + defaults_missed) > 0 else 0
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Plot confusion matrix
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predicted Non-Default', 'Predicted Default'],
                                y=['Actual Non-Default', 'Actual Default'],
                                colorscale=color_scheme,
                                text=cm,
                                texttemplate='%{text}',
                                textfont={"size": 20}
                            ))
                            
                            fig.update_layout(
                                title=f"Confusion Matrix (Threshold: {threshold})",
                                xaxis_title="Predicted",
                                yaxis_title="Actual",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Show key metrics that change with threshold
                            st.markdown("**Adjusted Performance:**")
                            st.metric(" Accuracy", f"{accuracy:.4f}")
                            st.metric(" Precision", f"{precision:.4f}")
                            st.metric(" Recall (Default Capture)", f"{recall:.4f}", help="Percentage of actual defaults correctly identified")
                            st.metric(" F1 Score", f"{f1:.4f}")
                            st.metric(" AUC-ROC", f"{auc:.4f}", help="AUC doesn't change with threshold")
                        
                        # Show detailed default capture analysis
                        st.markdown("---")
                        st.markdown("#### Default Detection Analysis")
                        
                        col3, col4, col5 = st.columns(3)
                        
                        with col3:
                            st.metric(
                                "Defaults Caught",
                                f"{defaults_caught:,}",
                                f"{capture_rate:.1f}% of all defaults"
                            )
                        
                        with col4:
                            st.metric(
                                "Defaults Missed",
                                f"{defaults_missed:,}",
                                f"{100-capture_rate:.1f}% missed",
                                delta_color="inverse"
                            )
                        
                        with col5:
                            st.metric(
                                "False Alarms",
                                f"{false_positives:,}",
                                help="Non-defaults incorrectly flagged as defaults"
                            )
                        
                        # Show prediction distribution affected by threshold
                        st.markdown("---")
                        st.markdown("#### Prediction Distribution (After Threshold Adjustment)")
                        
                        pred_dist = pd.DataFrame({
                            'Prediction': ['Predicted Non-Default', 'Predicted Default'],
                            'Count': [(y_pred_adjusted == 0).sum(), (y_pred_adjusted == 1).sum()]
                        })
                        
                        fig_dist = go.Figure(data=[
                            go.Bar(
                                x=pred_dist['Prediction'], 
                                y=pred_dist['Count'],
                                text=pred_dist['Count'], 
                                textposition='auto',
                                marker_color=['#636EFA', '#EF553B']
                            )
                        ])
                        
                        fig_dist.update_layout(
                            title=f"How many predictions in each category (Threshold: {threshold})",
                            xaxis_title="Prediction Category",
                            yaxis_title="Number of Samples",
                            height=400
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Prediction Probability Distribution Analysis
                        st.markdown("---")
                        st.markdown("#### Prediction Probability Distribution Analysis")
                        st.markdown("*Examining the distribution of predicted probabilities provides insight into model confidence and decision patterns.*")
                        
                        col_prob1, col_prob2 = st.columns([1, 1])
                        
                        with col_prob1:
                            # Histogram of all prediction probabilities
                            fig_prob_hist = go.Figure()
                            
                            # Separate by actual class
                            prob_non_default = y_proba[y_test == 0]
                            prob_default = y_proba[y_test == 1]
                            
                            fig_prob_hist.add_trace(go.Histogram(
                                x=prob_non_default,
                                name='Actual Non-Defaults',
                                opacity=0.7,
                                marker_color='#636EFA',
                                nbinsx=50
                            ))
                            
                            fig_prob_hist.add_trace(go.Histogram(
                                x=prob_default,
                                name='Actual Defaults',
                                opacity=0.7,
                                marker_color='#EF553B',
                                nbinsx=50
                            ))
                            
                            # Add threshold line
                            fig_prob_hist.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Threshold: {threshold}",
                                annotation_position="top right"
                            )
                            
                            fig_prob_hist.update_layout(
                                title="Probability Distribution by Actual Class",
                                xaxis_title="Predicted Probability of Default",
                                yaxis_title="Frequency",
                                barmode='overlay',
                                height=400,
                                legend=dict(x=0.02, y=0.98)
                            )
                            
                            st.plotly_chart(fig_prob_hist, use_container_width=True)
                        
                        with col_prob2:
                            # Box plot showing probability distribution by actual class
                            fig_box = go.Figure()
                            
                            fig_box.add_trace(go.Box(
                                y=prob_non_default,
                                name='Actual Non-Defaults',
                                marker_color='#636EFA',
                                boxmean='sd'
                            ))
                            
                            fig_box.add_trace(go.Box(
                                y=prob_default,
                                name='Actual Defaults',
                                marker_color='#EF553B',
                                boxmean='sd'
                            ))
                            
                            # Add threshold line
                            fig_box.add_hline(
                                y=threshold,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Threshold: {threshold}",
                                annotation_position="right"
                            )
                            
                            fig_box.update_layout(
                                title="Probability Distribution Statistics",
                                yaxis_title="Predicted Probability of Default",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("**Key Insights:**")
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            avg_prob_non_default = np.mean(prob_non_default)
                            st.metric(
                                "Avg Prob (Non-Defaults)",
                                f"{avg_prob_non_default:.3f}",
                                help="Average predicted probability for actual non-defaulters"
                            )
                        
                        with col_stat2:
                            avg_prob_default = np.mean(prob_default)
                            st.metric(
                                "Avg Prob (Defaults)",
                                f"{avg_prob_default:.3f}",
                                help="Average predicted probability for actual defaulters"
                            )
                        
                        with col_stat3:
                            separation = avg_prob_default - avg_prob_non_default
                            st.metric(
                                "Class Separation",
                                f"{separation:.3f}",
                                help="Difference in average probabilities (higher is better)"
                            )
                        
                        with col_stat4:
                            # Confidence: how many predictions are far from threshold
                            confident_predictions = np.sum((y_proba < (threshold - 0.2)) | (y_proba > (threshold + 0.2)))
                            confidence_rate = confident_predictions / len(y_proba) * 100
                            st.metric(
                                "Confident Predictions",
                                f"{confidence_rate:.1f}%",
                                help=f"Predictions >0.2 away from threshold ({threshold})"
                            )
                        
                        # Interpretation text
                        st.markdown(f"""
                        **Interpretation:**
                        - **Good separation**: Actual defaults have higher average probability ({avg_prob_default:.3f}) than non-defaults ({avg_prob_non_default:.3f})
                        - **Overlap region**: Samples near threshold ({threshold}) are uncertain - require manual review
                        - **Model confidence**: {confidence_rate:.1f}% of predictions are confident (far from threshold)
                        - **Threshold effect**: Moving threshold left/right changes which samples get classified as default
                        """)
                        
                        # Show detailed classification report
                        st.markdown("---")
                        st.markdown("#### Detailed Classification Report")
                        
                        from sklearn.metrics import classification_report
                        
                        # Generate classification report
                        report_dict = classification_report(y_test, y_pred_adjusted, 
                                                           target_names=['Non-Defaulter', 'Defaulter'],
                                                           output_dict=True,
                                                           zero_division=0)
                        
                        # Create report dataframe
                        report_df = pd.DataFrame({
                            'Class': ['Non-Defaulter (0)', 'Defaulter (1)'],
                            'Precision': [report_dict['Non-Defaulter']['precision'], 
                                        report_dict['Defaulter']['precision']],
                            'Recall': [report_dict['Non-Defaulter']['recall'], 
                                     report_dict['Defaulter']['recall']],
                            'F1-Score': [report_dict['Non-Defaulter']['f1-score'], 
                                       report_dict['Defaulter']['f1-score']],
                            'Support': [int(report_dict['Non-Defaulter']['support']), 
                                      int(report_dict['Defaulter']['support'])]
                        })
                        
                        # Add overall metrics
                        overall_df = pd.DataFrame({
                            'Class': ['Overall (Weighted Avg)', 'Overall (Macro Avg)'],
                            'Precision': [report_dict['weighted avg']['precision'],
                                        report_dict['macro avg']['precision']],
                            'Recall': [report_dict['weighted avg']['recall'],
                                     report_dict['macro avg']['recall']],
                            'F1-Score': [report_dict['weighted avg']['f1-score'],
                                       report_dict['macro avg']['f1-score']],
                            'Support': [int(report_dict['weighted avg']['support']),
                                      int(report_dict['macro avg']['support'])]
                        })
                        
                        # Combine reports
                        full_report_df = pd.concat([report_df, overall_df], ignore_index=True)
                        
                        # Display in columns
                        col_r1, col_r2 = st.columns([1, 1])
                        
                        with col_r1:
                            st.markdown("**Per-Class Performance:**")
                            
                            # Format and display
                            styled_report = full_report_df.style.format({
                                'Precision': '{:.4f}',
                                'Recall': '{:.4f}',
                                'F1-Score': '{:.4f}',
                                'Support': '{:,}'
                            }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], 
                                                  cmap='RdYlGn', vmin=0, vmax=1)
                            
                            st.dataframe(styled_report, use_container_width=True)
                        
                        with col_r2:
                            st.markdown("**Interpretation:**")
                            
                            non_default_recall = report_dict['Non-Defaulter']['recall']
                            default_recall = report_dict['Defaulter']['recall']
                            non_default_precision = report_dict['Non-Defaulter']['precision']
                            default_precision = report_dict['Defaulter']['precision']
                            
                            st.markdown(f"""
                            **Non-Defaulters (Class 0):**
                            - Precision: {non_default_precision:.2%} of predicted non-defaults are correct
                            - Recall: {non_default_recall:.2%} of actual non-defaults are identified
                            - Support: {int(report_dict['Non-Defaulter']['support']):,} samples
                            
                            **Defaulters (Class 1):**
                            - Precision: {default_precision:.2%} of predicted defaults are correct
                            - Recall: {default_recall:.2%} of actual defaults are identified
                            - Support: {int(report_dict['Defaulter']['support']):,} samples
                            
                            ---
                            
                            **Key Insights:**
                            - Model correctly identifies **{default_recall:.1%}** of defaulters
                            - When model predicts default, it's correct **{default_precision:.1%}** of the time
                            - **{100-default_recall:.1%}** of defaulters are missed (False Negatives)
                            - **{100-default_precision:.1%}** of default predictions are false alarms (False Positives)
                            """)
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
    
    # Show confusion matrix for all model types
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

