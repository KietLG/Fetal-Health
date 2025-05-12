import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import joblib

def preprocess_data(df):
    le = LabelEncoder()
    df['fetal_health'] = le.fit_transform(df['fetal_health'])
    return df

st.set_page_config(page_title="Fetal Health Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🩺 Fetal Health Prediction Dashboard")

st.sidebar.header("🔹 Dashboard Information")
st.sidebar.info("Giao diện dự đoán tình trạng sức khỏe của thai nhi và phân tích dữ liệu từ tập dữ liệu fetal_health.")

model_files = {
    "KNN": "knn.pkl",
    "LogisticRegression": "logistic.pkl",
    "DecisionTree": "decision.pkl",
    "RandomForest": "randomforest.pkl",
    "SVM": "svc.pkl",
    "LGBM": "lgbm.pkl",
    "XGB": "xgboost.pkl",
    "GradientBoosting": "gradient.pkl",
    "CatBoost": "catboost.pkl",
    "ExtraTrees": "extratree.pkl",
}

trained_models = {}
for model_name, file_name in model_files.items():
    try:
        trained_models[model_name] = joblib.load(file_name)
        st.sidebar.success(f"✅ {model_name} model load thành công.")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi load {model_name} model: " + str(e))
        trained_models[model_name] = None

try:
    scaler = joblib.load("scaler.pkl")
    st.sidebar.success("✅ Scaler load thành công.")
except Exception as e:
    st.sidebar.error("❌ Lỗi khi load scaler: " + str(e))
    scaler = None

dataset_path = r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project gấm\fetal_health.csv"
try:
    raw_df = pd.read_csv(dataset_path)
    st.sidebar.success("✅ Dataset load thành công.")
except Exception as e:
    st.sidebar.error("❌ Lỗi khi load dataset: " + str(e))
    raw_df = None

if raw_df is not None:
    processed_df = preprocess_data(raw_df.copy())
else:
    processed_df = None

if scaler is not None and hasattr(scaler, "feature_names_in_"):
    model_columns = list(scaler.feature_names_in_)
else:
    model_columns = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability', 'histogram_mode',
        'histogram_median', 'histogram_variance', 'histogram_skew', 'histogram_kurtosis',
        'histogram_max', 'histogram_min', 'histogram_mean'
    ]

tabs = st.tabs(["🩺 Prediction", "📊 EDA & Feature Relationships", "📈 Model Evaluation"])

with tabs[0]:
    st.header("🔎 Fetal Health Prediction")
    col_input, col_summary = st.columns(2)

    with col_input:
        st.subheader("📥 Nhập dữ liệu của thai nhi")
        baseline_value = st.number_input("Baseline Value", min_value=50.0, max_value=250.0, value=140.0, step=0.1)
        accelerations = st.number_input("Accelerations", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        fetal_movement = st.number_input("Fetal Movement", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        uterine_contractions = st.number_input("Uterine Contractions", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        abnormal_stv = st.number_input("Abnormal Short-Term Variability", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        mean_stv = st.number_input("Mean Value of Short-Term Variability", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        perc_abnormal_ltv = st.number_input("Percentage Time with Abnormal Long-Term Variability", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        histogram_mode = st.number_input("Histogram Mode", min_value=0.0, max_value=300.0, value=150.0, step=0.1)
        histogram_median = st.number_input("Histogram Median", min_value=0.0, max_value=300.0, value=150.0, step=0.1)
        histogram_variance = st.number_input("Histogram Variance", min_value=0.0, max_value=5000.0, value=1000.0, step=0.1)
        histogram_skew = st.number_input("Histogram Skew", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        histogram_kurtosis = st.number_input("Histogram Kurtosis", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        histogram_max = st.number_input("Histogram Max", min_value=0.0, max_value=300.0, value=200.0, step=0.1)
        histogram_min = st.number_input("Histogram Min", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
        histogram_mean = st.number_input("Histogram Mean", min_value=0.0, max_value=300.0, value=150.0, step=0.1)

        model_options = [name for name, model in trained_models.items() if model is not None]
        selected_model = st.selectbox("Chọn mô hình dự đoán", model_options, index=model_options.index("CatBoost") if "CatBoost" in model_options else 0)

    with col_summary:
        st.subheader("📋 Tóm tắt dữ liệu đầu vào")
        st.write("Baseline Value:", baseline_value)
        st.write("Accelerations:", accelerations)
        st.write("Fetal Movement:", fetal_movement)
        st.write("Uterine Contractions:", uterine_contractions)
        st.write("Abnormal STV:", abnormal_stv)
        st.write("Mean STV:", mean_stv)
        st.write("Perc. Abnormal LTV:", perc_abnormal_ltv)
        st.write("Histogram Mode:", histogram_mode)
        st.write("Histogram Median:", histogram_median)
        st.write("Histogram Variance:", histogram_variance)
        st.write("Histogram Skew:", histogram_skew)
        st.write("Histogram Kurtosis:", histogram_kurtosis)
        st.write("Histogram Max:", histogram_max)
        st.write("Histogram Min:", histogram_min)
        st.write("Histogram Mean:", histogram_mean)

    if st.button("🔍 Predict Fetal Health"):
        input_data = {col: 0 for col in model_columns}
        input_data['baseline value'] = baseline_value
        input_data['accelerations'] = accelerations
        input_data['fetal_movement'] = fetal_movement
        input_data['uterine_contractions'] = uterine_contractions
        input_data['abnormal_short_term_variability'] = abnormal_stv
        input_data['mean_value_of_short_term_variability'] = mean_stv
        input_data['percentage_of_time_with_abnormal_long_term_variability'] = perc_abnormal_ltv
        input_data['histogram_mode'] = histogram_mode
        input_data['histogram_median'] = histogram_median
        input_data['histogram_variance'] = histogram_variance
        input_data['histogram_skew'] = histogram_skew
        input_data['histogram_kurtosis'] = histogram_kurtosis
        input_data['histogram_max'] = histogram_max
        input_data['histogram_min'] = histogram_min
        input_data['histogram_mean'] = histogram_mean

        input_df = pd.DataFrame([input_data])
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        mapping = {0: "1 - Normal", 1: "2 - Suspect", 2: "3 - Pathological"}

        if trained_models[selected_model] is not None:
            model = trained_models[selected_model]
            prediction = model.predict(input_scaled)

            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()  
            pred_text = mapping.get(prediction, f"Label {prediction}")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
                proba_text = ", ".join([f"{mapping.get(i, i)}: {round(p * 100, 2)}%" for i, p in enumerate(proba)])
            else:
                proba = None
                proba_text = "N/A"

            st.subheader("📊 Kết quả dự đoán")
            st.write("🔍 Mô hình:", selected_model)
            st.write("🔍 Dự đoán:", pred_text)
            st.write("📈 Xác suất dự đoán:", proba_text)

            if proba is not None:
                max_proba = max(proba) * 100
                fig_prob = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=max_proba,
                    title={"text": "Prediction Confidence (%)"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "red"}}
                ))
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.error(f"⚠️ Mô hình {selected_model} không khả dụng.")

with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")
    if raw_df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.subheader("Data Distribution")
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'fetal_health' in numeric_cols:
            numeric_cols.remove('fetal_health')
        chosen_col = st.selectbox("Chọn cột số để xem phân bố", numeric_cols, key="dist_col")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig_hist = px.histogram(raw_df, x=chosen_col, nbins=30,
                                    title=f"Histogram of {chosen_col}",
                                    color_discrete_sequence=px.colors.sequential.Viridis)
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_dist2:
            fig_box = px.box(raw_df, y=chosen_col, color="fetal_health",
                             title=f"Boxplot of {chosen_col}",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = raw_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Correlation Heatmap", height=800)
        fig_corr.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Scatter Matrix")
        subset_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        fig_scatter = px.scatter_matrix(raw_df, dimensions=subset_cols, color="fetal_health",
                                        title="Scatter Matrix of Numeric Features",
                                        color_discrete_sequence=px.colors.qualitative.Set1,
                                        height=800)
        fig_scatter.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.error("Dataset không khả dụng.")

with tabs[2]:
    st.header("Model Evaluation")

    # Khởi tạo Session State cho các kết quả đánh giá nếu chưa có
    if 'loaded_eval' not in st.session_state:
        st.session_state.loaded_eval = False
        st.session_state.X_train_scaled = None
        st.session_state.X_test_scaled = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.detailed_results_train = {}
        st.session_state.detailed_results_test = {}
        st.session_state.cv_results = {}

    # Hiển thị biểu đồ radar tĩnh (static results)
    static_results = {
        "KNN": {"F1 Score": 0.86647, "Accuracy": 0.855799, "Precision": 0.892314, "Recall": 0.855799},
        "LogisticRegression": {"F1 Score": 0.850735, "Accuracy": 0.836991, "Precision": 0.880686, "Recall": 0.836991},
        "DecisionTree": {"F1 Score": 0.902612, "Accuracy": 0.901254, "Precision": 0.905491, "Recall": 0.901254},
        "RandomForest": {"F1 Score": 0.934279, "Accuracy": 0.934169, "Precision": 0.935172, "Recall": 0.934169},
        "SVM": {"F1 Score": 0.872276, "Accuracy": 0.862069, "Precision": 0.895241, "Recall": 0.862069},
        "LGBM": {"F1 Score": 0.959108, "Accuracy": 0.959248, "Precision": 0.959015, "Recall": 0.959248},
        "XGB": {"F1 Score": 0.951478, "Accuracy": 0.951411, "Precision": 0.95162, "Recall": 0.951411},
        "GradientBoosting": {"F1 Score": 0.945793, "Accuracy": 0.945141, "Precision": 0.946942, "Recall": 0.945141},
        "CatBoost": {"F1 Score": 0.95484, "Accuracy": 0.954545, "Precision": 0.955476, "Recall": 0.954545},
        "ExtraTrees": {"F1 Score": 0.94395, "Accuracy": 0.943574, "Precision": 0.944619, "Recall": 0.943574},
    }
    categories = ["F1 Score", "Accuracy", "Precision", "Recall"]
    radar_data_static = []
    for model_name, metrics in static_results.items():
        radar_data_static.append(go.Scatterpolar(
            r=[metrics[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=model_name
        ))
    fig_static = go.Figure(data=radar_data_static)
    fig_static.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Biểu đồ Radar: So sánh hiệu suất các mô hình (Static Results)"
    )
    st.plotly_chart(fig_static, use_container_width=True)
    st.markdown("---")

    # Đánh giá mô hình chỉ chạy một lần khi chưa load
    if processed_df is not None and scaler is not None and not st.session_state.loaded_eval:
        try:
            X_train = joblib.load('X_train.pkl')
            X_test = joblib.load('X_test.pkl')
            st.session_state.y_train = joblib.load('y_train.pkl')
            st.session_state.y_test = joblib.load('y_test.pkl')
            st.session_state.X_train_scaled = scaler.transform(X_train)
            st.session_state.X_test_scaled = scaler.transform(X_test)

            # Tính toán kết quả cho từng mô hình
            for model_name, model in trained_models.items():
                if model is not None:
                    # Cross-validation
                    cv_scores = cross_val_score(model, st.session_state.X_train_scaled, st.session_state.y_train, cv=5, scoring='accuracy', n_jobs=-1)
                    st.session_state.cv_results[model_name] = {
                        "CV Mean Accuracy": cv_scores.mean(),
                        "CV Std Dev": cv_scores.std()
                    }

                    # Đánh giá trên tập train
                    y_pred_train = model.predict(st.session_state.X_train_scaled)
                    y_prob_train = model.predict_proba(st.session_state.X_train_scaled) if hasattr(model, "predict_proba") else None
                    st.session_state.detailed_results_train[model_name] = {
                        "F1 Score": f1_score(st.session_state.y_train, y_pred_train, average='weighted', zero_division=0),
                        "Accuracy": accuracy_score(st.session_state.y_train, y_pred_train),
                        "Precision": precision_score(st.session_state.y_train, y_pred_train, average='weighted', zero_division=0),
                        "Recall": recall_score(st.session_state.y_train, y_pred_train, average='weighted', zero_division=0),
                        "y_pred": y_pred_train,
                        "y_prob": y_prob_train,
                        "y_true": st.session_state.y_train
                    }

                    # Đánh giá trên tập test
                    y_pred_test = model.predict(st.session_state.X_test_scaled)
                    y_prob_test = model.predict_proba(st.session_state.X_test_scaled) if hasattr(model, "predict_proba") else None
                    st.session_state.detailed_results_test[model_name] = {
                        "F1 Score": f1_score(st.session_state.y_test, y_pred_test, average='weighted', zero_division=0),
                        "Accuracy": accuracy_score(st.session_state.y_test, y_pred_test),
                        "Precision": precision_score(st.session_state.y_test, y_pred_test, average='weighted', zero_division=0),
                        "Recall": recall_score(st.session_state.y_test, y_pred_test, average='weighted', zero_division=0),
                        "y_pred": y_pred_test,
                        "y_prob": y_prob_test,
                        "y_true": st.session_state.y_test
                    }
            st.session_state.loaded_eval = True
        except Exception as e:
            st.error("Lỗi khi load tập dữ liệu hoặc tính toán đánh giá: " + str(e))

    # Hiển thị kết quả khi đã load
    if st.session_state.loaded_eval:
        selected_model = st.selectbox("Chọn mô hình để xem chi tiết đánh giá", options=list(st.session_state.detailed_results_test.keys()))
        st.subheader(f"Đánh giá chi tiết cho mô hình: {selected_model}")

        metrics_train = st.session_state.detailed_results_train[selected_model]
        metrics_test = st.session_state.detailed_results_test[selected_model]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Tập Train")
            st.write({k: round(metrics_train[k], 3) for k in ["F1 Score", "Accuracy", "Precision", "Recall"]})
        with col2:
            st.markdown("### Tập Test")
            st.write({k: round(metrics_test[k], 3) for k in ["F1 Score", "Accuracy", "Precision", "Recall"]})

        st.subheader("Cross-Validation Results")
        cv_metrics = st.session_state.cv_results[selected_model]
        st.write({
            "CV Mean Accuracy": round(cv_metrics["CV Mean Accuracy"], 3),
            "CV Std Deviation": round(cv_metrics["CV Std Dev"], 3)
        })

        st.subheader("So sánh Cross-Validation Accuracy giữa các mô hình")
        cv_df = pd.DataFrame({
            "Model": list(st.session_state.cv_results.keys()),
            "CV Accuracy": [st.session_state.cv_results[m]["CV Mean Accuracy"] for m in st.session_state.cv_results],
            "CV Std Dev": [st.session_state.cv_results[m]["CV Std Dev"] for m in st.session_state.cv_results]
        })
        fig_cv = px.bar(cv_df, x="Model", y="CV Accuracy", error_y="CV Std Dev",
                        title="Cross-Validation Accuracy per Model",
                        color="Model", height=400)
        st.plotly_chart(fig_cv, use_container_width=True)

        st.subheader("Biểu đồ Radar: Hiệu suất trên Tập Test")
        fig_radar_detail = go.Figure(data=[
            go.Scatterpolar(
                r=[metrics_test[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=selected_model
            )
        ])
        fig_radar_detail.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        st.plotly_chart(fig_radar_detail, use_container_width=True)

        st.subheader("Additional Evaluation Plots (Tập Test)")
        y_pred = metrics_test["y_pred"]
        y_prob = metrics_test["y_prob"]
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(metrics_test["y_true"], y_pred)
            mapping = {0: "1 - Normal", 1: "2 - Suspect", 2: "3 - Pathological"}
            labels = [mapping[i] for i in range(len(np.unique(st.session_state.y_test)))]
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels={"x": "Predicted", "y": "Actual"},
                               x=labels, y=labels,
                               title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_eval2:
            if y_prob is not None:
                st.markdown("**ROC Curve**")
                fpr_vals, tpr_vals = {}, {}
                roc_auc_val = {}
                for i in range(len(np.unique(st.session_state.y_test))):
                    fpr_vals[i], tpr_vals[i], _ = roc_curve(st.session_state.y_test, y_prob[:, i], pos_label=i)
                    roc_auc_val[i] = auc(fpr_vals[i], tpr_vals[i])
                fig_roc = go.Figure()
                for i in range(len(np.unique(st.session_state.y_test))):
                    fig_roc.add_trace(go.Scatter(x=fpr_vals[i], y=tpr_vals[i], mode='lines',
                                                 name=f"{mapping[i]} (AUC={roc_auc_val[i]:.2f})",
                                                 line=dict(width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                             name="Chance", line=dict(dash='dash', color='navy')))
                fig_roc.update_layout(
                    title="ROC Curve - Multiclass",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1.05])
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Model không hỗ trợ dự đoán xác suất để vẽ ROC curve.")

        if y_prob is not None:
            st.subheader("Precision-Recall Curve (Test)")
            precision_vals, recall_vals = {}, {}
            for i in range(len(np.unique(st.session_state.y_test))):
                precision_vals[i], recall_vals[i], _ = precision_recall_curve(st.session_state.y_test == i, y_prob[:, i])
            fig_pr = go.Figure()
            for i in range(len(np.unique(st.session_state.y_test))):
                fig_pr.add_trace(go.Scatter(x=recall_vals[i], y=precision_vals[i], mode='lines',
                                            name=f"{mapping[i]}",
                                            line=dict(width=2)))
            fig_pr.update_layout(
                title="Precision-Recall Curve - Multiclass",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.error("Dataset hoặc scaler không khả dụng để đánh giá mô hình.")
