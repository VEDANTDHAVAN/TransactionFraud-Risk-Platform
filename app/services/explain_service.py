import shap
import pandas as pd
from app.model_loader import get_model

model = get_model()

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

explainer = shap.TreeExplainer(classifier)

def explain_transaction(data: dict):
    input_df = pd.DataFrame([data])
    processed_input = preprocessor.transform(input_df)

    if hasattr(processed_input, "toarray"):
        processed_input = processed_input.toarray()

    shap_values = explainer.shap_values(processed_input)

    categorical_cols = input_df.select_dtypes(include=["object"]).columns
    numerical_cols = input_df.select_dtypes(exclude=["object"]).columns

    encoded_cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)

    feature_names = list(numerical_cols) + list(encoded_cat_features)
    contributions = []

    for feature, value in zip(feature_names, shap_values[0]):
        contributions.append({
            "feature": feature,
            "impact": round(float(value), 4)
        })

    contributions = sorted(
        contributions,
        key = lambda x: abs(x["impact"]),
        reverse = True
    )

    return {
        "top_reasons": contributions[:10]
    }