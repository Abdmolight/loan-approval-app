import streamlit as st
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

# ----------------------------
# Sample dataset
# ----------------------------
data = {
    "age": [25, 45, 35, 50, 23, 40, 60],
    "income": [30000, 80000, 50000, 100000, 25000, 70000, 40000],
    "credit_score": [580, 720, 650, 780, 560, 690, 600],
    "approved": [0, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df[["age", "income", "credit_score"]]
y = df["approved"]

# ----------------------------
# Train Decision Tree
# ----------------------------
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Loan Approval Explanation Demo")

# User inputs
age = st.slider("Age", 18, 70, 30)
income = st.slider("Income", 20000, 150000, 50000)
credit_score = st.slider("Credit Score", 300, 850, 650)

input_data = pd.DataFrame([[age, income, credit_score]], columns=X.columns)

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(input_data)[0]
st.subheader(f"Prediction: {'Approved' if prediction else 'Rejected'}")

# ----------------------------
# Interactive Decision Rules
# ----------------------------
st.subheader("Decision Rules Explanation (Interactive)")

tree_rules = export_text(model, feature_names=list(X.columns))
lines = tree_rules.split("\n")
current_input = input_data.iloc[0]

# Track current path conditions
path_conditions = []

for line in lines:
    line_strip = line.strip()
    highlight = False

    if "<=" in line_strip or ">" in line_strip:
        # Condition line
        if "<=" in line_strip:
            feature, value = line_strip.split(" <= ")
            op = "<="
        else:
            feature, value = line_strip.split(" > ")
            op = ">"
        feature = feature.replace("|---", "").strip()  # Remove tree prefix
        value = float(value.strip())
        condition_text = f"{feature} {op} {value}"
        path_conditions.append(condition_text)

        # Highlight if input satisfies this condition
        if (op == "<=" and current_input[feature] <= value) or (op == ">" and current_input[feature] > value):
            highlight = True

    elif "class:" in line_strip:
        # Leaf node
        class_val = int(line_strip.split("class:")[1].strip())
        decision_text = "loan approved" if class_val == 1 else "loan rejected"
        sentence = " AND ".join(path_conditions) + f" → {decision_text}"

        # Highlight the path that matches user's input
        if class_val == prediction:
            st.markdown(f"**✅ {sentence}**")
        else:
            st.write(sentence)

        # Pop last condition after reaching leaf
        if path_conditions:
            path_conditions.pop()

# ----------------------------
# SHAP Explanation
# ----------------------------
st.subheader("SHAP Explanation")

explainer = shap.TreeExplainer(model)
shap_values = explainer(input_data)

# Handle binary classification: pick predicted class
pred_class = prediction
single_class_values = shap_values.values[0, :, pred_class]
single_class_base_value = shap_values.base_values[0, pred_class]
single_class_data = shap_values.data[0]

single_class_expl = shap.Explanation(
    values=single_class_values,
    base_values=single_class_base_value,
    data=single_class_data,
    feature_names=X.columns
)

# Waterfall plot
ax = shap.plots.waterfall(single_class_expl, show=False)
fig = ax.figure  # Extract figure from Axes for Streamlit
st.pyplot(fig)

