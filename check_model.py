import joblib
import pandas as pd

artifact = joblib.load("model.joblib")

print("Features:", artifact["features"])
print("Parameters:", artifact["parameters"])

sample = pd.DataFrame([{
    "gender": "female",
    "race/ethnicity": "group B",
    "parental level of education": "bachelor's degree",
    "lunch": "standard",
    "test preparation course": "completed",
    "reading score": 72,
    "writing score": 74
}])

prediction = artifact["model"].predict(sample)

print("Predicted math score:", round(float(prediction[0]), 2))