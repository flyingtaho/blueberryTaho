# solver.py
import pandas as pd
import torch
import torch.nn as nn
import requests

class LogisticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeffs = {
            "age": 0.0156,
            "income": 2.2177,
            "credit_score": 2.6590,
            "travel_frequency": 1.1344,
            "owns_car": 0.0341,
            "is_employed": -0.0703,
            "location": {"International": -0.0333},
            "residential_status": {
                "Living with family/friends": 0.0201,
                "Owner": 0.0203,
                "Renter": 0.0247,
                "Other": -0.0629
            },
            "education": {
                "College": 1.9276,
                "High School": -3.8321,
                "Postgraduate": 1.9067
            }
        }

    def forward(self, x):
        z = (
            self.coeffs["age"] * x["age"] +
            self.coeffs["income"] * x["income"] +
            self.coeffs["credit_score"] * x["credit_score"] +
            self.coeffs["travel_frequency"] * x["travel_frequency"] +
            self.coeffs["owns_car"] * x["owns_car"] +
            self.coeffs["is_employed"] * x["is_employed"] +
            self.coeffs["location"].get(x["location"], 0) +
            self.coeffs["residential_status"].get(x["residential_status"], 0) +
            self.coeffs["education"].get(x["education"], 0)
        )
        return torch.sigmoid(torch.tensor(z))


def explore_ranges(app_path, travel_path):
    """Explore data ranges for guidance."""
    app_df = pd.read_csv(app_path)
    travel_df = pd.read_csv(travel_path)

    summary = {
        "age": (app_df["age"].min(), app_df["age"].max()),
        "income": (app_df["income"].min(), app_df["income"].max()),
        "credit_score": (app_df["credit_score"].min(), app_df["credit_score"].max()),
    }

    travel_freq = travel_df.groupby("id").apply(lambda x: (x["destination"] == "International").sum())
    summary["travel_frequency"] = (int(travel_freq.min()), int(travel_freq.max()))
    return summary


def find_optimal_input(ranges):
    """Try combinations within valid ranges to reach prob >= 0.999"""
    model = LogisticModel()

    # Try combinations within valid range (start near maximum positive features)
    for edu in ["College", "Postgraduate"]:
        for res in ["Renter", "Owner"]:
            for age in range(ranges["age"][1]-5, ranges["age"][1]+1):
                for freq in range(ranges["travel_frequency"][1]-2, ranges["travel_frequency"][1]+1):
                    x = {
                        "age": age,
                        "income": ranges["income"][1],
                        "credit_score": ranges["credit_score"][1],
                        "travel_frequency": freq,
                        "owns_car": 1,
                        "is_employed": 0,  # negative coefficient
                        "location": "Domestic",
                        "residential_status": res,
                        "education": edu
                    }
                    prob = model.forward(x).item()
                    if prob >= 0.999:
                        return x, prob
    return None, None


def send_to_api(data):
    """Send payload to prediction endpoint"""
    url = "https://mle-test-app-55hmh2trlq-as.a.run.app/predict"
    r = requests.post(url, json=data)
    return r.text
