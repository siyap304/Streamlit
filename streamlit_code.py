import os
import torch
import torch.nn as nn
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ------------------------------
# Configuration and Setup
# ------------------------------
st.set_page_config(page_title="EV Battery Temp Predictor", layout="centered")
st.title("🔋 Predict Internal Battery Temperature")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Battery dataset paths
battery_paths = {
    'LFP': 'battery/lfp_25degC',
    'NMC': 'battery/nmc_25degC',
    'NCA': 'battery/nca_25degC',
}

# Checkpoint paths
checkpoint_paths = {
    'LFP': 'checkpoints/lfp_model.pth',
    'NMC': 'checkpoints/nmc_model.pth',
    'NCA': 'checkpoints/nca_model.pth',
}

# ------------------------------
# Dataset + Model Definition
# ------------------------------
class BatterySequenceDataset(Dataset):
    def __init__(self, folder_path, external_temp, seq_len=20):
        self.seq_len = seq_len
        self.inputs, self.targets = [], []
        self.scaler_X, self.scaler_y = MinMaxScaler(), MinMaxScaler()

        data = []
        for fname in os.listdir(folder_path):
            if fname.endswith('.xlsx'):
                df = pd.read_excel(os.path.join(folder_path, fname))
                if all(c in df.columns for c in ['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Surface_Temp(degC)']):
                    df = df[['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Surface_Temp(degC)']]
                    df['Charging_Current'] = self.extract_charging_current(fname)
                    df['Ext_Temp'] = external_temp
                    data.append(df.dropna())
        df_full = pd.concat(data, ignore_index=True)

        features = ['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Charging_Current', 'Ext_Temp']
        X = self.scaler_X.fit_transform(df_full[features])
        y = self.scaler_y.fit_transform(df_full[['Surface_Temp(degC)']])

        for i in range(len(X) - seq_len):
            self.inputs.append(torch.tensor(X[i:i+seq_len], dtype=torch.float32))
            self.targets.append(torch.tensor(y[i+seq_len], dtype=torch.float32))

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return len(self.inputs)

    def extract_charging_current(self, fname):
        match = re.search(r'_(\d+\.?\d*)C', fname)
        return float(match.group(1)) if match else 0.0

class TempLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ------------------------------
# Prediction Function
# ------------------------------
def evaluate_from_checkpoint(battery, external_temp):
    model = TempLSTM(input_size=5)
    model.load_state_dict(torch.load(checkpoint_paths[battery], map_location=device))
    model.to(device)
    model.eval()

    dataset = BatterySequenceDataset(battery_paths[battery], external_temp)
    dataloader = DataLoader(dataset, batch_size=64)

    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.extend(out.cpu().numpy())

    return sum(preds) / len(preds)

# ------------------------------
# Rule‑Based Suggestion Logic
# ------------------------------
def suggest_best_battery(temp: float, available_chems: list[str]) -> str | None:
    """
    Choose the optimal battery chemistry based on ambient temperature, with fallback order:
      - Above 40°C: ["LFP", "NCA", "NMC"]
      - Below 10°C: ["NCA", "NMC", "LFP"]
      - Between 10°C and 40°C: ["NMC", "NCA", "LFP"]
    Returns the first chemistry in the preference list that’s also in available_chems.
    Returns None if there’s no overlap.
    """
    if temp > 40:
        preference = ["LFP", "NCA", "NMC"]
    elif temp < 10:
        preference = ["NCA", "NMC", "LFP"]
    else:
        preference = ["NMC", "NCA", "LFP"]

    for chem in preference:
        if chem in available_chems:
            return chem
    return None  # nothing matched

reasoning = {
    'LFP': "LFP batteries maintain better thermal stability under high temperatures, making them safer and more reliable.",
    'NMC': "NMC batteries strike a good balance between energy density and thermal control, performing well in moderate conditions.",
    'NCA': "NCA batteries perform better in cold conditions due to their higher energy density and low‑temperature capability."
}

# ------------------------------
# Streamlit UI
# ------------------------------
ambient_temp = st.number_input(
    "🌡️ Enter Ambient Temperature (°C):",
    min_value=-20.0,
    max_value=80.0,
    value=25.0,
    step=0.5
)

battery_options = st.multiselect(
    "🔌 Select Available Battery Chemistries:",
    options=["LFP", "NMC", "NCA"],
    default=[]
)

if st.button("🚀 Suggest Optimal Battery"):
    if not battery_options:
        st.error("⚠️ Please select at least one battery chemistry to proceed.")
    else:
        best = suggest_best_battery(ambient_temp, battery_options)
        if best is None:
            st.error("⚠️ None of your selected chemistries match the recommendation criteria. Please adjust your selection.")
        else:
            st.success("✅ Suggestion complete!")
            st.markdown(f"### 🏆 Best Chemistry at {ambient_temp:.1f}°C: **{best}**")
            st.info(f"**Why {best}?** {reasoning[best]}")
