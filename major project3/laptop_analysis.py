# ============================================================
# LAPTOP DATASET ANALYSIS & PRICE PREDICTION
# ============================================================
# This script performs full EDA, preprocessing, and ML
# classification/regression on the laptop dataset.
# ============================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves PNGs)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
import warnings
warnings.filterwarnings("ignore")

# ── helper ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def save(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [saved] {path}")

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

csv_path = os.path.join(BASE_DIR, "laptopData.csv")
df = pd.read_csv(csv_path)
print(f"Shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────────────────
# 2. BASIC CLEANING
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. CLEANING")
print("="*60)

# Drop unnamed index column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Drop fully-empty rows
before = len(df)
df.dropna(how="all", inplace=True)
print(f"Removed {before - len(df)} fully-null rows. Remaining: {len(df)}")

# Drop duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Removed {before - len(df)} duplicate rows. Remaining: {len(df)}")

# ─────────────────────────────────────────────────────────
# 3. TYPE CONVERSION
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. TYPE CONVERSION")
print("="*60)

# Inches → float
df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

# Ram  → int  (strip "GB")
df["Ram"] = (df["Ram"].astype(str)
               .str.replace("GB", "", regex=False)
               .pipe(pd.to_numeric, errors="coerce")
               .astype("Int64"))

# Weight → float  (strip "kg")
df["Weight"] = (df["Weight"].astype(str)
                  .str.replace("kg", "", regex=False)
                  .pipe(pd.to_numeric, errors="coerce"))

print("Missing values after conversion:")
print(df.isna().sum())

# Fill remaining numeric NaNs with column median
for col in ["Inches", "Weight"]:
    df[col].fillna(df[col].median(), inplace=True)

# ─────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING  (Memory)
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. FEATURE ENGINEERING")
print("="*60)

def parse_memory(mem_str):
    """Returns (size_GB, type_flag_dict)."""
    mem_str = str(mem_str).upper()
    size = 0
    flags = {"HDD": False, "SSD": False, "Hybrid": False, "Flash": False}

    # find all 'NGB' or 'N.NTB' patterns
    tokens = re.findall(r"(\d+\.?\d*)\s*(TB|GB)", mem_str)
    for val, unit in tokens:
        gb = float(val) * (1024 if unit == "TB" else 1)
        size += gb

    if "HDD" in mem_str:         flags["HDD"]    = True
    if "SSD" in mem_str:         flags["SSD"]    = True
    if "HYBRID" in mem_str:      flags["Hybrid"] = True
    if "FLASH" in mem_str:       flags["Flash"]  = True
    if not any(flags.values()):  flags["Flash"]  = True  # fallback

    return size, flags

parsed = df["Memory"].apply(parse_memory)
df["Memory_Size_GB"]    = parsed.apply(lambda x: x[0])
df["Memory_SSD"]        = parsed.apply(lambda x: x[1]["SSD"]).astype(int)
df["Memory_HDD"]        = parsed.apply(lambda x: x[1]["HDD"]).astype(int)
df["Memory_Hybrid"]     = parsed.apply(lambda x: x[1]["Hybrid"]).astype(int)
df["Memory_Flash"]      = parsed.apply(lambda x: x[1]["Flash"]).astype(int)

# CPU brand
df["CPU_Brand"] = df["Cpu"].str.split().str[0]

# GPU brand
df["GPU_Brand"] = df["Gpu"].str.split().str[0]

# Screen resolution pixels
def extract_res(s):
    m = re.search(r"(\d{3,4})x(\d{3,4})", str(s))
    if m:
        return int(m.group(1)) * int(m.group(2))
    return np.nan

df["Screen_Pixels"] = df["ScreenResolution"].apply(extract_res)
df["Screen_Pixels"].fillna(df["Screen_Pixels"].median(), inplace=True)

print("New columns:", [c for c in df.columns if c not in
      ["Company","TypeName","Inches","ScreenResolution","Cpu",
       "Ram","Memory","Gpu","OpSys","Weight","Price"]])

# ─────────────────────────────────────────────────────────
# 5. EDA  –  VISUALISATIONS
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. VISUALISATIONS  (saved to ./plots/)")
print("="*60)

# 5a. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Laptop Price Distribution")
plt.xlabel("Price"); plt.ylabel("Count")
save("01_price_distribution.png")

# 5b. Laptops per company
plt.figure(figsize=(10, 5))
order = df["Company"].value_counts().index
sns.countplot(x="Company", data=df, order=order)
plt.title("Laptops per Company")
plt.xticks(rotation=45)
save("02_laptops_per_company.png")

# 5c. Price by company (box)
plt.figure(figsize=(12, 5))
sns.boxplot(x="Company", y="Price", data=df,
            order=df.groupby("Company")["Price"].median().sort_values().index)
plt.title("Price Distribution by Company")
plt.xticks(rotation=45)
save("03_price_by_company.png")

# 5d. Price by TypeName
plt.figure(figsize=(10, 5))
sns.boxplot(x="TypeName", y="Price", data=df)
plt.title("Price Distribution by TypeName")
plt.xticks(rotation=30)
save("04_price_by_typename.png")

# 5e. Price vs RAM
plt.figure(figsize=(8, 5))
sns.boxplot(x="Ram", y="Price", data=df)
plt.title("Price vs RAM (GB)")
save("05_price_vs_ram.png")

# 5f. Correlation heatmap (numeric)
plt.figure(figsize=(10, 8))
num_cols = ["Inches", "Ram", "Weight", "Price",
            "Memory_Size_GB", "Memory_SSD", "Memory_HDD",
            "Screen_Pixels"]
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
save("06_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────
# 6.  PRICE CATEGORY  (for classification)
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. PRICE CATEGORY")
print("="*60)

# Bin into 3 categories: Budget / Mid-range / High-end
price_bins   = [0, 30000, 70000, df["Price"].max() + 1]
price_labels = ["Budget", "Mid-range", "High-end"]
df["Price_Category"] = pd.cut(df["Price"], bins=price_bins,
                               labels=price_labels)
print(df["Price_Category"].value_counts())

# ─────────────────────────────────────────────────────────
# 7.  ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. ENCODING")
print("="*60)

cat_cols = ["Company", "TypeName", "OpSys", "CPU_Brand", "GPU_Brand"]
label_encoders = {}

df_enc = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col + "_enc"] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col]  = le

# ─────────────────────────────────────────────────────────
# 8.  CLASSIFICATION  –  Price Category
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("8. CLASSIFICATION  (Price Category)")
print("="*60)

FEATURE_COLS = [
    "Inches", "Ram", "Weight",
    "Memory_Size_GB", "Memory_SSD", "Memory_HDD", "Screen_Pixels",
    "Company_enc", "TypeName_enc", "OpSys_enc",
    "CPU_Brand_enc", "GPU_Brand_enc",
]

X = df_enc[FEATURE_COLS].fillna(0)
y_cat = LabelEncoder().fit_transform(df_enc["Price_Category"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
y_pred_cat = rf_clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred_cat), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cat,
      target_names=["Budget", "High-end", "Mid-range"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_cat)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Budget","High-end","Mid-range"],
            yticklabels=["Budget","High-end","Mid-range"])
plt.title("Confusion Matrix – Price Category")
plt.ylabel("True"); plt.xlabel("Predicted")
save("07_confusion_matrix_classification.png")

# Feature importance
fi = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 6))
fi.plot(kind="barh")
plt.title("Feature Importance – Price Category Classifier")
save("08_feature_importance_classification.png")

# ─────────────────────────────────────────────────────────
# 9.  REGRESSION  –  Price
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("9. REGRESSION  (Price prediction)")
print("="*60)

y_reg = df_enc["Price"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_r, y_train_r)
y_pred_r = rf_reg.predict(X_test_r)

mae  = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2   = r2_score(y_test_r, y_pred_r)

print(f"MAE  : {mae:,.0f}")
print(f"RMSE : {rmse:,.0f}")
print(f"R²   : {r2:.4f}")

# Predicted vs actual scatter
plt.figure(figsize=(7, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.5, edgecolors="k", linewidths=0.3)
mn, mx = y_test_r.min(), y_test_r.max()
plt.plot([mn, mx], [mn, mx], "r--", lw=2)
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
plt.title("Regression: Actual vs Predicted Price")
save("09_regression_actual_vs_predicted.png")

# Feature importance
fi_r = pd.Series(rf_reg.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 6))
fi_r.plot(kind="barh")
plt.title("Feature Importance – Price Regressor")
save("10_feature_importance_regression.png")

# ─────────────────────────────────────────────────────────
# 10.  SAVE CLEANED DATA
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("10. SAVE CLEANED CSV")
print("="*60)

out_csv = os.path.join(BASE_DIR, "laptopData_cleaned.csv")
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

print("\n" + "="*60)
print("ALL DONE  –  Check the 'plots/' folder for visualisations.")
print("="*60)
