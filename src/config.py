from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CSV_DEFAULT_PATH = DATA_DIR / "HAM10000_metadata.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DX_FULLNAME = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

BINARY_HIGH_RISK = {"mel", "bcc", "akiec", "vasc"}
BINARY_BENIGN = {"nv", "df", "bkl"}

SUPPORTED_MULTICLASS_LABELS = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

MODEL_FEATURES = ["age", "sex", "localization"]
NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["sex", "localization"]
