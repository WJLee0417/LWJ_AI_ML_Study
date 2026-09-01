import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from train import ROOT, RESULTS, ASSETS, MODELS, load_and_split

x_train, x_valid, x_test, y_train, y_valid, y_test = load_and_split()
model = joblib.load(MODELS / "churn_model.joblib")
metrics = json.loads((RESULTS / "test-metrics.json").read_text(encoding="utf-8"))
ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(x_test)), display_labels=["Retained","Churned"]).plot(cmap="Blues")
plt.tight_layout(); plt.savefig(ASSETS / "confusion-matrix.png", dpi=160); plt.close()
comparison = Path(RESULTS / "validation-model-comparison.csv").read_text(encoding="utf-8")
report = "# 고객 이탈 모델 결과\n\n## 홀드아웃 테스트\n\n" + "\n".join(f"- {key}: {value:.3f}" if isinstance(value,float) else f"- {key}: {value}" for key,value in metrics.items()) + "\n\n이탈 고객을 놓치는 비용이 더 크다는 가정에서 validation recall을 우선해 모델을 선택했다. 전처리·스케일링은 Pipeline 내부에서 train 데이터로만 fit했다.\n"
(RESULTS / "model-report.md").write_text(report, encoding="utf-8")
print("Evaluation complete")
