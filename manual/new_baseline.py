"""
baseline with new workflow
"""

import time
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ensure project root is on import path so `prepare` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare import load_data, evaluate

def build_model():
	"""
    Return an sklearn Pipeline. This is what the agent improves.
	"""
	dr_model = Pipeline([
		('pca', PCA(n_components=250))
    ])
	
	cls_model = Pipeline([
        ('lr', LogisticRegression(max_iter=200, C=0.1, random_state=42))
    ])
	
	return dr_model, cls_model

def run_model():
	"""
	Run and evaluate model, and log results
	"""
	X_train, y_train, X_val, y_val = load_data()

	# train model
	dr_model, cls_model = build_model()
	t0 = time.time()
	X_train_dr = dr_model.fit_transform(X_train)
	cls_model.fit(X_train_dr, y_train)
	train_time = time.time() - t0
	print(f"Training time: {train_time:.2f} seconds")

	# evaluate model and log results
	sil, prec_10, acc = evaluate(dr_model, cls_model, X_val, y_val)

	print(f"Silhouette: {sil:.4f}, Precision@10: {prec_10:.4f}, Accuracy: {acc:.4f}")

if __name__ == '__main__':
	run_model()
