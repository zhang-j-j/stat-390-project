"""
baseline
"""

import time
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

# ensure project root is on import path so `prepare` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare import load_data, evaluate, log_result

def build_model():
	"""
    Return an sklearn Pipeline. This is what the agent improves.
	"""
	return Pipeline([
		('pca', PCA(n_components=250)),
		('lr', LogisticRegression(max_iter=200, C=0.1, random_state=42))
	])

def run_model():
	"""
	Run and evaluate model, and log results
	"""
	X_train, y_train, X_val, y_val = load_data()

	# train model
	model = build_model()
	t0 = time.time()
	model.fit(X_train, y_train)
	train_time = time.time() - t0
	print(f"Training time: {train_time:.2f} seconds")

	# evaluate model and log results
	acc = evaluate(model, X_val, y_val)
	# log_result(
	# 	experiment_id='baseline',
	# 	val_acc=acc,
	# 	status='success',
	# 	description='Baseline model with no preprocessing or feature engineering.'
	# )
	print(acc)

if __name__ == '__main__':
	run_model()
