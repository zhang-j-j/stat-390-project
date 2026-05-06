"""
Run one experiment: build model, train, evaluate, log result.

Usage:
    python model.py "description"              # logs as status=keep
    python model.py "description" --baseline   # logs as status=baseline
    python model.py "description" --discard    # logs as status=discard
"""

import sys
import time
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

from prepare import load_data, evaluate, log_result

# EDITABLE - change to the user-specified results file
# ONLY CHANGE THE SECOND PART OF THE PATH
results_file = 'results/' + 'second_run_results.tsv'

# NOTE: git-hash tracking removed — experiments are logged without auto-commits

def build_model():
    """
    EDITABLE: Return an sklearn Pipeline. This is what the agent improves.
    """
    return Pipeline([
		('pca', PCA(n_components=200)),
		('lr', LogisticRegression(max_iter=200, C=0.1, random_state=42))
	])

class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    """
    FROZEN: Sklearn-style transformer that fits a small Keras autoencoder and returns encoded features.

    If TensorFlow is not available, `fit` will raise an informative ImportError.
    """

    def __init__(self, latent_dim=64, hidden_units=128, depth=1, epochs=10, batch_size=64,
                 activation='relu', verbose=0, random_state=42, early_stopping=True):
        self.latent_dim = int(latent_dim)
        self.hidden_units = int(hidden_units)
        self.depth = int(depth)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.activation = activation
        self.verbose = int(verbose)
        self.random_state = random_state
        self.early_stopping = early_stopping

    def _check_tf(self):
        if tf is None:
            raise ImportError(
                "TensorFlow is required for AutoencoderTransformer.\n"
                "Install it in your environment, e.g. `pip install tensorflow`."
            )

    def _build_models(self, input_dim):
        self._check_tf()
        tf.random.set_seed(self.random_state)
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        for i in range(self.depth):
            x = tf.keras.layers.Dense(self.hidden_units, activation=self.activation)(x)
        latent = tf.keras.layers.Dense(self.latent_dim, activation='linear', name='latent')(x)
        x = latent
        for i in range(self.depth):
            x = tf.keras.layers.Dense(self.hidden_units, activation=self.activation)(x)
        outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
        autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')
        encoder = tf.keras.Model(inputs, latent, name='encoder')
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        input_dim = X_scaled.shape[1]
        autoencoder, encoder = self._build_models(input_dim)
        callbacks = []
        if self.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))
        autoencoder.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            callbacks=callbacks,
        )
        self.encoder_ = encoder
        self._is_fitted = True
        return self

    def transform(self, X):
        if not getattr(self, '_is_fitted', False):
            raise RuntimeError('AutoencoderTransformer is not fitted yet. Call fit() first.')
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        return self.encoder_.predict(X_scaled, batch_size=self.batch_size)

def run_model(results_file):
    """
    FROZEN: Run and evaluate model, and log results
    """

    # setup
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # load data
    X_train, y_train, X_val, y_val = load_data()

    # train model
    model = build_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} seconds")

    # evaluate model and log results
    acc = evaluate(model, X_val, y_val)
    log_result(
        results_file=results_file,
        experiment_id='baseline',
        val_acc=acc,
        status=status,
        description=description
    )
    print(f"Result logged (status={status})")

if __name__ == '__main__':
    run_model(results_file)
