"""
Run one experiment: build model, train, evaluate, log result.

Usage:
    python model.py "description"              # run a new experiment
    python model.py "description" --baseline   # run an experiment as the baseline
"""

import sys
import time
import numpy as np
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from prepare import load_data, evaluate, log_result

# EDITABLE - change to the user-specified run ID
RUN_ID = 'first_run'

def build_model():
    """
    EDITABLE: Return a tuple (dr_model, cls_model) where dr_model is a dimensionality reduction model and cls_model is a classifier.
    
    Both dr_model and cls_model should be sklearn Pipelines.
    """
    dr_model = Pipeline([
		('pca', PCA(n_components=250))
    ])
    
    cls_model = Pipeline([
        ('lr', LogisticRegression(max_iter=200, C=0.1, random_state=42))
    ])

    return dr_model, cls_model

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

def run_model():
    """
    FROZEN: Run and evaluate model, and log results
    """

    # setup
    args = sys.argv[1:]
    baseline = False
    description_parts = []
    for a in args:
        if a == "--baseline":
            baseline = True
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # load data
    X_train, y_train, X_val, y_val = load_data()

    # train model
    dr_model, cls_model = build_model()
    t0 = time.time()
    X_train_dr = dr_model.fit_transform(X_train)
    cls_model.fit(X_train_dr, y_train)
    train_time = time.time() - t0

    # evaluate model and log results
    sil, prec_10, acc = evaluate(cls_model, X_val, y_val)
    log_result(
        run_id=RUN_ID,
        train_time=train_time,
        val_metrics=(sil, prec_10, acc),
        description=description,
        baseline=baseline
    )

if __name__ == '__main__':
    run_model()
