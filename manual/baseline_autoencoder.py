"""Manual experiment: Small Keras autoencoder + LogisticRegression.

This file mirrors the style of other `manual/` experiments: a `build_model()` and
`run_model()` entrypoint. Use it for manual iteration before integrating into
the autoresearch loop.

Run:
    python manual/baseline_autoencoder.py

Optional args (example):
    python manual/baseline_autoencoder.py --latent 64 --epochs 10 --batch 64
"""
import time
from pathlib import Path
import sys

# ensure project root is on import path so `prepare` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import tensorflow as tf
except Exception:
    tf = None

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from prepare import load_data


def build_autoencoder(input_dim, latent_dim=64, hidden_units=128, depth=1, activation='relu'):
    if tf is None:
        raise ImportError('TensorFlow not installed. Install with `pip install tensorflow`')
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(depth):
        x = tf.keras.layers.Dense(hidden_units, activation=activation)(x)
    latent = tf.keras.layers.Dense(latent_dim, activation='linear', name='latent')(x)
    x = latent
    for _ in range(depth):
        x = tf.keras.layers.Dense(hidden_units, activation=activation)(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    autoencoder = tf.keras.Model(inputs, outputs)
    encoder = tf.keras.Model(inputs, latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def build_model(latent_dim=64, hidden_units=128, depth=1, epochs=10, batch_size=64):
    X_train, y_train, X_val, y_val = load_data()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    input_dim = X_train_s.shape[1]
    ae, encoder = build_autoencoder(input_dim, latent_dim=latent_dim,
                                    hidden_units=hidden_units, depth=depth)

    start = time.time()
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    ae.fit(
        X_train_s, X_train_s,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )
    train_time = time.time() - start

    Z_train = encoder.predict(X_train_s)
    Z_val = encoder.predict(X_val_s)

    clf = LogisticRegression(max_iter=500)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_val)
    acc = balanced_accuracy_score(y_val, y_pred)

    return dict(
        val_acc=acc,
        train_time=train_time,
        latent_dim=latent_dim,
        hidden_units=hidden_units,
        depth=depth,
        epochs=epochs,
        batch_size=batch_size,
    )


def run_model(latent_dim=64, hidden_units=128, depth=1, epochs=10, batch_size=64):
    res = build_model(latent_dim=latent_dim,
                      hidden_units=hidden_units,
                      depth=depth,
                      epochs=epochs,
                      batch_size=batch_size)

    print(f"AE train time: {res['train_time']:.2f} s")
    print(f"Validation balanced accuracy (AE -> LR): {res['val_acc']:.6f}")


if __name__ == '__main__':
    run_model(depth=3, epochs=30)
