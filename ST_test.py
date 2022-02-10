
# %% Imports
# %load_ext autoreload
# %autoreload 2

import tensorflow as tf

from tensorflow.keras.losses import MeanAbsoluteError

from set_transformers.tensorflow.models import SetTransformer

# %% Set Transformer build

d = 8
num_heads = 2
key_dim = 4
k = 1
m = 8
n_sabs = 2

rff_kwargs = dict(
    units_per_layers=[d]
)

mab_kwargs = dict(
    multi_head_attention_kwargs=dict(
        num_heads=num_heads,
        key_dim=key_dim
    ),
    rff_kwargs=rff_kwargs,
    layer_normalization_h_kwargs=dict(),
    layer_normalization_out_kwargs=dict(),
    normalize=False
)

isab_kwargs = dict(
    m=m,
    d=d,
    mab_h_kwargs=mab_kwargs,
    mab_out_kwargs=mab_kwargs
)

set_transformer_kwargs = dict(
    embedding_size=d,
    encoder_kwargs=dict(
        type="ISAB",
        kwargs_per_layer=[
            isab_kwargs
        ] * n_sabs
    ),
    decoder_kwargs=dict(
        pma_kwargs=dict(
            k=k,
            d=d,
            rff_kwargs=rff_kwargs,
            mab_kwargs=mab_kwargs,
        ),
        sab_kwargs=mab_kwargs,
        rff_kwargs=rff_kwargs
    )
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Masking(),
        SetTransformer(
            attention_axes=[-3],
            **set_transformer_kwargs
        ),
        tf.keras.layers.Dense(1)
    ]
)
# %% Dataset generation

n_samples = 1000

set_len = 20
min_mask_len, max_mask_len = 2, 21

min_center, max_center = 1, 100
min_spread, max_spread = -6, 6
min_scale, max_scale = 1, 10

len_mask = tf.random.uniform((n_samples,), min_mask_len, max_mask_len, dtype=tf.int32)
mask = tf.stack(
    [
        tf.concat([tf.ones((l,)), tf.zeros((max_mask_len - 1 - l,))], axis=0)
        for l in len_mask
    ]
)

x = (
    (
        tf.random.uniform((n_samples, 1, 1), min_scale, max_scale)  # set spread scale
        * tf.random.uniform((n_samples, set_len, 1), min_spread, max_spread)  # set spread
    )
    +
    tf.random.uniform((n_samples, 1, 1), min_center, max_center)  # set center
) * tf.reshape(mask, (n_samples, set_len, 1))

# %% Target invariant function

y = tf.math.reduce_mean(x, axis=1, keepdims=True)

# %% Model training

model.compile(
    loss="mean_absolute_error",
    optimizer="adam"
)
model.fit(
    x=x,
    y=y,
    epochs=100,
    batch_size=20
)

# %% Test set generation

test_set_len = 200

test_min_center, test_max_center = min_center, max_center
test_min_spread, test_max_spread = min_spread, max_spread
test_min_scale, test_max_scale = min_scale, max_scale

x_test = (
    (
        tf.random.uniform((n_samples, 1, 1), test_min_scale, test_max_scale)  # set spread scale
        * tf.random.uniform((n_samples, test_set_len, 1), test_min_spread, test_max_spread)  # set spread
    )
    +
    tf.random.uniform((n_samples, 1, 1), test_min_center, test_max_center)  # set center
)
y_test = tf.reduce_mean(x_test, axis=1, keepdims=True)

# %% Mean absolute error

mae = MeanAbsoluteError()
print(
    "Mean absolute error:\n"
    f"Train: {mae(x, y):.1f}\n"
    f" Test: {mae(x_test, y_test):.1f}"
)
# %% Visual test

idx = 238
print(
    "Regression:\n"
    f"Empyrical estimator: {y_test[idx][0, 0]:.1f}\n"
    "               vs\n"
    f"         Regression: {model(x_test[idx:idx+1])[0, 0, 0]:.1f}"
)

# %%
