
# %% Imports

from importlib.metadata import distribution
from numpy import ndim
import tensorflow as tf
import tensorflow_probability as tfp
from inference_gym.targets import (
    Banana
)

from tensorflow.keras.losses import MeanAbsoluteError

from set_transformers.tensorflow.models import SetTransformer

tfd = tfp.distributions

# %% Datasets

D = 2
train_size, val_size = 2_000, 200
base_set_length = 300
epochs = 100

datasets = {
    name: {
        "train": dist.sample((train_size, base_set_length)),
        "val": dist.sample((val_size, base_set_length))
    }
    for name, dist in [
        ("Banana", Banana(ndims=D)),
        (
            "Uniform", 
            tfd.Uniform(
                low=- 1 * tf.ones((D,)),
                high=1 *  tf.ones((D,))
            )
        ),
        (
            "Normal",
            tfd.Normal(
                loc=tf.zeros((D,)),
                scale=1.
            )
        )
    ]
}

for dist, dist_dataset in datasets.items():
    for split, data in dist_dataset.items():
        datasets[dist][split] = (
            (data - tf.reduce_mean(data, axis=(1, 2), keepdims=True))
            / tf.math.reduce_std(data, axis=(1, 2), keepdims=True)
        )

scales = {
    split: tfd.Uniform(low=1, high=10).sample((size, 1, 1))
    for split, size in [
        ("train", train_size),
        ("val", val_size)
    ] 
}

shifts = {
    split: tfd.Uniform(low=-50, high=50).sample((size, 1, 1))
    for split, size in [
        ("train", train_size),
        ("val", val_size)
    ] 
}

# %% Set sizes

small_sizes = dict(
    minval=1,
    maxval=10
)

medium_sizes = dict(
    minval=10,
    maxval=100
)

big_sizes = dict(
    minval=100,
    maxval=300
)

len_masks = {
    size: {
        split: tf.random.uniform(
            shape=shape,
            **size_kwargs,
            dtype=tf.int32
        )
        for split, shape in [
            ("train", (train_size,)),
            ("val", (val_size,))
        ]
    }
    for size, size_kwargs in [
        ("small", small_sizes),
        ("medium", medium_sizes),
        ("big", big_sizes)
    ]
}

masks = {
    size: {
        split: tf.stack(
            [
                tf.concat(
                    [
                        tf.ones((l, 1)),
                        tf.zeros((base_set_length - l, 1))
                    ],
                    axis=-2
                )
                for l in split_lens
            ]
        )
        for split, split_lens in size_lens.items()
    }
    for size, size_lens in len_masks.items()
}

set_generalizations = [
    (
        "small",
        [
            "small",
            "medium",
            "big"
        ]
    ) 
    
]


# %% Functions

def my_mean(tensor): return tf.reduce_mean(tensor, axis=1, keepdims=True)
def my_max(tensor): return tf.reduce_max(tensor, axis=1, keepdims=True)
def my_sum(tensor): return tf.reduce_sum(tensor, axis=1, keepdims=True)
def my_std(tensor): return tf.math.reduce_std(tensor, axis=1, keepdims=True)
def my_norm3(tensor): return tf.norm(tensor, ord=3, axis=1, keepdims=True)
def my_norm5(tensor): return tf.norm(tensor, ord=5, axis=1, keepdims=True)

functions = [
    ("Mean", my_mean),
    ("Max", my_max),
    ("Sum", my_sum),
    ("Std", my_std),
    ("3-Norm", my_norm3),
    ("5-Norm", my_norm5)
]

# %% Set transformer build

def get_ST(D: int) -> tf.keras.Model:
    d = 8  # ! should equal `num_heads * key_dim``
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
        normalize=True
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
            tf.keras.layers.Dense(D)
        ]
    )

    return model

# %% Massive test

mae = MeanAbsoluteError()
maes = {}

for dataset_name, dataset in datasets.items():
    print(f"Testing over dataset {dataset_name}...")
    for function_name, my_func in functions:
        print(f"\tTesting for function {function_name}...")
        for source_size, target_sizes in set_generalizations:
            print(f"\t\tTraining over sets of {source_size} size...")
            x_train = (
                (
                    dataset["train"] * scales["train"]
                    + shifts["train"]
                ) * masks[source_size]["train"]
            )
            y_train = my_func(x_train)

            model = get_ST(D)
            model.compile(
                loss="mean_absolute_error",
                optimizer="adam"
            )
            model.fit(
                x=x_train,
                y=y_train,
                epochs=epochs,
                batch_size=20,
                verbose=0
            )
            mae_train = mae(model(x_train), y_train)
            print(f"\t\t\t\tMAE train set: {mae_train:.2f}")

            for target_size in target_sizes:
                x_val = (
                    (
                        dataset["val"] * scales["val"]
                        + shifts["val"]
                    ) * masks[target_size]["val"]
                )
                y_val = my_func(x_val)

                mae_val = mae(model(x_val), y_val)

                print(
                    f"\t\t\t...Generalizing over sets of {target_size} size:\n"
                    f"\t\t\t\tMAE val set: {mae_val:.2f}"
                )

# %%
