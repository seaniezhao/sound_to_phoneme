import tensorflow as tf


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        layers=4,
        blocks=2,
        dilation_channels=128,
        residual_channels=128,
        skip_channels=256,
        input_channel=24,
        output_channel=86,
        initial_kernel=1,
        kernel_size=2,
        bias=True
    )

    if hparams_string:
        tf.logging.info('Parsing harmonic hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final harmonic hparams: %s', hparams.values())

    return hparams


