import abc
import tensorflow as tf
import tensorflow_probability as tfp
import tf_slim

ds = tfp.distributions


class BaseEncoder(object, metaclass=abc.ABCMeta):
  """Abstract encoder class.

    Implementations must define the following abstract methods:
     -`build`
     -`encode`
  """

  @abc.abstractproperty
  def output_depth(self):
    """Returns the size of the output final dimension."""
    pass

  @abc.abstractmethod
  def build(self, hparams, is_training=True):
    """Builder method for BaseEncoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      is_training: Whether or not the model is being used for training.
    """
    pass

  @abc.abstractmethod
  def encode(self, sequence, sequence_length):
    """Encodes input sequences into a precursors for latent code `z`.

    Args:
       sequence: Batch of sequences to encode.
       sequence_length: Length of sequences in input batch.

    Returns:
       outputs: Raw outputs to parameterize the prior distribution in
          MusicVae.encode, sized `[batch_size, N]`.
    """
    pass


class BaseDecoder(object, metaclass=abc.ABCMeta):
  """Abstract decoder class.

  Implementations must define the following abstract methods:
     -`build`
     -`reconstruction_loss`
     -`sample`
  """

  @abc.abstractmethod
  def build(self, hparams, output_depth, is_training=True):
    """Builder method for BaseDecoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model is being used for training.
    """
    pass

  @abc.abstractmethod
  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
          `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
          sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
          `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
    """
    pass

  @abc.abstractmethod
  def sample(self, n, max_length=None, z=None, c_input=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.

    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
    """
    pass


class MusicVAE(object):
  """Music Variational Autoencoder."""

  def __init__(self, encoder, decoder):
    """Initializer for a MusicVAE model.

    Args:
      encoder: A BaseEncoder implementation class to use.
      decoder: A BaseDecoder implementation class to use.
    """
    self._encoder = encoder
    self._decoder = decoder

  def build(self, hparams, output_depth, is_training):
    """Builds encoder and decoder.

    Must be called within a graph.

    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model will be used for training.
    """
    tf.logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                    self.encoder.__class__.__name__,
                    self.decoder.__class__.__name__, hparams.values())
    self.global_step = tf.train.get_or_create_global_step()
    self._hparams = hparams
    self._encoder.build(hparams, is_training)
    self._decoder.build(hparams, output_depth, is_training)

  @property
  def encoder(self):
    return self._encoder

  @property
  def decoder(self):
    return self._decoder

  @property
  def hparams(self):
    return self._hparams

  def encode(self, sequence, sequence_length, control_sequence=None):
    """Encodes input sequences into a MultivariateNormalDiag distribution.

    Args:
      sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
          containing the sequences to encode.
      sequence_length: The length of each sequence in the `sequence` Tensor.
      control_sequence: (Optional) A Tensor with shape
          `[num_sequences, max_length, control_depth]` containing control
          sequences on which to condition. These will be concatenated depthwise
          to the input sequences.

    Returns:
      A tfp.distributions.MultivariateNormalDiag representing the posterior
      distribution for each sequence.
    """
    hparams = self.hparams
    z_size = hparams.z_size

    sequence = tf.to_float(sequence)
    if control_sequence is not None:
      control_sequence = tf.to_float(control_sequence)
      sequence = tf.concat([sequence, control_sequence], axis=-1)
    encoder_output = self.encoder.encode(sequence, sequence_length)

    mu = tf.layers.dense(
        encoder_output,
        z_size,
        name='encoder/mu',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    sigma = tf.layers.dense(
        encoder_output,
        z_size,
        activation=tf.nn.softplus,
        name='encoder/sigma',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))

    return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

  def _compute_model_loss(
      self, input_sequence, output_sequence, sequence_length, control_sequence):
    """Builds a model with loss for train/eval."""
    hparams = self.hparams
    batch_size = hparams.batch_size

    input_sequence = tf.to_float(input_sequence)
    output_sequence = tf.to_float(output_sequence)

    max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

    input_sequence = input_sequence[:, :max_seq_len]

    if control_sequence is not None:
      control_depth = control_sequence.shape[-1]
      control_sequence = tf.to_float(control_sequence)
      control_sequence = control_sequence[:, :max_seq_len]
      # Shouldn't be necessary, but the slice loses shape information when
      # control depth is zero.
      control_sequence.set_shape([batch_size, None, control_depth])

    # The target/expected outputs.
    x_target = output_sequence[:, :max_seq_len]
    # Inputs to be fed to decoder, including zero padding for the initial input.
    x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                     [(0, 0), (1, 0), (0, 0)])
    x_length = tf.minimum(sequence_length, max_seq_len)

    # Either encode to get `z`, or do unconditional, decoder-only.
    if hparams.z_size:  # vae mode:
      q_z = self.encode(input_sequence, x_length, control_sequence)
      z = q_z.sample()

      # Prior distribution.
      p_z = ds.MultivariateNormalDiag(
          loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

      # KL Divergence (nats)
      kl_div = ds.kl_divergence(q_z, p_z)

      # Concatenate the Z vectors to the inputs at each time step.
    else:  # unconditional, decoder-only generation
      kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
      z = None

    r_loss, metric_map = self.decoder.reconstruction_loss(
        x_input, x_target, x_length, z, control_sequence)[0:2]

    free_nats = hparams.free_bits * tf.math.log(2.0)
    kl_cost = tf.maximum(kl_div - free_nats, 0)

    beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step)))
            * hparams.max_beta)
    self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

    scalars_to_summarize = {
        'loss': self.loss,
        'losses/r_loss': r_loss,
        'losses/kl_loss': kl_cost,
        'losses/kl_bits': kl_div / tf.math.log(2.0),
        'losses/kl_beta': beta,
    }
    return metric_map, scalars_to_summarize

  def train(self, input_sequence, output_sequence, sequence_length,
            control_sequence=None):
    """Train on the given sequences, returning an optimizer.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.

    Returns:
      optimizer: A tf.train.Optimizer.
    """

    _, scalars_to_summarize = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence)

    hparams = self.hparams
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
          tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
          hparams.min_learning_rate)

    optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    return optimizer

  def eval(self, input_sequence, output_sequence, sequence_length,
           control_sequence=None):
    """Evaluate on the given sequences, returning metric update ops.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
        identical).
      control_sequence: (Optional) sequence on which to condition the decoder.

    Returns:
      metric_update_ops: tf.metrics update ops.
    """
    metric_map, scalars_to_summarize = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence)

    for n, t in scalars_to_summarize.items():
      metric_map[n] = tf.metrics.mean(t)

    metrics_to_values, metrics_to_updates = (
        tf_slim.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in metrics_to_values.items():
      tf.summary.scalar(metric_name, metric_value)

    return list(metrics_to_updates.values())

  def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
    """Sample with an optional conditional embedding `z`."""
    if z is not None and int(z.shape[0]) != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0], n))

    if self.hparams.z_size and z is None:
      tf.logging.warning(
          'Sampling from conditional model without `z`. Using random `z`.')
      normal_shape = [n, self.hparams.z_size]
      normal_dist = tfp.distributions.Normal(
          loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape))
      z = normal_dist.sample()

    return self.decoder.sample(n, max_length, z, c_input, **kwargs)



class BaseLstmDecoder(BaseDecoder):
  """Abstract LSTM Decoder class.

  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=True):
    if hparams.use_cudnn:
      tf.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

    self._is_training = is_training

    tf.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n',
                    hparams.dec_rnn_size)

    self._sampling_probability = lstm_utils.get_sampling_probability(
        hparams, is_training)
    self._output_depth = output_depth
    self._output_layer = tf.layers.Dense(
        output_depth, name='output_projection')
    self._dec_cell = lstm_utils.rnn_cell(
        hparams.dec_rnn_size, hparams.dropout_keep_prob,
        hparams.residual_decoder, is_training)

  @property
  def state_size(self):
    return self._dec_cell.state_size

  @abc.abstractmethod
  def _sample(self, rnn_output, temperature):
    """Core sampling method for a single time step.

    Args:
      rnn_output: The output from a single timestep of the RNN, sized
          `[batch_size, rnn_output_size]`.
      temperature: A scalar float specifying a sampling temperature.
    Returns:
      A batch of samples from the model.
    """
    pass

  @abc.abstractmethod
  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    """Core loss calculation method for flattened outputs.

    Args:
      flat_x_target: The flattened ground truth vectors, sized
        `[sum(x_length), self._output_depth]`.
      flat_rnn_output: The flattened output from all timeputs of the RNN,
        sized `[sum(x_length), rnn_output_size]`.
    Returns:
      r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
      metric_map: A map of metric names to tuples, each of which contain the
        pair of (value_tensor, update_op) from a tf.metrics streaming metric.
    """
    pass

  def _decode(self, z, helper, input_shape, max_length=None):
    """Decodes the given batch of latent vectors vectors, which may be 0-length.

    Args:
      z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
        may be 0 for unconditioned decoding.
      helper: A seq2seq.Helper to use.
      input_shape: The shape of each model input vector passed to the decoder.
      max_length: (Optional) The maximum iterations to decode.

    Returns:
      results: The LstmDecodeResults.
    """
    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')

    decoder = lstm_utils.Seq2SeqLstmDecoder(
        self._dec_cell,
        helper,
        initial_state=initial_state,
        input_shape=input_shape,
        output_layer=self._output_layer)
    final_output, final_state, final_lengths = contrib_seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_length,
        swap_memory=True,
        scope='decoder')
    results = lstm_utils.LstmDecodeResults(
        rnn_input=final_output.rnn_input[:, :, :self._output_depth],
        rnn_output=final_output.rnn_output,
        samples=final_output.sample_id,
        final_state=final_state,
        final_sequence_lengths=final_lengths)

    return results

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
        `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.
    """
    batch_size = int(x_input.shape[0])

    has_z = z is not None
    z = tf.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf.tile(
        tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

    has_control = c_input is not None
    if c_input is None:
      c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

    sampling_probability_static = tf.get_static_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
      helper = contrib_seq2seq.TrainingHelper(x_input, x_length)
    else:
      # Use scheduled sampling.
      if has_z or has_control:
        auxiliary_inputs = tf.zeros([batch_size, tf.shape(x_input)[1], 0])
        if has_z:
          auxiliary_inputs = tf.concat([auxiliary_inputs, repeated_z], axis=2)
        if has_control:
          auxiliary_inputs = tf.concat([auxiliary_inputs, c_input], axis=2)
      else:
        auxiliary_inputs = None
      helper = contrib_seq2seq.ScheduledOutputTrainingHelper(
          inputs=x_input,
          sequence_length=x_length,
          auxiliary_inputs=auxiliary_inputs,
          sampling_probability=self._sampling_probability,
          next_inputs_fn=self._sample)

    decode_results = self._decode(
        z, helper=helper, input_shape=helper.inputs.shape[2:])
    flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
    flat_rnn_output = flatten_maybe_padded_sequences(
        decode_results.rnn_output, x_length)
    r_loss, metric_map = self._flat_reconstruction_loss(
        flat_x_target, flat_rnn_output)

    # Sum loss over sequences.
    cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf.reduce_sum(r_loss[b:e]))
    r_loss = tf.stack(r_losses)

    return r_loss, metric_map, decode_results

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             start_inputs=None, end_fn=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when sampling, if
        applicable.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      end_fn: (Optional) A callable that takes a batch of samples (sized
        `[n, output_depth]` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`.
    """
    if z is not None and int(z.shape[0]) != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0], n))

    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    if c_input is not None:
      # Tile control sequence across samples.
      c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

    # If not given, start with zeros.
    if start_inputs is None:
      start_inputs = tf.zeros([n, self._output_depth], dtype=tf.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf.concat([start_inputs, z], axis=-1)
    if c_input is not None:
      start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)
    initialize_fn = lambda: (tf.zeros([n], tf.bool), start_inputs)

    sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
    end_fn = end_fn or (lambda x: False)

    def next_inputs_fn(time, outputs, state, sample_ids):
      del outputs
      finished = end_fn(sample_ids)
      next_inputs = tf.concat([sample_ids, z], axis=-1)
      if c_input is not None:
        # We need to stop if we've run out of control input.
        finished = tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                           lambda: finished,
                           lambda: True)
        next_inputs = tf.concat([
            next_inputs,
            tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                    lambda: c_input[time + 1],
                    lambda: tf.zeros_like(c_input[0]))  # should be unused
        ], axis=-1)
      return (finished, next_inputs, state)

    sampler = contrib_seq2seq.CustomHelper(
        initialize_fn=initialize_fn, sample_fn=sample_fn,
        next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
        sample_ids_dtype=tf.float32)

    decode_results = self._decode(
        z, helper=sampler, input_shape=start_inputs.shape[1:],
        max_length=max_length)

    return decode_results.samples, decode_results