# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import estimators


_SimAgentMetricsConfig = sim_agents_metrics_pb2.SimAgentMetricsConfig
_FeatureConfig = _SimAgentMetricsConfig.FeatureConfig
_HistogramEstimate = _SimAgentMetricsConfig.HistogramEstimate
_KernelDensityEstimate = _SimAgentMetricsConfig.KernelDensityEstimate

TEST_HISTOGRAM = _HistogramEstimate(
    min_val=0.0, max_val=1.0, num_bins=10, additive_smoothing_pseudocount=0.0)


class EstimatorsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(0.0, 0.0001)
  def test_histogram_estimate_likelihood_equals_one_with_batchsize3_1sample(
      self, additive_smoothing_pseudocount: float
  ):
    log_samples = tf.constant([[0.05], [0.15], [0.25]], dtype=tf.float32)
    sim_samples = tf.convert_to_tensor(
        np.array([[0.05], [0.15], [0.25]]), dtype=tf.float32
    )
    histogram_config = _HistogramEstimate(
        min_val=0.0,
        max_val=1.0,
        num_bins=10,
        additive_smoothing_pseudocount=additive_smoothing_pseudocount,
    )
    log_likelihood = estimators.histogram_estimate(
        histogram_config, log_samples, sim_samples
    )
    expected_likelihoods = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    self.assertAllClose(
        tf.exp(log_likelihood), expected_likelihoods, atol=0.001
    )

  @parameterized.named_parameters(
      {'testcase_name': 'no_time_collapse', 'independent_timesteps': False},
      {'testcase_name': 'time_collapse', 'independent_timesteps': True},
  )
  def test_log_likelihood_estimate_timeseries_returns_correct_shape(
      self, independent_timesteps):
    feature_config = _FeatureConfig(
        histogram=TEST_HISTOGRAM,
        independent_timesteps=independent_timesteps,
        metametric_weight=0.0)
    log_likelihood = estimators.log_likelihood_estimate_timeseries(
        feature_config=feature_config,
        log_values=tf.zeros((4, 80)),
        sim_values=tf.zeros((32, 4, 80)))
    self.assertEqual(log_likelihood.shape, (4, 80))

  def test_log_likelihood_estimate_single_value_returns_correct_shape(self):
    feature_config = _FeatureConfig(
        histogram=TEST_HISTOGRAM,
        metametric_weight=0.0)
    log_likelihood = estimators.log_likelihood_estimate_scenario_level(
        feature_config=feature_config,
        log_values=tf.zeros((4,)),
        sim_values=tf.zeros((32, 4,)))
    self.assertEqual(log_likelihood.shape, (4,))

  def test_histogram_returns_correctly(self):
    sim_samples = tf.convert_to_tensor([
        tf.random.uniform((10000,)),
        tf.random.normal((10000,), mean=0.5, stddev=0.2)
    ])
    log_samples = tf.convert_to_tensor([
        [0.0, 0.15, 0.5, 1.0],
        [0.0, 0.15, 0.5, 1.0],
    ])

    log_likelihood = estimators.histogram_estimate(
        TEST_HISTOGRAM, log_samples, sim_samples)

    self.assertEqual(log_likelihood.shape, (2, 4))
    # The first population is a uniform distribution, so we expect similar log
    # probabilities (with a certain margin) for all the buckets.
    batch0_diff = log_likelihood[0] - tf.reduce_mean(log_likelihood[0])
    self.assertAllClose(batch0_diff, tf.zeros_like(batch0_diff), atol=0.1)
    # The second population is a normal distribution, so we only expect the
    # maximum to be on the 0.5 bucket.
    self.assertEqual(tf.argmax(log_likelihood[1]), 2)

  def test_histogram_estimate_raises_wrong_batch_size(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Log and Sim batch sizes must be equal.'):
      estimators.histogram_estimate(
          TEST_HISTOGRAM, tf.zeros((3, 5)), tf.zeros((5, 5)))

  def test_kernel_density_returns_correctly(self):
    sim_samples = tf.convert_to_tensor([
        tf.random.uniform((10000,)),
        tf.random.normal((10000,), mean=0.5, stddev=0.2)
    ])
    log_samples = tf.convert_to_tensor([
        [0.15, 0.25, 0.5, 0.75],
        [0.0, 0.15, 0.5, 1.0],
    ])
    log_likelihood = estimators.kernel_density_estimate(
        _KernelDensityEstimate(bandwidth=0.01), log_samples, sim_samples)

    self.assertEqual(log_likelihood.shape, (2, 4))
    # The first population is a uniform distribution, so we expect similar log
    # probabilities (with a certain margin) for all the buckets.
    batch0_diff = log_likelihood[0] - tf.reduce_mean(log_likelihood[0])
    self.assertAllClose(batch0_diff, tf.zeros_like(batch0_diff), atol=0.1)
    # The second population is a normal distribution, so we only expect the
    # maximum to be on the 0.5 bucket.
    self.assertEqual(tf.argmax(log_likelihood[1]), 2)

  def test_kernel_density_raises_wrong_bandwidth(self):
    with self.assertRaisesRegex(
        ValueError,
        'Bandwidth needs to be positive for KernelDensity estimation.'):
      estimators.kernel_density_estimate(
          _KernelDensityEstimate(bandwidth=0.0),
          tf.zeros((1, 1)),
          tf.zeros((1, 1)),
      )

  def test_kernel_density_raises_wrong_batch_size(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Log and Sim batch sizes must be equal.'):
      estimators.kernel_density_estimate(
          _KernelDensityEstimate(bandwidth=1.0),
          tf.zeros((3, 5)), tf.zeros((5, 5)))

  @parameterized.named_parameters(
      {'testcase_name': 'missing_field', 'pseudocounts': None, 'inf': False},
      {'testcase_name': 'zero_counts', 'pseudocounts': 0.0, 'inf': True},
      {'testcase_name': 'nonzero_counts', 'pseudocounts': 1.0, 'inf': False},
  )
  def test_histogram_adds_pseudocount(self, pseudocounts, inf):
    if pseudocounts is not None:
      config = _HistogramEstimate(min_val=0.0, max_val=1.0, num_bins=10,
                                  additive_smoothing_pseudocount=pseudocounts)
    else:
      config = _HistogramEstimate(min_val=0.0, max_val=1.0, num_bins=10)

    sim_samples = tf.convert_to_tensor([[0.0, 0.0, 0.0, 1.0],])
    log_samples = tf.convert_to_tensor([[0.5],])
    log_likelihood = estimators.histogram_estimate(
        config, log_samples, sim_samples)
    if inf:
      self.assertEqual(log_likelihood[0, 0], -np.inf)
    else:
      self.assertNotEqual(log_likelihood[0, 0], -np.inf)

  def test_bernoulli_returns_correctly(self):
    sim_samples = tf.convert_to_tensor([
        [False, False, False, False],
        [True, True, True, True],
        [False, False, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ])
    log_samples = tf.convert_to_tensor([
        [False], [True], [True], [False], [True]
    ])
    log_likelihood = estimators.bernoulli_estimate(
        sim_agents_metrics_pb2.SimAgentMetricsConfig.BernoulliEstimate(),
        log_samples, sim_samples)
    self.assertEqual(log_likelihood.shape, (5, 1))
    # The first 2 elements should have a likelihood of almost 1.
    self.assertAllClose(tf.exp(log_likelihood[:2]),
                        tf.ones_like(log_likelihood[:2]),
                        atol=1e-2)
    # The second 2 elements should have a likelihood of almost 0.
    self.assertAllClose(tf.exp(log_likelihood[2:4]),
                        tf.zeros_like(log_likelihood[2:4]),
                        atol=1e-2)
    # The last element should have a likelihood of ~0.25.
    self.assertAllClose(tf.exp(log_likelihood[4:]),
                        tf.fill(log_likelihood[4:].shape, 0.25),
                        atol=1e-2)


if __name__ == '__main__':
  tf.test.main()
