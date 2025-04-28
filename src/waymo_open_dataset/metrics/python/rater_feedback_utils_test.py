# Copyright 2025 The Waymo Open Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Tests for rater feedback score.


"""

from typing import List, Tuple
import numpy as np
from absl.testing import absltest
from waymo_open_dataset.metrics.python import rater_feedback_utils

np.set_printoptions(precision=4, suppress=True)


TEST_DATA = [
    {
        "index": 654,
        "speed": 7.716310501098633,
        "inference_trajectories_MODEL1": [
            [
                [6.171509239622196, -0.009397453076275042],
                [8.000396968767177, 0.015122155172520024],
                [8.127134351157906, 0.013057147475194597],
                [8.127134351157906, 0.013057147475194597],
                [8.127134351157906, 0.013057147475194597],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL1": 10.0,
        "inference_trajectories_MODEL2": [
            [
                [5.99, 0.04],
                [10.71, 0.09],
                [15.76, 0.16],
                [21.19, 0.29],
                [26.88, 0.46],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL2": 6.0,
        "inference_trajectories_MODEL3": [
            [
                [6.49681282043457, -0.08341606706380844],
                [11.86185359954834, -0.7157369256019592],
                [16.411989212036133, -2.882720470428467],
                [19.67278480529785, -7.174043655395508],
                [21.36919403076172, -13.386308670043945],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL3": 0.0857031037460114,
        "rater_specified_trajectories": [
            np.array([
                [5.560453799210904, -0.004478094274702471],
                [8.225178876764971, 0.06945586454486374],
                [8.981938003877872, 0.13376779402065608],
                [9.019658779669157, 0.17555393283325316],
                [9.225985951119242, 0.23028404494718302],
                [1000, 1000],  # Add an additional trajectory.
            ]),
            np.array([
                [5.916309994790822, -0.026716178826546866],
                [9.30287526924758, -0.12584874734611162],
                [11.293785754835312, -0.27650364630528657],
                [12.9184486854341, -0.48288228974388403],
                [14.58230516729327, -0.7568973709328475],
            ]),
            np.array([
                [6.141386389118566, -0.020121756825716375],
                [10.561775492192169, -0.09057494033771718],
                [14.474645343934071, -0.2161768449137753],
                [18.882845193998037, -0.35637872008717864],
                [24.11468565241205, -0.3829896472898042],
            ]),
        ],
        "rater_feedback_labels": np.array([10, 8, 6]),
    },
    {
        "index": 665,
        "speed": 2.1179442405700684,
        "inference_trajectories_MODEL1": [
            [
                [2.843017676842237, -0.37735536030049843],
                [6.38129387739059, 1.286901696548739],
                [8.794416704687592, 7.077898292320242],
                [9.18251237084678, 16.29023700124344],
                [8.504202822321759, 27.60476714770084],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL1": 9.975806567307454e-08,
        "inference_trajectories_MODEL2": [
            [
                [3.01, -1.84],
                [5.85, -5.91],
                [8.08, -11.99],
                [10.27, -19.74],
                [12.73, -28.7],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL2": 0.1888432558844814,
        "inference_trajectories_MODEL3": [
            [
                [2.8002960681915283, -1.4907801151275635],
                [5.121310234069824, -5.308871269226074],
                [6.296884059906006, -11.264496803283691],
                [7.084940433502197, -19.01148223876953],
                [7.9615983963012695, -28.370180130004883],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL3": 5.228241007761699,
        "rater_specified_trajectories": [
            np.array([
                [1.7595119940760924, -0.14755874950424186],
                [2.7480205686624686, -0.32863112759878277],
                [3.3625390147151393, -0.39949733974754054],
                [4.182183364439197, -0.337427855545684],
                # [5.5415976026697535, 0.3037444224610226],
            ]),
            np.array([
                [2.59763517447891, -0.6775495835981928],
                [5.754005458843949, -4.707164778688821],
                [6.891353861038624, -12.898902197828647],
                [7.305693692876957, -24.033863794415083],
                [8.146774067708066, -37.432607402115536],
            ]),
            np.array([
                [2.213074454070238, -0.38524633570341393],
                [4.294703086304935, -1.9280354641468875],
                [5.370505334488826, -4.004647880774428],
                [5.742787436107619, -5.969209589298771],
                [5.9755163585334685, -7.5328635370988195],
            ]),
        ],
        "rater_feedback_labels": np.array([6, 10, 9]),
    },
    {
        "index": 689,
        "speed": 0.0,
        "inference_trajectories_MODEL1": [
            [
                [0.047916457508563326, -0.002053818058584511],
                [0.047916457508563326, -0.002053818058584511],
                [0.047916457508563326, -0.002053818058584511],
                [0.047916457508563326, -0.002053818058584511],
                [0.047916457508563326, -0.002053818058584511],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL1": 10.0,
        "inference_trajectories_MODEL2": [
            [
                [0.34, -0.0],
                [1.83, -0.06],
                [4.5, -0.34],
                [8.32, -1.12],
                [13.28, -2.25],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL2": 3.03901054409306,
        "inference_trajectories_MODEL3": [
            [
                [7.659792754566297e-05, 0.0007740182336419821],
                [-0.0010324065806344151, -0.01936725527048111],
                [-0.01153242401778698, -0.0009276904165744781],
                [0.002999512478709221, -0.005620112176984549],
                [0.04409359022974968, -0.002266160910949111],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL3": 10.0,
        "rater_specified_trajectories": [
            np.array([
                [0.021765578061604174, -0.0009883759189506236],
                [0.0704513775876876, -0.0009584886752236343],
                [0.14182930844481234, -0.0010432547105665435],
                [0.22239692046559867, -0.0011236832513077388],
                [0.30971526501389235, -0.0009913351880186383],
            ]),
            np.array([
                [0.02437357686130781, 0.004737612713597628],
                [0.15121775384659486, 0.003419895889237523],
                [0.462070832894824, -0.005463375944827931],
                [0.9242727706432561, -0.0357403913849339],
                [1.7397894186888152, -0.1495510390718664],
            ]),
            np.array([
                [0.0385635132306561, -0.004438296846046796],
                [0.45706518174256416, -0.006832136379443909],
                [1.90967754585472, -0.0567133154977455],
                [4.645350163917101, -0.2937360710000121],
                [8.768589639407764, -0.7562402387763996],
            ]),
        ],
        "rater_feedback_labels": np.array([10, 9, 6]),
    },
    {
        "index": 1271,
        "speed": 6.091951370239258,
        "inference_trajectories_MODEL1": [
            [
                [6.315426478653762, 0.043856417986262386],
                [12.975259841223988, 0.42393014905155724],
                [19.67305036031098, 1.8076265692643574],
                [26.044775930205105, 4.679681408367287],
                [31.774181598299265, 8.93842790671124],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL1": 10.0,
        "inference_trajectories_MODEL2": [
            [
                [6.22, 0.06],
                [12.87, 0.42],
                [19.53, 1.58],
                [25.81, 3.93],
                [31.42, 7.51],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL2": 7.307854143879492,
        "inference_trajectories_MODEL3": [
            [
                [6.233642578125, 0.12418137490749359],
                [12.584113121032715, 0.8310785293579102],
                [18.869394302368164, 2.611462354660034],
                [25.005125045776367, 5.8199005126953125],
                [31.0169620513916, 10.563572883605957],
            ],
        ],
        "hard_rater_feedback_score_decay_10_MODEL3": 10.0,
        "rater_specified_trajectories": [
            np.array([
                [6.116130709045365, -0.0064596553261253575],
                [11.283734185302137, 0.06930987395980992],
                [14.445738365972602, 0.22190593950062976],
                [15.863924647712793, 0.3513780388188934],
                [16.220629990528778, 0.3979054272808753],
            ]),
            np.array([
                [6.3023491185483635, 0.027379927538731863],
                [13.151817455762284, 0.5152881368030648],
                [20.08838570878538, 2.2066481916076555],
                [26.655734316468852, 5.669334168663681],
                [32.556009257534924, 10.914624712820569],
            ]),
            # np.array([
            #     [6.319490477111003, -0.0015768621483402967],
            #     [13.059201327919482, 0.15629147996469328],
            #     [19.736942500581563, 0.8357351084546281],
            #     [26.002607822890013, 2.49883944469002],
            #     [31.577330942158596, 5.3225048541467],
            # ]),
        ],
        "rater_feedback_labels": np.array([8, 10]),
    },
]


class MetricTest(absltest.TestCase):

  def test_get_rater_feedback_score_with_zeros(self):
    batch_size = 2
    num_inference_trajectories = 2
    num_rater_specified_trajectories = 3
    frequency = 4  # Default frequency. Shouldn't be changed.
    num_timesteps = 5 * frequency

    # Consider trajectories staying at the same location.
    inference_trajectories: np.ndarray = np.zeros(
        (batch_size, num_inference_trajectories, num_timesteps, 2),
        dtype=np.float64,
    )

    # Uniformly distributed probabilities.
    inference_probs: np.ndarray = (
        np.ones((batch_size, num_inference_trajectories), dtype=np.float64)
        / num_inference_trajectories
    )

    rater_specified_trajectories: List[List[np.ndarray]] = []
    for _ in range(batch_size):
      current_batch_rater_trajs: List[np.ndarray] = []
      for _ in range(num_rater_specified_trajectories):
        # Trajectories staying at the same location (zeros)
        trajectory_data = np.zeros((num_timesteps, 2), dtype=np.float64)
        current_batch_rater_trajs.append(trajectory_data)
      rater_specified_trajectories.append(current_batch_rater_trajs)

    rater_feedback_labels: List[np.ndarray] = []
    for _ in range(batch_size):
      # Scores are all 10
      labels_data = (
          np.ones(num_rater_specified_trajectories, dtype=np.int64) * 10
      )
      rater_feedback_labels.append(labels_data)
    # Initial speed is 0.0.
    init_speed: np.ndarray = np.zeros((batch_size,), dtype=np.float64)

    # Threshold multipliers and decay factor.
    # These default values should be used for all cases.
    lat_lng_threshold_multipliers: Tuple[float, float] = (1.0, 4.0)
    decay_factor: float = 0.1

    outputs = rater_feedback_utils.get_rater_feedback_score(
        inference_trajectories,
        inference_probs,
        rater_specified_trajectories,
        rater_feedback_labels,
        init_speed,
        lat_lng_threshold_multipliers=lat_lng_threshold_multipliers,
        decay_factor=decay_factor,
        frequency=frequency,
        output_trust_region_visualization=False,
        length_seconds=5,
    )

    rater_feedback_score = outputs["rater_feedback_score"]

    # assert that the rater feedback score is 10.0.
    np.testing.assert_array_equal(
        rater_feedback_score, np.ones((batch_size,)) * 10.0
    )

    # Test the case where the rater trajectories have varied lengths.
    rater_specified_trajectories: List[List[np.ndarray]] = []
    for i in range(batch_size):
      current_batch_rater_trajs: List[np.ndarray] = []
      for _ in range(i + 2):
        # Trajectories staying at the same location (zeros)
        trajectory_data = np.zeros((num_timesteps - i, 2), dtype=np.float64)
        current_batch_rater_trajs.append(trajectory_data)
      rater_specified_trajectories.append(current_batch_rater_trajs)

    rater_feedback_labels: List[np.ndarray] = []
    for i in range(batch_size):
      # Scores are all 10
      labels_data = np.ones(i + 2, dtype=np.int64) * 10
      rater_feedback_labels.append(labels_data)

    # assert that the rater feedback score is 10.0.
    np.testing.assert_array_equal(
        rater_feedback_score, np.ones((batch_size,)) * 10.0
    )

  def test_get_rater_feedback_score_with_samples(self):
    """Test with real samples."""

    rater_specified_trajectories = [
        data["rater_specified_trajectories"] for data in TEST_DATA
    ]
    rater_feedback_labels = [
        data["rater_feedback_labels"] for data in TEST_DATA
    ]

    init_speed = np.array(
        [data["speed"] for data in TEST_DATA], dtype=np.float64
    )

    for model in ["MODEL1", "MODEL2", "MODEL3"]:
      inference_trajectories = np.array(
          [data["inference_trajectories_" + model] for data in TEST_DATA],
          dtype=np.float64,
      )
      inference_probs = np.array([1.0] * len(TEST_DATA))[..., None]

      for min_nontrust_score in [0.0, 4.0]:
        outputs = rater_feedback_utils.get_rater_feedback_score(
            inference_trajectories,
            inference_probs,
            rater_specified_trajectories,
            rater_feedback_labels,
            init_speed,
            frequency=1,  # Default is 4.
            output_trust_region_visualization=False,
            # Default is 4.
            minimum_score_outside_trust_region=min_nontrust_score,
        )
        rater_feedback_score = np.array(outputs["rater_feedback_score"])

        rater_feedback_score_expected = np.array([
            data[f"hard_rater_feedback_score_decay_10_{model}"]
            for data in TEST_DATA
        ])
        if min_nontrust_score <= 4.0:
          # In our samples, no sample within trust region has score <= 4.0
          # Therefore, we can clip the final score to get the expected output.
          rater_feedback_score_expected = np.maximum(
              rater_feedback_score_expected, min_nontrust_score
          )
        else:
          raise NotImplementedError(
              "Test not implemented for min_nontrust_score > 4.0"
          )
        np.testing.assert_array_almost_equal(
            rater_feedback_score_expected,
            rater_feedback_score,
            decimal=15,
            err_msg=(
                f"Test failed for: hard_rater_feedback_score_decay_10_{model}"
            ),
        )

  def test_get_rater_feedback_score_with_grid(self):
    # [num_rater_specified_trajectories=1, T, 2]
    rater_specified_trajectories = np.array(
        [[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]],
        dtype=np.float64,
    )
    # [num_rater_specified_trajectories=1]
    rater_feedback_labels = np.array([1], dtype=np.int64)

    # Scale factor's boundaries are 1.4 and 11.0, so we pick 5 values.
    init_speed_list = np.array([0.0, 1.4, 6.2, 11.0, 22.0])

    # A set of hyperparameters for
    # - lat_lng_threshold_multipliers
    # - decay_factor
    lat_lng_threshold_multipliers_list = [(1.0, 2.0), (1.0, 4.0)]
    decay_factor_list = [0.1, 0.2, 0.3]
    frequency = 1

    # A set of hyperparamters for minimum score outside trust region.
    minimum_score_outside_trust_region_list = [0.0 / 10.0, 4.0 / 10.0]

    for lat_lng_threshold_multipliers in lat_lng_threshold_multipliers_list:
      for decay_factor in decay_factor_list:
        for min_nontrust_score in minimum_score_outside_trust_region_list:
          # Get lat and lng thresholds for each init_speed.
          # [num_init_speed=5, num_trust_regions=2], [5, 2]
          lat_thresholds, lng_thresholds = (
              rater_feedback_utils.get_lat_lng_thresholds(
                  init_speed_list, lat_lng_threshold_multipliers
              )
          )

          # Define 9x9 grid of shift relative to thresholds.
          # [9]
          lat_shifts = np.linspace(-2.0, 2.0, 9, endpoint=True)
          lng_shifts = np.linspace(-2.0, 2.0, 9, endpoint=True)
          # [9, 9], [9, 9]
          lat_shifts, lng_shifts = np.meshgrid(lat_shifts, lng_shifts)
          # [81]
          lat_shifts = lat_shifts.flatten()
          lng_shifts = lng_shifts.flatten()

          # Multiply by thresholds.
          # [81, 1, 1] * [1, 5, 2] --> [81, 5, 2]
          lat_shifts = lat_shifts[..., None, None] * lat_thresholds[None, ...]
          lng_shifts = lng_shifts[..., None, None] * lng_thresholds[None, ...]
          # [81, 5]
          init_speed = np.repeat(
              init_speed_list[None, ...], lat_shifts.shape[0], axis=0
          )

          # Reshape: [batch_size=405, ...]
          lat_shifts = np.reshape(lat_shifts, (-1, 2))
          lng_shifts = np.reshape(lng_shifts, (-1, 2))
          init_speed = init_speed.flatten()

          batch_size = init_speed.shape[0]

          # [num_rater_specified_trajectories=1, T, 2]
          inference_trajectories = np.copy(rater_specified_trajectories)
          # [batch_size=405, num_rater_specified_trajectories=1, T, 2]
          inference_trajectories = np.repeat(
              inference_trajectories[None, ...], batch_size, axis=0
          )
          # We assume frequence=1.
          indices = rater_feedback_utils._THRESHOLD_TIME_SECONDS - 1

          # Get shifted inference trajectories.
          # Shift x at 3s and 5s
          inference_trajectories[:, 0, indices[0], 0] += lng_shifts[:, 0]
          inference_trajectories[:, 0, indices[1], 0] += lng_shifts[:, 1]
          # Shift y at 3s and 5s
          inference_trajectories[:, 0, indices[0], 1] -= lat_shifts[:, 0]
          inference_trajectories[:, 0, indices[1], 1] -= lat_shifts[:, 1]

          # Get inference probs.
          # [1, 1]
          inference_probs = np.array([[1.0]], dtype=np.float64)
          # [batch_size=405, 1]
          inference_probs = np.repeat(inference_probs, batch_size, axis=0)

          # Use the same rater specified trajectories for all batches.
          repeated_rater_specified_trajectories_np = np.repeat(
              rater_specified_trajectories[None, ...],
              batch_size,
              axis=0,
          )
          # Convert to the format that the rater metric can use.
          repeated_rater_trajectories_list: List[List[np.ndarray]] = []
          for i in range(batch_size):  # Iterate for each item in the batch
            # Each batch item will have its own list of rater trajectories
            trajectories_for_this_batch_item: List[np.ndarray] = []
            for j in range(repeated_rater_specified_trajectories_np.shape[1]):
              trajectory_slice = np.copy(
                  repeated_rater_specified_trajectories_np[i, j, :, :]
              )
              trajectories_for_this_batch_item.append(trajectory_slice)
            repeated_rater_trajectories_list.append(
                trajectories_for_this_batch_item
            )

          repeated_rater_feedback_labels_np = np.repeat(
              rater_feedback_labels[None, ...],
              batch_size,
              axis=0,
          )
          repeated_rater_labels_list: List[np.ndarray] = []
          for i in range(batch_size):  # Iterate for each item in the batch
            # Extract the (1,) label array for batch item i
            label_slice = np.array(repeated_rater_feedback_labels_np[i])
            repeated_rater_labels_list.append(label_slice)
          outputs = rater_feedback_utils.get_rater_feedback_score(
              inference_trajectories,
              inference_probs,
              repeated_rater_trajectories_list,
              repeated_rater_labels_list,
              init_speed,
              lat_lng_threshold_multipliers=lat_lng_threshold_multipliers,
              decay_factor=decay_factor,
              frequency=frequency,
              output_trust_region_visualization=False,
              minimum_score_outside_trust_region=min_nontrust_score,
              default_num_of_rater_specified_trajectories=3,
          )
          rater_feedback_score = np.array(outputs["rater_feedback_score"])
          # [num_lng_grids, num_lat_grids, num_init_speed]
          rater_feedback_score = np.reshape(rater_feedback_score, (9, 9, -1))

          # You can print out the rater feedback score for each init_speed.
          # An example output is shown below. This example shows the raw
          # decay without clipping. If `min_nontrust_score` is set, these scores
          # will be clipped to a minimum of `min_nontrust_score`.
          # [[0.3    0.3    0.3    0.3    0.3    0.3    0.3    0.3    0.3   ]
          #  [0.3    0.5477 0.5477 0.5477 0.5477 0.5477 0.5477 0.5477 0.3   ]
          #  [0.3    0.5477 1.     1.     1.     1.     1.     0.5477 0.3   ]
          #  [0.3    0.5477 1.     1.     1.     1.     1.     0.5477 0.3   ]
          #  [0.3    0.5477 1.     1.     1.     1.     1.     0.5477 0.3   ]
          #  [0.3    0.5477 1.     1.     1.     1.     1.     0.5477 0.3   ]
          #  [0.3    0.5477 1.     1.     1.     1.     1.     0.5477 0.3   ]
          #  [0.3    0.5477 0.5477 0.5477 0.5477 0.5477 0.5477 0.5477 0.3   ]
          #  [0.3    0.3    0.3    0.3    0.3    0.3    0.3    0.3    0.3   ]]
          for i in range(rater_feedback_score.shape[-1]):
            print(rater_feedback_score[..., i])

          # (TEST1) we can see that the rater feedback score is
          # flat within the trust region for list of hyperparameters
          # (init_speed, lat_lng_threshold_multipliers, decay_factor).
          assert np.all(rater_feedback_score[2:7, 2:7, :] == 1.0)

          # (TEST2) we can see that the rater feedback score is the same
          # for given lat_lng_threshold_multipliers and decay_factor.
          for i in range(rater_feedback_score.shape[-1]):
            if i > 0:
              assert np.allclose(
                  rater_feedback_score[..., i], rater_feedback_score[..., i - 1]
              )

          # (TEST3) we can see that the rater feedback score is symmetric
          # w.r.t. vertical and horizontal centerlines.
          for i in range(rater_feedback_score.shape[-1]):
            matrix = rater_feedback_score[..., i]
            assert np.allclose(matrix, np.flipud(matrix))
            assert np.allclose(matrix, np.fliplr(matrix))

          # (TEST4) we can see that the score decays as expected.
          for i in range(rater_feedback_score.shape[-1]):
            matrix = rater_feedback_score[..., i]
            assert np.allclose(
                matrix[:, 0],
                max(min_nontrust_score, decay_factor),
            )
            assert np.allclose(
                matrix[:, 8],
                max(min_nontrust_score, decay_factor),
            )
            assert np.allclose(
                matrix[0, :],
                max(min_nontrust_score, decay_factor),
            )
            assert np.allclose(
                matrix[8, :],
                max(min_nontrust_score, decay_factor),
            )

            assert np.allclose(
                matrix[1:-1, 1],
                max(min_nontrust_score, np.sqrt(decay_factor)),
            )
            assert np.allclose(
                matrix[1:-1, 7],
                max(min_nontrust_score, np.sqrt(decay_factor)),
            )
            assert np.allclose(
                matrix[1, 1:-1],
                max(min_nontrust_score, np.sqrt(decay_factor)),
            )
            assert np.allclose(
                matrix[7, 1:-1],
                max(min_nontrust_score, np.sqrt(decay_factor)),
            )


if __name__ == "__main__":
  absltest.main()
