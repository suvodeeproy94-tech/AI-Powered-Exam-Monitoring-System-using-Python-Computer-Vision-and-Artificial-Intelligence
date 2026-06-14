"""Tests for honest cross-person gesture-model evaluation."""

import unittest

import numpy as np

from tools.train_gesture_model import split_dataset


class TrainingToolTests(unittest.TestCase):
    """Verify participant groups are kept separate when enough people exist."""

    def test_split_prefers_unseen_participant_evaluation(self):
        features = []
        labels = []
        participants = []
        for participant in ("P01", "P02", "P03", "P04"):
            for label_index, label in enumerate(("Open Palm", "Closed Fist")):
                for sample_index in range(5):
                    features.append(
                        [label_index + sample_index * 0.01 for _ in range(42)]
                    )
                    labels.append(label)
                    participants.append(participant)

        result = split_dataset(
            np.asarray(features),
            np.asarray(labels),
            np.asarray(participants),
        )

        self.assertEqual(result[-1], "participant_group_split")
        self.assertGreater(len(result[0]), 0)
        self.assertGreater(len(result[1]), 0)


if __name__ == "__main__":
    unittest.main()
