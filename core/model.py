#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
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
#

from maxfw.model import MAXModelWrapper
import tensorflow as tf
import numpy as np
import core.utils as utils
import logging
from config import DEFAULT_MODEL_PATH

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = {
        'id': 'None',
        'name': 'CodeNet Language Classification',
        'description': 'Simple convolutional deep neural network to classify snippets of code',
        'languages': str(utils.langs),
        'type': 'TensorFlow',
        'source': 'IBM',
        'license': 'Apache 2.0'
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        # Load the graph
        self.model = tf.keras.models.load_model(path)

        # Set up instance variables and required inputs for inference
        logger.info('Loaded model')

    def _pre_process(self, inp):
        return utils.turn_file_to_vectors(inp)

    def _post_process(self, preds):
        return [utils.langs[np.argmax(preds)], np.max(preds)]

    def _predict(self, file_path):
        vectors = self._pre_process(file_path)
        preds = self.model.predict(np.array(vectors))
        return self._post_process(preds)
