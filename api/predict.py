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

from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
import os

# Set up parser for input data <http://flask-restx.readthedocs.io/en/stable/parsing.html>
input_parser = MAX_API.parser()
# Example parser for file input
input_parser.add_argument('file', type=FileStorage, location='files', required=True)


# Creating a JSON response model: https://flask-restx.readthedocs.io/en/stable/marshalling.html#the-api-model-factory
label_prediction = MAX_API.model('LabelPrediction', {
    'language': fields.String(required=True, description='Predicted Language'),
    'probability': fields.Float(required=True)
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction), description='Predicted language and probability')
})


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        args['file'].save(args['file'].filename)
        preds = self.model_wrapper._predict(os.path.abspath(args['file'].filename))

        # Modify this code if the schema is changed
        label_preds = [{'language': p[0], 'probability': p[1]} for p in preds]
        result['predictions'] = label_preds
        result['status'] = 'ok'

        return result
