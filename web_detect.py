#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Demonstrates web detection using the Google Cloud Vision API.

Example usage:
  python web_detect.py https://goo.gl/X4qcB6
  python web_detect.py ../detect/resources/landmark.jpg
  python web_detect.py gs://your-bucket/image.png
"""
# [START full_tutorial]
# [START imports]
import numpy as np
import io

import time
from google.cloud import vision
from google.cloud.vision.feature import Feature
from google.cloud.vision.feature import FeatureTypes

# [END imports]

#CONSTANTS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

project_name = 'fakenewsll-169502'
limit = 10

def annotate(path):
    """Returns web annotations given the path to an image."""
    # [START get_annotations]
    image = None
    vision_client = vision.Client(project_name)

    if path.startswith('http') or path.startswith('gs:'):
        image = vision_client.image(source_uri=path)

    else:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision_client.image(content=content)

    features = []

    features.append(Feature(feature_type=FeatureTypes.WEB_DETECTION,max_results=100000))
    features.append(Feature(feature_type=FeatureTypes.FACE_DETECTION,max_results=100000))
    features.append(Feature(feature_type=FeatureTypes.LANDMARK_DETECTION,max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.IMAGE_PROPERTIES,max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.TEXT_DETECTION, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.LABEL_DETECTION, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.SAFE_SEARCH_DETECTION, max_results=limit))

    results = image.detect(features)


    return results[0]
    # [END get_annotations]


def report(annotations):
    """Prints detected features in the provided web annotations."""
    # [START print_annotations]
    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images retrieved')

        for page in annotations.pages_with_matching_images:
            print('Score : {}'.format(page.score))
            print('Url   : {}'.format(page.url))

    if annotations.full_matching_images:
        print ('\n{} Full Matches found: '.format(
               len(annotations.full_matching_images)))

        for image in annotations.full_matching_images:
            print('Score:  {}'.format(image.score))
            print('Url  : {}'.format(image.url))

    if annotations.partial_matching_images:
        print ('\n{} Partial Matches found: '.format(
               len(annotations.partial_matching_images)))

        for image in annotations.partial_matching_images:
            print('Score: {}'.format(image.score))
            print('Url  : {}'.format(image.url))

    if annotations.web_entities:
        print ('\n{} Web entities found: '.format(
            len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description))
    # [END print_annotations]


def convert_likelihood_to_int(likelihood):
    from google.cloud.vision.likelihood import Likelihood
    if likelihood == Likelihood.VERY_UNLIKELY:
        return 1
    elif likelihood == Likelihood.UNLIKELY:
        return 2
    elif likelihood == Likelihood.POSSIBLE:
        return 3
    elif likelihood == Likelihood.LIKELY:
        return 4
    elif likelihood == Likelihood.VERY_LIKELY:
        return 5
    else:
        return 0
    pass


def extract_features(annotations):
    featurevector = []
    # Web feats
    web_feats = []
    web_feats.append(len(annotations.web.full_matching_images))
    web_feats.append(len(annotations.web.pages_with_matching_images))
    web_feats.append(len(annotations.web.partial_matching_images))
    web_feats.append(len(annotations.web.web_entities))
    for entity in annotations.web.web_entities:
        #web_feats.append(get_word_vector(entity.description))
        pass
    # Safesearch
    safe_search_feats = []
    safe_search_feats.append(convert_likelihood_to_int(annotations.safe_searches.adult))
    safe_search_feats.append(convert_likelihood_to_int(annotations.safe_searches.medical))
    safe_search_feats.append(convert_likelihood_to_int(annotations.safe_searches.spoof))
    safe_search_feats.append(convert_likelihood_to_int(annotations.safe_searches.violence))
    # properties
    properties_feats = []
    for colorindex in range(0,5):
        color = annotations.properties.colors[colorindex].color
        properties_feats.append(color.alpha)
        properties_feats.append(color.blue)
        properties_feats.append(color.green)
        properties_feats.append(color.red)
    # logos
    logos_feats = []
    logos_feats.append(len(annotations.logos))
    # landmark
    landmark_feats = []
    landmark_feats.append(len(annotations.landmarks))
    # labels
    #TODO
    # faces
    faces_feats = []
    faces_feats.append(len(annotations.faces))

    featurevector = web_feats + safe_search_feats + \
                    properties_feats + logos_feats + landmark_feats + \
                    faces_feats
    return featurevector



if __name__ == '__main__':
    # [START run_web]
    #Initialize web environment


    # Get a list of images/paths to images
    images = ['https://assets.merriam-webster.com/mw/images/article/art-wap-article-main/disinformation-3378-30b12acfed3c4540ab101702aaf23744@1x.jpg',
              'https://www.petdrugsonline.co.uk/images/page-headers/cats-master-header',
              './resources/cat.jpg']
    # Feature list
    imgfeats = np.array([])

    # Iterate through images and get the annotations and the features
    for image in images:
        annotations = annotate(image)
        featurevector = extract_features(annotations)
        if len(imgfeats) == 0:
            imgfeats = featurevector
        else:
            imgfeats = np.vstack((imgfeats,featurevector))
    # Classify whether true or not
    imgfeats = imgfeats.transpose()
    X = imgfeats[0:1]
    Y = [1, 0, 0]
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y[0:1])
    print classif.predict(imgfeats[2])

    # [END run_web]
# [END full_tutorial]
