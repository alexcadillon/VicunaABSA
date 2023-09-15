# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import xml.etree.ElementTree as ET
import os

import datasets
import datasets.features.features
from datasets import ClassLabel


_CITATION = """\
@inproceedings{pontiki-etal-2015-semeval,
    title = "{S}em{E}val-2015 Task 12: Aspect Based Sentiment Analysis",
    author = "Pontiki, Maria  and
      Galanis, Dimitris  and
      Papageorgiou, Haris  and
      Manandhar, Suresh  and
      Androutsopoulos, Ion",
    booktitle = "Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)",
    month = jun,
    year = "2015",
    address = "Denver, Colorado",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S15-2082",
    doi = "10.18653/v1/S15-2082",
    pages = "486--495",
}
"""

_DESCRIPTION = """\
These are the datasets for Aspect Based Sentiment Analysis (ABSA), Task 12 of SemEval-2015.
"""

_HOMEPAGE = "https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "restaurants": {"trial": "absa-2015_restaurants_trial.xml",
                    "train": "ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml",
                    "test": "ABSA15_Restaurants_Test.xml"},
    "laptops": {"trial": "absa-2015_laptops_trial.xml",
                "train": "ABSA15_LaptopsTrain/ABSA-15_Laptops_Train_Data.xml",
                "test": "ABSA15_Laptops_Test.xml"},
    "hotels": {"test": "ABSA15_Hotels_Test.xml"},
}


class SemEval2015Task12(datasets.GeneratorBasedBuilder):
    """These are the datasets for Aspect Based Sentiment Analysis (ABSA), Task 12 of SemEval-2015."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="restaurants", version=VERSION, description="Restaurant reviews"),
        datasets.BuilderConfig(name="laptops", version=VERSION, description="Laptop reviews"),
        datasets.BuilderConfig(name="hotels", version=VERSION, description="Hotel reviews"),
    ]

    # DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        categories = {
            "restaurants": {
                "entities": ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"],
                "attributes": ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"]
            },
            "laptops": {
                "entities": ["LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING", "PORTS",
                             "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISC",
                             "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING", "SUPPORT",
                             "COMPANY"],
                "attributes": ["GENERAL", "PRICE", "QUALITY", "OPERATION_PERFORMANCE", "USABILITY", "DESIGN_FEATURES",
                               "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"]
            },
            "hotels": {
                "entities": ["HOTEL", "ROOMS", "FACILITIES", "ROOMS_AMENITIES", "SERVICE", "LOCATION", "FOOD_DRINKS"],
                "attributes": ["GENERAL", "PRICES", "COMFORT", "CLEANLINESS", "QUALITY", "DESIGN_FEATURES",
                               "STYLE_OPTIONS", "MISCELLANEOUS"]
            },
        }
        polarities = ["positive", "negative", "neutral"]
        if self.config.name == "restaurants":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "reviewId": datasets.Value(dtype="string"),
                    "sentences": [
                        {
                            "sentenceId": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "opinions": [
                                {
                                    "target": datasets.Value("string"),
                                    "category": {
                                        "entity": datasets.Value("string"),
                                        "attribute": datasets.Value("string")
                                    },
                                    "polarity": datasets.Value("string"),
                                    "from": datasets.Value("string"),
                                    "to": datasets.Value("string"),
                                }
                            ]
                        }
                    ]
                }
            )
        elif self.config.name == "laptops":
            features = datasets.Features(
                {
                    "reviewId": datasets.Value(dtype="string"),
                    "sentences": [
                        {
                            "sentenceId": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "opinions": [
                                {
                                    "category": {
                                        "entity": datasets.Value("string"),
                                        "attribute": datasets.Value("string")
                                    },
                                    "polarity": datasets.Value("string"),
                                }
                            ]
                        }
                    ]
                }
            )
        elif self.config.name == "hotels":
            features = datasets.Features(
                {
                    "reviewId": datasets.Value(dtype="string"),
                    "sentences": [
                        {
                            "sentenceId": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "opinions": [
                                {
                                    "target": datasets.Value("string"),
                                    "category": {
                                        "entity": datasets.Value("string"),
                                        "attribute": datasets.Value("string")
                                    },
                                    "polarity": datasets.Value("string"),
                                    "from": datasets.Value("string"),
                                    "to": datasets.Value("string"),
                                }
                            ]
                        }
                    ]
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        if self.config.name in ["restaurants", "laptops"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split("trial"),
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": data_dir['trial'],
                        "split": "trial",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": data_dir['train'],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": data_dir['test'],
                        "split": "test"
                    },
                ),
            ]
        elif self.config.name == "hotels":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": data_dir['test'],
                        "split": "test"
                    },
                ),
            ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `id_` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        tree = ET.parse(filepath)
        root = tree.getroot()
        for id_, review in enumerate(root.iter("Review")):
            reviewId = review.attrib.get("rid")
            sentences = []
            for sentence in review.iter("sentence"):
                sentence_dict = {}
                sentence_dict["sentenceId"] = sentence.get("id")
                sentence_dict["text"] = sentence.find("text").text
                opinions = []
                for opinion in sentence.iter("Opinion"):
                    opinion_dict = opinion.attrib
                    opinion_dict["category"] = dict(zip(["entity", "attribute"], opinion_dict["category"].split("#")))
                    opinions.append(opinion_dict)
                sentence_dict["opinions"] = opinions
                sentences.append(sentence_dict)
            yield id_, {
                "reviewId": reviewId,
                "sentences": sentences
            }
