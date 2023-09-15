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
@inproceedings{pontiki-etal-2016-semeval,
    title = "{S}em{E}val-2016 Task 5: Aspect Based Sentiment Analysis",
	author = {Pontiki, Maria and Galanis, Dimitris and Papageorgiou, Haris and Androutsopoulos, Ion and Manandhar, Suresh and AL-Smadi, Mohammad and Al-Ayyoub, Mahmoud and Zhao, Yanyan and Qin, Bing and De Clercq, Orphée and Hoste, Véronique and Apidianaki, Marianna and Tannier, Xavier and Loukachevitch, Natalia and Kotelnikov, Evgeniy and Bel, Nuria and Jiménez-Zafra, Salud María and Eryiğit, Gülşen},
    booktitle = "Proceedings of the 10th International Workshop on Semantic Evaluation ({S}em{E}val-2016)",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S16-1002",
    doi = "10.18653/v1/S16-1002",
    pages = "19--30",
}
"""

_DESCRIPTION = """\
These are the datasets for Aspect Based Sentiment Analysis (ABSA), Task 5 of SemEval-2016.
"""

_HOMEPAGE = "https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "restaurants": {
        "trial": {
            "SB1": "restaurants_trial_english_sl.xml",
            "SB2": "restaurants_trial_english_tl.xml"
        },
        "train": {
            "SB1": "ABSA16_Restaurants_Train_SB1_v2.xml",
            "SB2": "ABSA16_Restaurants_Train_English_SB2.xml"
        },
        "test": {
            "SB1": "EN_REST_SB1_TEST.xml.gold",
            "SB2": "EN_REST_SB2_TEST.xml.gold"
        }
    },
    "laptops": {
        "trial": {
            "SB1": "laptops_trial_english_sl.xml",
            "SB2": "laptops_trial_english_tl.xml"
        },
        "train": {
            "SB1": "ABSA16_Laptops_Train_SB1_v2.xml",
            "SB2": "ABSA16_Laptops_Train_English_SB2.xml"
        },
        "test": {
            "SB1": "EN_LAPT_SB1_TEST_.xml.gold",
            "SB2": "EN_LAPT_SB2_TEST.xml.gold"
        }
    },
}


class SemEval2016Task5(datasets.GeneratorBasedBuilder):
    """These are the datasets for Aspect Based Sentiment Analysis (ABSA), Task 5 of SemEval-2016."""

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
                "entities": ["LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISC", "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING", "SUPPORT", "COMPANY"],
                "attributes": ["GENERAL", "PRICE", "QUALITY", "OPERATION_PERFORMANCE", "USABILITY", "DESIGN_FEATURES", "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"]
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
                    ],
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
                    ],
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

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `id_` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        tree_SB1 = ET.parse(filepath["SB1"])
        tree_SB2 = ET.parse(filepath["SB2"])
        root_SB1 = tree_SB1.getroot()
        root_SB2 = tree_SB2.getroot()
        for id_, review_SB1 in enumerate(root_SB1.iter("Review")):
            reviewId = review_SB1.attrib.get("rid")
            sentences = []
            for sentence in review_SB1.iter("sentence"):
                sentence_dict = {}
                sentence_dict["sentenceId"] = sentence.get("id")
                sentence_dict["text"] = sentence.find("text").text
                sentence_opinions = []
                for sentence_opinion in sentence.iter("Opinion"):
                    sentence_opinion_dict = sentence_opinion.attrib
                    sentence_opinion_dict["category"] = dict(zip(["entity", "attribute"], sentence_opinion_dict["category"].split("#")))
                    sentence_opinions.append(sentence_opinion_dict)
                sentence_dict["opinions"] = sentence_opinions
                sentences.append(sentence_dict)

            review_opinions = []
            for review_SB2 in root_SB2.iter("Review"):
                if review_SB2.attrib.get("rid") == reviewId:
                    for review_opinion in review_SB2.iter("Opinion"):
                        review_opinion_dict = review_opinion.attrib
                        review_opinion_dict["category"] = dict(zip(["entity", "attribute"], review_opinion_dict["category"].split("#")))
                        review_opinions.append(review_opinion_dict)
                    break

            yield id_, {
                "reviewId": reviewId,
                "sentences": sentences,
                "opinions": review_opinions,
            }
