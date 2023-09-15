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
from datasets import ClassLabel


_CITATION = """\
@inproceedings{pontiki-etal-2014-semeval,
    title = "{S}em{E}val-2014 Task 4: Aspect Based Sentiment Analysis",
    author = "Pontiki, Maria  and
      Galanis, Dimitris  and
      Pavlopoulos, John  and
      Papageorgiou, Harris  and
      Androutsopoulos, Ion  and
      Manandhar, Suresh",
    booktitle = "Proceedings of the 8th International Workshop on Semantic Evaluation ({S}em{E}val 2014)",
    month = aug,
    year = "2014",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S14-2004",
    doi = "10.3115/v1/S14-2004",
    pages = "27--35",
}
"""

_DESCRIPTION = """\
These are the datasets for Aspect Based Sentiment Analysis (ABSA), Task 4 of SemEval-2014.
"""

_HOMEPAGE = "https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "restaurants": {"trial": "restaurants-trial.xml",
                    "train": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml",
                    "test": "ABSA_Gold_TestData/Restaurants_Test_Gold.xml"},
    "laptops": {"trial": "laptops-trial.xml",
                "train": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml",
                "test": "ABSA_Gold_TestData/Laptops_Test_Gold.xml"},
}


class SemEval2014Task4(datasets.GeneratorBasedBuilder):
    """These are the datasets for Aspect Based Sentiment Analysis (ABSA), Task 4 of SemEval-2014."""

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
        datasets.BuilderConfig(name="restaurants", version=VERSION, description="Restaurant review sentences"),
        datasets.BuilderConfig(name="laptops", version=VERSION, description="Laptop review sentences"),
    ]

    # DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "restaurants":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {'sentenceId': datasets.Value(dtype='string'),
                'text': datasets.Value(dtype='string'),
                'aspectTerms': [
                    {'term': datasets.Value(dtype='string'),
                    # 'polarity': ClassLabel(num_classes=4, names=['positive', 'negative', 'neutral', 'conflict']),
                    'polarity': datasets.Value(dtype='string'),
                    'from': datasets.Value(dtype='string'),
                    'to': datasets.Value(dtype='string')}
                ],
                'aspectCategories': [
                    # {'category': ClassLabel(num_classes=5, names=['food', 'service', 'price', 'ambience', 'anecdotes/miscellaneous']),
                    # 'polarity': ClassLabel(num_classes=4, names=['positive', 'negative', 'neutral', 'conflict'])}
                    {'category': datasets.Value(dtype='string'),
                     'polarity': datasets.Value(dtype='string')}
                ],
                # 'domain': ClassLabel(num_classes=2, names=['restaurants', 'laptops'])
                 }
            )
        elif self.config.name == "laptops":
            features = datasets.Features(
                {'sentenceId': datasets.Value(dtype='string'),
                 'text': datasets.Value(dtype='string'),
                 'aspectTerms': [
                     {'term': datasets.Value(dtype='string'),
                      # 'polarity': ClassLabel(num_classes=4, names=['positive', 'negative', 'neutral', 'conflict']),
                      'polarity': datasets.Value(dtype='string'),
                      'from': datasets.Value(dtype='string'),
                      'to': datasets.Value(dtype='string')}
                 ],
                 # 'domain': ClassLabel(num_classes=2, names=['restaurants', 'laptops'])
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
                    "split": "trial"
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
        tree = ET.parse(filepath)
        root = tree.getroot()
        for id_, sentence in enumerate(root.iter("sentence")):
            sentenceId = sentence.attrib.get("id")
            text = sentence.find("text").text
            aspectTerms = []
            for aspectTerm in sentence.iter("aspectTerm"):
                aspectTerms.append(aspectTerm.attrib)
            if self.config.name == "restaurants":
                aspectCategories = []
                for aspectCategory in sentence.iter("aspectCategory"):
                    aspectCategories.append(aspectCategory.attrib)
                yield id_, {
                    "sentenceId": sentenceId,
                    "text": text,
                    "aspectTerms": aspectTerms,
                    "aspectCategories": aspectCategories,
                    # "domain": self.config.name,
                }
            elif self.config.name == 'laptops':
                yield id_, {
                    "sentenceId": sentenceId,
                    "text": text,
                    "aspectTerms": aspectTerms,
                    # "domain": self.config.name,
                }