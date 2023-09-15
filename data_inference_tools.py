"""
Tools for running inference on datasets.
"""
from datasets import load_dataset
from huggingface_hub import AsyncInferenceClient
import asyncio
import json
import jsonschema
from tqdm.asyncio import tqdm_asyncio
import logging
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

# Set up logging
logging.basicConfig(filename='data_inference.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MaxTriesReached(Exception):
    """Exception raised when the maximum number of tries is reached without a valid output."""
    def __init__(self, max_tries):
        self.max_tries = max_tries

    def __str__(self):
        return "Maximum tries of {max_tries} reached without a valid output.".format(max_tries=self.max_tries)


class DatasetInferenceHandler:
    """Inference handler parent class."""

    @staticmethod
    def micro_avg_precision_recall_f1(predicted, ground_truth):
        """Compute micro-averaged precision, recall, and F1 score."""
        tp = 0
        fp = 0
        fn = 0

        for p, gt in zip(predicted, ground_truth):
            tp += len(set(p) & set(gt))
            fp += len(set(p) - set(gt))
            fn += len(set(gt) - set(p))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1

    @staticmethod
    def accuracy(predicted, ground_truth):
        """Compute accuracy."""
        correct = 0
        total = 0
        for p, gt in zip(predicted, ground_truth):
            correct += sum([p_i == gt_i for p_i, gt_i in zip(p, gt)])
            total += len(gt)
        return correct / total if total != 0 else 0

    pass


class SemEval2014Task4InferenceHandler(DatasetInferenceHandler):
    """Inference handler for SemEval2014Task4 dataset."""

    def __init__(self, endpoint_url, hf_token, gen_kwargs, prompts_file_path="prompt_structures.json", validation_file_path="validation_schemas.json"):
        # super().__init__(endpoint_url, hf_token, gen_kwargs, prompts_file_path)
        self.client = AsyncInferenceClient(endpoint_url, token=hf_token)
        self.gen_kwargs = gen_kwargs
        with open(prompts_file_path, 'r') as f:
            self.prompt_structures = json.load(f)["SemEval2014Task4"]
        with open(validation_file_path, 'r') as f:
            self.validation_schemas = json.load(f)["SemEval2014Task4"]
        self.dataset_name = "SemEval2014Task4"
        self.dataset_script_path = "datasets/SemEval2014Task4/SemEval2014Task4.py"

        logging.info("Initialized %s with endpoint: %s", self.__class__.__name__, endpoint_url)

    def generate_prompt(self, domain, review, task, examples_dataset, n_examples=5, vicuna_wrapper=False):
        """Generate a prompt for a given review and task. The prompt contains random examples from the TRIAL dataset."""
        # Task description
        prompt_description = self.prompt_structures[domain][task]["description"]
        # Examples
        prompt_examples_list = []
        for example in examples_dataset.sample(n_examples):
            text = example["text"]
            output_json = []
            if task in ["TSD", "TD"]:
                for aspect_term in example["aspectTerms"]:
                    target = aspect_term["term"]
                    if task == "TSD":
                        sentiment = aspect_term["polarity"]
                        output_json.append({"target": target, "sentiment": sentiment})
                    else:
                        output_json.append({"target": target})
            elif task in ["ASD", "AD"]:
                for aspect_category in example["aspectCategories"]:
                    aspect = aspect_category["category"]
                    if task == "ASD":
                        sentiment = aspect_category["polarity"]
                        output_json.append({"aspect": aspect, "sentiment": sentiment})
                    else:
                        output_json.append({"aspect": aspect})
            prompt_examples_list.append(self.prompt_structures[domain][task]["example_structure"].format(text=text, output=json.dumps(output_json)))
        prompt_examples = "\n\n".join("Example {i}:\n\n{example}".format(i=i+1, example=example) for i, example in enumerate(prompt_examples_list))
        # Input
        prompt_input = self.prompt_structures[domain][task]["input_structure"].format(text=review["text"])
        # Put prompt together
        prompt = "{description}\n\n{examples}\n\nYour turn:\n\n{input}\nOutput:".format(description=prompt_description, examples=prompt_examples, input=prompt_input)
        # Add Vicuna wrapper
        if vicuna_wrapper:
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n\n{prompt}\n\nASSISTANT:".format(prompt=prompt)
        return prompt

    async def generate_output(self, prompt):
        """Generate output for a given prompt using the Hugging Face Inference API."""
        return await self.client.text_generation(prompt, stream=False, details=False, **self.gen_kwargs)

    def validate_output(self, output, domain, task):
        """Validate the output of the model using a JSON schema."""
        try:
            parsed_output = json.loads(output.replace("\\", ""))
            jsonschema.validate(parsed_output, self.validation_schemas[domain][task])
            return parsed_output
        except Exception as e:
            raise e

    async def inference_pipeline(self, domain, review, task, examples_dataset, max_tries=1, n_examples=5, vicuna_wrapper=False):
        """Run the inference pipeline for a given review and task. Inference is retried if the output is not valid."""
        prompt = self.generate_prompt(domain, review, task, examples_dataset, n_examples=n_examples, vicuna_wrapper=vicuna_wrapper)
        tries = 0
        while tries < max_tries:
            try:
                output = await self.generate_output(prompt)
                parsed_output = self.validate_output(output, domain, task)
                return parsed_output
            except Exception as e:
                tries += 1
        raise MaxTriesReached(max_tries)

    def compute_metrics(self, results, domain, task):
        """Compute metrics for a given task."""
        tasks_attributes = {
            "TD": ["target"],
            "AD": ["aspect"],
            "TSD": ["target", "sentiment"],
            "ASD": ["aspect", "sentiment"],
        }
        try:
            attributes = tasks_attributes[task]
        except KeyError:
            raise NotImplementedError("Task {task} not implemented.".format(task=task))
        try:
            predicted = [[json.dumps({k: opinion[k] for k in attributes}) for opinion in result["predicted_output"]] for result in results]
            ground_truth = [[json.dumps({k: opinion[k] for k in attributes}) for opinion in result["gold_output"]] for result in results]
            precision, recall, f1 = self.micro_avg_precision_recall_f1(predicted, ground_truth)
            return {"precision": precision, "recall": recall, "f1": f1}
        except KeyError:
            raise NotImplementedError("Metrics cannot be computed for {dataset} [{domain}] with task: {task}.".format(dataset=self.dataset_name, domain=domain, task=task))


class SemEval2015Task12InferenceHandler(DatasetInferenceHandler):
    """Inference handler for SemEval2015Task12 dataset."""

    def __init__(self, endpoint_url, hf_token, gen_kwargs, prompts_file_path="prompt_structures.json", validation_file_path="validation_schemas.json"):
        # super().__init__(endpoint_url, hf_token, gen_kwargs, prompts_file_path)
        self.client = AsyncInferenceClient(endpoint_url, token=hf_token)
        self.gen_kwargs = gen_kwargs
        with open(prompts_file_path, 'r') as f:
            self.prompt_structures = json.load(f)["SemEval2015Task12"]
        with open(validation_file_path, 'r') as f:
            self.validation_schemas = json.load(f)["SemEval2015Task12"]
        self.dataset_name = "SemEval2015Task12"
        self.dataset_script_path = "datasets/SemEval2015Task12/SemEval2015Task12.py"

        logging.info("Initialized %s with endpoint: %s", self.__class__.__name__, endpoint_url)

    def generate_prompt(self, domain, review, task, examples_dataset, n_examples=5, vicuna_wrapper=False):
        """Generate a prompt for a given review and task. The prompt contains random examples from the TRIAL dataset."""
        # Task description
        prompt_description = self.prompt_structures[domain][task]["description"]
        # Examples
        prompt_examples_list = []
        for example in examples_dataset.sample(n_examples):
            text = example["text"]
            output_json = []
            for opinion in example["opinions"]:
                aspect_entity = opinion["category"]["entity"]
                aspect_attribute = opinion["category"]["attribute"]
                sentiment = opinion["polarity"]
                if task == "TASD":
                    target = opinion["target"]
                    output_json.append({"aspect": {"entity": aspect_entity, "attribute": aspect_attribute}, "target": target, "sentiment": sentiment})
                elif task == "ASD":
                    output_json.append({"aspect": {"entity": aspect_entity, "attribute": aspect_attribute}, "sentiment": sentiment})
            prompt_examples_list.append(self.prompt_structures[domain][task]["example_structure"].format(text=text, output=json.dumps(output_json)))
        prompt_examples = "\n\n".join("Example {i}:\n\n{example}".format(i=i+1, example=example) for i, example in enumerate(prompt_examples_list))
        # Input
        prompt_input = self.prompt_structures[domain][task]["input_structure"].format(text=review["text"])
        # Put prompt together
        prompt = "{description}\n\n{examples}\n\nYour turn:\n\n{input}\nOutput:".format(description=prompt_description, examples=prompt_examples, input=prompt_input)
        # Add Vicuna wrapper
        if vicuna_wrapper:
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n\n{prompt}\n\nASSISTANT:".format(prompt=prompt)
        return prompt

    async def generate_output(self, prompt):
        """Generate output for a given prompt using the Hugging Face Inference API."""
        return await self.client.text_generation(prompt, stream=False, details=False, **self.gen_kwargs)

    def validate_output(self, output, domain, task):
        """Validate the output of the model using a JSON schema."""
        try:
            parsed_output = json.loads(output.replace("\\", ""))
            jsonschema.validate(parsed_output, self.validation_schemas[domain][task])
            return parsed_output
        except Exception as e:
            raise e

    async def inference_pipeline(self, domain, review, task, examples_dataset, max_tries=1, n_examples=5, vicuna_wrapper=False):
        """Run the inference pipeline for a given review and task. Inference is retried if the output is not valid."""
        prompt = self.generate_prompt(domain, review, task, examples_dataset, n_examples=n_examples, vicuna_wrapper=vicuna_wrapper)
        tries = 0
        while tries < max_tries:
            try:
                output = await self.generate_output(prompt)
                parsed_output = self.validate_output(output, domain, task)
                return parsed_output
            except Exception as e:
                tries += 1
        raise MaxTriesReached(max_tries)

    def compute_metrics(self, results, domain, task):
        """Compute metrics for a given task."""
        tasks_attributes = {
            "TD": ["target"],
            "AD": ["aspect"],
            "TAD": ["target", "aspect"],
            "TSD": ["target", "sentiment"],
            "ASD": ["aspect", "sentiment"],
            "TASD": ["aspect", "target", "sentiment"],
        }
        try:
            attributes = tasks_attributes[task]
        except KeyError:
            raise NotImplementedError("Task {task} not implemented.".format(task=task))
        try:
            predicted = [[json.dumps({k: opinion[k] for k in attributes}) for opinion in result["predicted_output"]] for result in results]
            ground_truth = [[json.dumps({k: opinion[k] for k in attributes}) for opinion in result["gold_output"]] for result in results]
            precision, recall, f1 = self.micro_avg_precision_recall_f1(predicted, ground_truth)
            return {"precision": precision, "recall": recall, "f1": f1}
        except KeyError:
            raise NotImplementedError("Metrics cannot be computed for {dataset} [{domain}] with task: {task}.".format(dataset=self.dataset_name, domain=domain, task=task))


class SemEval2016Task5InferenceHandler(SemEval2015Task12InferenceHandler):
    """Inference handler for SemEval2016Task5 dataset."""

    def __init__(self, endpoint_url, hf_token, gen_kwargs, prompts_file_path="prompt_structures.json", validation_file_path="validation_schemas.json"):
        super().__init__(endpoint_url, hf_token, gen_kwargs, prompts_file_path, validation_file_path)
        self.dataset_name = "SemEval2016Task5"
        self.dataset_script_path = "datasets/SemEval2016Task5/SemEval2016Task5.py"



class DatasetAnalyzer:
    """Dataset analyzer class"""

    def __init__(self, inference_handler, split="test", max_concurrent_tasks=100):
        self.inference_handler = inference_handler
        self.loading_script_path = self.inference_handler.dataset_script_path
        self.split = split
        self.max_concurrent_tasks = max_concurrent_tasks

    async def inference(self, domain, task, max_tries=1, n_examples=5, vicuna_wrapper=False, use_tqdm=True):
        """Run inference on a dataset for a given domain and task."""
        # Load dataset
        dataset = load_dataset(self.loading_script_path, name=domain, split=self.split)
        # Load flattened examples dataset
        if self.inference_handler.dataset_name == "SemEval2014Task4":
            examples_dataset = load_dataset(self.loading_script_path, name=domain, split="trial").with_format("pandas")[:].apply(lambda sentence: sentence.to_dict(), axis=1)
        else:
            examples_dataset = load_dataset(self.loading_script_path, name=domain, split="trial").with_format("pandas").map(lambda reviews: {"sentences": [sentence for review in reviews["sentences"] for sentence in review]}, batched=True)["sentences"]
        # Run inference
        # Use asyncio.gather to run inference concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        results = []

        async def inference_helper(sentence):
            """Helper function to run inference on a single review and manage the semaphore."""
            async with semaphore:
                try:
                    output = await asyncio.wait_for(
                        self.inference_handler.inference_pipeline(domain, sentence, task, examples_dataset,
                                                                  max_tries=max_tries, n_examples=n_examples,
                                                                  vicuna_wrapper=vicuna_wrapper),
                        timeout=120
                    )# format the gold output according to the task
                    if self.inference_handler.dataset_name == "SemEval2014Task4":
                        if task in ["TD", "TSD"]:
                            gold_output = [{"target": target["term"], "sentiment": target["polarity"]} for target in sentence["aspectTerms"]]
                        elif task in ["AD", "ASD"]:
                            gold_output = [{"aspect": aspect["category"], "sentiment": aspect["polarity"]} for aspect in sentence["aspectCategories"]]
                    else:
                        if task == "TASD":
                            gold_output = [{"aspect": opinion["category"], "target": opinion["target"], "sentiment": opinion["polarity"]} for opinion in sentence["opinions"]]
                        elif task == "ASD":
                            gold_output = [{"aspect": opinion["category"], "sentiment": opinion["polarity"]} for opinion in sentence["opinions"]]
                    results.append({
                        "sentenceId": sentence["sentenceId"],
                        "text": sentence["text"],
                        "predicted_output": output,
                        "gold_output": gold_output,
                    })
                except Exception as e:
                    logging.error("Error processing review %s: %s", sentence["sentenceId"], e)
        if use_tqdm:
            if self.inference_handler.dataset_name == "SemEval2014Task4":
                await tqdm_asyncio.gather(*(inference_helper(sentence) for sentence in dataset), desc="Processing reviews", unit="review")
            else:
                await tqdm_asyncio.gather(*(inference_helper(sentence) for review in dataset for sentence in review["sentences"]), desc="Processing reviews", unit="review")
        else:
            if self.inference_handler.dataset_name == "SemEval2014Task4":
                await asyncio.gather(*(inference_helper(sentence) for sentence in dataset))
            else:
                await asyncio.gather(*(inference_helper(sentence) for review in dataset for sentence in review["sentences"]))

        logging.info("Inference completed for %s [%s] with task: %s", self.inference_handler.dataset_name, domain, task)

        return results

    def compute_metrics(self, results, domain, task):
        """Compute metrics for a given task."""
        return self.inference_handler.compute_metrics(results, domain, task)


# Function to generate a handler from dataset name
def get_handler(dataset_name, endpoint_url, hf_token, gen_kwargs):
    if dataset_name == "SemEval2014Task4":
        return SemEval2014Task4InferenceHandler(endpoint_url, hf_token, gen_kwargs)
    elif dataset_name == "SemEval2015Task12":
        return SemEval2015Task12InferenceHandler(endpoint_url, hf_token, gen_kwargs)
    elif dataset_name == "SemEval2016Task5":
        return SemEval2016Task5InferenceHandler(endpoint_url, hf_token, gen_kwargs)
    else:
        raise NotImplementedError("Dataset {dataset} not implemented.".format(dataset=dataset_name))