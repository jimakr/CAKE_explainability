import numpy as np
from torch import tensor
from torch.utils.data import Dataset as TDataset
from transformers import Trainer, TrainingArguments


class MyDataset(TDataset):
    """MyDataset class is used to transform an instance (input sequence) to be appropriate for use in transformers
    """

    def __init__(self, encodings, labels, tokenizer):
        self.encodings = tokenizer(
            list(encodings), truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ModelWrap:
    """MyModel class loads the transformer model, and setups the myPredict function to later use in the explanation techniques"""

    def __init__(self, trainer, tokenizer, task, labels):
        """Init function
        Args:
            path: The path of the folder with the trained models
            dataset_name: The name of the dataset to be used in conjunction to the path to find and load the model
            model_name: The transformer's name (currently only 'bert' and 'distilbert' are available)
            task: 'single_label' or 'multi_label' task
            labels: The number of the labels from our dataset (integer)
            cased: Boolean for cased (True) or uncased (False)
        Attributes:
            trainer: The huggingface trainer module (which includes the loaded model) -> initiated through the __load_model__ function
            tokenizer: The model specific tokenizer -> initiated through the __load_model__ function
            key_list: The list of the key-related linear layers -> initiated through the __get_additional_info_from_trainer__ function
            query_list: The list of the query-related linear layers -> initiated through the __get_additional_info_from_trainer__ function
            layers: The number of layers
            heads: The number of heads
            embedding_size: The size of the embedding
            ehe: The size of the embedding per head
        """
        self.task = task
        self.labels = labels
        self.tokenizer = tokenizer
        self.trainer = trainer

    def single_predict(self, instance):
        """This function allows the prediction for a single instance (a single input sequence).
        It returns the prediction, the attention matrices, and the hidden states
        """
        # We double the instance, because the trainer module needs a "dataset"
        instance = [instance, instance]
        # We also add dummy labels, which are not used, but needed by the module
        if self.task == 'single_label':
            instance_labels = [0, 0]
        else:
            instance_labels = [[0] * self.labels, [0] * self.labels]
        instance_dataset = MyDataset(
            instance, instance_labels, self.tokenizer)
        outputs = self.trainer.predict(instance_dataset)
        predictions = outputs.predictions[0]
        return predictions
