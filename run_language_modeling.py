# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tune LambdaBert using MLM + NSP or MLM + SOP loss
Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import ConcatDataset

from datasets import load_dataset

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForNextSentencePrediction,
    DataCollatorForSOP,
    BertTokenizerFast,
    set_seed,
)
import wandb

from custom_datasets import SOPTextDataset, NSPTextDataset
from configuration_lambdabert import LambdaBertConfig
from modeling_lambdabert import LambdaBertForPreTrainingSOP, LambdaBertForPreTrainingNSP


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    sop: bool = field(
        default=False,
        metadata={"help": "Whether SOP should be performed instead of MLM + NSP"},
    )


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    wandb.login()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(42)

    data_path = "data/wikitext2"
    train_path, eval_path = f"{data_path}train.txt", f"{data_path}eval.txt"

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train, eval = dataset["train"], dataset["validation"]
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        with open(train_path, "w", encoding='utf-8') as f:
            f.write("\n".join([x["text"] for x in train]))
        with open(eval_path, "w", encoding='utf-8') as f:
            f.write("\n".join([x["text"] for x in eval]))
        logger.info(f"Dataset files were saved to {data_path}")

    config = LambdaBertConfig()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    if model_args.sop:
        model = LambdaBertForPreTrainingSOP(config)
        train_dataset = SOPTextDataset(tokenizer, file_path=train_path, block_size=tokenizer.max_len)
        eval_dataset = SOPTextDataset(tokenizer, file_path=eval_path, block_size=tokenizer.max_len)
        data_collator = DataCollatorForSOP(tokenizer=tokenizer)
        logger.info("SOP + MLM pretraining")
    else:
        model = LambdaBertForPreTrainingNSP(config)
        train_dataset = NSPTextDataset(tokenizer, file_path=train_path, block_size=tokenizer.max_len)
        eval_dataset = NSPTextDataset(tokenizer, file_path=eval_path, block_size=tokenizer.max_len)
        data_collator = DataCollatorForNextSentencePrediction(tokenizer=tokenizer)
        logger.info("NSP + MLM pretraining")

    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()
