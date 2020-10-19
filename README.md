# LambdaBERT

### A 🤗transformers-style implementation of BERT using LambdaNetworks instead of self-attention

The `LambdaLayer` implementation was adapted from [lucidrains' implementation](https://github.com/lucidrains/lambda-networks) to work with 1D sequences, following the directives taken from the original paper (currently under review). The 🤗`transformer` architecture is used to minimize reimplementation.

*__Motivation:__ Linear lambda functions use the key-value product __λc__ as a learned projection matrix that is input-independent (i.e. a subset of learned latent dimensions over training examples), as opposed to the token-dependent nature of self-attention. From an interpretability POV, pre-trained __λc__ parameters may then be a more interesting subject to probe for linguistic structures than attention weights.*

### Content

Run `./setup.sh` to install all dependencies in a local Python3 environment.


- `configuration_lambdabert.py`, `modeling_lambdabert.py`: 🤗`transformer`-compliant files containing the model implementation, which is a light adaptation of the standard `BertModel` class for supporting lambda computations.

- `custom_datasets.py`: Reimplementation of NSP and SOP datasets from 🤗`transformer` to work with the demo language modeling script (`wikitext2`)

- `run_language_modeling.py`: Stripped down version from 🤗`transformer` examples, allows to pretrain `LambdaBert` on WikiText2 using the 🤗`datasets` library.

Run pre-training as:

```shell
# Customize with HuggingFace training args
python run_language_modeling.py \
  --output_dir="models" \
  --do_train \
  --do_eval \
  --eval_steps=100 \
  --sop # Omit for MLM + NSP instead of MLM + SOP
```

*TODO: Language modeling benchmarks on WikiText2, WikiText103*

### Example usage

```python
from configuration_lambdabert import LambdaBertConfig
from modeling_lambdabert import LambdaBertModel
from transformers import BertTokenizer
from datasets import load_dataset

config = LambdaBertConfig(local_context_size=25, output_lambda_params=True)

# Instantiate the untrained model
# model = LambdaBertModel.from_pretrained()
model = LambdaBertModel(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
inputs = tokenizer("Is attention really all you need?", return_tensors="pt")

# Tuple containing
# 1) Pooled output at last layer [batch_size x 1 x hidden_size]
# 2) Sequence output at last layer [batch_size x seq_len x hidden_size]
# 3) λc parameters across all model's layers
# (e.g. tuple of 12 [batch_size x key_depth x num_lambda_queries] tensors)
out = model(**inputs)
```

### Links

[Yannic Kilcher's video on LambdaNetworks](https://www.youtube.com/watch?v=3qxJ2WD8p4w&t=3081s)

[lucidrain's LambdaNetworks original implementation](https://github.com/lucidrains/lambda-networks)

[Original LambdaNetworks paper under review at ICLR 2021](https://openreview.net/forum?id=xTJEN-ggl1b)

### Citations

```bibtex
@inproceedings{anonymous-2021-lambdanetworks,
    title={LambdaNetworks: Modeling Long-range Interactions without Attention},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=xTJEN-ggl1b},
    note={under review}
}

@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and Chang, Ming-Wei  and  Lee, Kenton  and  Toutanova, Kristina",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1423",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
}

@article{Wolf2019HuggingFacesTS,
    title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
    author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
    journal={ArXiv},
    year={2019},
    volume={abs/1910.03771}
}
```
