# DLCF-DCA-Mindspore
> Here is the implementation code for DLCF-DCA, based on MindSpore.
>
> paper: Combining dynamic local context focus and dependency cluster attention for aspect-level sentiment classification
>
> Original author code: https://github.com/XuMayi/DLCF-DCA

## Requirement

* mindspore
* numpy
* sklearn
* python 3.8
* mindnlp
* transformers
* mindformers

## Usage

### Training

```sh
python main.py
```

* There are adjustable hyperparameters in the main.py
* DLCF-DCA_x2ms is an implementation based on mindformers and DLCF-DCA_x2ms2 is an implementation based on mindnlp

### Tips

There are two ways we implement DLCF-DCA, bert based on mindnlp and bert based on mindstudio, and both use MindStudio to help us move models, but in both versions, We found that the reasoning speed of the trainer migrated with mindstudio was very slow, so we rewrote the trainer in mindformer version, and the speed was greatly improved. For details, please see apc_trainer.py, we also wrote a method for saving the model and reading the best model. We will provide more detailed usage methods in the future.

## Model implemented in DLCF-DCA-x2ms

### BERT for Sentence Pair Classification ([bert_spc.py](./models/bert_spc.py))

BERT paper



## Licence

MIT
