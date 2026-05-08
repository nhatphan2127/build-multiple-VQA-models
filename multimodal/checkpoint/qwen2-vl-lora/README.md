---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2-VL-2B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2-VL-2B-Instruct
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: finetuned-gwen
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# finetuned-gwen

This model is a fine-tuned version of [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8494

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.PAGED_ADAMW with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.3213        | 0.3628 | 100  | 1.2525          |
| 1.2177        | 0.7256 | 200  | 1.0929          |
| 0.9299        | 1.0871 | 300  | 1.0242          |
| 0.8772        | 1.4499 | 400  | 0.9701          |
| 0.8868        | 1.8127 | 500  | 0.9160          |
| 0.6841        | 2.1741 | 600  | 0.8915          |
| 0.6179        | 2.5370 | 700  | 0.8670          |
| 0.6057        | 2.8998 | 800  | 0.8518          |
| 0.6832        | 3.0    | 828  | 0.8494          |


### Framework versions

- PEFT 0.19.1
- Transformers 5.8.0
- Pytorch 2.11.0+cu130
- Datasets 4.8.5
- Tokenizers 0.22.2