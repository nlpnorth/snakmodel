![SnakModel Logo](snakmodel.png)

**SnakModel** is a 7B-parameter, autoregressive language model specifically designed for Danish. There are both an instruction-tuned variant, as well as a base version for further fine-tuning. Our models build upon [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf), which we continuously pre-train on a diverse collection of Danish corpora comprising 350M documents and 13.6B words, before tuning it on 3.7M Danish instruction-answer pairs.

**Developers**

[**🧭 NLPnorth** research unit](https://nlpnorth.github.io) at the [IT University of Copenhagen](https://itu.dk), Denmark.  
[**🌊 AAU-NLP** research unit](https://aaunlp.github.io) at [Aalborg University Copenhagen](https://aau.dk), Denmark.

[Mike Zhang](https://jjzha.github.io)\*, [Max Müller-Eberstein](https://mxij.me)\*, [Elisa Bassignana](http://elisabassignana.github.io), [Rob van der Goot](https://robvanderg.github.io).  
\*equal contribution.

## Resources

* 💬 SnakModeller:
  * **[SnakModel-7B (base)](https://huggingface.co/NLPnorth/snakmodel-7b-base)**: The base LM trained on Danish text completion + its intermediate checkpoints.
  * **[SnakModel-7B (instruct)](https://huggingface.co/NLPnorth/snakmodel-7b-instruct)**: An instruction-tuned variant of the base model + its intermediate checkpoints.
* ⚙️ Model Training and Analysis Code:
  * **Research Paper**: To appear in the Proceedings of NoDaLiDa/Baltic-HLT 2025 (pre-print coming soon).
  * **Codebase**: this repository.
* 🇩🇰 Cultural Awareness Evaluation:
  * **Research Paper**: coming in Q1 2025 (pre-print coming soon).
  * **Codebase**: coming soon to this repository.
  * **Web-based LLM Evaluation Interface**: coming soon.


## Codebase

This codebase contains code to replicate all experiments from the research papers related to SnakModel. This includes:

* `analyze/`:
  * [`divergence/`](analyze/divergence/README.md): contains scripts to analyze the divergence of embedding and model weights before and after adaptation.
  * [`leakage/`](analyze/leakage/README.md): contains the scripts we used to estimate train/test data leakage.
  * `prompt.py`: is an interactive script, which allows for prompting model checkpoints locally.
* `evaluate/`: contains a script to run ScandEval benchmarking on all relevant checkpoints.
* [`finetune/`](finetune/README.md): contains a script to run instruction fine-tuning using LoRA.
* `plot/`: contains scripts to replicate plots from the SnakModel research papers.
* [`pretrain/`](pretrain/README.md): contains our multi-GPU pre-training scripts adapted from Megatron-LLM. Also include data collection and pre-processing.

### Installation

For pre-training, please refer to the instructions in the [`pretrain/`](pretrain/README.md) folder. We recommend running pre-training in a separate installation in Docker.

For instruction-tuning, and subsequent analyses, simply install the required packages (ideally, in a virtual environment):
```bash
(venv) $ pip install -r requirements.txt
```

## Citation

If you find the work in this repository useful, please don't forget to cite:

```bibtex
@inproceedings{snakmodel,
  title={{S}nak{M}odel: Lessons Learned from Training an Open Danish Large Language Model},
  author={Mike Zhang and Max M{\"u}ller-Eberstein and Elisa Bassignana and Rob van der Goot},
  booktitle={The Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies},
  year={2024},
  url={https://openreview.net/forum?id=YxzfgQGpRQ}
}
```
