# Build Your Own GPT From Scratch

Two small language models, written and trained completely from scratch with PyTorch. There are no pretrained weights, no API and no borrowed model. Every pattern the models know came from training on text.

| | Model 1 | Model 2 |
|---|---|---|
| Tokenizer | single characters (109 symbols) | word pieces (BPE, 8,192 tokens) |
| Size | 836,205 parameters | about 30 million parameters |
| Layers / width | 4 layers, 128 wide, 4 heads | 8 layers, 512 wide, 8 heads |
| Context | 128 characters | 256 tokens (about 1,000 characters) |
| Training text | 4 books, about 1.1 million characters | about 200 MB of dialogue, science and history (54 million tokens) |
| Hardware | laptop CPU, about 42 minutes | free Colab T4 GPU, about 16 minutes for 6,000 steps |
| Result | invents word-shaped text in the style of its books | fluent, conversation-shaped English |

The point of the project is to understand how a GPT works from the inside and to see what a hobbyist can really do on a laptop and a free GPU.

> **Honest limits:** these are base models. They *continue* text, they do not *answer* questions. Type `Hi` and Model 2 writes a plausible conversation, not a reply. Turning a base model into a chat partner needs a second training stage (chat fine-tuning, see the roadmap). At 30 million parameters they also say wrong or odd things confidently.

## How it works

1. **Tokenize.** Text becomes numbers. Model 1 gives every character a number. Model 2 uses a byte-pair-encoding tokenizer that learns common word pieces, so `the` is one token and rare words split into a few. That is about 3.9 characters per token, which is why Model 2 learns faster.
2. **The model.** A decoder-only transformer, the same architecture family as ChatGPT and Claude, only far smaller:
   - embeddings turn each token into a vector and add its position;
   - each block has **causal self-attention** (every token looks back at earlier tokens and weighs which matter) and a feed-forward network, both with residual connections and layer norm;
   - a final layer scores every possible next token.
3. **Train.** Take random snippets, predict the next token at every position, measure the error (cross-entropy **loss**), and nudge all weights slightly to reduce it with AdamW. Repeat thousands of times. Nothing about English is programmed in.
4. **Generate.** Feed in some text, get probabilities for the next token, sample one (with temperature and, for Model 2, top-k), append it, and repeat.

Model 2 adds what bigger models use: GELU, weight tying between the input embedding and output layer, flash-style attention (`scaled_dot_product_attention`), mixed precision, warmup plus cosine learning-rate decay, gradient clipping, and a validation set spread over the whole file.

## Results

**Model 1** (character level, CPU, 3,000 steps): loss fell from 4.86 (random guessing) to 1.405 train / 1.406 val. Train and val stayed equal, so it was not memorizing. Output has real word shapes, quotes and dialogue rhythm, but no meaning.

**Model 2** (word pieces, T4 GPU, 6,000 steps): loss fell from 9.1 to about 3.4 train. Its first sample for the prompt `Hi, how are you?`:

```
Hi, how are you?
I'm good. How about you?
I'm good. I just wanted to talk to you for a minute before my birthday party. How was your day?
```

Loss numbers are not comparable between the two models because they use different tokenizers.

## Repository layout

```
requirements.txt
model_1/
  model.py      character-level GPT
  train.py      train or continue training
  chat.py       type a start, the model continues it
  prepare.py    optional: rebuild train.txt from a "books" folder
model_2/
  model.py      BPE-token GPT
  train.py      builds tok.json and tokens.npy if missing, then trains
  chat.py       type a start, the model continues it
  collect.py    downloads and cleans the 200 MB training text
```

Not in the repo (too big or not mine to publish): `train.txt`, `tokens.npy`, the trained `*.pt` weights, and the books. See [Data and licenses](#data-and-licenses).

## Quick start (Windows; Linux and macOS work the same)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Model 1 (CPU is fine)**

1. Put your own text into `model_1/train.txt` (about 1 MB works), or put books into `model_1/books/` and run `python model_1/prepare.py`.
2. `python model_1/train.py` trains (about 40 minutes for 3,000 steps on a laptop). Run it again to keep training.
3. `python model_1/chat.py` to talk to it.

**Model 2 (needs a GPU to be practical)**

1. `python model_2/collect.py` builds `model_2/train.txt` (about 200 MB, needs `datasets` and internet).
2. Run `model_2/train.py` on a GPU, for example in Google Colab (set the runtime to T4 GPU, mount Drive, upload the folder, then `!python train.py`). It trains the tokenizer, converts the text, and trains. Use `--batch 16` if the GPU runs out of memory.
3. Download `model_2.pt` and `tok.json`, then `python model_2/chat.py` runs it on any laptop CPU.

Options for both `train.py` files: `--steps N` for the number of steps, `--fresh` to start a new model instead of continuing. Training resumes from the saved model by default, and `model_backup.pt` is a one-time safety copy that is never overwritten.

In `chat.py`, `/temp 0.5` makes output safer and more repetitive, `/temp 1.1` makes it wilder, and `/len 300` changes the length.

## Data and licenses

Model 2's training text is collected by `collect.py` from public Hugging Face datasets:

| Share | Source | License |
|---|---|---|
| 30% | [SODA](https://huggingface.co/datasets/allenai/soda) synthetic dialogue | CC BY 4.0 |
| 20% | [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (Stanford subset) | Apache 2.0 |
| 25% | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By |
| 25% | [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (English, 2023-11-01) | CC BY-SA |

Check each dataset page for the current terms before you redistribute anything derived from it. Model 1 was trained on a private collection of books that are not included here. Use only text you have the right to use, for example public-domain books from Project Gutenberg.

GitHub rejects files over 100 MB, so share trained weights through GitHub Releases or Hugging Face instead of committing them.

## Roadmap

- [x] Character-level GPT on a laptop
- [x] BPE tokenizer and a larger GPT trained on a free GPU
- [ ] Chat fine-tuning on `User:` / `AI:` dialogue so the model answers once and stops
- [ ] Train longer and on more data (a bigger model needs a bigger GPU)

## Credits

Built step by step with **Claude** (Anthropic), which wrote and explained the code while the project owner collected the data, ran the training and tested the models. The architecture follows the ideas in "Attention Is All You Need" and Andrej Karpathy's *Let's build GPT* and nanoGPT, which are the best places to learn more.

## License

Code: MIT (add a `LICENSE` file with your name). Training data and any weights trained on it follow the licenses in the table above.
