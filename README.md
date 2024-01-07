## Tiny Transformer Code Completion

The goal of the project is a code completion helper, powered by a language 
model and executed on a client-side in a browser, without any backend: 

Demo: https://code-completion.tymur-prorochenko.com/

- `tab` to accept a suggestion
- `esc` / `mouse click` / `arrow keys` to clear suggestion
- only printable ASCII characters (codes 32 to 127) are supported
- suggestions are generated only when the caret is at the end of the string
- up to 2048 characters 

---
### Some inspiration
I had a cool mindset shift recently: you do not need to be a large corporation 
to do interesting things with language models or to train them from scratch.

For example, in this [paper](https://arxiv.org/abs/2305.07759), a study is 
made: given a dataset of tiny stories written in a language that could be 
understood by 3-4-year-old children, even language models with less than 10 mln 
parameters could coherently complete the texts.
Such models could be trained in a few hours on a single GPU from scratch.

And [here](https://github.com/dmarcos/llama2.c-web) is a port of 
[llama2.c](https://github.com/karpathy/llama2.c) to Web Assembly, which just 
works in any browser!

---
### How is it done
For the training data, I used 
[the Stack](https://huggingface.co/datasets/bigcode/the-stack)
dataset and took 10 languages more or less by popularity: 
C, C++, Go, Java, Javascript, Kotlin, PHP, Python, Rust, TypeScript.

I also did a minor extra filtration on my side, I dropped samples that: 
- contained non-standard ANSII characters
- size above 95th or below 1st percentile
- max line length above 95th or below 1st percentile
- for Python code: cannot be parsed by ast 

For the model, I reused a 6mln parameter llama2, 288 dimension embeddings, 
6 layers, 6 attention heads from the examples in llama2.c project. I found that
it is fast enough to be triggered after every keystroke and weighs only 6.7 MB 
when quantized to 8 bits. 

The tokenization I've changed a lot: looks like to give consistent predictions 
after every letter you need character-sized tokens. Hence the tokenization here 
is a direct conversion to ANSII codes, giving a vocabulary size of 128.

I first pre-trained a baseline model on a subset of my dataset: 
10mln samples, 25bln tokens. It took 9 hours on RTX4090: training in bfloat16, 
using a batch size of 96 and sequence length of 2048. Learning rate 5e-4, 
with 1% warmup and a cosine lr decay towards zero.

Next, I created a separate fine-tuned version of the model for each of 
the 10 languages, training on all the data for every language.

Next, all the models are converted to WASM with Emscripten and called after 
every keystroke. Also, an interesting optimization is done: transformer state
is reused, so every type a key is pressed, we trigger forward only on the 
tokens that are not in the model state, which makes completion in a constant 
time for any input size (as long as the new input only differs by a few 
characters in the end).

---
### To do
- improve state reuse to include completion
- properly handle out-of-vocabulary characters
- faster matrix multiplication
- do not use emscripten
- research hyperparameters vs suggestion quality/model size/latency
