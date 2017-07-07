# Show, Ask, Attend, and Answer

Unofficial implementation of Show, Ask, Attend, and Answer VQA model in Keras. I have not fully tested this implementation due to hardware constraints. The only testing done is the model instantiation to make sure all layers and tensor dimensions were compatible. Hopefully this can serve as a starter code for people to fully implement the model

Paper:
* Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering [[arXiv](https://arxiv.org/abs/1704.03162)][[pdf](https://arxiv.org/pdf/1704.03162.pdf)]
  * Vahid Kazemi and Ali Elqursh
  * Google Research

If you have any questions about the code feel free to contact me, open an issue, pull request, etc.

Disclaimer: I’m not affiliated with the authors of the paper or Google Research. This is an unofficial implementation and results with this code may not directly reflect what is achieved in the paper.

## Requirements ##
- Python 3.6 (2.7 should work too)
- Keras 2.0.2 (used Tensorflow 1.0.0 for the backend)

## Possible Issues ##
The default model in the paper uses 2 glimpses in their attention mechanism. The way I implemented this was by taking the weighted average of the (14x14x2048) image features with each attention map to form 2 (1x2048) vectors that were both concatenated along with the (1x1024) question vector from the LSTM which forms a (1x5120) vector. I’m not completely sure this is correct so if anyone can either confirm or correct that, it would be much appreciated.

