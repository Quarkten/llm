import numpy as np
from model import Seq2Seq

def main():
    with open('corpus.txt', 'r') as f:
        data = f.read()

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    hidden_size = 100
    sequence_length = 25
    learning_rate = 1e-1

    model = Seq2Seq(vocab_size, hidden_size, sequence_length, learning_rate)

    n, p = 0, 0
    h_prev = np.zeros((hidden_size, 1))
    while n <= 10000:
        if p + sequence_length * 2 + 1 >= len(data):
            h_prev = np.zeros((hidden_size, 1))
            p = 0

        inputs = [char_to_ix[ch] for ch in data[p:p + sequence_length]]
        targets = [char_to_ix[ch] for ch in data[p + sequence_length:p + sequence_length * 2]]

        if n % 100 == 0:
            sample_ix = model.sample(inputs, h_prev, 50)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f'---\n {txt} \n---')

        loss, ys, hs = model.forward(inputs, targets, h_prev)
        dV, dc = model.backward(None, hs, ys, targets)
        model.update(dV, dc)

        h_prev = hs[sequence_length - 1]
        p += sequence_length * 2
        n += 1

if __name__ == '__main__':
    main()
