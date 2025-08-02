import numpy as np
from model import SimpleRNN

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

    model = SimpleRNN(vocab_size, hidden_size, sequence_length, learning_rate)

    n, p = 0, 0
    h_prev = np.zeros((hidden_size, 1))
    while n <= 10000:
        if p + sequence_length + 1 >= len(data) or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            p = 0
        inputs = [char_to_ix[ch] for ch in data[p:p + sequence_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + sequence_length + 1]]

        if n % 100 == 0:
            sample_ix = model.sample(h_prev, inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f'---\n {txt} \n---')

        xs, hs, ys = model.forward(inputs, h_prev)
        dU, dW, dV, db, dc = model.backward(xs, hs, ys, targets)
        model.update(dU, dW, dV, db, dc)

        h_prev = hs[sequence_length - 1]
        p += sequence_length
        n += 1

if __name__ == '__main__':
    main()
