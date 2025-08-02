import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_z = np.random.randn(hidden_size, input_size) * 0.01
        self.U_z = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))

        self.W_r = np.random.randn(hidden_size, input_size) * 0.01
        self.U_r = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((hidden_size, 1))

        self.W_h = np.random.randn(hidden_size, input_size) * 0.01
        self.U_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        z = 1 / (1 + np.exp(-(np.dot(self.W_z, x) + np.dot(self.U_z, h_prev) + self.b_z)))
        r = 1 / (1 + np.exp(-(np.dot(self.W_r, x) + np.dot(self.U_r, h_prev) + self.b_r)))
        h_hat = np.tanh(np.dot(self.W_h, x) + np.dot(self.U_h, r * h_prev) + self.b_h)
        h = (1 - z) * h_prev + z * h_hat
        return h

class Seq2Seq:
    def __init__(self, vocab_size, hidden_size, sequence_length, learning_rate):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        self.encoder = GRU(vocab_size, hidden_size)
        self.decoder = GRU(vocab_size, hidden_size)
        self.V = np.random.randn(vocab_size, hidden_size) * 0.01
        self.c = np.zeros((vocab_size, 1))

    def forward(self, inputs, targets, h_prev):
        xs, hs, os, ys = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = self.encoder.forward(xs[t], hs[t-1])

        decoder_input = np.zeros((self.vocab_size, 1))
        decoder_input[targets[0]] = 1
        decoder_hidden = hs[len(inputs)-1]

        for t in range(len(targets)):
            decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
            os[t] = np.dot(self.V, decoder_hidden) + self.c
            ys[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))
            loss += -np.log(ys[t][targets[t], 0])
            decoder_input = np.zeros((self.vocab_size, 1))
            decoder_input[targets[t]] = 1

        return loss, ys, hs

    def backward(self, xs, hs, ys, targets):
        # This is a simplified backward pass for demonstration purposes.
        # A full backward pass for a GRU is quite complex.
        dV = np.zeros_like(self.V)
        dc = np.zeros_like(self.c)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(targets))):
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1
            dV += np.dot(dy, hs[t].T)
            dc += dy
            dh = np.dot(self.V.T, dy) + dh_next
            dh_next = dh

        return dV, dc

    def update(self, dV, dc):
        self.V -= self.learning_rate * dV
        self.c -= self.learning_rate * dc

    def sample(self, inputs, h_prev, n):
        xs, hs = {}, {}
        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = self.encoder.forward(xs[t], hs[t-1])

        decoder_input = np.zeros((self.vocab_size, 1))
        decoder_input[inputs[0]] = 1
        decoder_hidden = hs[len(inputs)-1]

        ixes = []
        for t in range(n):
            decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
            o = np.dot(self.V, decoder_hidden) + self.c
            y = np.exp(o) / np.sum(np.exp(o))
            ix = np.random.choice(range(self.vocab_size), p=y.ravel())
            decoder_input = np.zeros((self.vocab_size, 1))
            decoder_input[ix] = 1
            ixes.append(ix)
        return ixes
