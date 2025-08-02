import numpy as np

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, sequence_length, learning_rate):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        self.U = np.random.randn(hidden_size, vocab_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((vocab_size, 1))

        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)

    def forward(self, inputs, h_prev):
        xs, hs, os, ys = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.U, xs[t]) + np.dot(self.W, hs[t-1]) + self.b)
            os[t] = np.dot(self.V, hs[t]) + self.c
            ys[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))
        return xs, hs, ys

    def backward(self, xs, hs, ys, targets):
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        dh_next = np.zeros_like(hs[0])
        for t in reversed(range(self.sequence_length)):
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1
            dV += np.dot(dy, hs[t].T)
            dc += dy
            dh = np.dot(self.V.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            db += dh_raw
            dU += np.dot(dh_raw, xs[t].T)
            dW += np.dot(dh_raw, hs[t-1].T)
            dh_next = np.dot(self.W.T, dh_raw)
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)
        return dU, dW, dV, db, dc

    def update(self, dU, dW, dV, db, dc):
        for param, dparam, mem in zip([self.U, self.W, self.V, self.b, self.c],
                                      [dU, dW, dV, db, dc],
                                      [self.mU, self.mW, self.mV, self.mb, self.mc]):
            mem += dparam * dparam
            param -= self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
            o = np.dot(self.V, h) + self.c
            y = np.exp(o) / np.sum(np.exp(o))
            ix = np.random.choice(range(self.vocab_size), p=y.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes
