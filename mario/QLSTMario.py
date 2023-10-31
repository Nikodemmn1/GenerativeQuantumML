import torch
import torch.nn as nn
from pennylane import numpy as np
import pennylane as qml


n_qubits = 2
str_entangled_layers = 1


def q_lstm_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))
    qml.StronglyEntanglingLayers(weights, range(n_qubits))
    #qml.RandomLayers(weights, range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, quantum_dummy=False, q_dev=None, qubits_dummy=None):
        super(QLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        weight_shapes = {'weights': (str_entangled_layers, n_qubits, 3)}
        #weight_shapes = {'weights': (str_entangled_layers, 8 * str_entangled_layers)}

        if qubits_dummy is None:
            self.xh1 = nn.Linear(input_size, n_qubits, bias=bias)
        else:
            self.xh1 = nn.Linear(input_size, qubits_dummy, bias=bias)
        if quantum_dummy:
            if qubits_dummy is None:
                self.xh2 = nn.Linear(n_qubits, n_qubits, bias=bias)
            else:
                self.xh2 = nn.Linear(qubits_dummy, qubits_dummy, bias=bias)
        else:
            self.q_node = qml.QNode(q_lstm_circuit, q_dev, diff_method="best", interface="torch")
            self.xh2 = qml.qnn.TorchLayer(self.q_node, weight_shapes)
        if qubits_dummy is None:
            self.xh3 = nn.Linear(n_qubits, hidden_size * 4, bias=bias)
            self.hh1 = nn.Linear(hidden_size, n_qubits, bias=bias)
        else:
            self.xh3 = nn.Linear(qubits_dummy, hidden_size * 4, bias=bias)
            self.hh1 = nn.Linear(hidden_size, qubits_dummy, bias=bias)
        if quantum_dummy:
            if qubits_dummy is None:
                self.hh2 = nn.Linear(n_qubits, n_qubits, bias=bias)
            else:
                self.hh2 = nn.Linear(qubits_dummy, qubits_dummy, bias=bias)
        else:
            self.hh2 = qml.qnn.TorchLayer(self.q_node, weight_shapes)
        if qubits_dummy is None:
            self.hh3 = nn.Linear(n_qubits, hidden_size * 4, bias=bias)
        else:
            self.hh3 = nn.Linear(qubits_dummy, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, states=None):
        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size)
        #       cy: of shape (batch_size, hidden_size)

        if states is None:
            hx = x.new_zeros(size=(x.size(0), self.hidden_size), requires_grad=True)
            states = (hx, hx)

        hx, cx = states
        gates = self.xh3(self.xh2(self.xh1(x))) + self.hh3(self.hh2(self.hh1(hx)))

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t
        hy = o_t * torch.tanh(cy)

        return hy, cy


class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, dropout, quantum_dummy=False, qubits_dummy=None):
        super(QLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.rnn_cell_list = nn.ModuleList()

        if quantum_dummy:
            q_dev = None
        else:
            q_dev = qml.device('default.qubit',
                               wires=n_qubits,
                               c_dtype=np.complex64,
                               r_dtype=np.float32)

        self.rnn_cell_list.append(QLSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias,
                                            quantum_dummy,
                                            q_dev,
                                            qubits_dummy))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(QLSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias,
                                                quantum_dummy,
                                                q_dev,
                                                qubits_dummy))

    def forward(self, x, initial_states=None):
        # Input of shape (batch_size, sequence_length, input_size)
        # Output of shape (batch_size, output_size)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        outs = []
        hidden = []
        next_states = None

        if initial_states is not None:
            h, c = initial_states
        else:
            h = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device, requires_grad=True)
            c = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device, requires_grad=True)

        for layer in range(self.num_layers):
            hidden.append((h[layer, ...], c[layer, ...]))

        for t in range(x.size(1)):
            for l in range(self.num_layers):
                if l == 0:
                    next_states = self.rnn_cell_list[l](x[:, t, :], (hidden[l][0], hidden[l][1]))
                else:
                    next_states = self.rnn_cell_list[l](nn.functional.dropout(hidden[l - 1][0], p=self.dropout),
                                                        (hidden[l][0], hidden[l][1]))
                hidden[l] = next_states
            outs.append(next_states[0])

        return torch.stack(outs, 1), (torch.stack([hs[0] for hs in hidden]), torch.stack([hs[1] for hs in hidden]))


class LSTMArio(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, quantum_dummy=False, qubits_dummy=None):
        super(LSTMArio, self).__init__()
        self.LSTM = QLSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bias=False,
                          dropout=dropout,
                          quantum_dummy=quantum_dummy,
                          qubits_dummy=qubits_dummy)
        self.out = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.out_acc = nn.Softmax(dim=-1)

    def forward(self, x, softmax=False, states=None):
        if states is not None:
            y, states = self.LSTM(x, states)
        else:
            y, states = self.LSTM(x)
        y = self.out(y)
        if softmax:
            y = self.out_acc(y)
        return y, states
