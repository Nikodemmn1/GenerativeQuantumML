import math
import torch
import torch.nn as nn
from torch.nn.functional import fold
from pennylane import numpy as np
import pennylane as qml


def q_lstm_circuit(n_qubits, inputs, random_layer_weights):
    qml.AngleEmbedding(inputs, range(n_qubits), rotation='Y')
    qml.RandomLayers(random_layer_weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class BaseQ2dLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, quantum_device, random_rotations, stride):
        super(BaseQ2dLayer, self).__init__()

        if isinstance(kernel_size, int) and kernel_size > 0:
            kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
        elif not isinstance(kernel_size, tuple) or (isinstance(kernel_size, int) and kernel_size <= 0):
            raise Exception("kernel_size must be a positive integer, a tuple or a list")
        if len(kernel_size) != 2:
            raise Exception("the kernel is 2-dimensional, "
                            "so the kernel_size must contain two values (unless it's an integer)!")
        self.kernel_size = kernel_size

        if not (isinstance(input_channels, int) and input_channels > 0 and
                isinstance(output_channels, int) and output_channels > 0):
            raise Exception("Numbers of input and output channels must be positive integers")
        self.input_channels = input_channels
        self.output_channels = output_channels

        if isinstance(stride, int) and stride > 0:
            stride = (stride, stride)
        elif isinstance(stride, list):
            stride = tuple(stride)
        elif not isinstance(stride, tuple) or (isinstance(stride, int) and stride <= 0):
            raise Exception("stride must be a positive integer, a tuple or a list")
        if len(stride) != 2:
            raise Exception("stride must contain two values (unless it's an integer)!")
        self.stride = stride

        self.n_qubits = kernel_size[0] * kernel_size[1]
        self.q_circuit = lambda inputs, random_layer_weights: (
            q_lstm_circuit(self.n_qubits, inputs, random_layer_weights))
        self.q_device = quantum_device
        self.q_weights_shape = {'random_layer_weights': (1, random_rotations)}

    def calculate_fragment_q_conv(self, inp):
        inp = torch.flatten(inp, start_dim=2)
        out = []
        for o in range(self.output_channels):
            out_channel_o = None
            kernels_output_tmp = getattr(self, f"kernels_output_{o}")
            for i, q_layer in enumerate(kernels_output_tmp):
                if out_channel_o is None:
                    out_channel_o = q_layer(inp[:, i, ...])
                else:
                    out_channel_o += q_layer(inp[:, i, ...])
            out.append(out_channel_o.sum(dim=1))
        out = torch.stack(out, dim=1)
        return out


class SimpleQConv2dLayer(BaseQ2dLayer):
    def __init__(self, input_channels, output_channels, kernel_size, quantum_device, random_rotations, stride=1):
        super(SimpleQConv2dLayer, self).__init__(input_channels, output_channels, kernel_size, quantum_device,
                                                 random_rotations, stride)

        for o in range(output_channels):
            kernels_output_tmp = nn.ModuleList()
            for _ in range(input_channels):
                quantum_node = qml.QNode(self.q_circuit, self.q_device, diff_method="best", interface="torch")
                quantum_layer = qml.qnn.TorchLayer(quantum_node, self.q_weights_shape)
                kernels_output_tmp.append(quantum_layer)
            setattr(self, f"kernels_output_{o}", kernels_output_tmp)

        self.u_kernel = math.floor((self.kernel_size[0] - 1) / 2)
        self.d_kernel = self.kernel_size[0] - 1 - self.u_kernel
        self.l_kernel = math.floor((self.kernel_size[1] - 1) / 2)
        self.r_kernel = self.kernel_size[1] - 1 - self.l_kernel

        self.last_size = (-1, -1)
        self.out_size = (-1, -1)

    # inp of shape: (batch_size, channels, height, width)
    def forward(self, inp: torch.Tensor):
        if inp.size()[2:] != self.last_size:
            self.out_size = (math.floor((inp.size(2) - self.kernel_size[0]) / self.stride[0] + 1),
                             math.floor((inp.size(3) - self.kernel_size[1]) / self.stride[1] + 1))
            self.last_size = inp.size()[2:]

        out = []
        y = self.u_kernel
        for _ in range(self.out_size[0]):
            x = self.l_kernel
            for __ in range(self.out_size[1]):
                fragment = inp[:, :, y - self.u_kernel:y + self.d_kernel + 1, x - self.l_kernel:x + self.r_kernel + 1]
                out.append(self.calculate_fragment_q_conv(fragment))
                x += self.stride[1]
            y += self.stride[0]

        out = torch.stack(out, dim=-1).reshape((out[0].size(0),
                                                out[0].size(1),
                                                self.out_size[0],
                                                self.out_size[1]))
        return out
