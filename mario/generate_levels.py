import json
import numpy as np
import torch
from tqdm import tqdm
from txt_to_png import txt_levels_to_img


def onehot_to_string(ix_to_char, onehot, cut_off_last=True):
    ints = np.argmax(onehot, axis=-1)
    chars = [ix_to_char[str(ix)] for ix in ints]
    string = "".join(chars)
    char_array = []
    lines = string.rstrip().split('\n')
    if cut_off_last:
        lines = lines[:-1]
    for line in lines:
        if len(line) == 16:
            char_array.append(list(line))
        elif len(line) > 16:
            char_array.append(list(line[:16]))
        elif len(line) < 16:
            char_array.append(['-'] * (16 - len(line)) + list(line))
    char_array = np.array(char_array).T
    string = ""
    for row in char_array:
        string += "".join(row) + "\n"
    return string


def get_seed(char_to_ix):
    cols_num = 8
    seed = np.loadtxt('generate_seed.txt', dtype=float)[:cols_num*17 - 1]
    for i in range(cols_num - 1):
        seed[17 + i*17 + 14] = 0
        seed[17 + i*17 + 14][char_to_ix['x']] = 1
        seed[17 + i*17 + 4] = 0
        seed[17 + i*17 + 4][char_to_ix['-']] = 1

    # cegły ze znakiem
    seed[17 + 1 * 17 + 11] = 0
    seed[17 + 1 * 17 + 11][char_to_ix['S']] = 1
    seed[17 + 2 * 17 + 11] = 0
    seed[17 + 2 * 17 + 11][char_to_ix['Q']] = 1
    seed[17 + 3 * 17 + 11] = 0
    seed[17 + 3 * 17 + 11][char_to_ix['S']] = 1
    seed[17 + 4 * 17 + 11] = 0
    seed[17 + 4 * 17 + 11][char_to_ix['S']] = 1

    # monetki
    seed[17 + 1 * 17 + 10] = 0
    seed[17 + 1 * 17 + 10][char_to_ix['o']] = 1
    seed[17 + 2 * 17 + 10] = 0
    seed[17 + 2 * 17 + 10][char_to_ix['o']] = 1
    seed[17 + 3 * 17 + 10] = 0
    seed[17 + 3 * 17 + 10][char_to_ix['o']] = 1
    seed[17 + 4 * 17 + 10] = 0
    seed[17 + 4 * 17 + 10][char_to_ix['o']] = 1

    # wrog
    seed[17 + 5 * 17 + 14] = 0
    seed[17 + 5 * 17 + 14][char_to_ix['E']] = 1
    seed[17 + 5 * 17 + 12] = 0
    seed[17 + 5 * 17 + 12][char_to_ix['x']] = 1
    seed[17 + 4 * 17 + 13] = 0
    seed[17 + 4 * 17 + 13][char_to_ix['x']] = 1
    seed[17 + 4 * 17 + 14] = 0
    seed[17 + 4 * 17 + 14][char_to_ix['-']] = 1
    seed[17 + 6 * 17 + 13] = 0
    seed[17 + 6 * 17 + 13][char_to_ix['x']] = 1
    seed[17 + 6 * 17 + 14] = 0
    seed[17 + 6 * 17 + 14][char_to_ix['-']] = 1
    return torch.from_numpy(seed.astype('float32'))


def generate_levels(model_path, model_name):
    with open('ix_to_char.json', 'r') as json_f:
        ix_to_char = json.load(json_f)

    with open('char_to_ix.json', 'r') as json_f:
        char_to_ix = json.load(json_f)

    model = torch.load(model_path).cpu()
    model.eval()

    seed = get_seed(char_to_ix)

    num_levels_to_gen = 10
    num_chunks = 3
    num_cols_per_chunk = 16
    num_rows_per_col = 17
    num_chars_to_gen = num_chunks * num_cols_per_chunk * num_rows_per_col - len(seed)
    hidden_size = 128
    vocab_size = 15
    num_layers = 1
    reverse_temperature = 1

    seed = torch.unsqueeze(seed, dim=0)
    seed = seed.repeat((num_levels_to_gen, 1, 1))

    gen = seed.clone()

    # initialize all hidden and cell states to zeros
    lstm_h = torch.zeros((num_layers, num_levels_to_gen, hidden_size))
    lstm_c = torch.zeros_like(lstm_h)

    for _ in tqdm(range(num_chars_to_gen), leave=False):
        # predict probas and update hidden and cell states
        with torch.no_grad():
            probas, (lstm_h, lstm_c) = model(seed, softmax=True, states=(lstm_h, lstm_c))

        probas = probas[:, -1]  # all batches, last timestep
        # before: probas.shape == (num_levels_to_gen, length_of_seed, vocab_size)
        # after: probas.shape == (num_levels_to_gen, vocab_size)

        seed = torch.zeros((num_levels_to_gen, 1, vocab_size))
        for b in range(num_levels_to_gen):
            p = probas[b]
            p = torch.exp(reverse_temperature*torch.log(p)) / torch.sum(torch.exp(reverse_temperature*torch.log(p)))
            idx = np.random.choice(np.arange(len(p)), p=p.numpy())
            seed[b][0] = 0
            seed[b][0][idx] = 1

        gen = np.concatenate([gen, seed], axis=1)

    text_levels = [onehot_to_string(ix_to_char, g) for g in gen]
    txt_levels_to_img(text_levels, model_name)
