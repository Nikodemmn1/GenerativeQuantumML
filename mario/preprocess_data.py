import os
import json
import numpy as np
from parse_preprocessed_data import get_inputs_and_targets


def load_txt(txt):
    with open(txt, 'r') as txt_f:
        return txt_f.readlines()


def main():
    seq_length = 200
    dir_1 = "data_raw/super_mario_bros"
    dir_2 = "data_raw/super_mario_bros_2_japan"
    annot_txts = [os.path.join(dir_1, fn) for fn in os.listdir(dir_1) if fn.split('.')[-1] == 'txt']
    annot_txts += [os.path.join(dir_2, fn) for fn in os.listdir(dir_2) if fn.split('.')[-1] == 'txt']

    with open('data_preprocessed/mario.txt', 'w+') as txt_f:
        for i, fp in enumerate(annot_txts):
            infile = load_txt(fp)

            lines = []
            for line in infile:
                lines.append(list(line.rstrip()))

            infile_transposed = np.array(lines).T

            for line in infile_transposed:  # each line represents a column
                num_chars_to_add = 16 - len(lines)
                txt_f.write("".join(['-'] * num_chars_to_add + list(line)))
                txt_f.write("\n")

            if i + 1 == len(annot_txts):  # separate each level with the ")" character
                txt_f.write(")")
            else:
                txt_f.write(")\n")

    char_to_ix, ix_to_char, \
        vocab_size, inputs, targets = get_inputs_and_targets('data_preprocessed/mario.txt', seq_length)

    first_three_cols = inputs[0][:10 * 17]
    np.savetxt('data_preprocessed/seed.txt', first_three_cols)
    with open('data_preprocessed/char_to_ix.json', 'w+') as json_f:
        json.dump(char_to_ix, json_f)
    with open('data_preprocessed/ix_to_char.json', 'w+') as json_f:
        json.dump(ix_to_char, json_f)


if __name__ == '__main__':
    main()
