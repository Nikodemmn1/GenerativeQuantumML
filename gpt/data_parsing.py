import json
import torch
from parser_TL2 import print_quest
from transformers import AutoTokenizer
import itertools
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_torchlight_2():
    with open("video-game-text-corpora-master/torchlight2/data/TL2_quests.json", "r") as f:
        torchlight_quests = json.load(f)
    torchlight_quests_raw = [print_quest([], q).strip() for q in torchlight_quests]
    torchlight_quests_raw = [t for t in torchlight_quests_raw if len(t) > 50] # odrzucanie zadań niezawierających tekstu 53

    tokenizer = AutoTokenizer.from_pretrained("openai-gpt", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=350
    )
    torchlight_quests_split = [text_splitter.split_text(q_raw) for q_raw in torchlight_quests_raw]
    torchlight_quests_pre_parsed = list(itertools.chain.from_iterable(torchlight_quests_split))
    torchlight_quests_pre_parsed = [t for t in torchlight_quests_pre_parsed if len(t) > 35]

    torchlight_quests_parsed = tokenizer(torchlight_quests_pre_parsed,
                                         padding=True,
                                         truncation=True,
                                         return_tensors="pt")

    return torchlight_quests_parsed.data


def main():
    tokens_t2 = parse_torchlight_2()
    torch.save(tokens_t2, "data/torchlight_2_parsed.pt")


if __name__ == '__main__':
    main()
