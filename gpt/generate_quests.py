import os
import torch
import re
from transformers import AutoTokenizer
from openai_quantum import OpenAIGPTLMHeadModel, OpenAIGPTConfig


def file_name_to_run_info(file_path):
    file_path = re.sub(r".+(lora_quantum=.+)", r"\g<1>", file_path)
    file_path = re.sub(r"_timestamp=\d+/.+\.pt", "", file_path)
    file_path = file_path.replace("lora_quantum=", "")
    file_path = file_path.replace("lora_rank=", "")
    file_path = file_path.replace("str_ent_layers=", "")
    info_list = file_path.split(sep="_")#[:-1]
    info = {"is_quantum": info_list[0] == "True",
            "lora_rank": int(info_list[1]),
            "str_ent_layers": int(info_list[2])}
    return info



def generate():
    generation_temperature = 0.825
    section_lengths = {"intro": 117,
                       "return": 111,
                       "complete": 78,
                       "details": 47,
                       "moredetails": 121,
                       "huddetails": 36}
    sections_big = ["intro", "return", "complete", "details", "huddetails"]
    model_paths = [
        # ścieżki do modeli
    ]

    for model_path in model_paths:
        info = file_name_to_run_info(model_path)

        if not os.path.exists(f"quests/q={str(info['is_quantum'])}_l={str(info['str_ent_layers'])}_r={str(info['lora_rank'])}"):
            os.makedirs(f"quests/q={str(info['is_quantum'])}_l={str(info['str_ent_layers'])}_r={str(info['lora_rank'])}")

        model_config = OpenAIGPTConfig(lora_quantum=info["is_quantum"],
                                       lora_rank=info["lora_rank"],
                                       lora_alpha=16,
                                       str_ent_layers=info["str_ent_layers"])

        model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt", config=model_config)
        model.load_state_dict(torch.load(model_path))
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained("openai-gpt", use_fast=False)
        tokenizer.pad_token = tokenizer.unk_token

        quests_to_generate = 1

        for q_num in range(quests_to_generate):
            g = ""
            for s in sections_big:
                if g != "":
                    g += "\n\n"
                g += f"{s}\n" + "-" * len(s) + "\n"
                x = tokenizer(g,
                              padding=True,
                              truncation=True,
                              return_tensors="pt").data["input_ids"]
                section_generate_finished = False

                while not section_generate_finished:
                    generated = model.generate(x.cuda(),
                                               max_new_tokens=section_lengths[s],
                                               do_sample=True,
                                               temperature=generation_temperature,
                                               begin_suppress_tokens=[0])
                    generated_decoded = tokenizer.batch_decode(generated,
                                                               skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True)
                    generated_section = re.search(s + " *\n" + r" *-+ *" + "\n" + r"(?s:.*?)($|-----)", generated_decoded[0])
                    if generated_section is None:
                        continue
                    generated_section = generated_section.group(0).rstrip()
                    if len(generated_section.splitlines()) <= 2:
                        continue
                    section_generate_finished = True

                if "---" in generated_section.splitlines()[-1]:
                    generated_section = "\n".join(generated_section.splitlines()[:-2])
                g += "\n".join(generated_section.splitlines()[2:])
            with open(
                    f"quests/q={str(info['is_quantum'])}_l={str(info['str_ent_layers'])}_r={str(info['lora_rank'])}/"
                    f"quest_{q_num}.txt",
                    "w") as f:
                f.write(g)


if __name__ == "__main__":
    generate()

