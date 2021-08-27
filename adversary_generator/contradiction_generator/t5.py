import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Adafactor
import json
from tqdm import tqdm

def load_data(path):
    data = []
    total_count = 0
    selected_count = 0
    with open(path) as data_file:
        for line in data_file:
            total_count += 1
            dp = json.loads(line.strip())
            if dp["gold_label"] != "contradiction":
                continue
            data.append(({"inputs" : dp["sentence1"], "labels" : dp["sentence2"]}))
            selected_count += 1
    print("Selected {} examples out of {}.".format(selected_count, total_count))
    return data

def run_data_loop(data_loader, model, tokenizer, optimizer, phase="train", epoch=0):
    avg_pe_loss = 0.0
    count = 0
    with tqdm(data_loader, unit="batch") as bprogress:
        if phase == "train":
            bprogress.set_description(f"Train Epoch {epoch + 1}")
        else:
            bprogress.set_description(f"Validation Epoch {epoch + 1}")
        for batch in bprogress:
            batch_dict = dict()
            for k, v in batch.items():
                batch_dict[k] = tokenizer(v, return_tensors="pt", padding="max_length", truncation=True).input_ids.to("cuda:0")
            loss = model(input_ids=batch_dict["inputs"], labels=batch_dict["labels"]).loss
            loss = loss.mean()
            avg_pe_loss += float(loss.item())

            if phase == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            count += 1
            bprogress.set_postfix(loss=avg_pe_loss/count)
    del loss
    del batch_dict["inputs"]
    del batch_dict["labels"]
    # print("Average loss after epoch {} is: {}".format(epoch + 1, avg_pe_loss/count))
    return model, avg_pe_loss/count

if __name__ == "__main__":
    parallel = True
    lr_rate = 1e-4
    batch_size = 4 #maximum size that can be run on one gpu
    if parallel:
        batch_size = 16

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    model.to("cuda:0")
    model.train()
    optimizer = Adafactor(model.parameters(), lr=lr_rate, relative_step=False)

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    train_set_path = "/home/ubuntu/contradiction/multinli_1.0/multinli_1.0_train.jsonl"
    dev_set_path = "/home/ubuntu/contradiction/multinli_1.0/multinli_1.0_dev_matched.jsonl"
    train_data = load_data(train_set_path)
    dev_data = load_data(dev_set_path)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    
    num_epochs = 3

    min_validation_loss = float("inf")

    for i in tqdm(range(num_epochs)):
        model.train()
        model, avg_train_loss = run_data_loop(train_loader, model, tokenizer, optimizer, "train", i)
        
        with torch.no_grad():
            model, avg_val_loss = run_data_loop(dev_loader, model, tokenizer, optimizer, "val", i)
        if avg_val_loss < min_validation_loss:
            torch.save(model.state_dict(), "./t5_contradiction_gen.pt")
            min_validation_loss = avg_val_loss
