import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
import random
import time
import logging
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

Params = {
    "save_dir": "model_output",
    "data_file": "文本分类练习.csv",
    "vocab_file": "vocab_chars.txt",
    "model_kind": "fast_text",
    "seq_len": 50,
    "embed_dim": 128,
    "filter_size": 3,
    "num_layers": 2,
    "epochs": 5,
    "batch_size": 64,
    "pool_type": "avg",
    "opt_name": "adam",
    "lr": 1e-3,
    "pretrain_path": None,
    "random_seed": 987,
    "num_classes": 2
}


class TextDataset(Dataset):
    def __init__(self, df, params, build_vocab=False):
        self.params = params
        self.df = df
        self.vocab = {}

        if build_vocab:
            self.create_vocabulary()
        else:
            self.vocab = self.load_vocabulary(params["vocab_file"])

        self.params["vocab_size"] = len(self.vocab)
        self.samples = []
        self.process_data()

    def create_vocabulary(self):
        all_chars = "".join([str(text) for text in self.df['review']])
        char_counts = Counter(all_chars)
        vocab = {char: idx + 2 for idx, (char, _) in enumerate(char_counts.most_common(4000))}
        vocab['[PAD]'] = 0
        vocab['[UNK]'] = 1
        self.vocab = vocab

        with open(self.params["vocab_file"], 'w', encoding='utf-8') as f:
            for k, v in self.vocab.items():
                f.write(json.dumps([k, v]) + "\n")

    def load_vocabulary(self, vocab_path):
        token_map = {}
        try:
            with open(vocab_path, encoding="utf8") as f:
                for line in f:
                    pair = json.loads(line)
                    token_map[pair[0]] = pair[1]
        except FileNotFoundError:
            pass
        return token_map

    def process_data(self):
        for _, row in self.df.iterrows():
            text = str(row['review'])
            label = int(row['label'])

            token_ids = self.tokenize(text)
            token_ids = torch.LongTensor(token_ids)
            label_tensor = torch.LongTensor([label])
            self.samples.append([token_ids, label_tensor])

    def tokenize(self, text):
        ids = []
        for char in text:
            ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        ids = self.pad_sequence(ids)
        return ids

    def pad_sequence(self, ids):
        ids = ids[:self.params["seq_len"]]
        ids += [0] * (self.params["seq_len"] - len(ids))
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_data_loader(df, params, shuffle=True, build_vocab=False):
    dataset = TextDataset(df, params, build_vocab=build_vocab)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=shuffle)
    return loader


class Conv1dEncoder(nn.Module):
    def __init__(self, params):
        super(Conv1dEncoder, self).__init__()
        embed_dim = params["embed_dim"]
        filter_size = params["filter_size"]
        pad = (filter_size - 1) // 2
        self.conv = nn.Conv1d(embed_dim, embed_dim, filter_size, bias=False, padding=pad)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class TextClassifier(nn.Module):
    def __init__(self, params):
        super(TextClassifier, self).__init__()
        embed_dim = params["embed_dim"]
        vocab_size = params["vocab_size"] + 1
        num_classes = params["num_classes"]
        model_kind = params["model_kind"]
        num_layers = params["num_layers"]
        self.use_bert = False

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if model_kind == "fast_text":
            self.encoder = lambda x: x
        elif model_kind == "lstm":
            self.encoder = nn.LSTM(embed_dim, embed_dim, num_layers=num_layers,
                                   batch_first=True, bidirectional=True)
            embed_dim *= 2
        elif model_kind == "cnn":
            self.encoder = Conv1dEncoder(params)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.pool_type = params["pool_type"]
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        x = self.embed(x)
        x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pool_type == "max":
            pool = nn.MaxPool1d(x.shape[1])
        else:
            pool = nn.AvgPool1d(x.shape[1])

        x = x.transpose(1, 2)
        x = pool(x).squeeze()
        logits = self.fc(x)

        if target is not None:
            return self.criterion(logits, target.squeeze())
        else:
            return logits


def select_optimizer(params, model):
    opt_name = params["opt_name"]
    lr = params["lr"]

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


class ModelEvaluator:
    def __init__(self, params, model, logger, valid_df):
        self.params = params
        self.model = model
        self.logger = logger
        self.valid_loader = get_data_loader(valid_df, params, shuffle=False, build_vocab=False)
        self.metrics = {"correct": 0, "wrong": 0}

    def evaluate(self, epoch):
        self.logger.info(f"Testing epoch {epoch} model performance:")
        self.model.eval()
        self.metrics = {"correct": 0, "wrong": 0}

        for _, batch in enumerate(self.valid_loader):
            if torch.cuda.is_available():
                batch = [d.cuda() for d in batch]
            input_ids, labels = batch

            with torch.no_grad():
                outputs = self.model(input_ids)
            self.update_metrics(labels, outputs)

        acc = self.calculate_accuracy()
        return acc

    def update_metrics(self, labels, outputs):
        for true, pred in zip(labels, outputs):
            pred_label = torch.argmax(pred)
            if int(true) == int(pred_label):
                self.metrics["correct"] += 1
            else:
                self.metrics["wrong"] += 1

    def calculate_accuracy(self):
        correct = self.metrics["correct"]
        wrong = self.metrics["wrong"]
        total = correct + wrong

        self.logger.info(f"Total samples: {total}")
        self.logger.info(f"Correct: {correct}, Wrong: {wrong}")
        self.logger.info(f"Accuracy: {correct / total:.4f}")

        return correct / total

    def test_speed(self):
        self.model.eval()
        sample_input = torch.randint(0, self.params["vocab_size"],
                                     (100, self.params["seq_len"]))
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()

        self.model(sample_input)

        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                self.model(sample_input)
        end = time.time()

        avg_time = (end - start) / 10 * 1000
        self.logger.info(f"Time for 100 samples: {avg_time:.2f} ms")
        return avg_time


def train_model(params, train_df, valid_df, logger):
    if not os.path.isdir(params["save_dir"]):
        os.mkdir(params["save_dir"])

    train_loader = get_data_loader(train_df, params, build_vocab=True)
    model = TextClassifier(params)

    if torch.cuda.is_available():
        logger.info("GPU available, moving model to GPU")
        model = model.cuda()

    optimizer = select_optimizer(params, model)
    evaluator = ModelEvaluator(params, model, logger, valid_df)

    for epoch in range(params["epochs"]):
        epoch += 1
        model.train()
        losses = []

        for _, batch in enumerate(train_loader):
            if torch.cuda.is_available():
                batch = [d.cuda() for d in batch]

            optimizer.zero_grad()
            input_ids, labels = batch
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        logger.info(f"Epoch {epoch} average loss: {np.mean(losses):.4f}")
        acc = evaluator.evaluate(epoch)

    infer_time = evaluator.test_speed()
    return acc, infer_time


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)

    seed = Params["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Loading data...")
    df = pd.read_csv(Params["data_file"])
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=seed)
    logger.info(f"Data loaded. Train: {len(train_df)}, Valid: {len(valid_df)}")

    model_names = ["fast_text", "cnn", "lstm"]
    results = []

    for name in model_names:
        logger.info("-" * 40)
        logger.info(f"Training model: {name}")
        Params["model_kind"] = name

        if name == "fast_text":
            Params["pool_type"] = "avg"
        else:
            Params["pool_type"] = "max"

        acc, time_cost = train_model(Params, train_df, valid_df, logger)

        results.append({
            "Model": name,
            "Accuracy": f"{acc:.4f}",
            "Time(100 samples)": f"{time_cost:.2f} ms"
        })

    print("\n" + "=" * 50)
    print("Final Results Summary")
    print("=" * 50)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))