from paddle.io import Dataset
import os.path as osp
import json
import os

class SST2(Dataset):
    def __init__(self, dataroot, split_name, text_to_input_fn=None):
        assert split_name in ["train", "dev", "test"]
        super().__init__()
        self.dataroot = dataroot        
        self.transform = text_to_input_fn
        self.split_name = split_name

        filename = osp.join(dataroot, self.split_name + '.tsv')
        with open(filename, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()[1:]

    def __getitem__(self, idx):
        if self.split_name == 'test':
            line_num, raw_text = self.lines[idx].strip().split('\t')
            encoded_input = self.transform(raw_text)
            return encoded_input, -1
        else:
            raw_text, label = self.lines[idx].strip().split('\t')
            encoded_input = self.transform(raw_text)
            return encoded_input, label

    def __len__(self):
        return len(self.lines)

def annotations_from_jsonl(fp: str):
    ret = []
    with open(fp, 'r') as inf:
        for line in inf:
            content = json.loads(line)
            ret.append(content)
            # ev_groups = []
            # for ev_group in content['evidences']:
            #     ev_group = tuple([Evidence(**ev) for ev in ev_group])
            #     ev_groups.append(ev_group)
            # content['evidences'] = frozenset(ev_groups)
            # ret.append(Annotation(**content))
    return ret

def load_datasets(data_dir: str, split_name: str):
    """Loads a training, validation, and test dataset

    Each dataset is assumed to have been serialized by annotations_to_jsonl,
    that is it is a list of json-serialized Annotation instances.
    """
    return annotations_from_jsonl(os.path.join(data_dir, f'{split_name}.jsonl'))

def load_documents(data_dir: str, docids=None):
    """Loads a subset of available documents from disk.

    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
    """

    docs_dir = os.path.join(data_dir, 'docs')
    res = dict()
    if docids is None:
        docids = sorted(os.listdir(docs_dir))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        with open(os.path.join(docs_dir, d), 'r') as inf:
            res[d] = inf.read()
    return res

class MovieReview(Dataset):
    def __init__(self, dataroot, split_name, text_to_input_fn=None):
        assert split_name in ["train", "val", "test"]
        super().__init__()

        self.dataroot = dataroot
        self.split_name = split_name
        self.transform = text_to_input_fn

        self.dataset_info = load_datasets(dataroot, split_name)  # list(dict:{classification, annotation, annotation_id})
        self.documents = load_documents(dataroot, None)  # a dict: {annotation_id: text}

        # loading texts and labels
        self.samples = []
        self.labels = []
        for sample_info in self.dataset_info:
            self.labels.append(1 if sample_info['classification'] == 'POS' else 0)
            self.samples.append(self.documents[sample_info['annotation_id']])

    def __getitem__(self, idx):
        raw_text = self.samples[idx]
        encoded_input = self.transform(raw_text)
        label = self.labels[idx]
        return encoded_input, label

    def __len__(self):
        return len(self.samples)