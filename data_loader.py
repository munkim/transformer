import torch
import torch.utils.data as data


class WMT_Dataset(data.Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self, index):
        source = torch.Tensor(self.src)
        target = torch.Tensor(self.tgt)

        return source[index], target[index]

    def __len__(self):
        return len(self.src)


def get_loader(src, tgt, src_vocabs, tgt_vocabs, batch_size, use_cuda, num_workers=2, shuffle=True):
    assert len(src) >= batch_size, "Training data size (%d) < batch Size (%d)" % (len(src), batch_size)
    assert len(src) == len(tgt), "Number of source (%d) and target (%d) do not match" % (len(src), len(tgt))

    # If the source or target is in words but not list of word indices, get the index of the words
    if not isinstance(src, int):
        src = [[src_vocabs[i] for i in src[k]] for k in range(len(src))]
        tgt = [[tgt_vocabs[i] for i in tgt[k]] for k in range(len(tgt))]

    print("Number of Data: SRC=%d and TGT=%d \n" % (len(src), len(tgt)))
    # Load WMT_Dataset
    wmt = WMT_Dataset(
        src=src,
        tgt=tgt
    )

    # Pytorch DataLoader Doc: http://pytorch.org/docs/data.html#torch.utils.data.DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset=wmt,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    batches = []
    for i, batch in enumerate(data_loader):
        source = batch[0]
        target = batch[1]
        max_src_len = max(torch.nonzero(source).transpose(0, 1)[1][:]) + 1  # Find all indices (tuples) of nonzero elements, and get the
        max_tgt_len = max(torch.nonzero(target).transpose(0, 1)[1][:]) + 1  # maximum index of the sequence. Add 1 since index starts from 0.
        source_trimmed = source.narrow(1, 0, max_src_len)  # Trim the source tensor to the maximum sequence length
        target_trimmed = target.narrow(1, 0, max_tgt_len)  # Trim the target tensor to the maximum sequence length
        batches += zip([source_trimmed], [target_trimmed])  # List of tensors in varying sizes.

        print ("Max SRC length of batch %d = %d" % (i, max_src_len))
        print ("Max TGT length of batch %d = %d" % (i, max_tgt_len))

    # Return a list of batches, a list of src_seq_len, and a list tgt_seq_len for every batches.
    return batches