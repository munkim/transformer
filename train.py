import argparse
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_loader
from model import Transformer
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def convert2text(arr, vocab):
    #assert isinstance(arr, int), "The output array is not a list of vocab indices"
    # arr = int(arr)
    sentence = [vocab.keys()[vocab.values().index(i)] for i in arr]
    sentence = ' '.join(sentence)
    return (sentence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='./data/data.pt',
                        help='Path to the source data. The default is ./data/data.pt, which is the output of preprocessing.')
    parser.add_argument('-epoch', default=10000)
    parser.add_argument('-log_step', default=5)
    parser.add_argument('-save_model_epoch', default=1)
    parser.add_argument('-save_model_path', default='./saved_model/')
    args = parser.parse_args()

    dataset = torch.load(args.data)

    batch_size = 4
    src_vocab = dataset['dict']['src']
    tgt_vocab = dataset['dict']['tgt']
    print("\n\nBatch Size = %d" % batch_size)
    print("Source Vocab Size = %d" % len(src_vocab))
    print("Target Vocab Size = %d" % len(tgt_vocab))

    print("\nLoading Training Data ... ")
    training_batches = get_loader(
        src=dataset['train']['src'],
        tgt=dataset['train']['tgt'],
        src_vocabs=dataset['dict']['src'],
        tgt_vocabs=dataset['dict']['tgt'],
        batch_size=batch_size,
        use_cuda=True,
        shuffle=True
    )

    # print("\nLoading Validation Data ... ")
    # validation_data = get_loader(
    #     src=dataset['valid']['src'],
    #     tgt=dataset['valid']['tgt'],
    #     src_vocabs=dataset['dict']['src'],
    #     tgt_vocabs=dataset['dict']['tgt'],
    #     batch_size=batch_size,
    #     use_cuda=False,
    #     shuffle=False
    # )

    # For python 2
    transformer_config = [6,512,512,8, batch_size, len(src_vocab), len(tgt_vocab), 100, 0.1, True]

    # For python 3
    # transformer_config = {
    #     'N': 6,
    #     'd_model': int(512),
    #     'd_ff': 512,
    #     'H': 8,
    #     'batch_size': batch_size,
    #     'src_vocab_size': int(len(src_vocab)),
    #     'tgt_vocab_size': int(len(tgt_vocab)),
    #     'max_seq': 100,
    #     'dropout': 0.1,
    #     'use_cuda': True
    # }

    transformer = Transformer(transformer_config)
    if torch.cuda.is_available():
        print("CUDA enabled.")
        transformer.cuda()

    optimizer = optim.Adam(
        transformer.parameters(),
        lr=0.001,
        # betas=(0.9, 0.98),
        # eps=1e-09
    )

    criterion = nn.CrossEntropyLoss()

    # Prepare a txt file to print training log
    if not os.path.exists(args.save_model_path):
        print("\nCreated a directory (%s) for saving model since it does not exist.\n" %args.save_model_path)
        os.makedirs(args.save_model_path)

    f = open('%s/train_log.txt' %args.save_model_path, 'w')

    # Train the model
    for e in range(args.epoch):
        for i, batch in enumerate(tqdm(training_batches, mininterval=2, desc='  Training  ', leave=False)):
            # print ("BATCH")
            # print(batch[0][0])
            # exit()
            sources = to_var(batch[0])
            targets = to_var(batch[1])
            src_seq_len = targets.size()[1]
            tgt_seq_len = targets.size()[1]


            if torch.cuda.is_available():
                sources = sources.cuda()
                targets = targets.cuda()


            optimizer.zero_grad()
            outputs = transformer(sources, targets)

            # print("\n\n\n########### OUTPUT ###########")
            # print(len(outputs))
            # print(outputs.max(1)[1].data.tolist() )
            # exit()
            #
            # print("\n\n\n########### TARGET ###########")
            # print(len(targets))
            # print(targets)

            # print(" \n\n TARGETS %d " %i)
            # print(targets)
            # print(targets.contiguous().view(-1).long())
            # exit()

            targets = targets.contiguous().view(-1).long()
            loss = criterion(outputs, targets)

            # backprop
            loss.backward()

            # optimize params
            optimizer.step()

            # Print log info to both console and file
            if i % args.log_step == 0:
                print("\n\n\n\n#################################################################################")
                log = ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f\n'
                      % (e, args.epoch, i, len(training_batches), loss.data[0], np.exp(loss.data[0])))
                print(log)
                f.write("{}".format(log))

                # Print the first sentence of the batch (The first sentence of the batch)
                src_indices = sources.data.tolist()[0][:src_seq_len]  # Variable -> Tensor -> List
                src_sentence = convert2text(src_indices, src_vocab)  # Get sentence

                pred_indices = outputs.max(1)[1].data.tolist()  # Variable -> Tensor -> List
                pred_indices = [i[0] for i in pred_indices[:tgt_seq_len]]  # Get data of index until the max_seq_length of target (i.e. first sentence of the batch).
                pred_sentence = convert2text(pred_indices, tgt_vocab)  # Get sentence

                tgt_indices = targets.data.tolist()[:tgt_seq_len]  # Variable -> Tensor -> List
                tgt_sentence = convert2text(tgt_indices, tgt_vocab)  # Get sentence

                original =  ("ORIGINAL:  {}\n".format(src_sentence))
                predicted = ("PREDICTED: {}\n".format(pred_sentence))
                truth =     ("TRUTH:     {}\n\n".format(tgt_sentence))
                print (original)
                print (predicted)
                print (truth)
                f.write("{}".format(original))
                f.write("{}".format(predicted))
                f.write("{}".format(truth))

        # Save the models
        if (e) % args.save_model_epoch == 0:
            torch.save(transformer.state_dict(),
                       os.path.join(args.save_model_path,
                                    'transformer-%d-%d.pkl' % (e+1, i+1)))



if __name__ == '__main__':
    main()
