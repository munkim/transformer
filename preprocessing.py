import torch
import nltk
import argparse
from collections import Counter

def load_vocab(vocabfile):
    return vocabfile


def make_vocab(datafile, vocab_size):
    
    def get_vocabs(corpus, vocab_size, thres):
        words = []
        #corpus = ["Hi, my name is Mun Kim.", "Mun is my nickname."] # For testing...
        for line in corpus:
            words.extend(nltk.tokenize.word_tokenize(line.lower())) # Lower case all words in the sentence (or each line of corpus) and split it by space.
            #words.extend(line.lower().split(" ")) # This will not tokenize apostrophes, commas, periods, etc.

        counter = Counter() # Count the number of occurence of each element throughout the array.
        counter.update(words)
        vocabs = [w for w, c in counter.items() if c >= thres]       
            
        return vocabs
    
    def add_label(vocabs):
        word2idx = {}
        for i, vocab in enumerate(vocabs):
            word2idx[vocab] = i+4 
            
        return word2idx

    vocabs = {}
    vocabs.update({'<blank>': 0}) # For padding
    vocabs.update({'<unk>': 1}) # For the unknown words in English like baegopa
    vocabs.update({'<s>': 2}) # For the star of sentence
    vocabs.update({'</s>': 3}) # For the end of sentence 
    
    
    with open(datafile) as f:
        corpus = f.readlines() # Read the whole corpus
        corpus_vocabs = get_vocabs(corpus, vocab_size, thres=1) # Get rid of the repeated words in the corpus
        vocabs.update(add_label(corpus_vocabs))

    return vocabs



def make_data(datafile, vocabs):
    # Include <unk> for the words that are unknown
    # Include padding. The length of the data is equal to the length of the longest sentence. 

    with open(datafile) as f:
        corpus = f.readlines() # Read the whole corpus

    sentences = []
    # <unk> padding
    for line in corpus:
        #tokens = line.lower().split() # simplified version
        tokens = nltk.tokenize.word_tokenize(line.lower())
        sent = []
        for token in tokens:

            if token in vocabs:
                sent.append(token)
            else:
                sent.append('<unk>')
        sentences.append(sent)

    # <s> and </s> padding
    for i, arr in enumerate(sentences):
        sentences[i].insert(0, '<s>')
        sentences[i].append('</s>')
        #sentences[i] = ["<s>", arr, "/s"]

    # <blank> padding (or zero padding)
    max_len = len(max(sentences, key=len))
    # max_len = 11

    for i, arr in enumerate(sentences):
        while max_len > len(arr):
            sentences[i].append("<blank>")

    #print(sentences[0])

    return sentences
    
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', help="(Required) Path to training source data", default="./data/toy-src-train.txt")
    parser.add_argument('-train_tgt', help="(Required) Path to training target data", default="./data/toy-tgt-train.txt")
    parser.add_argument('-valid_src', help="(Required) Path to validation source data", default="./data/src-val.txt")
    parser.add_argument('-valid_tgt', help="(Required) Path to validation target data", default="./data/tgt-val.txt")
    parser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
    parser.add_argument('-tgt_vocab', help="Path to an existing target vocabulary")
    parser.add_argument('-src_vocab_size', type=int, default=50000, help="Size of the source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int, default=50000, help="Size of the target vocabulary")
    args = parser.parse_args()

    assert args.train_src is not None
    assert args.train_tgt is not None

    # word to index dictionary {"word": 10, "index": 20}
    dicts = {}
    dicts['src'] = load_vocab(args.src_vocab) if (args.src_vocab is not None) else make_vocab(args.train_src, args.src_vocab_size)
    dicts['tgt'] = load_vocab(args.tgt_vocab) if (args.tgt_vocab is not None) else make_vocab(args.train_tgt, args.tgt_vocab_size)
    #assert dicts['src'] == dicts['tgt'], 'The source and target of dictionary do not match.'
    print("Vocabs")
    print("Vocab size of src = %d " % len(dicts['src']))
    print("Vocab size of tgt = %d " % len(dicts['tgt']))
    #print(list(dicts['src'].keys())[0])
    print("\n")

    # Prepare training data
    print("Reading Training data...")
    train = {}
    train['src'] = make_data(args.train_src, dicts['src'])
    train['tgt'] = make_data(args.train_tgt, dicts['tgt'])
    print ("Number of training data (src, tgt)", len(train['src']), "  ", len(train['tgt']))
    assert len(train['src']) == len(train['tgt']), 'The source and target of training data do not match.'


    # # Prepare validation data
    # print("Reading Validation data...")
    # valid = {}
    # valid['src'] = make_data(args.valid_src, dicts['src'])
    # valid['tgt'] = make_data(args.valid_tgt, dicts['tgt'])
    # print ("Number of validation data (src, tgt)", len(valid['src']), "  ", len(valid['tgt']))
    # assert len(valid['src']) == len(valid['tgt']), 'The source and target of validation data do not match.'
    #
    # JSON data
    # data = {
    #     'dict': dicts,
    #     'train': train,
    #     'valid': valid
    # }

    data = {
        'dict': dicts,
        'train': train
    }
    print('\n(Note) For testing, please run this again with testing data')
    print('\nSaving to data.pt \n')
    torch.save(data, './data/data.pt')
    print('DONE. Saved to data.pt.\n')

if __name__ == '__main__':
    main()