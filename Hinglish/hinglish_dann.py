import os
import math
import numpy as np
from scipy import sparse
from collections import Counter
import random
from tqdm import trange
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Function

# set the seed

def main(LAM,VOC,H, BS, ALP):
    tm = datetime.datetime.now()
    manual_seed = 321
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed(manual_seed)

    # location of the data and size of the vocabulary
    DATA_DIR = "data"
    #VOCAB_SIZE = 5000
    VOCAB_SIZE = VOC

    # creates the vocab from the preprocessed features
    def create_vocab(documents):
        # count all the tokens in both the files (document frequency)
        vocab_count = Counter()
        for doc in documents:
            doc = doc.strip()
            tokens = [token.split(":")[0] for token in doc.split()[0:-1]]  # last token is the label so we ignore it
            for token in set(tokens):
                vocab_count[token] += 1
        # create the token to id and id to token mappings
        t2i, i2t = {}, {}
        for token, _ in vocab_count.most_common()[:VOCAB_SIZE]:
            t2i[token] = len(i2t)
            i2t[t2i[token]] = token
        print("created vocab. of size %d" % len(t2i))
        print("top 10 tokens ...")
        print(list(t2i)[0:10])
        return t2i, i2t, vocab_count


    def tfidf_docs(documents, t2i, i2t, vocab_count, do_unlabeled=False):
        labels_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
        coords_1, coords_2, values, y = [], [], [], []
        row_id = 0
        n, d = len(documents), len(t2i)
        for doc in documents:
            items = doc.split()
            for item in items[0:-1]:
                token, freq = item.split(":")
                if token in t2i:
                    col_id = t2i[token]
                    # we will use the weighing scheme 2 from the recommended options
                    # ref: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    tf_score = 1.0 + math.log(float(freq))
                    idf_score = math.log(1.0 + float(n) / vocab_count[token])
                    coords_1.append(row_id)
                    coords_2.append(col_id)
                    values.append(tf_score * idf_score)
            if do_unlabeled:
                y = ""
            else:
                label = labels_dict[items[-1].split(":")[1]]
                y.append(label)
                row_id = row_id + 1
        X = sparse.coo_matrix((values, (coords_1, coords_2)), shape=(n, d))
        y = np.array(y)
        print("shape of inputs = %d, %d" % (X.shape[0], X.shape[1]))
        print("number of non-zero entries = %d" % (X.count_nonzero()))
        return X, y


    # reads the labeled documents to X, y
    def read_labeled_dir(domain, vocab=None):
        print("processing the labeled %s domain" % domain)

        # file paths
        positive_f = os.path.join(DATA_DIR, domain, 'positive.txt')
        negative_f = os.path.join(DATA_DIR, domain, 'negative.txt')
        neutral_f = os.path.join(DATA_DIR, domain, 'neutral.txt')

        # load both files to memory
        positive_documents = [line.strip() for line in open(positive_f,encoding="utf-8")]
        negative_documents = [line.strip() for line in open(negative_f,encoding="utf-8")]
        neutral_documents = [line.strip() for line in open(neutral_f,encoding="utf-8")]
        total_documents = positive_documents + negative_documents+neutral_documents
        random.shuffle(total_documents)
        #total_documents = total_documents[:15131]

        if not vocab:
            # read the vocab
            t2i, i2t, vocab_count = create_vocab(total_documents)
        else:
            t2i, i2t, vocab_count = vocab['t2i'], vocab['i2t'], vocab['vocab_count']

        # create the tf-idf representation for all the documents
        X, y = tfidf_docs(total_documents, t2i, i2t, vocab_count)

        return {'inputs': X, 'labels': y}, {'t2i': t2i, 'i2t': i2t, 'vocab_count': vocab_count}


    # read the unlabeled documents to X
    def read_unlabeled_file(domain, vocab):
        print("processing the unlabeled %s domain" % domain)

        # file paths
        unlab_f = os.path.join(DATA_DIR, domain, 'unlabeled.txt')

        # load the content to memory
        unlab_documents = [line.strip() for line in open(unlab_f, encoding="utf-8")]
        random.shuffle(unlab_documents)
        #unlab_documents = unlab_documents[0:2000]

        t2i, i2t, vocab_count = vocab['t2i'], vocab['i2t'], vocab['vocab_count']

        # create the tf-idf representation for all the unlabeled documents
        X, _ = tfidf_docs(unlab_documents, t2i, i2t, vocab_count,do_unlabeled=True)

        return {'inputs': X}



    #hindi_mixed = English semeval corpus, mixed with Hindi translated+transliterated version
    #hindi_twitter = codemixed hindi downloaded data from twitter (some transliterated) [labeled here coming from this years semeval]
    #hindi_sentimix = codemixed hindi from this years semeval competition
    # read the labeled data from source domain
    source_labeled_data, source_vocab = read_labeled_dir("mixed")

    # read the labeled data from target domain
    target_labeled_data, _ = read_labeled_dir("hindi_twitter", vocab=source_vocab)

    # read the unlabeled data from target domain
    target_unlabeled_data = read_unlabeled_file("hindi_twitter", source_vocab)

    '''
    Create the PyTorch modules
    '''


    # gradient reversal layer
    class ReverseLayerF(Function):
        @staticmethod
        def forward(ctx, x, lmbda):
            ctx.lmbda = lmbda
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.lmbda
            return output, None


    # DANN layers
    class DANN(nn.Module):
        def __init__(self, d, h, lmbda):
            super(DANN, self).__init__()
            self.lmbda = lmbda
            self.input_to_hidden = nn.Linear(d, h)
            self.class_classifier = nn.Linear(h, 2)  # classes: positive vs negative
            self.domain_classifier = nn.Linear(h, 2)  # classes: source vs target

        def forward(self, input_data):
            hidden_rep = self.input_to_hidden(input_data)
            class_output = self.class_classifier(hidden_rep)
            reverse_feature = ReverseLayerF.apply(hidden_rep, self.lmbda)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output


    '''
    create the model instance, optimizer, loss and so on
    '''

    # hyperparameters for training the model
    #ALPHA = 0.001  # learning rate
    ALPHA = ALP
    #HIDDEN_SIZE = 200  # search space [1, 5, 12, 25, 50, 75, 100, 150, 200]

    #HIDDEN_SIZE = 1  # search space [1, 5, 12, 25, 50, 75, 100, 150, 200]
    HIDDEN_SIZE = H
    BATCH_SIZE = BS
    EPOCHS = 5
    #LAMBDA = 0.1  # search space among 9 values between 10^{−2} and 1 on a logarithmic scale
    LAMBDA = LAM

    # create the model instance
    n, d = source_labeled_data["inputs"].shape
    print("num: "+str(n) +" dim:"+str(d))
    model = DANN(d, HIDDEN_SIZE, LAMBDA)
    model.to(device)

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    # setup both the loss
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    # ensure all the parameters of the model are learnable
    for p in model.parameters():
        p.requires_grad = True

    '''
    train the model
    '''

    # collect the training data
    # TODO: process the inputs as sparse array instead of making them dense
    X_src = torch.from_numpy(source_labeled_data["inputs"].toarray()).float()
    y_src_class = torch.from_numpy(source_labeled_data["labels"])
    X_targ = torch.from_numpy(target_unlabeled_data["inputs"].toarray()).float()

    # placeholders for holding the current batch
    cur_X_src = torch.zeros(BATCH_SIZE, d, device=device)
    cur_y_src_class = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    cur_y_src_domain = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    cur_X_targ = torch.zeros(BATCH_SIZE, d, device=device)
    cur_y_targ_domain = torch.ones(BATCH_SIZE, dtype=torch.long, device=device)

    # start training
    print('training...')
    model.train()
    num_batches = n // BATCH_SIZE
    print(str(num_batches)+" number of batches")
    for epoch in trange(EPOCHS):
        rand_idx = np.random.permutation(n)
        for bi in range(num_batches):
            # prepare batch
            for sample_i in range(BATCH_SIZE):
                cur_idx = rand_idx[BATCH_SIZE * bi + sample_i]
                cur_X_src[sample_i] = X_src[cur_idx]
                cur_y_src_class[sample_i] = y_src_class[cur_idx]
                cur_X_targ[sample_i] = X_targ[cur_idx]
            # train the model using this batch
            model.zero_grad()  # clears the gradient buffer
            # source side losses
            class_output, domain_output = model(cur_X_src)
            error_src_class = loss_class(class_output, cur_y_src_class)
            error_src_domain = loss_domain(domain_output, cur_y_src_domain)
            # target side losses
            _, domain_output = model(cur_X_targ)
            error_src_domain = loss_domain(domain_output, cur_y_targ_domain)
            # total losses
            error_this_batch = error_src_class + error_src_domain + error_src_domain
            # backward prop.
            error_this_batch.backward()
            optimizer.step()

    '''
    evaluate the model
    '''

    # collect the evaluation data
    # TODO: process the inputs as sparse array instead of making them dense
    X_targ = torch.from_numpy(target_labeled_data["inputs"].toarray()).float()
    y_targ_class = torch.from_numpy(target_labeled_data["labels"])

    # placeholders for holding the current batch
    cur_X_targ = torch.zeros(BATCH_SIZE, d, device=device)
    cur_y_targ_class = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

    model.eval()
    num_test_instances = X_targ.shape[0]
    num_batches = num_test_instances // BATCH_SIZE
    errors = 0.0
    with torch.no_grad():
        for bi in range(num_batches):
            # prepare batch
            for sample_i in range(BATCH_SIZE):
                cur_idx = BATCH_SIZE * bi + sample_i
                cur_X_targ[sample_i] = X_targ[cur_idx]
                cur_y_targ_class[sample_i] = y_targ_class[cur_idx]
            pred_class_output, _ = model(cur_X_targ)
            # update errors
            for sample_i in range(BATCH_SIZE):
                cur_label = 0 if pred_class_output[sample_i][0] > pred_class_output[sample_i][1] else 1
                if cur_y_targ_class[sample_i] != cur_label:
                    errors += 1.0
    print("evaluation error in target labeled samples = %.3f" % (errors / num_test_instances))
    print("for params LAMBDA =" + str(LAM)+ " Vocab Size=" +str(VOC) +" Hidden size="+str(H))
    print("Time taken:"+str(datetime.datetime.now() - tm))
    return ((errors/num_test_instances),str(LAM)+"_"+str(VOC)+"_"+str(H))