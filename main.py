import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from feature_extraction import (NUM_DEPS, SHIFT, DataConfig, Flags,
                                load_datasets, pos_prefix, punc_pos)
from general_utils import get_minibatches
from model import ParserModel
from test_functions import compute_dependencies, get_UAS, parse_sentence
from torch import nn

import pandas as pd


def load_embeddings(config, emb_type='new', emb_file_name=None):
    if emb_type == 'new':
        print('Creating new trainable embeddings')
        word_embeddings = nn.Embedding(config.word_vocab_size,
                                       config.embedding_dim)
        pos_embeddings = nn.Embedding(config.pos_vocab_size,
                                      config.embedding_dim)
        dep_embeddings = nn.Embedding(config.dep_vocab_size,
                                      config.embedding_dim)
    elif emb_type == 'twitter':
        pass
    elif emb_type == 'wiki' or emb_type == 'wikipedia':
        pass
    else:
        raise Error('unknown embedding type!: "%s"' % emb_type)

    return word_embeddings, pos_embeddings, dep_embeddings


def train(save_dir='saved_weights',
          parser_name='parser',
          num_epochs=5,
          max_iters=-1,
          print_every_iters=10):
    """
    Trains the model.

    parser_name is the string prefix used for the filename where the parser is
    saved after every epoch
    """

    # load dataset
    load_existing_dump = False
    print('Loading dataset for training')
    dataset = load_datasets(load_existing_dump)
    config = dataset.model_config

    print('Loading embeddings')
    word_embeddings, pos_embeddings, dep_embeddings = load_embeddings(config)

    if False:
        # Switch to True if you want to print examples of feature types
        print('words: ', len(dataset.word2idx))
        print('examples: ', [(k, v)
                             for i, (k,
                                     v) in enumerate(dataset.word2idx.items())
                             if i < 30])
        print('\n')
        print('POS-tags: ', len(dataset.pos2idx))
        print(dataset.pos2idx)
        print('\n')
        print('dependencies: ', len(dataset.dep2idx))
        print(dataset.dep2idx)
        print('\n')
        print("some hyperparameters")
        print(vars(config))

    # load parser object
    parser = ParserModel(config, word_embeddings, pos_embeddings,
                         dep_embeddings)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser.to(device)

    # set save_dir for model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create object for loss function
    loss_fn = nn.CrossEntropyLoss()


    # create object for an optimizer that updated the weights of parser model.  
    optimizer = torch.optim.SGD(parser.parameters(), lr=config.lr)  
    
    loss_list = []
    acc_list = []
    uas_list = []
    for epoch in range(1, num_epochs + 1):

        ###### Training #####

        # load training set in minibatches
        for i, (train_x, train_y) in enumerate(get_minibatches([dataset.train_inputs,
                                                                dataset.train_targets], \
                                                               config.batch_size,
                                                               is_multi_feature_input=True)):

            word_inputs_batch, pos_inputs_batch, dep_inputs_batch = train_x

            # Convert the numpy data to pytorch's tensor represetation.  They're
            # numpy objects initially.  
            word_inputs_batch = torch.tensor(word_inputs_batch).to(device)
            pos_inputs_batch = torch.tensor(pos_inputs_batch).to(device)
            dep_inputs_batch = torch.tensor(dep_inputs_batch).to(device)

            # Convert the labels from 1-hot vectors to a list of which index was
            # 1, which is what Pytorch expects.  
            labels = np.argmax(train_y, axis=1)  

            # Convert the label to pytorch's tensor
            labels = torch.tensor(labels)  

            if max_iters >= 0 and i > max_iters:
                break
            if i == 0 and epoch == 1:
                print("size of word inputs: ", word_inputs_batch.size())
                print("size of pos inputs: ", pos_inputs_batch.size())
                print("size of dep inputs: ", dep_inputs_batch.size())
                print("size of labels: ", labels.size())

          
            #### Backprop & Update weights ####

            # Before the backward pass, use the optimizer object to zero all of
            # the gradients for the variables

            optimizer.zero_grad()

            # For the current batch of inputs, run a full forward pass through the
            # data and get the outputs for each item's prediction.
            # These are the raw outputs, which represent the activations for
            # prediction over valid transitions.

            outputs = parser(word_inputs_batch, pos_inputs_batch, dep_inputs_batch) # TODO

            # Compute the loss for the outputs with the labels.  

            loss = None  
            loss = loss_fn(outputs, labels)
    
            # Backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()
            # Perform 1 update using the optimizer


            optimizer.step()
            # Every 10 batches, print out some reporting so I can see convergence
            if i % print_every_iters == 0:
                print ('Epoch: %d [%d], loss: %1.3f, acc: %1.3f' \
                       % (epoch, i, loss.item(),
                          int((outputs.argmax(1)==labels).sum())/len(labels)))

        print("End of epoch")

        # save model
        save_file = os.path.join(save_dir, '%s-epoch-%d.mdl' % (parser_name,
                                                                epoch))
        print('Saving current state of model to %s' % save_file)
        torch.save(parser, save_file)

        ###### Validation #####
        print('Evaluating on valudation data after epoch %d' % epoch)

        # Once we're in test/validation time, we need to indicate that we are in
        # "evaluation" mode.  This will turn off things like Dropout so that
        # we're not randomly zero-ing out weights when it might hurt performance
        parser.eval()

        # Compute the current model's UAS score on the validation (development)
        # dataset. 
        compute_dependencies(parser, device, dataset.valid_data, dataset)
        valid_UAS = get_UAS(dataset.valid_data)
        print("- validation UAS: {:.2f}".format(valid_UAS * 100.0))
        loss_list.append(loss.item())
        acc_list.append(int((outputs.argmax(1)==labels).sum())/len(labels))
        uas_list.append(valid_UAS * 100.0)

        # Once we're done with test/validation, we need to indicate that we are back in
        # "train" mode.  This will turn back on things like Dropout
        parser.train()

    score = pd.DataFrame({'loss': loss_list, 'acc': acc_list, 'uas': uas_list})
    score.to_csv(r"score.csv", index = True, header = True) 

    return parser


def test(parser):

    # load dataset
    print('Loading data for testing')
    dataset = load_datasets()
    config = dataset.model_config
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Make sure the parser is in evaluation mode so it's not using things like dropout
    parser.eval()

    # Compute UAS (unlabeled attachment score), which is the standard evaluate metric for parsers.
    #
    # For details see
    # http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002
    # Chapter 6.1
    compute_dependencies(parser, device, dataset.test_data, dataset)
    valid_UAS = get_UAS(dataset.test_data)
    print("- test UAS: {:.2f}".format(valid_UAS * 100.0))

    parser.eval()
    test_string = "I shot an elephant with a banana"
    parse_sentence(test_string, parser, device, dataset)

def parse_example(parser, sentence):

    # load dataset
    print('Loading embeddings and ids for parsing')
    dataset = load_datasets()
    config = dataset.model_config
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Make sure the parser is in evaluation mode so it's not using things like dropout
    parser.eval()

    parse_sentence(sentence, parser, device, dataset)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--max_train_iters",
        help="Maximum training " + "iterations during one epoch (debug only)",
        type=int,
        default=-1,
        required=False)
    argparser.add_argument(
        "--parser_name",
        help="Name used to save parser",
        type=str,
        default="parser",
        required=False)
    argparser.add_argument(
        "--num_epochs",
        help="Number of epochs",
        type=int,
        default=50,
        required=False)
    argparser.add_argument(
        "--print_every_iters",
        help="How often to print " + "updates during training",
        type=int,
        default=50,
        required=False)
    argparser.add_argument(
        "--train", help="Train the model", action='store_true')
    argparser.add_argument(
        "--test", help="Test the model", action='store_true')
    argparser.add_argument(
        "--load_model_file",
        help="Load the specified " + "saved model for testing",
        type=str,
        default=None)
    argparser.add_argument(
        "--parse_sentence",
        help="Parses the example sentence using a trained parser",
        type=str,
        required=False)    

    args = argparser.parse_args()
    parser = None
    if args.train:
        parser = train(
            max_iters=args.max_train_iters,
            num_epochs=args.num_epochs,
            parser_name=args.parser_name,
            print_every_iters=args.print_every_iters)
    if args.test:
        if parser is None or args.load_model_file is not None:
            # load parser object
            print('Loading saved parser for testing')
            load_file = args.load_model_file

            if load_file is None:
                # Back off to see if we can keep going
                load_file = 'saved_weights/parser-epoch-1.mdl'

            print('Testing using model saved at %s' % load_file)
            parser = torch.load(load_file)
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
            parser.to(device)

        test(parser)
    if args.parse_sentence:
        if parser is None or args.load_model_file is not None:
            # load parser object
            print('Loading saved parser for testing')
            load_file = args.load_model_file

            if load_file is None:
                # Back off to see if we can keep going
                load_file = 'saved_weights/parser-epoch-1.mdl'

            print('Testing using model saved at %s' % load_file)
            parser = torch.load(load_file)
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
            parser.to(device)

        parse_example(parser, args.parse_sentence)

    if not (args.train or args.test or args.parse_sentence):
        print('None of --train, --test, or --parse_sentence specified! Doing nothing...')
        argparser.print_usage()
