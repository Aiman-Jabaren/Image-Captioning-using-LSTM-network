from freshDecoder import *
from encoder import *
from fresh_data_loader import *
import pickle
import random
import torch.optim as optim
from torch.autograd import Variable
import csv
import time
from tqdm import tqdm
import gc
import os
import torchvision.transforms as tf
import json
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from matplotlib import pyplot as plt
import sys




def validate_test(val_loader, encoder, decoder, criterion, maxSeqLen,
             vocab, batch_size, use_gpu = True, calculate_bleu = True):

    save_generated_imgs = True
    #Evaluation Mode
    decoder.eval()
    encoder.eval()

    
    references = list()
    hypotheses = list() 
   
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
        
    with torch.no_grad():
        
        count    = 0
        loss_avg = 0
        bleu1_avg = 0
        bleu4_avg = 0
                
        for i, (inputs, caps, allcaps) in enumerate(val_loader):
            
            
            
            # Move to device, if available
            if use_gpu:
                inputs = inputs.to(device)
                caps = caps.to(device)

                        
            enc_out = encoder(inputs)
            actual_lengths = allcaps
            
            
            
            temperature = 1
            test_pred = decoder.generate_caption(enc_out, maxSeqLen, temperature)

            test_pred_sample = test_pred[0].cpu().numpy()          
        
            k = 0
            for b in range(inputs.shape[0]):
                caption = (" ").join([vocab.idx2word[x.item()] for x in test_pred[b]])
                img = tf.ToPILImage()(inputs[b,:,:,:].cpu())
                plt.imshow(img)
                    
                plt.show()
                print("Caption: " + caption)
                if save_generated_imgs:
                    file = "./generated_imgs/" + "test_im_"+ str(k) 
                    img.save(file + ".png", "PNG")
                    k+=1
                    #with open(generated_imgs_filename, "a") as file:
                        #file.write("writing! " + "train_epoch" + str(epoch) + "im_"+ str(k) + "\n")            
                        #file.write("Caption: " + caption +"\n \n")
                    
                    
                    
            

            
            #Build a list of the predicted sentences
            # Convert word_ids to words
            sampled_caption = []

            for word_id in test_pred_sample:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            hypotheses.append(sampled_caption) 

            
            #print('i: ', i)
            #print('len(sampled_caption): ',len(sampled_caption))
             
            
            decoder.resetHidden(inputs.shape[0])
            outputs = decoder(caps, enc_out, actual_lengths)
            
            loss = criterion(outputs, Variable(caps.long()))
            loss_avg += loss
            count+=1
            
            #del outputs            
            
            #print('VAL: loss: ', loss)


            caps_array = caps.cpu().numpy()  
            # Convert word_ids to words
            reference_caption = []
            sampled_caption = []
            
            for word_id in caps_array[0]:
                word = vocab.idx2word[word_id]
                reference_caption.append(word)
                if word == '<end>':
                    break
            ref_sentence = ' '.join(reference_caption)
            #if i % 500 == 0:
                #print('ref_sentence: ', ref_sentence)
                #print('len(ref_sentence): ',len(reference_caption))
            references.append(reference_caption)   
            #print('len(reference_caption): ',len(reference_caption))
        
            #print('len(references)', len(references))
            #print('len(hypotheses)', len(hypotheses))
            #print('references: ', references)
            #print('hypotheses: ', hypotheses)
      
            del caps
            del outputs            
            
            
            #if i % 10 == 0:
            #    break
             
        # Calculate BLEU-4 scores
        if calculate_bleu:
            #TODO
            #print('len(references)',len(references))
            #print('len(hypotheses)',len(hypotheses))
            bleu4 = corpus_bleu(references, hypotheses)                
            bleu1 = corpus_bleu(references, hypotheses,weights=(1.0, 0, 0, 0))
            #bleu4 = corpus_bleu(reference_caption, sampled_caption)                
            #bleu1 = corpus_bleu(reference_caption, sampled_caption,weights=(1.0, 0, 0, 0))
            #print('bleu4: ', bleu4)        
            #print('bleu1: ', bleu1)  
#            bleu4_avg+=bleu4
#            bleu1_avg+=bleu1
                            
                
        loss_avg  = loss_avg/count
        print('VAL: loss_avg: ', loss_avg)

        if calculate_bleu:
            
            bleu4_avg = bleu4
            bleu1_avg = bleu1 
            
            print('VAL: bleu4_avg: ', bleu4_avg)
            print('VAL: bleu1_avg: ', bleu1_avg)
        
        
        
            
    return loss_avg, bleu1_avg, bleu4_avg




if __name__=='__main__':
    name = "lstm"

    with open('TrainImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        trainIds = list(reader)[0]
        
    with open('TestImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        testIds = list(reader)[0]

        
    if len(sys.argv) > 1:
        name = sys.argv[1]
        
        
    trainIds = [int(i) for i in trainIds]
    testIds = [int(i) for i in testIds]
    
    # Will shuffle the trainIds incase of ordering in csv
    random.shuffle(trainIds)
    splitIdx = int(len(trainIds)/5)
    
    # Selecting 1/5 of training set as validation
    valIds = trainIds[:splitIdx]
    trainIds = trainIds[splitIdx:]
    
    
    trainValRoot = "./data/images/train/"
    testRoot = "./data/images/test/"
    
    trainValJson = "./data/annotations/captions_train2014.json"
    testJson = "./data/annotations/captions_val2014.json"
    
    
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    img_side_length = 256
    transform = tf.Compose([
        tf.Resize(img_side_length),
        #tf.RandomCrop(img_side_length),
        tf.CenterCrop(img_side_length),
        tf.ToTensor(),
    ])
    batch_size = 10
    shuffle = True
    num_workers = 5
    
    
    trainDl = get_loader(trainValRoot, trainValJson, trainIds, vocab, 
                         transform=transform, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers)
    valDl = get_loader(trainValRoot, trainValJson, valIds, vocab, 
                         transform=transform, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers)
    testDl = get_loader(testRoot, testJson, testIds, vocab, 
                        transform=transform, batch_size=batch_size, 
                        shuffle=shuffle, num_workers=num_workers)
    
    encoded_feature_dim = 1024
    maxSeqLen = 56
    hidden_dim = 1500
    depth = 1
    
    encoder = Encoder(encoded_feature_dim)
    decoder = Decoder(encoded_feature_dim, hidden_dim, depth, vocab.idx, batch_size)    
    
    encoder = torch.load('weights_base/' + name + 'encoder_best')
    decoder = torch.load('weights_base/' + name + 'decoder_best')    
    criterion = nn.CrossEntropyLoss()
    
    
    
    val_loss, val_bleu1, val_bleu4  = validate_test(testDl, encoder, decoder, criterion,maxSeqLen,
                             vocab, batch_size, use_gpu= True, calculate_bleu = True) 
