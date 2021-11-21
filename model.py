import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_size = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def hidden_layer(self, batch_size):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embed_size(captions)
        
        embed_concatenated = torch.cat((features.unsqueeze(1), embed),1)
        lstm_output, self.hidden = self.lstm(embed_concatenated)
        voc_out = self.fc(lstm_output)
        return voc_out 
        
    def sample(self, inputs, states=None, max_len= 15):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res=[]
        for i in range(max_len):
            lstm_output, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_output)
            outputs = outputs.squeeze(1)
            wordid = outputs.argmax(1)
            res.append(wordid.item())
            inputs = self.embed_size(wordid).unsqueeze(1)
        return res 