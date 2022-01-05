import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from IPython import display
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)

# def use_svg_display():
#     """Use the svg format to display a plot in Jupyter."""
#     display.set_matplotlib_formats('svg')

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    #use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

def get_word_dict(english,chinese):
    eng_to_index = {"<PAD>":0}
    index_to_eng = ["<PAD>"]
    for word in english:
        for e in word:
            if e not in eng_to_index:
                eng_to_index[e] = len(eng_to_index)
                index_to_eng.append(e)

    ch_to_index = {"<PAD>":0,"<BEG>":1,"<END>":2}
    index_to_ch = ["<PAD>","<BEG>","<END>"]
    for word in chinese:
        for e in word:
            if e not in ch_to_index:
                ch_to_index[e] = len(ch_to_index)
                index_to_ch.append(e)

    return eng_to_index,index_to_eng,ch_to_index,index_to_ch

class MyModel(nn.Module):
    def __init__(self,eng_to_index, index_to_eng, ch_to_index, index_to_ch,embedding_num,hidden_num):
        super().__init__()
        self.eng_to_index = eng_to_index
        self.index_to_eng = index_to_eng
        self.ch_to_index = ch_to_index
        self.index_to_ch = index_to_ch

        self.encoder = nn.LSTM(input_size=embedding_num,hidden_size=hidden_num,num_layers=1,bidirectional=False,batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_num+hidden_num,hidden_size=hidden_num,num_layers=1,bidirectional=False,batch_first=True)
        self.linear = nn.Linear(hidden_num,len(ch_to_index)+3)
        self.corss_loss = nn.CrossEntropyLoss()

    def forward(self,encoder_input,decoder_input,label,decoder_hidden=None):
        encoder_output,encoder_hidden = self.encoder(encoder_input)
        if decoder_hidden == None:
            decoder_hidden = encoder_hidden[0]
        query = decoder_hidden.permute(1,0,2)
        key = encoder_output
        attweight = torch.softmax(torch.sum(query*key,dim=-1).unsqueeze(-1),dim=1)
        value = torch.bmm(attweight.permute(0,2,1),encoder_output)
        context = torch.repeat_interleave(value, decoder_input.shape[1], dim=1)
        feature = torch.cat((decoder_input, context), dim=2)

        decoder_output, (decoder_hidden, _) = self.decoder(feature, encoder_hidden)
        pre = self.linear(decoder_output)
        loss = self.corss_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))

        return loss,attweight,decoder_hidden

class MyDataSet(Dataset):
    def __init__(self,english,chinese,eng_to_index,ch_to_index,embedding_num,eng_max_len,ch_max_len):
        self.english = english
        self.chinese = chinese
        self.eng_to_index = eng_to_index
        self.ch_to_index = ch_to_index
        self.eng_max_len = eng_max_len
        self.ch_max_len = ch_max_len

        self.eng_embedding = nn.Embedding(len(eng_to_index),embedding_num)
        self.ch_embedding = nn.Embedding(len(ch_to_index),embedding_num)

    def __getitem__(self,index):
        e_data = self.english[index]
        c_data = self.chinese[index]

        e_index = torch.tensor([self.eng_to_index[i] for i in e_data] + [self.eng_to_index["<PAD>"]]* (self.eng_max_len - len(e_data)))
        c_index = torch.tensor([self.ch_to_index["<BEG>"]]+[self.ch_to_index[i] for i in c_data]+ [self.ch_to_index["<END>"]] + [self.ch_to_index["<PAD>"]]* (self.ch_max_len - len(c_data)))

        e_embedding = self.eng_embedding(e_index)
        c_embedding = self.ch_embedding(c_index)

        return e_embedding,c_embedding[:-1],c_index[1:]

    def __len__(self):
        return len(self.english)

def translate(mydataset,ch_to_index,index_to_ch):
    print("****************************")
    test_dataloader = DataLoader(mydataset,1, False)
    for eng_embedding,_,_ in test_dataloader:
        result = []
        encoder_output,encoder_hidden = model.encoder(eng_embedding)
        decoder_input = mydataset.ch_embedding(torch.tensor(ch_to_index["<BEG>"])).reshape(1,1,-1)
        decoder_hid = encoder_hidden
        while True:

            query = decoder_hid[0].permute(1, 0, 2)
            key = encoder_output
            attweight = torch.softmax(torch.sum(query*key,dim=-1).unsqueeze(-1), dim=1)
            value = torch.bmm(attweight.permute(0, 2, 1), encoder_output)
            context = torch.repeat_interleave(value, decoder_input.shape[1], dim=1)
            feature = torch.cat((decoder_input, context), dim=2)

            decoder_output, decoder_hid = model.decoder(feature, decoder_hid)
            pre = model.linear(decoder_output)
            w_index = int(torch.argmax(pre,dim=-1))
            ch = index_to_ch[w_index]

            if ch == "<END>" or len(result)> 10:
                print("".join(result),end=" ")
                break
            result.append(ch)
            decoder_input = mydataset.ch_embedding(torch.tensor(w_index)).reshape(1,1,-1)
    print("\n****************************")

if __name__ == '__main__':
    english = ["apple","banana","orange","pear","black","red","white","pink","green","blue"]
    chinese = ["苹果","香蕉","橙子","梨子","黑色","红色","白色","粉红色","绿色","蓝色"]

    eng_to_index, index_to_eng, ch_to_index, index_to_ch = get_word_dict(english,chinese)
    eng_max_len = 6
    ch_max_len = 3
    embedding_num = 15
    hidden_num = 20
    batch_size = 2
    epoch = 200
    lr = 0.01

    mydataset = MyDataSet(english,chinese,eng_to_index,ch_to_index,embedding_num,eng_max_len,ch_max_len)
    dataloader = DataLoader(mydataset,batch_size,False)

    model = MyModel(eng_to_index, index_to_eng, ch_to_index, index_to_ch,embedding_num,hidden_num)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    decoder_hidden = None
    for e in range(epoch):
        trained_attn = []
        for eng_embedding,x_embedding,y_embedding in dataloader:
            loss,attweight,decoder_hidden = model(eng_embedding,x_embedding,y_embedding,decoder_hidden)
            decoder_hidden = decoder_hidden.detach()
            loss.backward()
            opt.step()
            opt.zero_grad()
            trained_attn.append(attweight.cpu().detach())
        print(f"loss:{loss:.3f}")
        translate(mydataset, ch_to_index, index_to_ch)

    a,b,c = attweight.shape
    matrix = torch.empty((a * len(trained_attn), b,c))

    for idx,t in enumerate(trained_attn):
        matrix[a*idx:a*(idx+1),:,:] = t

    show_heatmaps(matrix.reshape((1, 1, 10, 6)),
                      xlabel='Keys', ylabel='Queries')
