import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class Gaussian_Orthogonal_MultiHeadSA(nn.Module):
    def __init__(self,para):
        self.p = para
        super(Gaussian_Orthogonal_MultiHeadSA, self).__init__()
        self.B = para.B # batch size
        self.D = para.D # seq dim
        self.E = para.head_nums
        self.theta = torch.nn.Parameter(torch.tensor([para.init_gaussion_theta]), requires_grad=True)
        self.D_E = self.D // para.head_nums
        self.is_punish = para.is_punish
        self.query_ = nn.Linear(self.D,self.D)
        self.key_ = nn.Linear(self.D, self.D)
        self.value_ = nn.Linear(self.D, self.D)
        self.linear_o_ = nn.Linear(self.D, self.D)
        self.softmax = nn.Softmax(2)
        self.this_batch_padlen = 1653
        self.gaussian_matrix = Gaussian_matrix(self.this_batch_padlen, self.this_batch_padlen, scaling=0,
                                          device=torch.device('cuda')).cuda()
        self.learnable_sigmoid = Learnable_sigmoid()

    def forward(self,input):
        Q = self.query_(input).permute(0,2,1).reshape(self.B*self.E,self.this_batch_padlen,self.D_E)
        K = self.key_(input).permute(0,2,1).reshape(self.B * self.E,self.this_batch_padlen,self.D_E).permute(0,2,1)
        V = self.value_(input).permute(0,2,1).reshape(self.B * self.E,self.this_batch_padlen,self.D_E)

        mask_q = torch.ones(self.B*self.E,self.this_batch_padlen,self.D_E,device=Q.device)

        Q = Q.mul(mask_q)
        K = K.mul(mask_q.transpose(2,1))
        Q = self.learnable_sigmoid(Q)
        K = self.learnable_sigmoid(K)

        att_score = torch.matmul(Q,K)/torch.sqrt(torch.tensor(self.this_batch_padlen,device=Q.device).float())

        if self.is_punish:
            punish_matrix = self.gaussian_matrix()
            punish_matrix = punish_matrix.repeat(self.B*self.E,1,1)
            att_score_orthogonal = torch.nn.init.orthogonal_(att_score, gain=1).permute(0,2,1)

            att_score = torch.mul(att_score,punish_matrix) 
            att_score = torch.matmul(att_score,att_score_orthogonal)

        att_score = att_score.masked_fill(att_score == 0.0, -2e20)

        A =self.linear_o_(torch.matmul(att_score,V).permute(0,2,1).reshape(self.B,self.D,self.this_batch_padlen).permute(0,2,1))  # re col shape

        return A

    def example_plot(self,ax, input, fontsize=12, hide_labels=False):
        # pc = ax.pcolormesh(np.random.randn(10, 10), vmin=-2.5, vmax=2.5)
        pc = ax.pcolormesh(input, vmin=-2.5, vmax=2.5)

        if not hide_labels:
            ax.set_xlabel('x-label', fontsize=fontsize)
            ax.set_ylabel('y-label', fontsize=fontsize)
            ax.set_title('Title', fontsize=fontsize)
        return pc

    def dispaly_feature_color(self,input, title="none"):
        fig = plt.figure(constrained_layout=True, figsize=(10, 5))

        axsLeft = fig.subplots(1, 2, sharey=True)
        fig.set_facecolor('0.75')
        fig.set_facecolor('w')

        ####
        input = input[0]  # delete batch_size
        input = input.cpu().detach().numpy()
        ####
        for ax in axsLeft:
            pc = self.example_plot(ax, input)

        fig.suptitle('Figure suptitle', fontsize='xx-large')
        plt.savefig(title + "_img.png")
        plt.show()


class Learnable_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.2 / (1 + torch.exp(-(1.6) * x))

class attention_hparams():
    def __init__(self, B=10, D=1024):
        self.head_nums = 4
        self.D = D
        self.D_E = self.D // self.head_nums
        self.E = self.head_nums
        self.B = B
        self.init_gaussion_theta: float = 1.5
        self.batch_size = self.B
        self.is_punish = True

class Gaussian_matrix(nn.Module):
    def __init__(self,nb_rows,nb_columns,scaling,device):
        super(Gaussian_matrix, self).__init__()
        self.device = device
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.scaling = scaling

    def forward(self):
        nb_full_blocks = int(self.nb_rows / self.nb_columns)

        block_list = []

        for _ in range(nb_full_blocks):
            q = self.orthogonal_matrix_chunk(self.nb_columns, device = self.device)
            block_list.append(q)

        remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
        if remaining_rows > 0:
            q = self.orthogonal_matrix_chunk(self.nb_columns, device = self.device)
            block_list.append(q[:remaining_rows])

        final_matrix = torch.cat(block_list)

        if self.scaling == 0:
            multiplier = torch.randn((self.nb_rows, self.nb_columns), device = self.device).norm(dim = 1)
        elif self.scaling == 1:
            multiplier = math.sqrt((float(self.nb_columns))) * torch.ones((self.nb_rows,), device = self.device)
        else:
            raise ValueError(f'Invalid scaling {self.scaling}')
        return torch.diag(multiplier) @ final_matrix

    def orthogonal_matrix_chunk(self,cols, device = None):
        unstructured_block = torch.randn((cols, cols), device = device)
        q, r = torch.qr(unstructured_block.cpu(), some = True)
        q, r = map(lambda t: t.to(device), (q, r))
        return q.t()



class MLP(nn.Module):
    def __init__(self,in_features,hidden_features = None,out_features = None,act_layer = nn.GELU,dropout = 0.1):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return  x

class AMB(nn.Module): # attention with mlp Block
    def __init__(self,batch_size=None,channels=None,dims=None):
        super(AMB, self).__init__()
        self.attention_msha = Gaussian_Orthogonal_MultiHeadSA(
            para=attention_hparams(B=batch_size,D=channels * dims))  # D=channels * dims)
        self.mlp2 = MLP(in_features=channels * dims, hidden_features=int(4.0 * channels * dims), act_layer=nn.GELU,
                   dropout=0.1)
        self.LayerNorm = nn.LayerNorm

    def forward(self,x):
        # print(x.shape, "1") #[2, 1653, 1024]
        att = self.LayerNorm(x.shape[-1], eps=1e-6, elementwise_affine=False)(x)
        att = self.attention_msha(att)
        att_x = att + x
        return att_x

class MAMB(nn.Module): # attention with mlp Block
    def __init__(self,batch_size=None,att_dim=None):
        super(MAMB, self).__init__()
        self.attention_msha = Gaussian_Orthogonal_MultiHeadSA(
            para=attention_hparams(B=batch_size,D=att_dim))  # D=channels * dims)
        self.LayerNorm = nn.LayerNorm

    def forward(self,x):
        # x = torch.complex()
        print(x.shape)
        x = self.LayerNorm(x.shape[-1],eps=1e-6,elementwise_affine=False)(x)
        print(x.shape,"2")
        att = self.attention_msha(x)
        att_x = att + x

        return att_x

class cAMB(nn.Module): # attention with mlp Block
    def __init__(self,batch_size=None,channels=None,dims=None):
        super(cAMB, self).__init__()
        self.c_channels = channels
        self.complex_att = AMB(batch_size=batch_size,channels=channels//2,dims=dims)

    def forward(self,x):
        if isinstance(x, list): 
            real, imag = x
        elif isinstance(x, torch.Tensor):
            real, imag = torch.chunk(x, 2)

        r2r_out = self.complex_att(real)[0]  # real + real lstm
        r2i_out = self.complex_att(real)[0]  # real + imag lstm
        i2r_out = self.complex_att(imag)[0]  # imag + real lstm
        i2i_out = self.complex_att(imag)[0]  # imag + imag lstm

        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        output = [real_out, imag_out]

        return output