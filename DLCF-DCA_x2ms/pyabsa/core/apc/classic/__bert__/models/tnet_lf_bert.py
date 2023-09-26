
from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class Absolute_Position_Embedding(nn.Cell):
    def __init__(self, opt, size=None, mode='sum'):
        self.opt = opt
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def construct(self, x, pos_inx):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x2ms_adapter.tensor_api.x2ms_size(x, -1))
        batch_size, seq_len = x2ms_adapter.tensor_api.x2ms_size(x)[0], x2ms_adapter.tensor_api.x2ms_size(x)[1]
        weight = x2ms_adapter.to(self.weight_matrix(pos_inx, batch_size, seq_len), self.opt.device)
        x = x2ms_adapter.tensor_api.unsqueeze(weight, 2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = x2ms_adapter.tensor_api.numpy(pos_inx)
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)
        weight = x2ms_adapter.x2ms_tensor(weight)
        return weight


class TNet_LF_BERT(nn.Cell):
    inputs = ['text_indices', 'aspect_indices', 'aspect_boundary']

    def __init__(self, bert, opt):
        super(TNet_LF_BERT, self).__init__()
        print("this is TNet_LF model")
        self.embed = bert
        self.position = Absolute_Position_Embedding(opt)
        self.opt = opt
        D = opt.embed_dim  # 模型词向量维度
        C = opt.polarities_dim  # 分类数目
        L = opt.max_seq_len
        HD = opt.hidden_dim
        self.lstm1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.convs3 = x2ms_nn.Conv1d(2 * HD, 50, 3, padding=1)
        self.fc1 = x2ms_nn.Linear(4 * HD, 2 * HD)
        self.fc = x2ms_nn.Linear(50, C)

    def construct(self, inputs):
        text_raw_indices, aspect_indices, aspect_in_text = inputs[0], inputs[1], inputs[2]
        feature_len = x2ms_adapter.x2ms_sum(text_raw_indices != 0, dim=-1)
        aspect_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1)
        feature = self.embed(text_raw_indices)['last_hidden_state']
        aspect = self.embed(aspect_indices)['last_hidden_state']
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = x2ms_adapter.tensor_api.transpose(v, 1, 2)
        e = x2ms_adapter.tensor_api.transpose(e, 1, 2)
        for i in range(2):
            a = x2ms_adapter.bmm(x2ms_adapter.tensor_api.transpose(e, 1, 2), v)
            a = x2ms_adapter.nn_functional.softmax(a, 1)  # (aspect_len,context_len)
            aspect_mid = x2ms_adapter.bmm(e, a)
            aspect_mid = x2ms_adapter.tensor_api.transpose(x2ms_adapter.cat((aspect_mid, v), dim=1), 1, 2)
            aspect_mid = x2ms_adapter.nn_functional.relu(x2ms_adapter.tensor_api.transpose(self.fc1(aspect_mid), 1, 2))
            v = aspect_mid + v
            v = x2ms_adapter.tensor_api.transpose(self.position(x2ms_adapter.tensor_api.transpose(v, 1, 2), aspect_in_text), 1, 2)
            e = x2ms_adapter.tensor_api.x2ms_float(e)
            v = x2ms_adapter.tensor_api.x2ms_float(v)

        z = x2ms_adapter.nn_functional.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.nn_functional.max_pool1d(z, x2ms_adapter.tensor_api.x2ms_size(z, 2)), 2)
        out = self.fc(z)
        return out
