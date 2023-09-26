import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindformers.models.base_model import BaseModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.bert.bert_config import BertConfig
from mindformers import BertModel
from mindformers import BertForPreTraining
from mindformers import BaseModel


class BertForPreTraining2(BaseModel):


    _support_list = MindFormerBook.get_model_support_list()['bert']
    def __init__(self, config=BertConfig()):
        config.seq_length = 80
        super(BertForPreTraining2, self).__init__(config)
        # super().__init__()
        self.is_training = config.is_training
        # self.bert = BertScore(config, config.is_training, config.use_one_hot_embeddings)
        self.bert = BertModel(config, config.is_training, config.use_one_hot_embeddings)
        # self.loss = BertLoss(config)
        self.cast = P.Cast()
        # self.use_moe = (config.parallel_config.moe_config.expert_num > 1)
        self.add = P.Add().shard(((1,), ()))
        self.load_checkpoint(config)

    def bert_forward(self, input_ids, input_mask, token_type_id):
        """connect backbone and heads."""
        moe_loss = 0
        # if self.use_moe:
        #     sequence_output, pooled_output, embedding_table, moe_loss = \
        #         self.bert(input_ids, token_type_id, input_mask)
        # else:
        sequence_output, pooled_output, embedding_table = \
                self.bert(input_ids, token_type_id, input_mask)

        if not self.is_training:
            print("eval")
            return sequence_output, pooled_output
        return sequence_output , pooled_output

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels=None,
                  masked_lm_positions=None,
                  masked_lm_ids=None,
                  masked_lm_weights=None):
        """Get pre-training loss"""
        if not self.is_training:
            return self.bert_forward(input_ids, input_mask, token_type_id, masked_lm_positions)

        prediction_scores, seq_relationship_score, moe_loss = \
            self.bert_forward(input_ids, input_mask, token_type_id, masked_lm_positions)
        total_loss = self.loss(prediction_scores, seq_relationship_score,
                               masked_lm_ids, masked_lm_weights, next_sentence_labels)
        if self.use_moe:
            total_loss = self.add(total_loss, moe_loss)
        return self.cast(total_loss, mstype.float32)
