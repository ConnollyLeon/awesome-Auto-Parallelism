'''
An implementation of Bert_CLS model
'''

import torch
from torch import nn
from transformers import BertModel


class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size, vocab_size, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.output_bias


    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        return sequence_output


class BertLargeCls(nn.Module):
    def __init__(self, config):
        super().__init__()

        mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.max_predictions_per_seq = 80

        def get_masked_lm_loss(
                logit_blob,
                masked_lm_positions,
                masked_lm_labels,
                label_weights,
                max_predictions_per_seq,
        ):
            # gather valid position indices
            logit_blob = torch.gather(
                logit_blob,
                index=masked_lm_positions.unsqueeze(2).to(
                    dtype=torch.int64).repeat(1, 1, 30522),
                dim=1,
            )
            logit_blob = torch.reshape(logit_blob, [-1, 30522])
            label_id_blob = torch.reshape(masked_lm_labels, [-1])

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            pre_example_loss = mlm_criterion(logit_blob, label_id_blob.long())
            pre_example_loss = torch.reshape(
                pre_example_loss, [-1, max_predictions_per_seq])
            sum_label_weight = torch.sum(label_weights, dim=-1)
            sum_label_weight = sum_label_weight / label_weights.shape[0]
            numerator = torch.sum(pre_example_loss * label_weights)
            denominator = torch.sum(label_weights) + 1e-5
            loss = numerator / denominator
            return loss

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config.hidden_size, config.vocab_size)
        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.masked_lm_criterion = get_masked_lm_loss

    def forward(self, labels, id, pos, weight, inputs, return_outputs=False):
        outputs = self.bert(**inputs)
        prediction_scores, seq_relationship_scores = self.cls(
            outputs[0], outputs[1])  # last_hidden_state, pooler_output
        next_sentence_loss = self.ns_criterion(
            seq_relationship_scores.view(-1, 2), labels.long().view(-1)
        )
        masked_lm_loss = self.masked_lm_criterion(
            prediction_scores, pos, id, weight, max_predictions_per_seq=self.max_predictions_per_seq
        )

        total_loss = next_sentence_loss + masked_lm_loss
        return (total_loss, outputs) if return_outputs else total_loss