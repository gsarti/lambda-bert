"""
Classes for the LambdaBert model
Mostly similar to HuggingFace BERT, except for the Lambda class replacing self-attention
Lambda forward is a 1d adaptation of lucidrains implementation: https://github.com/lucidrains/lambda-networks/blob/main/.gitignore
Please raise an issue if you spot a bug in the computations!
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from configuration_lambdabert import LambdaBertConfig

from transformers.modeling_bert import (
    BertLMPredictionHead,
    BertSelfOutput, 
    BertIntermediate, 
    BertOutput, 
    BertEmbeddings, 
    BertPooler, 
    BertPreTrainingHeads, 
    BertForPreTrainingOutput
)
from transformers.modeling_albert import AlbertForPreTrainingOutput
from transformers.modeling_utils import apply_chunking_to_forward, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

class Lambda(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_lambda_queries != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of lambda query "
                "heads (%d)" % (config.hidden_size, config.num_lambda_queries)
            )
        self.lambda_query_size = config.hidden_size // config.num_lambda_queries
        self.intra_depth = config.intra_depth
        self.key_depth = config.key_depth
        self.heads = config.num_lambda_queries

        self.query = nn.Conv1d(config.hidden_size, self.key_depth * self.heads, 1, bias=False)
        self.key = nn.Conv1d(config.hidden_size, self.key_depth * self.intra_depth, 1, bias=False)
        self.value = nn.Conv1d(config.hidden_size, self.lambda_query_size * self.intra_depth, 1, bias=False)

        self.norm_q = nn.BatchNorm1d(self.key_depth * self.heads)
        self.norm_v = nn.BatchNorm1d(self.lambda_query_size * self.intra_depth)
        
        self.do_local_context = config.local_context_size is not None
        if self.do_local_context:
            if config.local_context_size % 2 != 1:
                raise ValueError(
                    f"Local context size must be odd. Current value: {config.local_context_size}" 
                )
            self.padding = config.local_context_size // 2
            self.rel_pos_embed = nn.Parameter(torch.randn(self.key_depth, self.intra_depth, 1, config.local_context_size))
        else:
            self.pos_embed = nn.Parameter(torch.randn(config.max_position_embeddings, config.max_position_embeddings, self.intra_depth, self.key_depth))

    def forward(
        self,
        x,
        token_mask=None,
        output_lambda_params=False
    ):
        n = x.shape[1]
        q, k, v = map(lambda hid: rearrange(hid, 'b n d -> b d n'), (x, x, x))
        q, k, v = self.query(q), self.key(k), self.value(v)

        q, v = self.norm_q(q), self.norm_v(v)

        q = rearrange(q, 'b (h k) n -> b h k n', h = self.heads)
        k = rearrange(k, 'b (k u) n -> b u k n', u = self.intra_depth)
        v = rearrange(v, 'b (v u) n -> b u v n', u = self.intra_depth)

        k = nn.Softmax(dim=-1)(k)

        λc = einsum('b u k n, b u v n -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b n h v', q, λc)
        if self.do_local_context:
            λp = F.conv2d(v, self.rel_pos_embed, padding = (0, self.padding))
            Yp = einsum('b h k n, b k v n -> b n h v', q, λp)
        else:
            λp = einsum('n n u k, b u v n -> b n k v', self.pos_embed[:n,:n,:,:], v)
            Yp = einsum('b h k n, b n k v -> b n h v', q, λp)

        Y = Yc + Yp
        Y = rearrange(Y, 'b n h v -> b n (h v)')

        outputs = (Y, λc) if output_lambda_params else (Y,)
        return outputs


class LambdaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = Lambda(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        token_mask=None,
        output_lambda_params=False,
    ):
        self_outputs = self.self(
            hidden_states,
            token_mask=token_mask,
            output_lambda_params=output_lambda_params
        )
        lambda_output = self.output(self_outputs[0], hidden_states)
        outputs = (lambda_output,) + self_outputs[1:] 
        return outputs


class LambdaBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lambdalayer = LambdaLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        token_mask=None,
        output_lambda_params=False,
    ):
        lambda_layer_outputs = self.lambdalayer(
            hidden_states,
            token_mask=token_mask,
            output_lambda_params=output_lambda_params,
        )
        lambda_output = lambda_layer_outputs[0]
        outputs = lambda_layer_outputs[1:] # add lambda weights if we output them

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, lambda_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, lambda_output):
        intermediate_output = self.intermediate(lambda_output)
        layer_output = self.output(intermediate_output, lambda_output)
        return layer_output


class LambdaBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LambdaBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        token_mask=None,
        output_lambda_params=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_lambda_params = () if output_lambda_params else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_lambda_params)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    token_mask=token_mask,
                    output_lambda_params=output_lambda_params,
                )
            hidden_states = layer_outputs[0]
            if output_lambda_params:
                all_lambda_params = all_lambda_params + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_lambda_params] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_lambda_params
        )


class LambdaBertPreTrainedModel(PreTrainedModel):
    config_class = LambdaBertConfig
    base_model_prefix = "lambdabert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class LambdaBertModel(LambdaBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = LambdaBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_lambda_params=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_lambda_params = output_lambda_params if output_lambda_params is not None else self.config.output_lambda_params
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            token_mask=attention_mask,
            output_lambda_params=output_lambda_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class LambdaBertForPreTrainingNSP(LambdaBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.lambdabert = LambdaBertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_lambda_params=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            labels = kwargs.pop("masked_lm_labels")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lambdabert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_lambda_params=output_lambda_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LambdaBertSOPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class LambdaBertForPreTrainingSOP(LambdaBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.lambdabert = LambdaBertModel(config)
        self.predictions = BertLMPredictionHead(config)
        self.sop_classifier = LambdaBertSOPHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def get_input_embeddings(self):
        return self.lambdabert.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        sentence_order_label=None,
        output_lambda_params=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            labels = kwargs.pop("masked_lm_labels")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lambdabert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_lambda_params=output_lambda_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


