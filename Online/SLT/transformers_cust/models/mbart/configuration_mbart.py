# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2021, National Institute of Information and Communication Technology (Raj Dabre)
# Modified portions by Raj Dabre are indicated as so.  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MBART model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json",
    # See all MBART models at https://huggingface.co/models?filter=mbart
}


class MBartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.MBartModel`. It is used to
    instantiate an MBART model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MBART `facebook/mbart-large-cc25
    <https://huggingface.co/facebook/mbart-large-cc25>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the MBART model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.MBartModel` or
            :class:`~transformers.TFMBartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models)

    Example::

        >>> from transformers import MBartModel, MBartConfig

        >>> # Initializing a MBART facebook/mbart-large-cc25 style configuration
        >>> configuration = MBartConfig()

        >>> # Initializing a model from the facebook/mbart-large-cc25 style configuration
        >>> model = MBartModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        gradient_checkpointing=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        ## Modified by Raj Dabre. Start.
        decoder_tying_config=None, ## Argument to control parameter tying in encoder. According to my RSNMT paper.
        encoder_tying_config=None, ## Argument to control parameter tying in encoder. According to my RSNMT paper. 
        features_vocab_sizes=None, ## Argument to control feature based NMT. According to my paper with Abhisek.
        features_embed_dims=None, ## Argument to control feature based NMT. According to my paper with Abhisek.
        multilayer_softmaxing=False, ## Argument to control multi layer softmaxing. According to my multilayer softmaxing paper.
        wait_k=-1, ## Argument to control whether we will be doing SNMT or not.
        unidirectional_encoder=False, ## Argument to indicate whether we will train a unidirectional encoder or not.
        multi_source=False, ## Argument to control whether we do multi source or not.
        multi_source_method=None, ## Argument to control the multi source combination method. Should be a string.
        additional_source_wait_k=-1, ## Argument to indicate whether the additional source also a wait-k input.
        softmax_temperature=1.0, ## Argument to indicate the softmax temperature.
        temperature_calibration=False, ## Argument to indicate whether the softmax temperature should be calibrated (aka learned) or not.
        no_embed_norm=False, ## Argument to stop embedding normalization.
        no_scale_attention_embedding=False, ## Argument to stop attention embeddings from being scaled.
        num_domains_for_domain_classifier=-1, ## Argument to indicate number of domains for domain classifier.
        gradient_reversal_for_domain_classifier=False, ## Argument to indicate whether we should do gradient reversal for domain classifier.
        positional_encodings=False, ## Argument to indicate whether we should do use positional encodings or embeddings.
        ## Modified by Raj Dabre. End.
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        ## Modified by Raj Dabre. Start.
        self.encoder_tying_config = encoder_tying_config ## Argument to control parameter tying in encoder. According to my RSNMT paper.
        self.decoder_tying_config = decoder_tying_config ## Argument to control parameter tying in encoder. According to my RSNMT paper. 
        self.features_vocab_sizes = features_vocab_sizes  ## Argument to control feature based NMT. According to my paper with Abhisek. 
        self.features_embed_dims = features_embed_dims ## Argument to control feature based NMT. According to my paper with Abhisek.
        self.multilayer_softmaxing = multilayer_softmaxing ## Argument to control multi layer softmaxing. According to my multilayer softmaxing paper.
        self.wait_k = wait_k ## Argument to control whether we will be doing SNMT or not.
        self.unidirectional_encoder = unidirectional_encoder ## Argument to indicate whether we will train a unidirectional encoder or not.
        self.multi_source = multi_source ## Argument to control whether we do multi source or not.
        self.multi_source_method = multi_source_method ## Argument to control the multi source combination method. Should be a string.
        self.additional_source_wait_k = additional_source_wait_k ## Argument to indicate whether the additional source also a wait-k input.
        self.softmax_temperature = softmax_temperature ## Argument to indicate the softmax temperature.
        self.temperature_calibration = temperature_calibration ## Argument to indicate whether the softmax temperature should be calibrated (aka learned) or not.
        self.no_embed_norm = no_embed_norm ## Argument to stop embedding normalization.
        self.no_scale_attention_embedding = no_scale_attention_embedding ## Argument to stop attention embeddings from being scaled.
        self.num_domains_for_domain_classifier = num_domains_for_domain_classifier ## Argument to indicate number of domains for domain classifier.
        self.gradient_reversal_for_domain_classifier = gradient_reversal_for_domain_classifier ## Argument to indicate whether we should do gradient reversal for domain classifier.
        self.positional_encodings = positional_encodings ## Argument to indicate whether we should do use positional encodings or embeddings.
        ## Modified by Raj Dabre. End.
        
    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
