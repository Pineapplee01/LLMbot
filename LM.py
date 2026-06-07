from transformers import AutoModel
import torch.nn as nn
from torch_geometric.nn.models import MLP
import torch


def _load_local_encoder(model_config, fallback_source):
    model_source = model_config.get('pretrained_model_source') or fallback_source
    kwargs = {"local_files_only": True} if model_config.get('pretrained_model_source') else {}
    return AutoModel.from_pretrained(model_source, **kwargs)


class LM_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.LM_model_name = model_config['lm_model']
        self.detach_embeddings = bool(model_config.get('detach_embeddings', False))
        if self.LM_model_name == 'deberta':
            self.LM = _load_local_encoder(model_config, 'microsoft/deberta-v3-base')
        elif self.LM_model_name == 'roberta':
            self.LM = _load_local_encoder(model_config, 'roberta-base')
        elif self.LM_model_name in {'roberta-f', 'roberta_finetuned'}:
            self.LM = _load_local_encoder(model_config, 'yzxjb/roberta-finetuned-20')
        elif self.LM_model_name == 'bert':
            self.LM = _load_local_encoder(model_config, 'bert-base-uncased')
        elif self.LM_model_name == 'twhin-bert':
            self.LM = _load_local_encoder(model_config, 'Twitter/twhin-bert-base')
        elif self.LM_model_name == 'xlm-roberta':
            self.LM = _load_local_encoder(model_config, 'xlm-roberta-base')
        elif self.LM_model_name in {'qwen_peft', 'qwen3_peft'}:
            from transformers import AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, TaskType
            _qwen_path = model_config.get(
                'qwen_model_path',
                '/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af',
            )
            device = model_config.get('device', torch.device('cpu'))
            use_auto_dtype = isinstance(device, torch.device) and device.type == 'cuda'
            base = AutoModelForCausalLM.from_pretrained(
                _qwen_path,
                trust_remote_code=bool(model_config.get('qwen_trust_remote_code', False)),
                torch_dtype="auto" if use_auto_dtype else torch.float32,
                low_cpu_mem_usage=True,
            )
            base.config.use_cache = False
            lora_cfg = LoraConfig(
                r=model_config.get('peft_rank', 8),
                lora_alpha=model_config.get('peft_alpha', 16.0),
                target_modules=["q_proj", "v_proj"],
                lora_dropout=model_config.get('peft_dropout', 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.LM = get_peft_model(base, lora_cfg)
            self.LM.enable_input_require_grads()
            self.LM.gradient_checkpointing_enable()
            self.is_decoder = True
        else:
            raise ValueError(f"Unsupported LM model: {self.LM_model_name}")

        self.classifier = MLP(
            in_channels=self.LM.config.hidden_size,
            hidden_channels=model_config['classifier_hidden_dim'],
            out_channels=2,
            num_layers=model_config['classifier_n_layers'],
            act=model_config['activation'],
            norm=None if getattr(self, 'is_decoder', False) else 'batch_norm',
        )

        self.LM.config.hidden_dropout_prob = model_config['lm_dropout']
        self.LM.attention_probs_dropout_prob = model_config['att_dropout']

    def forward(self, tokenized_tensors):
        if getattr(self, 'is_decoder', False):
            out = self.LM(**tokenized_tensors, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # [B, seq_len, H]
            mask = tokenized_tensors['attention_mask']
            last_idx = mask.sum(dim=1) - 1  # last non-padding token index
            embedding = hidden[torch.arange(hidden.size(0)), last_idx].float()
        else:
            out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states']
            embedding = out[-1].mean(dim=1)
        classifier_embedding = embedding.detach() if self.detach_embeddings else embedding
        return embedding.detach(), self.classifier(classifier_embedding)
