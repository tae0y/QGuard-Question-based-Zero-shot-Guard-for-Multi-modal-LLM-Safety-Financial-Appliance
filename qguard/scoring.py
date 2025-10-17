from typing import Dict, Any, List, Tuple, Optional
import torch
from .token_utils import yes_no_probs_from_logits

def _model_device_dtype(model):
    try:
        p = next(model.parameters())
        return p.device, p.dtype
    except StopIteration:
        dev = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dt = getattr(model, "dtype", torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        return dev, dt

_PATCHED = False
_CHAT_PATCHED = False

def _ensure_tuple_generate(model):
    global _PATCHED
    if _PATCHED:
        return

    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "generate"):
        return

    orig_generate = lm.generate

    def wrapped_generate(*args, **kwargs):
        kwargs.setdefault("return_dict_in_generate", True)
        kwargs.setdefault("output_scores", True)
        out = orig_generate(*args, **kwargs)

        if hasattr(out, "sequences") and hasattr(out, "scores"):
            sequences = out.sequences
            scores = out.scores
            return (sequences, scores)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            return (out[0], out[1])

        sequences = getattr(out, "sequences", out)
        scores = getattr(out, "scores", None)
        return (sequences, scores)

    lm.generate = wrapped_generate
    _PATCHED = True

def _patch_chat_method(model, tokenizer):
    """Patch the model.chat method to fix the decoding bug"""
    global _CHAT_PATCHED
    if _CHAT_PATCHED:
        return
    
    if not hasattr(model, 'chat'):
        return
    
    original_chat = model.chat
    
    def patched_chat(tokenizer_arg, pixel_values, question, generation_config, history=None, return_history=False):
        # Call the original internals but catch the decode error
        try:
            # Try to use the model's internal generation
            response, history = model.generate(
                tokenizer=tokenizer_arg,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                history=history,
                return_history=True
            )
            
            # The generate method should return properly
            # But we need to extract logits separately
            return response, None  # Will handle logits in fallback
            
        except:
            # Full fallback path
            raise TypeError("Forcing fallback to direct generation")
    
    model.chat = patched_chat
    _CHAT_PATCHED = True

def _setup_tokenizer(tokenizer):
    """Ensure tokenizer has necessary tokens configured"""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Last resort: use token 0
            tokenizer.pad_token_id = 0
    return tokenizer

@torch.no_grad()
def _first_step_logits_via_chat(
    model,
    tokenizer,
    question_text: str,
    images: torch.Tensor, 
    max_new_tokens: int = 1,
) -> Tuple[torch.Tensor, str]: 
    _ensure_tuple_generate(model)
    tokenizer = _setup_tokenizer(tokenizer)

    gen_cfg = dict(
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    model_device, model_dtype = _model_device_dtype(model)
    if model_device.type == "cpu" and model_dtype == torch.bfloat16:
        model_dtype = torch.float32
    images = images.to(device=model_device, dtype=model_dtype).contiguous()

    # Direct generation path - bypass chat method entirely
    if hasattr(model, 'build_conversation_input_ids'):
        input_ids = model.build_conversation_input_ids(
            tokenizer, 
            query=question_text, 
            history=None, 
            images=images
        )
    else:
        # Create image tokens if needed
        messages = [{"role": "user", "content": question_text}]
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = question_text
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    input_ids = input_ids.to(model_device)
    attention_mask = torch.ones_like(input_ids)
    
    # 입력 길이 저장 (나중에 생성된 부분만 추출하기 위해)
    input_length = input_ids.shape[1]
    
    # Generate with proper configuration
    outputs = model.language_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_cfg
    )
    
    # Extract sequences and scores
    if isinstance(outputs, tuple) and len(outputs) == 2:
        sequences, scores = outputs
    else:
        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        scores = outputs.scores if hasattr(outputs, 'scores') else None
    
    # 생성된 토큰만 디코딩 (입력 프롬프트 제외)
    generated_tokens = sequences[0, input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Handle logits extraction
    if scores is None:
        raise ValueError("No logits returned from model")
    
    step0 = scores[0][0] if isinstance(scores, (list, tuple)) else scores[0]
    if not isinstance(step0, torch.Tensor):
        step0 = torch.tensor(step0, device=model_device, dtype=torch.float32)

    return step0, (response or "")


@torch.no_grad()
def _first_step_logits_text_only(
    model,
    tokenizer,
    question_text: str,
    max_new_tokens: int = 1,
) -> Tuple[torch.Tensor, str]: 
    _ensure_tuple_generate(model)
    tokenizer = _setup_tokenizer(tokenizer)

    gen_cfg = dict(
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    model_device, model_dtype = _model_device_dtype(model)
    if model_device.type == "cpu" and model_dtype == torch.bfloat16:
        model_dtype = torch.float32
    
    # Build input properly
    if hasattr(model, 'build_conversation_input_ids'):
        input_ids = model.build_conversation_input_ids(
            tokenizer, 
            query=question_text, 
            history=None, 
            images=None
        )
    else:
        # Apply chat template if available
        messages = [{"role": "user", "content": question_text}]
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = question_text
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    input_ids = input_ids.to(model_device)
    attention_mask = torch.ones_like(input_ids)
    
    # 입력 길이 저장 (나중에 생성된 부분만 추출하기 위해)
    input_length = input_ids.shape[1]
    
    # Generate directly
    outputs = model.language_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_cfg
    )
    
    # Extract sequences and scores
    if isinstance(outputs, tuple) and len(outputs) == 2:
        sequences, scores = outputs
    else:
        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        scores = outputs.scores if hasattr(outputs, 'scores') else None
    
    # 생성된 토큰만 디코딩 (입력 프롬프트 제외)
    generated_tokens = sequences[0, input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Handle logits extraction
    if scores is None:
        raise ValueError("No logits returned from model")
    
    step0 = scores[0][0] if isinstance(scores, (list, tuple)) else scores[0]
    if not isinstance(step0, torch.Tensor):
        step0 = torch.tensor(step0, device=model_device, dtype=torch.float32)

    return step0, (response or "")


def score_guard_question_yes_prob(
    model,
    tokenizer,
    guard_q: str,
    prompt_text: str,
    yes_ids: List[int],
    no_ids: List[int],
    images: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    q_full = f"{guard_q} (You must answer with only Yes or No).\n\nprompt: {prompt_text}"

    if images is not None:
        step0, decoded = _first_step_logits_via_chat(model, tokenizer, q_full, images, max_new_tokens=4)
    else:
        step0, decoded = _first_step_logits_text_only(model, tokenizer, q_full, max_new_tokens=4)

    y_prob, n_prob = yes_no_probs_from_logits(step0, yes_ids, no_ids)

    return {
        "question": q_full,
        "yes_prob": y_prob,
        "no_prob": n_prob,
        "response": decoded,
    }