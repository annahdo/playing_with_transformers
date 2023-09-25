
import torch
from tqdm import tqdm

def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def get_difference(wrapped_model, prompt1, prompt2, layer_ids, block_names):
    differences = {}
    for block_name in block_names:
        differences[block_name] = {}
    
        for layer_id in layer_ids:
            wrapped_model.wrap_block(layer_id, block_name=block_name)
            
            # get internal activations
            wrapped_model.reset()
            wrapped_model.run_prompt(prompt1)
            activations1 = wrapped_model.get_activations(layer_id, block_name=block_name)
            wrapped_model.reset()
            wrapped_model.run_prompt(prompt2)
            activations2 = wrapped_model.get_activations(layer_id, block_name=block_name)

            diff = activations1[0,-1,:]-activations2[0,-1,:]
            differences[block_name][layer_id] = diff


    wrapped_model.reset()
    return differences

def activation_editing(wrapped_model, sentences, directions, coeff=1, coeff_dict=None,
                                token_pos=None, max_new_tokens=30, textfile=None, printout=True, calc_neutral=True, batch_size=32, use_cache=True):

    def append_to_file(text):
        if textfile:
            with open(textfile, 'a', encoding='utf-8') as f:
                f.write(text)

    num_batches = len(sentences)//batch_size
            
    generations = {"neutral": [], "positive": [], "negative": []}
    wrapped_model.reset()
    if calc_neutral:
        print("No activation addition\n")
        append_to_file("No activation addition\n")
        for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches, disable=printout):
            generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
            generations["neutral"].extend(generated)
            for g in generated:
                if printout:
                    print(f"{g}\n")
                append_to_file(f"{g}\n\n")
          
    set_activations_to_add(wrapped_model, directions, coeff=coeff, coeff_dict=coeff_dict, token_pos=token_pos)
    
    print("-" * 30 + "\nPositive activation addition\n")
    append_to_file("-" * 30 + "\nPositive activation addition\n")
    for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches, disable=printout):
        generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
        generations["positive"].extend(generated)
        for g in generated:
            if printout:
                print(f"{g}\n")
            append_to_file(f"{g}\n\n")
    
    
    wrapped_model.reset()
    set_activations_to_add(wrapped_model, directions, coeff=-coeff, coeff_dict=coeff_dict, token_pos=token_pos)

    print("-" * 30 + "\nNegative activation addition\n")
    append_to_file("-" * 30 + "\nNegative activation addition\n")
    for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches, disable=printout):
        generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
        generations["negative"].extend(generated)
        for g in generated:
            if printout:
                print(f"{g}\n")
            append_to_file(f"{g}\n\n")
    
    wrapped_model.reset()

    return generations 


def set_activations_to_add(wrapped_model, directions, coeff=1, coeff_dict=None, token_pos=None):
    for block_name in directions.keys():
        for layer_id, activation in directions[block_name].items():
            if len(activation.shape) > 1:
                # activations and coeff_dict are shaped as heads need to flatten
                if coeff_dict:
                    activation = activation * coeff_dict[block_name][layer_id].unsqueeze(1)
                activation = activation.flatten()

            else:
                if coeff_dict:
                    activation = activation*coeff_dict[block_name][layer_id]
            wrapped_model.set_to_add(layer_id, coeff*activation.to(wrapped_model.device), token_pos=token_pos, block_name=block_name)