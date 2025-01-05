from datasets import load_dataset

def sst2_dataset_view():
    """View the SST-2 dataset."""
    dataset = load_dataset("glue", "sst2")
    print(dataset)

    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    print(train_data)
    print(valid_data)
    print(test_data)

    train_data = train_data.select(range(10))
    valid_data = valid_data.select(range(10))
    test_data = test_data.select(range(10))

    print('This is training data:')
    for example in train_data:
        print(example)
    
    print('This is valid data:')
    for example in valid_data:
        print(example)
    
    print('This is test data:')
    for example in test_data:
        print(example)

def print_dataset():
    
    glue_list =['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    
    for glue in glue_list:
        dataset = load_dataset("glue", glue)
        print("----------------------------------------------\n")
        print(glue)
        print(dataset)
        print("----------------------------------------------\n")

    print("----------------------------------------------\n")
    print("----------------------------------------------\n")
    dataset = load_dataset("glue", "sst2")
    print("sst2")
    print(dataset['train'].select(range(10)))
    print("----------------------------------------------\n")
    print("----------------------------------------------\n")


# print_dataset()
# sst2_dataset_view()


def print_batch_info(input_ids, model):
    print("\n\n\n---------------------------------------------------------\n")
    print("input_ids: ", input_ids)
    print("input_ids shape: ", input_ids.shape)
    print("model named modules: ", model.named_modules())
    print("\n---------------------------------------------------------\n\n\n")

def print_MHA_info(name, query, key, value, attn_output, attn_over_seq, attn_over_batch, head_importance):
    print("\n\n\n---------------------------------------------------------\n")
    print(f"Computing importance scores for module: {name}")
    print("projection layer's shpae is (B, S, D)")
    print("shape of query: ", query.shape)
    print("shape of key: ", key.shape)
    print("shape of value: ", value.shape)
    print("attn_output: ", attn_output)
    print("attn_output shape: ", attn_output.shape, "(Expected: [B, S, D])")
    print("attn_over_seq: ", attn_over_seq)
    print("attn_over_seq shape: ", attn_over_seq.shape, "(Expected: [B, D])")
    print("attn_over_batch: ", attn_over_batch)
    print("attn_over_batch shape: ", attn_over_batch.shape, "(Expected: [D])")
    print("head_importance: ", head_importance)
    print("\n---------------------------------------------------------\n\n\n")

def print_MLP_info(neuron_output, neuron_over_seq, neuron_over_batch, neuron_importance):
    print("\n\n\n---------------------------------------------------------\n")
    print("neuron_output: ", neuron_output)
    print("neuron_output shape: ", neuron_output.shape, "(Expected: [B, S, D])")
    print("neuron_over_seq: ", neuron_over_seq)
    print("neuron_over_seq shape: ", neuron_over_seq.shape, "(Expected: [B, D])")
    print("neuron_over_batch: ", neuron_over_batch)
    print("neuron_over_batch shape: ", neuron_over_batch.shape, "(Expected: [D])")
    print("neuron_importance: ", neuron_importance)
    print("\n---------------------------------------------------------\n\n\n")

def print_LN_info(name, emb_output, emb_over_seq, emb_over_batch, emb_importance):
    print("\n\n\n---------------------------------------------------------\n")
    print(f"Computing importance scores for module: {name}")
    print("emb_output: ", emb_output)
    print("emb_output shape: ", emb_output.shape, "(Expected: [B, S, D])")
    print("emb_over_seq: ", emb_over_seq)
    print("emb_over_seq shape: ", emb_over_seq.shape, "(Expected: [B, D])")
    print("emb_over_batch: ", emb_over_batch)
    print("emb_over_batch shape: ", emb_over_batch.shape, "(Expected: [D])")
    print("emb_importance: ", emb_importance)
    print("\n---------------------------------------------------------\n\n\n")