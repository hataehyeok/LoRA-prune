from datasets import load_dataset

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

print_dataset()

