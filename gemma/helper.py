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