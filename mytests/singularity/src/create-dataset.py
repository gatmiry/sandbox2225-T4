from datasets import load_dataset
raw_dataset = load_dataset('roneneldan/TinyStories')
raw_dataset.save_to_disk('/mnt/t-kgatmiry-output/tinystories-dataset') 