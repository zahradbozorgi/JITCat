def get_processor(dataset_name, *args, **kwargs):
    if dataset_name == '5L':
        from processors.CSL_5LProcessor import CSL_5LProcessor
        return CSL_5LProcessor(dataset_name, *args, **kwargs)
    elif dataset_name == 'Astrazeneca':
        from processors.AstraProcessor import AstraProcessor
        return AstraProcessor(dataset_name, *args, **kwargs)
    elif dataset_name == 'CSL312':
        from processors.CSL312Processor import CSL312Processor
        return CSL312Processor(dataset_name, *args, **kwargs)
    elif dataset_name == 'AMBR':
        from processors.AMBRProcessor import AMBRProcessor
        return AMBRProcessor(dataset_name, *args, **kwargs)
    elif dataset_name == 'bpic2011_f1':
        from processors.BPICProcessor import BPICProcessor
        return BPICProcessor(dataset_name, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")