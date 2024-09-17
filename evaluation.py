import time

def evaluate_retrieval(retrieved_docs, relevant_docs):
    relevant = set(relevant_docs)
    retrieved = set(retrieved_docs)
    
    precision = len(relevant.intersection(retrieved)) / len(retrieved) if retrieved else 0
    recall = len(relevant.intersection(retrieved)) / len(relevant) if relevant else 0
    return precision, recall

def evaluate_model(response, expected_response):
    return 1.0 if response == expected_response else 0.0

def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time
