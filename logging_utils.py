import logging
import os
from evaluation import evaluate_retrieval, evaluate_model


# Initialize logging
log_path = os.path.expanduser("~/evals.log")  # This writes the log file to the home directory

logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

def log_and_evaluate(query, retrieved_docs, relevant_docs, response, expected_response, retrieval_time, response_time):
    precision, recall = evaluate_retrieval(retrieved_docs, relevant_docs)
    model_accuracy = evaluate_model(response, expected_response)
    
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved Documents: {retrieved_docs}")
    logging.info(f"Relevant Documents: {relevant_docs}")
    logging.info(f"Precision: {precision}, Recall: {recall}")
    logging.info(f"Model Response: {response}")
    logging.info(f"Expected Response: {expected_response}")
    logging.info(f"Model Accuracy: {model_accuracy}")
    logging.info(f"Retrieval Time: {retrieval_time}s, Model Response Time: {response_time}s")
