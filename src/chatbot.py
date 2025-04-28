from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

#LSTM model for text embedding
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = self.dropout(x)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        out, (hn, cn) = self.lstm(x, hidden)
        mean_out = torch.mean(out, dim=1)
        return mean_out, (hn, cn)

vocab = {}
current_idx = 0

def get_word_index(word):
    global current_idx
    if word not in vocab:
        vocab[word] = current_idx
        current_idx += 1
    return vocab[word]

def text_to_tensor(text):
    words = text.lower().split()
    indices = [get_word_index(word) for word in words]
    return torch.tensor(indices).unsqueeze(0)

def calculate_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

def main():
    input_size = 100
    hidden_size = 256
    num_layers = 2
    vocab_size = 10000
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, vocab_size)
    lstm_model.eval()

    conversation_history = []
    max_history_length = 5

    tokenizer  = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    eos_token_id = tokenizer.eos_token_id
    print(f"EOS token id is: {eos_token_id}")
    
    ## Load the LLM and embedding model
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
        max_new_tokens=500,
        context_window=2048,
        generate_kwargs={"temperature": 0.3, "top_p": 0.8,"eos_token_id": eos_token_id, "do_sample": True},
    )

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader('data').load_data()

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir='storage')

    storage_context = StorageContext.from_defaults(persist_dir='storage')
    index = load_index_from_storage(storage_context)

    while True:
        user_query = input("\nAsk something (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break

        response = index.as_query_engine().query(user_query)
        
        query_tensor = text_to_tensor(user_query)
        response_tensor = text_to_tensor(str(response))
        
        if conversation_history:
            context_embedding = torch.zeros(1, hidden_size)
            for text in conversation_history:
                text_tensor = text_to_tensor(text)
                mean_out, _ = lstm_model(text_tensor)
                context_embedding = mean_out
        else:
            context_embedding = torch.zeros(1, hidden_size)

        response_embedding, _ = lstm_model(response_tensor)
        
        similarity_score = calculate_similarity(
            context_embedding.detach().numpy().flatten(), 
            response_embedding.detach().numpy().flatten()
        )

        if similarity_score > 0.7:
            print("\n Warning: Response may be too similar to previous context.")
        
        print("\nGenerated Answer:\n", response)
        approval = input("\nApprove this answer? (y/n): ").strip().lower()
        
        if approval == 'y':
            conversation_history.append(user_query)
            conversation_history.append(str(response))
            if len(conversation_history) > max_history_length * 2:
                conversation_history = conversation_history[-max_history_length * 2:]
            print("\n Response added to conversation memory.")
        else:
            print("\n Rejected. Skipping...")
            continue

if __name__ == "__main__":
    main()
