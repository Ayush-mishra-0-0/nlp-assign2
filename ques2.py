import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import spacy
from collections import defaultdict

# Custom dataset class for question-query pairs
class QuestionQueryDataset(Dataset):
    def __init__(self, questions, queries, tokenizer):
        self.questions = questions
        self.queries = queries
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        question = self.questions[idx]
        query = self.queries[idx]
        
        # Tokenize inputs
        question_tokens = self.tokenizer.encode(
            question, 
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        query_tokens = self.tokenizer.encode(
            query,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question': question_tokens.squeeze(),
            'query': query_tokens.squeeze(),
            'question_text': question,
            'query_text': query
        }

# Encoder with Bi-LSTM
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        
        # Pack padded sequence
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack outputs
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # Combine bidirectional states
        hidden = self._combine_bidirectional_states(hidden)
        cell = self._combine_bidirectional_states(cell)
        
        return outputs, (hidden, cell)
    
    def _combine_bidirectional_states(self, state):
        """Combines forward and backward states for bidirectional LSTM"""
        # state shape: (num_layers * 2, batch_size, hidden_size)
        forward_state = state[::2]
        backward_state = state[1::2]
        combined_state = torch.cat([forward_state, backward_state], dim=2)
        return combined_state  # shape: (num_layers, batch_size, hidden_size * 2)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # Calculate attention scores
        # hidden shape: (batch_size, hidden_size * 2)
        # encoder_outputs shape: (batch_size, seq_len, hidden_size * 2)
        
        energy = torch.tanh(self.attention(encoder_outputs))  # (batch_size, seq_len, hidden_size)
        energy = energy.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attention = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        
        # Apply softmax to get attention weights
        return torch.softmax(attention, dim=2)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        
        # Modified LSTM input size to account for concatenated context vector
        self.lstm = nn.LSTM(
            embed_size + hidden_size * 2,  # input size: embedding + context
            hidden_size * 2,  # hidden size matches encoder's bidirectional output
            num_layers=num_layers,  # match encoder's number of layers
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x shape: (batch_size, 1)
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))  # (batch_size, 1, embed_size)
        
        # Calculate attention weights
        attention_weights = self.attention(hidden, encoder_outputs)
        
        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)
        
        # Combine embedded input and context vector
        rnn_input = torch.cat([embedded, context], dim=2)
        
        # Pass through LSTM
        # No need to unsqueeze hidden and cell as they're already the right shape
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Generate prediction
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell
# Complete Seq2Seq model
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.fc.out_features
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Get encoder outputs
        encoder_outputs, (hidden, cell) = self.encoder(
            source,
            torch.tensor([source.size(1)] * batch_size)
        )
        
        # First input to decoder is start token
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(
                decoder_input,
                hidden,
                cell,
                encoder_outputs
            )
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
            
        return outputs

# Entity and relation linking
class EntityLinker:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def link_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        source = batch['question'].to(device)
        target = batch['query'].to(device)
        
        output = model(source, target)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        target = target[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_queries = 0
    total_queries = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            source = batch['question'].to(device)
            target = batch['query'].to(device)
            
            output = model(source, target, teacher_forcing_ratio=0)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predictions = output.argmax(1)
            correct = (predictions == target).sum().item()
            total = target.size(0)
            
            correct_queries += correct
            total_queries += total
            
    accuracy = correct_queries / total_queries
    return epoch_loss / len(test_loader), accuracy

# Main training loop
def main():
    # Load and preprocess data
    with open('qald_9_plus_train_wikidata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract questions and queries
    questions = []
    queries = []
    for item in data['questions']:
        en_question = next(
            (q['string'] for q in item['question'] if q['language'] == 'en'),
            None
        )
        sparql_query = item.get('query', {}).get('sparql', '')
        
        if en_question and sparql_query:
            questions.append(en_question)
            queries.append(sparql_query)
    
    # Split data
    train_questions, test_questions, train_queries, test_queries = train_test_split(
        questions, queries, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = QuestionQueryDataset(train_questions, train_queries, tokenizer)
    test_dataset = QuestionQueryDataset(test_questions, test_queries, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=2
    )
    
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=2
    )
    
    model = Seq2SeqModel(encoder, decoder, device).to(device)
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    n_epochs = 10
    best_accuracy = 0
    
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Query Generation Accuracy: {accuracy:.4f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pt')

if __name__ == '__main__':
    main()
