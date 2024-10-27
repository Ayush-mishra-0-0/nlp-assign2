import torch
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.amp as amp
import os

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.preprocess_data(data)
    
    def preprocess_data(self, data):
        examples = []
        for entry in data['data']:
            for paragraph in entry['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    if not qa['is_impossible']:
                        answer = qa['answers'][0]
                        answer_start = answer['answer_start']
                        answer_text = answer['text']
                        answer_end = answer_start + len(answer_text)
                        
                        examples.append({
                            'context': context,
                            'question': qa['question'],
                            'answer_text': answer_text,
                            'answer_start': answer_start,
                            'answer_end': answer_end
                        })
        return examples[:10000]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize question and context
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            max_length=self.max_length,
            truncation='only_second',
            stride=128,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Find start and end token positions for the answer
        start_positions = torch.tensor([0])
        end_positions = torch.tensor([0])
        
        offset_mapping = encoding.pop('offset_mapping')
        
        # Find token positions that correspond to the answer
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= example['answer_start'] <= end:
                start_positions = torch.tensor([idx])
            if start <= example['answer_end'] <= end:
                end_positions = torch.tensor([idx])
                break
                
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'start_positions': start_positions,
            'end_positions': end_positions
        }

class ModelTrainer:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())
        
        # Create directory for saving results
        os.makedirs('results', exist_ok=True)
    
    def train(self, train_dataloader, eval_dataloader, epochs=3, lr=2e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_dataloader),
            epochs=epochs,
            pct_start=0.1
        )
        
        training_stats = []
        best_eval_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            
            # Training
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc='Training')
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Automatic mixed precision training
                with amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_dataloader)
            
            # Evaluation
            eval_loss = self.evaluate(eval_dataloader)
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_loss,
                }, 'results/best_model.pt')
            
            # Save training stats
            training_stats.append({
                'epoch': epoch + 1,
                'training_loss': avg_train_loss,
                'eval_loss': eval_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Evaluation loss: {eval_loss:.4f}')
            
            # Save current epoch stats
            pd.DataFrame(training_stats).to_csv('results/training_stats.csv', index=False)
        
        return training_stats
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        self.model.eval()
        total_eval_loss = 0
        
        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(**batch)
                total_eval_loss += outputs.loss.item()
        
        return total_eval_loss / len(eval_dataloader)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    with open('train-v2.0.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('dev-v2.0.json', 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = ModelTrainer()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = QADataset(train_data, trainer.tokenizer)
    eval_dataset = QADataset(eval_data, trainer.tokenizer)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced batch size for single GPU
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True
    )
    
    # Train model
    print("Starting training...")
    training_stats = trainer.train(train_dataloader, eval_dataloader)
    
    # Plot and save results
    stats_df = pd.DataFrame(training_stats)
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['epoch'], stats_df['training_loss'], label='Training Loss')
    plt.plot(stats_df['epoch'], stats_df['eval_loss'], label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.savefig('results/training_loss.png')
    plt.close()
    
    print("Training complete! Results saved in 'results' directory.")

if __name__ == "__main__":
    main()