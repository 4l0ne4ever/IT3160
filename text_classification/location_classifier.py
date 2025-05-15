import torch
import gc
import pandas as pd
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer

class LocationClassifier():
    def __init__(self, 
                model_path,
                data_path=None,
                context_column_name = 'context',
                label_column_name = 'label',
                model_name = "distilbert/distilbert-base-uncased",
                test_size=0.2,
                num_labels=4,
                huggingface_token = None):
        # Detect available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model_path = model_path
        self.data_path = data_path
        self.context_column_name = context_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        
        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        self.tokenizer = self.load_tokenizer()
        
        if not huggingface_hub.repo_exists(self.model_path):
            if data_path is None:
                raise ValueError("Data path is required to train the model")
            
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            
            # Pass the detected device to get_class_weights
            class_weights = get_class_weights(all_data, device=str(self.device))
            
            self.train_model(train_data, test_data, class_weights)
            
        self.model = self.load_model(self.model_path)
    
    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict)
        model = model.to(self.device)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.set_device(str(self.device))
        trainer.set_class_weights(class_weights)

        trainer.train()

        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        trainer.push_to_hub()

        # Clean up memory
        del model, trainer
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def load_model(self, model_path):
        model = pipeline('text-classification', model=model_path, top_k=None)
        return model
        
    def load_data(self, data_path):
        df = pd.read_json(data_path, lines=True)
        df['context'] = df['location_name'] + '. ' + df['location_description']
        df['label'] = df['location_type']
        df = df[['context', 'label']]
        df = df.dropna()

        # Label encoding
        le = preprocessing.LabelEncoder()
        le.fit(df['label'].tolist())       
        label_dict = {index: label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df["label_dtframe"] = le.transform(df["label"].tolist())

        # Train/Test split
        test_size = 0.2
        df_train, df_test = train_test_split(df, 
                                             test_size=test_size,
                                             stratify=df['label_dtframe'],
                                             )

        # Convert pandas dataframe to Hugging Face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # Loại bỏ cột không cần thiết và đảm bảo định dạng đúng
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ['context', 'label_dtframe']])
        test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in ['context', 'label_dtframe']])

        # Tokenize the dataset
        tokenizer_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)
        tokenizer_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)

        # Đảm bảo định dạng tensor cho PyTorch
        tokenizer_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenizer_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        return tokenizer_train, tokenizer_test
    def preprocess_function(self, tokenizer, examples):
        # Mã hóa cột context
        tokenized_inputs = tokenizer(examples["context"], truncation=True, padding=True)
        # Thêm cột label dưới dạng số nguyên
        tokenized_inputs["labels"] = examples["label_dtframe"]  # Sử dụng cột đã mã hóa
        return tokenized_inputs


    def load_tokenizer(self):
        try:
            if huggingface_hub.repo_exists(self.model_path):
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except ValueError as e:
            print(f"Error loading tokenizer from {self.model_path}: {e}. Falling back to {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def postprocess(self, model_output):
        output = []
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output
       
    def classify_location(self, context):
        model_output = self.model(context)
        predictions = self.postprocess(model_output)
        return predictions