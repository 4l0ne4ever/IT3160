import torch
import gc
import os
import glob
import huggingface_hub
import pandas as pd
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import transformers

class CharacterChatbotQwen():
    def __init__(self,
                 model_path,
                 data_path="/content/data/transcripts/",
                 huggingface_token=None):
        self.model_path = model_path
        data_path = os.path.abspath(data_path.rstrip('/')) + '/'
        if not os.path.isdir(data_path):
            raise ValueError(f"Data directory does not exist: {data_path}")
        self.data_files = glob.glob(os.path.join(data_path, "*.csv"))
        if not self.data_files:
            raise ValueError(f"No CSV files found in {data_path}. Directory exists: {os.path.exists(data_path)}")
        print(f"Successfully found {len(self.data_files)} CSV files in {data_path}:")
        self.huggingface_token = huggingface_token
        self.base_model_path = "Qwen/Qwen3-4B"  # Confirmed model ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print(f"Model {self.model_path} not found. We will train our model.")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.model = self.load_model(self.model_path)
            
    def load_model(self, model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": bnb_config,
            },
        )
        return pipeline
            
    def load_data(self):
        transcript_df = pd.concat(
            [pd.read_csv(file, on_bad_lines="skip") for file in self.data_files],
            ignore_index=True
        )
        transcript_df = transcript_df.dropna()
        transcript_df['number_of_words'] = transcript_df['line'].str.strip().str.split(" ")
        transcript_df['number_of_words'] = transcript_df['number_of_words'].apply(lambda x: len(x))
        transcript_df['eleven_res'] = 0
        transcript_df.loc[(transcript_df['name']=="Eleven")&(transcript_df['number_of_words']>=1),'eleven_res']=1
        
        indexes_to_take = list(transcript_df[(transcript_df['eleven_res'] == 1)&(transcript_df.index>0)].index)
        system_prompt = """ You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""
        prompts = []
        for ind in indexes_to_take:
            prompt = system_prompt
            prompt += transcript_df.iloc[ind -1]['line']
            prompt += "\n"
            prompt += transcript_df.iloc[ind]['line']
            prompts.append(prompt)
            
        df = pd.DataFrame({"prompt": prompts})
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def train(self,
              base_model_name_or_path,
              dataset,
              output_dir="./results",
              per_device_train_batch_size=1,
              gradient_accumulation_steps=1,
              optimizer="paged_adamw_32bit",
              save_steps=200,
              logging_steps=10,
              learning_rate=5e-5,
              max_grad_norm=0.3,
              max_steps=600,
              warmup_ratio=0.1,
              lr_scheduler_type="cosine"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading model {base_model_name_or_path}: {e}")
            raise
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        def preprocess_function(examples):
            return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)

        dataset = dataset.map(preprocess_function, batched=True)

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )

        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optimizer,
                save_steps=save_steps,
                logging_steps=logging_steps,
                learning_rate=learning_rate,
                fp16=True,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                report_to="none",
                gradient_checkpointing=True,
                eval_strategy="steps",  # Updated from evaluation_strategy
                eval_steps=50,
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            )
        except Exception as e:
            print(f"Error in TrainingArguments: {e}")
            raise

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
        )
        trainer.train()

        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model = model.merge_and_unload()
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del model, base_model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        
    def chat(self, message, history):
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Tạo danh sách messages
        messages = [{"role": "system", "content": """You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""}]
        
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})
        
        # Sử dụng apply_chat_template để định dạng đầu vào
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        
        # Load model directly instead of using pipeline
        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16)
        
        # Tạo phản hồi
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        # Giải mã phản hồi
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Trích xuất phần phản hồi của assistant
        return response.split("assistant:")[-1].strip() if "assistant:" in response else response
        