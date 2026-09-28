import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate
import numpy as np

# ---------------------------------------------------------
# 1. Configuration & Models
# ---------------------------------------------------------
TEACHER_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
STUDENT_MODEL_NAME = "distilbert-base-multilingual-cased"
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"

TEMPERATURE = 2.0
ALPHA = 0.5  # Weighting between standard loss and distillation loss

print("1. Loading Tokenizers and Models...")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

# Load Teacher (Pre-trained on sentiment analysis)
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL_NAME, num_labels=3
)
teacher_model.eval()

# Freeze teacher parameters
for param in teacher_model.parameters():
    param.requires_grad = False

# Load Student (To be distilled)
student_model = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL_NAME, num_labels=3
)

# ---------------------------------------------------------
# 2. Custom Distillation Trainer
# ---------------------------------------------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, teacher_tokenizer=None, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_tokenizer = teacher_tokenizer
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels
        labels = inputs.get("labels")

        # 1. Student Forward Pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # 2. Teacher Forward Pass (re-tokenize text for teacher architecture)
        # Decode student input tokens back to raw text for teacher tokenization
        raw_texts = self.processing_class.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        
        teacher_inputs = self.teacher_tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.args.device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        # 3. Standard Cross-Entropy Loss
        ce_loss = F.cross_entropy(student_logits, labels)

        # 4. KL Divergence Soft Distillation Loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = self.kl_div_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # Combined Loss
        loss = (self.alpha * ce_loss) + ((1.0 - self.alpha) * distill_loss)

        return (loss, student_outputs) if return_outputs else loss

# ---------------------------------------------------------
# 3. Dataset Preprocessing
# ---------------------------------------------------------
print("2. Preparing Dataset...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

def preprocess_function(examples):
    tokenized = student_tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )
    # Ensure label column is named 'labels'
    tokenized["labels"] = examples["label"]
    return tokenized

processed_ds = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

train_dataset = processed_ds["train"]
eval_dataset = processed_ds["validation"]

# ---------------------------------------------------------
# 4. Evaluation Metrics & Setup
# ---------------------------------------------------------
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./distilled-student-checkpoint",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

trainer = DistillationTrainer(
    model=student_model,
    teacher_model=teacher_model,
    teacher_tokenizer=teacher_tokenizer,
    alpha=ALPHA,
    temperature=TEMPERATURE,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=student_tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=student_tokenizer),
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------
# 5. Execute Training & Save
# ---------------------------------------------------------
if __name__ == "__main__":
    print("3. Starting Distillation Training...")
    trainer.train()

    print("4. Saving Final Distilled Student Model...")
    student_model.save_pretrained("./distilled-student-final")
    student_tokenizer.save_pretrained("./distilled-student-final")
    print("Distillation Complete!")