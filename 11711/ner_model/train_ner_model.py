import os
import argparse
from transformers import AutoTokenizer, TrainingArguments   
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, Trainer
from helper_functions import create_directory, get_tag_dict, get_dataset, get_train_val_dataset 
from helper_functions import tokenize_and_align_labels, compute_metrics
import evaluate

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root', type=str, default='./', help='Project root directory')
    parser.add_argument('--ner-model-name', dest="model_name", type=str, default='bert-finetuned-ner', help='Name you want to give to ner model that is about to be trained.')
    parser.add_argument('--pre-trained-lm', dest="pre_trained_lm", type=str, default='bert-base-cased', help='Name of the pre-trained language model in Hugging Face that will be used as backbone.')
    parser.add_argument('--train-data', dest="train_file", type=str, help='Name of .conll file with the training data.')
    parser.add_argument('--val-data', dest="val_file", type=str, help='Name of .conll file with the validation data.')
    parser.add_argument('--batch-size', dest="batch_size", type=int, default = 8)
    parser.add_argument('--learning-rate', dest="learning_rate", type=float, default = 2e-5)
    parser.add_argument('--num-train-epochs', dest="num_train_epochs", type=int, default = 10)
    parser.add_argument('--weight-decay', dest="weight_decay", type=float, default = 0.01)
    parser.add_argument('--max-number-tokens', dest="max_number_tokens", type=int, default = 350, help='Max number of tokens in a sentence')
    return parser.parse_args()

def train_and_save_model(root = './',    
                         model_name = 'bert-finetuned-ner',
                         pre_trained_lm = 'bert-base-cased',
                         train_file = 'train_processed.conll',
                         val_file = 'validation_processed.conll',
                         batch_size = 8,
                         learning_rate = 2e-5,
                         num_train_epochs = 10,
                         weight_decay = 0.01,
                         max_number_tokens = 350):

  if not os.path.exists(root):
    print(f"Cannot access root at '{root}'.")
    return False

  # check if training file exists
  if not os.path.exists(root + train_file):
    print(f"Data file '{train_file}' not found.")
    return False
  
  # check if training file exists
  if not os.path.exists(root + val_file):
    print(f"Data file '{val_file}' not found.")
    return False

  model_save_path = root + model_name 

  # create directory to save model
  create_directory(directory_path = model_save_path)

  # create a dictionary from ner tags (string) to id (an integer)
  label2id = get_tag_dict()
  label_names = list(label2id.keys())

  # create a dictionary from id (an integer) to the ner tags (string)
  id2label = {i: label for i, label in enumerate(label_names)}

  # load backbone language model and tokenizer
  try:
      # Attempt to load the model and tokenizer
      model = AutoModelForTokenClassification.from_pretrained(
          pre_trained_lm,
          id2label = id2label,
          label2id = label2id,
          num_labels = len(label_names))
      if "scideberta" in pre_trained_lm:
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_lm, add_prefix_space=True)
      else:
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_lm)
  except Exception as e:
      print(f"The model '{pre_trained_lm}' is not available on the Hugging Face Model Hub.")
      return False

  # setting training arguments
  args = TrainingArguments(
      model_name,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      learning_rate=learning_rate,
      num_train_epochs=num_train_epochs,
      weight_decay=weight_decay,
      per_device_train_batch_size = batch_size, 
      push_to_hub=False,
  )

  # get data ready
  #train_val_dataset = get_dataset(input_file = root + train_val_file, val_size=val_size, is_test=False)
  train_val_dataset = get_train_val_dataset(input_train_file = root + train_file, 
                                            input_val_file = root + val_file, 
                                            max_number_tokens = max_number_tokens)

  # create tokenized datasets
  tokenized_datasets = train_val_dataset.map(
      tokenize_and_align_labels,
      batched=True,
      remove_columns=train_val_dataset["train"].column_names,
      fn_kwargs={"tokenizer": tokenizer}
  )

  # data collator to build paddings accordinly to the backbone model
  data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

  def compute_metrics_wrapper(p):
    metric = evaluate.load("seqeval")
    return compute_metrics(p, label_names, metric)

  trainer = Trainer(
      model=model,
      args=args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["validation"],
      data_collator=data_collator,
      compute_metrics=compute_metrics_wrapper,
      tokenizer=tokenizer,
  )

  # train model
  trainer.train()

  # save model
  print("Saving model and tokenizer at " + model_save_path)
  model.save_pretrained(model_save_path)
  tokenizer.save_pretrained(model_save_path)

  return True

if __name__ == '__main__':
    #get arguments
    args = parse_arguments()

    # for arg_name, arg_value in vars(args).items():
    #     print(f"{arg_name}: {arg_value}")

    model_trained = train_and_save_model(root = args.root,    
                                         model_name = args.model_name,
                                         pre_trained_lm = args.pre_trained_lm,
                                         train_file = args.train_file,
                                         val_file = args.val_file,
                                         batch_size = args.batch_size,
                                         learning_rate = args.learning_rate,
                                         num_train_epochs = args.num_train_epochs,
                                         weight_decay = args.weight_decay,
                                         max_number_tokens = args.max_number_tokens)
    
    if not model_trained:
      print("Training failed")
    else:   
      print("Done")

    