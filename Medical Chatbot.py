import pandas as pd
from datasets import load_dataset
dataset = load_dataset("danielpark/MQuAD-v1")

# Assuming 'dataset' is your DatasetDict and you're interested in the 'train' split
train_dataset = dataset['train']

# Extract 'question' and 'answer' fields
user_text = train_dataset['question']
bot_text = train_dataset['answer']

# Create a pandas DataFrame with 'user_text' and 'bot_text' columns
df = pd.DataFrame({
    'User': user_text,
    'Bot': bot_text
})

# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# Optionally, save the DataFrame to a CSV file
df.to_csv('medical_conversation_data.csv', index=False)

# Display the first few rows of the DataFrame to verify its diversity
print(df.head())

# Assuming 'df' is your Pandas DataFrame with columns 'User' and 'Bot'
df['formatted'] = "User: " + df['User'] + " \nBot: " + df['Bot'] + "\n"

training_text = "\n".join(df['formatted'].tolist())

from transformers import GPT2Tokenizer
import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Instantiate a tokenizer for gpt-2 model
"""
- GPT2Tokenizer is a class from the Transformers library.
It is designed to handle tokenization for the GPT-2 model.
- Tokenization is the process of converting raw text into a format that can
be input into a neural network.
It typically transforms the text into a sequence of integers called tokens.
- from_pretrained('gpt2') is a method use to load t pre-trained tokenizer.
The 'gpt2' argument specifies that it should load the tokenizer pre-configured
and trained to work with the gpt-2 model.
This includes all the speciific settings like vocab size, special tokens, other configurations that are unique to gpt-2.
- tokenizer is the tokenizer object that is now ready to be used to tokenize text.
Ex: methods like tokenizer.encode('your text here') can be used to convert text into tokens.
Ex: methods like tokenizer.decode([list of tokens]) can be used to convert the tokens back into text.
"""
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
"""
The tokenizer has a padding token and a end of sequence tokenizer.
Set these two values to be the same.
- Set the padding token of the tokenizer to the same value of the end of sequence token.
- End of sequence token: A special token used by models like gpt2 to indicate the end of a text sequence.
It is used by the model to determine when a sentence or a text has finished.
- Padding token: Used to fill up sequences to a uniform length.
This is necessary because most neural networks require inputs of the same size.
If you have sentences or inputs of different lengths you use the padding token to equalize their lengths by padding shorter sequences.
Normally the padding token might be a special token that is specific for padding purposes.
- When you set tokenizer.pad_token=tokenizer.eos_token, you tell the tokenizer to use the EOS token also as the padding token.
This is useful in scenarios where you want to simplify the token set or when the model's behavior aligns better with using the eos token for padding.
This is not typical for most applicaitons.
- Causes every added padding token to be treated as signaling the end of a sequence. 
This affects how the model interprets sequences.

"""
tokenizer.pad_token = tokenizer.eos_token  

# Tokenize the training text
"""
- Prepare the text data for input into a neural network/here we specifically prepare it for models like gpt-2.
- tokenizer(training_text): calls the tokenizer on training_text (could be a single string or a batch of texts/strings).
The tokenizer converts the text into tokens that the model can understand aka it converts it to numerical representations. 
- return_tensors='pt': Specifies the format of the returned data.
pt stands for PyTorch tensors.
Other options include tf/TensorFlow tensors, np/numpy arrays.
This ensures the output is compatible with PyTorch models as tensors.
- padding=True:Ensures that all sequences in the batch are padded to the length of the longest sequence
in the case that training_text contains multiple texts.
- max_length: Pads all sequences to the length specified. 
Note that this uniformity in sequence length is required for batch processing in neural networks.
This is the max length of the sequences.
Any text longer than 512 tokens will be truncated.
All texts will be padded to this length since padding=True.
The choice of 512 tokens aligns with typical configurations of many transformer based models like gpt2
because they often have a max input size of 512 tokens. 
- inputs: the output will be a batch of tokenized text data that is formated as PyTorch tensors. 
They are ready to be fed into a model for tasks like training.  
"""
inputs = tokenizer(training_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

"""
- Define a custom dataset implementation designed to work with PyTorch.
- This class inherits from torch.utils.data.Dataset (the base class for all datasets in PyTorch).
- This prepares data for use in training a model that requires inputs and corresponding labels where labels are the inputs themselves.
- Helps us train the model to understand and generate text based on the context provided by the inputs.
"""
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # Set labels to be the same as input_ids
        return item

# Create the dataset
"""
Create an instance of the ConversationDataset class using the inputs variable as its parameter.
Prepares your data for neural network training within the PyTorch framework.
"""
dataset = ConversationDataset(inputs)

"""
Use the Hugging Face transformers library to laod a pre-trained model.
- Here we laod the pre-trained model gpt2.
"""
model = GPT2LMHeadModel.from_pretrained('gpt2')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

prompt = "What causes nausea?"
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

# Generate a response
output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=1000,
    temperature=0.3,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,  # Prevent repeating n-grams
    num_beams=5,  # Beam search
    length_penalty=0.8,  # Adjust length of responses
    do_sample=True,
    num_return_sequences=1,
)

# Decode the generated sequence to text
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

# Extract the text after the prompt
response_text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

print(response_text)