import argparse
import torch
import os
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Oh, and maybe a dash of PEFT for that QLoRA goodness later?
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# --- Custom SFT Data Collator ---
class SFTDataCollator:
    """ 
    A custom data collator that ensures SFT-masked labels are correctly padded.
    It expects 'input_ids', 'attention_mask', and SFT-masked 'labels' from the preprocessor.
    It then uses a standard DataCollatorForLanguageModeling for padding and tensorization.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Instantiate the standard collator that will do the actual work
        self.standard_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features: list[dict[str, any]]) -> dict[str, any]:
        # At this point, each dictionary in 'features' should already have:
        # - input_ids: List[int] (or List[List[int]] if nested)
        # - attention_mask: List[int] (or List[List[int]] if nested)
        # - labels: List[int] (SFT-masked, or List[List[int]] if nested)
        
        # --- Debugging of received features before unwrapping ---
        # print(f"DEBUG SFTDataCollator: Received batch of {len(features)} features for collating.")
        # for i, feature_dict_debug in enumerate(features):
        #     for key_to_log in ['input_ids', 'attention_mask', 'labels']:
        #         if key_to_log in feature_dict_debug:
        #             val = feature_dict_debug[key_to_log]
        #             print(f"DEBUG SFTDataCollator (Feature {i}) PRE-UNWRAP '{key_to_log}': Type: {type(val)}, Len: {len(val) if isinstance(val, list) else 'N/A'}, Content (sample): {str(val)[:100]}")
        #             if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
        #                 print(f"DEBUG SFTDataCollator (Feature {i}) PRE-UNWRAP '{key_to_log}': Nested list detected. Inner list type: {type(val[0])}")
        # --- End of Pre-Unwrap Debugging ---

        unwrapped_features = []
        for i, feature_dict in enumerate(features):
            new_feature_dict = feature_dict.copy() # Shallow copy is fine as we replace lists, not modify them in-place.
            for key_to_check in ['input_ids', 'attention_mask', 'labels']:
                if key_to_check in new_feature_dict:
                    current_value = new_feature_dict[key_to_check]
                    # Check if it's a list, containing exactly one other list
                    if isinstance(current_value, list) and len(current_value) == 1 and isinstance(current_value[0], list):
                        # print(f"SFTDataCollator: Unwrapping {key_to_check} for feature {i}. Original length 1, inner list type {type(current_value[0])}")
                        new_feature_dict[key_to_check] = current_value[0]
            unwrapped_features.append(new_feature_dict)
        
        # --- Debugging after unwrap ---
        # print(f"DEBUG SFTDataCollator: Post-unwrap, passing {len(unwrapped_features)} features to standard_collator.")
        # if unwrapped_features:
        #     for i, feature_dict_debug in enumerate(unwrapped_features[:1]): # Log first feature post-unwrap
        #         for key_to_log in ['input_ids', 'attention_mask', 'labels']:
        #             if key_to_log in feature_dict_debug:
        #                 val = feature_dict_debug[key_to_log]
        #                 print(f"DEBUG SFTDataCollator (Feature {i}) POST-UNWRAP '{key_to_log}': Type: {type(val)}, IsList: {isinstance(val, list)}, Content (sample): {str(val)[:100]}")
        #                 if isinstance(val, list) and val and not all(isinstance(x, int) for x in val):
        #                     print(f"CRITICAL SFTDataCollator (Feature {i}) POST-UNWRAP '{key_to_log}': Still contains non-integers after unwrap!")
        # --- End of Post-Unwrap Debugging ---

        return self.standard_collator(unwrapped_features)


# --- Argument Parsing Chapel ---
# Here we define the sacred scrolls (arguments) our script will accept.
def parse_arguments():
    """Parses command-line arguments for the training script.
    Think of this as the dojo's intake form for new training regimens.
    """
    parser = argparse.ArgumentParser(
        description="Simple Fine-Tuning Script - The Lonn-San Edition"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",  # The true target: Phi-4 Mini Instruct!
        help="Path to pretrained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="persona_data/zen_coder_generated.jsonl", # Path to our Zen Coder data!
        help="Path to dataset or dataset identifier from Hugging Face Hub / local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine_tuned_model_dojo",
        help="Directory where the fine-tuned model and checkpoints will be stored. Our trophy room!",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,  # Optimal number of epochs based on previous runs.
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,  # How many scrolls to read at once, per GPU sensei.
        help="Batch size per GPU/CPU for training.",
    )
    # TODO: Add more arguments as needed, Master! Learning rate, LoRA configs, etc.

    args = parser.parse_args()
    print(f"📜 Training Configuration Scrolls Read: {args}")
    return args


# --- Main Training Ceremony ---
# This is where the real magic happens! (Or will happen, with your guidance)
def main_training_ritual(args):
    """The core function where the model learns and evolves.
    Currently, a tranquil meditation spot awaiting your fine-tuning incantations.
    """
    # Load environment variables from .env file
    load_dotenv()
    hf_username = os.getenv("HF_USERNAME")
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    # Define a model name for Hugging Face Hub and W&B run name
    # You can customize this further if needed, Lonn-San!
    base_model_name = args.model_name_or_path.split("/")[-1] # e.g., "Phi-4-mini-instruct"
    hub_model_id_template = f"{hf_username}/{base_model_name}-persona-qlora" if hf_username else None
    wandb_run_name = f"{base_model_name}-persona-qlora-training"

    print(f"🧘 Initiating training with model: {args.model_name_or_path}")
    print(f"💾 Outputting to: {args.output_dir}")

    # TODO: Step 1: Load the tokenizer, like finding the right calligraphy brush.
    print(f"🖌️ Loading tokenizer for model: {args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            print("PAD_TOKEN not found, setting to EOS_TOKEN. Common for autoregressive models.")
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded successfully. Pad token: {tokenizer.pad_token}, EOS token: {tokenizer.eos_token}")
        print(f"Special tokens map: {tokenizer.special_tokens_map}")

        # Verify chat template and model_max_length
        print(f"DEBUG: Tokenizer chat template looks like: {tokenizer.chat_template}")
        if not tokenizer.chat_template:
            print("WARNING: Tokenizer chat_template is not set! This might cause issues with apply_chat_template.")
            # Attempt to set a default one based on Phi-4 documentation if it's missing, though AutoTokenizer should handle this.
            # tokenizer.chat_template = "<|system|>{{message.content}}<|end|><|user|>{{message.content}}<|end|><|assistant|>{{message.content}}"

        if tokenizer.model_max_length is None or tokenizer.model_max_length > 8192: # Check for unusually large values
            print(f"WARNING: tokenizer.model_max_length is {tokenizer.model_max_length}. Setting to 4096 for Phi-4-mini-instruct.")
            tokenizer.model_max_length = 4096
        print(f"Using tokenizer.model_max_length: {tokenizer.model_max_length}")

    except Exception as e:
        print(f"😭 Oh dear! Failed to load tokenizer: {e}")
        print("Ensure the model name is correct and you have an internet connection if downloading.")
        return # Exit early if tokenizer loading fails

    # TODO: Step 2: Load the model. Is it a mighty pre-trained sensei or a fresh apprentice?
    print(f"🧘 Summoning the model: {args.model_name_or_path} with QLoRA enchantment!")

    # Configure BitsAndBytes for 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"🧮 BitsAndBytesConfig for QLoRA: {bnb_config}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config, # Apply QLoRA quantization
            torch_dtype=torch.bfloat16, # Still specify compute dtype here for consistency
            device_map="auto"           # Automatically map model to available devices
            # trust_remote_code=True # Phi-4 might need this if not using the latest transformers
        )
        print(f"✅ Model loaded successfully with 4-bit quantization! It's on device: {model.device}")
        # print(f"Model configuration: {model.config}") # We can keep this, or remove if too verbose now

    except Exception as e:
        print(f"😭 Alas! Failed to load model with QLoRA: {e}")
        print("Things to check: model name, internet connection, GPU memory (if using CUDA), bitsandbytes installation.")
        print("If it's a new model like Phi-4, you might need 'trust_remote_code=True' or a transformers library update.")
        return # Exit early if model loading fails

    # TODO: Step 2.5 (For QLoRA/PEFT): Prepare model for k-bit training and apply LoRA config.
    print("✨ Preparing model for k-bit training (PEFT)...")
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")

    # Define LoRA configuration
    # For Phi-3/4, common target modules are often related to attention and MLP layers.
    # These are educated guesses; actual module names can be found by inspecting model.named_modules()
    lora_config = LoraConfig(
        r=16,  # Rank of the LoRA matrices
        lora_alpha=32,  # Scaling factor for LoRA matrices
        target_modules=[
            "qkv_proj",     # Combined query, key, value projection in some architectures
            "o_proj",       # Output projection in attention
            "gate_up_proj", # Gated linear unit's up projection in MLP (common in Llama/Phi)
            "down_proj"     # Down projection in MLP
            # For other models, you might see "query_key_value", "dense", "attention.dense", etc.
            # Or more specific like "q_proj", "k_proj", "v_proj" if not combined.
        ],
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",  # Typically, biases are not trained in LoRA
        task_type="CAUSAL_LM",  # Specify the task type
    )
    print(f"📜 LoRA Config: {lora_config}")

    print("Applying LoRA configuration to the model...")
    model = get_peft_model(model, lora_config)
    print("✅ LoRA applied successfully!")

    print("📊 Trainable parameters after QLoRA:")
    model.print_trainable_parameters()
    # model = prepare_model_for_kbit_training(model) # This was the old location
    # lora_config = LoraConfig(...) # Define your LoRA scroll here! # This was the old location
    # model = get_peft_model(model, lora_config) # This was the old location
    # model.print_trainable_parameters() # See how much we're actually training - efficiency is key!

    # --- Preprocessing Function Definition ---
    # This function will be applied to the dataset to format it using the chat template.
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    if assistant_token_id == tokenizer.unk_token_id:
        print("WARNING: '<|assistant|>' token not found in tokenizer. SFT Label masking might be incorrect or ineffective.")

    def preprocess_function(examples_batch):
        batch_messages = []
        # Ensure all content is string type
        instructions = [str(item) if item is not None else "" for item in examples_batch["instruction"]]
        inputs = [str(item) if item is not None else "" for item in examples_batch["input"]]
        outputs = [str(item) if item is not None else "" for item in examples_batch["output"]]

        for i in range(len(instructions)):
            messages = [
                {"role": "system", "content": instructions[i]},
                {"role": "user", "content": inputs[i]},
                {"role": "assistant", "content": outputs[i]}
            ]
            batch_messages.append(messages)

        try:
            # Tokenize the batch of messages
            tokenized_result = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=False, # Crucial for supervised fine-tuning
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=False, # Data collator will handle padding per batch
                return_attention_mask=True, # Request attention mask
                return_tensors=None, # Return list of lists for .map()
            )
        except Exception as e:
            print(f"ERROR during apply_chat_template: {e}")
            print(f"Problematic batch_messages (first item): {batch_messages[0] if batch_messages else 'empty batch'}")
            raise # Re-raise the exception to stop execution

        print(f"DEBUG: Type of tokenized_result from apply_chat_template: {type(tokenized_result)}")
        if isinstance(tokenized_result, dict):
            print(f"DEBUG: tokenized_result is a dict. Keys: {list(tokenized_result.keys())}")
            # Ensure attention_mask is present
            if 'input_ids' in tokenized_result and 'attention_mask' not in tokenized_result:
                print("DEBUG: Manually creating attention_mask for dict output.")
                tokenized_result['attention_mask'] = [[1] * len(ids) for ids in tokenized_result['input_ids']]

            # Ensure SFT labels are present and correctly masked
            if 'input_ids' in tokenized_result:
                if 'labels' not in tokenized_result:
                    print("DEBUG: 'labels' key missing from dict output. Creating SFT-masked labels.")
                    sft_labels_batch = []
                    for id_sequence in tokenized_result['input_ids']:
                        labels_seq = list(id_sequence) # copy
                        if assistant_token_id != tokenizer.unk_token_id:
                            try:
                                assistant_idx = id_sequence.index(assistant_token_id)
                                for j in range(assistant_idx + 1): labels_seq[j] = -100
                            except ValueError: # assistant token not found
                                for j in range(len(labels_seq)): labels_seq[j] = -100 # mask all
                        sft_labels_batch.append(labels_seq)
                    tokenized_result['labels'] = sft_labels_batch
                else:
                    print("DEBUG: 'labels' key already present in dict output. Assuming they are SFT-masked.")
                    # Optionally, could re-verify/re-mask here if unsure about the source's SFT masking.
            return tokenized_result
        
        elif isinstance(tokenized_result, list):
            print(f"DEBUG: tokenized_result is a list. Assuming it's a batch of input_ids sequences.")
            if len(tokenized_result) > 0 and isinstance(tokenized_result[0], list):
                input_ids_batch = tokenized_result
                attention_mask_batch = [[1] * len(ids) for ids in input_ids_batch]
                labels_batch = []
                for id_sequence in input_ids_batch:
                    labels_sequence = list(id_sequence) # Make a mutable copy
                    if assistant_token_id != tokenizer.unk_token_id:
                        try:
                            assistant_idx = id_sequence.index(assistant_token_id)
                            for j in range(assistant_idx + 1):
                                labels_sequence[j] = -100
                        except ValueError:
                            print(f"WARNING: '<|assistant|>' token ID {assistant_token_id} not found in sequence: {id_sequence[:20]}... Masking all labels for this sample to be safe.")
                            for j in range(len(labels_sequence)):
                                labels_sequence[j] = -100
                    else:
                        print(f"DEBUG: Assistant token unknown. No SFT masking applied to labels for sequence.")
                    labels_batch.append(labels_sequence)

                print("DEBUG (List Path): Manually creating dict with 'input_ids', 'attention_mask', and SFT-masked 'labels'.")
                return {
                    "input_ids": input_ids_batch,
                    "attention_mask": attention_mask_batch,
                    "labels": labels_batch,
                }
            else:
                print(f"ERROR: tokenized_result is a list, but not structured as expected (list of lists). Content (first 5): {tokenized_result[:5]}")
                # Return empty or minimal dict to avoid crashing .map immediately, but this is an error state.
                return {}

        # Fallback for unexpected types from apply_chat_template
        print(f"ERROR: Unexpected output type from apply_chat_template: {type(tokenized_result)}. Returning empty dict.")
        return {}
    # --- End of Preprocessing Function ---

    # TODO: Step 3: Load and preprocess the dataset. What wisdom will our model learn?
    print(f"WISDOM SEEKING: Attempting to load dataset from: {args.dataset_name_or_path}")
    train_dataset = None
    eval_dataset = None
    try:
        # Load the full dataset
        full_dataset = load_dataset("json", data_files=args.dataset_name_or_path)
        print(f"📚 Full dataset loaded successfully! Content: {full_dataset}")
        original_columns = full_dataset["train"].column_names
        print(f"Original columns: {original_columns}")

        print("Applying preprocessing function to dataset...")
        processed_dataset = full_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=original_columns, # Remove original text columns
            desc="Running tokenizer on dataset", # Add a description for the progress bar
        )
        print(f"Dataset processed. New columns: {processed_dataset['train'].column_names}")
        # print(f"🔍 First processed train example structure: {processed_dataset['train'][0]}") # Shows keys and types/lengths
        print(f"Processed dataset features: {processed_dataset['train'].features}")

        # Split the processed dataset into training and evaluation sets
        if "train" in processed_dataset:
            print("Splitting processed dataset into train and evaluation sets (90/10 split)...")
            # Ensure the split is reproducible if needed by setting a seed
            split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"] # train_test_split names the new eval set 'test'
            print(f"Split complete. Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")
            # print(f"🔍 First example from processed train split (input_ids): {train_dataset[0]['input_ids'][:50]}...") # Print part of it
            # print(f"🔍 First example from processed eval split (input_ids): {eval_dataset[0]['input_ids'][:50]}...")
        else:
            print("Could not find a 'train' split in the loaded dataset to perform a train/eval split.")
            print("Proceeding without an evaluation set, using the full processed 'train' split for training.")
            train_dataset = processed_dataset["train"] # Or handle error appropriately

    except Exception as e:
        print(f"😭 Oh no! Failed to load dataset: {e}")
        print("Please ensure the path is correct and the file is a valid JSONL.")
        # We should probably exit or handle this more gracefully in a real script
        return # Exit early if dataset loading fails

    # TODO: Step 4: Define Training Arguments. The rules of the dojo!
    print("📜 Defining Training Arguments...")
    
    # Prepare for Hugging Face Hub push and W&B logging
    push_to_hub_enabled = False
    hub_model_id_actual = None
    if hf_username and hf_token:
        push_to_hub_enabled = True
        hub_model_id_actual = hub_model_id_template
        print(f"✅ Hugging Face Hub push enabled. Model ID: {hub_model_id_actual}")
    else:
        print("⚠️ Hugging Face Hub push disabled: HF_USERNAME or HUGGINGFACE_API_KEY not found in environment.")

    report_to_wandb = "none"
    if wandb_api_key:
        report_to_wandb = "wandb"
        print(f"✅ Weights & Biases logging enabled. Run name: {wandb_run_name}")
        # Note: W&B project name can be set via environment variable WANDB_PROJECT
        # or you can add an argument for it if you prefer more explicit control.
        # For now, it will use the default project or one set by WANDB_PROJECT env var.
    else:
        print("⚠️ Weights & Biases logging disabled: WANDB_API_KEY not found in environment.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1, # Set eval batch size to 1
        learning_rate=2e-4,  # Common learning rate for QLoRA
        weight_decay=0.01,  # Standard weight decay
        optim="paged_adamw_8bit", # Optimizer for QLoRA
        bf16=True,  # Use bfloat16 precision if available (matches model load)
        # fp16=False, # Set to True if bf16 is not available and you have a T4/V100 etc.
        logging_strategy="steps",
        logging_steps=10,        # Log every 10 steps
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",   # Save checkpoints every epoch
        load_best_model_at_end=True, # Load the best model found during training
        # metric_for_best_model="loss", # Default is loss, explicitly stating for clarity if needed
        gradient_accumulation_steps=1, # Accumulate gradients (effective batch size = N * per_device_batch_size)
        lr_scheduler_type="cosine", # Cosine learning rate scheduler
        warmup_ratio=0.03,          # Warmup ratio for the scheduler
        report_to=report_to_wandb, # report_to="wandb" or "none"
        remove_unused_columns=False, # Keep all columns from the dataset; after preprocessing, only relevant ones exist
        # ddp_find_unused_parameters=False, # Set to False if not using DDP or if encountering issues
        push_to_hub=push_to_hub_enabled,
        hub_model_id=hub_model_id_actual,
        hub_token=hf_token, # Pass the token for push_to_hub
        # run_name=wandb_run_name, # Set run name for W&B if enabled (and supported directly by TrainingArguments)
                                  # Alternatively, W&B picks up run names through env vars or its own init.
                                  # If WANDB_RUN_NAME is set, it will be used.
    )
    print(f"🏋️ TrainingArguments defined: {training_args}")

    # TODO: Step 5: Initialize the Trainer. The head sensei for our training session.
    print("🥋 Initializing the Trainer...")

    # Initialize our SFT Data Collator
    sft_data_collator = SFTDataCollator(tokenizer=tokenizer)
    print(f"SFT Data collator initialized: {sft_data_collator}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,   
        data_collator=sft_data_collator, # Use our custom SFT collator
    )
    print("Trainer initialized.")

    # TODO: Step 6: Begin the training! LET THE LEARNING COMMENCE!
    print("💪 Starting the grand training ceremony!")
    try:
        trainer.train()
        print("🎉 Training ceremony complete!")
    except Exception as e:
        print(f"😭 Catastrophe during training ceremony: {e}")
        print("Check error messages above. GPU memory, dataset issues, or deeper configuration problems could be the cause.")
        return # Exit if training fails catastrophically

    # TODO: Step 7: Save the model. Its newfound wisdom, preserved for eternity (or until the next fine-tune).
    print(f"💾 Saving the enlightened model to {args.output_dir}...")
    try:
        model.save_pretrained(args.output_dir)
        if tokenizer: # Only save tokenizer if it was loaded
             tokenizer.save_pretrained(args.output_dir) # Don't forget the brush!
        print(f"✅ Model and tokenizer saved successfully to {args.output_dir}")
    except Exception as e:
        print(f"😭 Failed to save the final model/tokenizer: {e}")

    print(
        "\n📜 This Padawan has laid the groundwork. The main training ritual awaits your masterful touch, Lonn-San! 📜"
    )
    print("Consider uncommenting and filling in the TODO sections above.")


# --- Script Execution Gateway ---
# This ensures our main ritual is only performed when the script is directly invoked.
if __name__ == "__main__":
    print("✨ Greetings, Master Lonn-San! Your humble fine-tuning script awakes! ✨\n")
    parsed_args = parse_arguments()
    main_training_ritual(parsed_args)
    print("\n🙏 May your model achieve enlightenment! 🙏")
