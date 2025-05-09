import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Oh, and maybe a dash of PEFT for that QLoRA goodness later?
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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
        default="mistralai/Mistral-7B-v0.1",  # A sensible default, but choose your fighter!
        help="Path to pretrained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        # default="your_favorite_dataset_here", # TODO: Master Lonn-San, guide us to the sacred data!
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
        default=1,  # Let's start with one lap around the training track.
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,  # How many scrolls to read at once, per GPU sensei.
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
    print(f"🧘 Initiating training with model: {args.model_name_or_path}")
    print(f"💾 Outputting to: {args.output_dir}")

    # TODO: Step 1: Load the tokenizer, like finding the right calligraphy brush.
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token # Common practice for autoregressive models

    # TODO: Step 2: Load the model. Is it a mighty pre-trained sensei or a fresh apprentice?
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     # torch_dtype=torch.bfloat16, # For faster training on compatible GPUs!
    #     # device_map="auto" # Let the Hugging Face spirits decide GPU allocation.
    # )

    # TODO: Step 2.5 (For QLoRA/PEFT): Prepare model for k-bit training and apply LoRA config.
    # model = prepare_model_for_kbit_training(model)
    # lora_config = LoraConfig(...) # Define your LoRA scroll here!
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters() # See how much we're actually training - efficiency is key!

    # TODO: Step 3: Load and preprocess the dataset. What wisdom will our model learn?
    # train_dataset = ...
    # eval_dataset = ... (Optional, but good for checking progress!)

    # TODO: Step 4: Define Training Arguments. The rules of the dojo!
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     num_train_epochs=args.num_train_epochs,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     # ... other vital parameters like learning_rate, weight_decay, logging_steps, etc.
    #     report_to="none", # Or "wandb", "tensorboard" if you have those altars set up.
    # )

    # TODO: Step 5: Initialize the Trainer. The head sensei for our training session.
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     # eval_dataset=eval_dataset, # If you have one!
    #     # tokenizer=tokenizer, # Useful for some Trainer features
    # )

    # TODO: Step 6: Begin the training! LET THE LEARNING COMMENCE!
    # print("💪 Starting the grand training ceremony!")
    # trainer.train()

    # TODO: Step 7: Save the model. Its newfound wisdom, preserved for eternity (or until the next fine-tune).
    # print(f"🎉 Training complete! Saving the enlightened model to {args.output_dir}")
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir) # Don't forget the brush!

    print(
        "\\n📜 This Padawan has laid the groundwork. The main training ritual awaits your masterful touch, Lonn-San! 📜"
    )
    print("Consider uncommenting and filling in the TODO sections above.")
    print(
        "You might need to install: pip install torch transformers datasets accelerate peft bitsandbytes"
    )


# --- Script Execution Gateway ---
# This ensures our main ritual is only performed when the script is directly invoked.
if __name__ == "__main__":
    print("✨ Greetings, Master Lonn-San! Your humble fine-tuning script awakes! ✨\\n")
    parsed_args = parse_arguments()
    main_training_ritual(parsed_args)
    print("\\n🙏 May your model achieve enlightenment! 🙏")
