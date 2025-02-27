import os
import sys
import time
import shutil
from datetime import datetime
import subprocess

def print_header(text):
    print("\n" + "="*80)
    print(f" {text} ".center(80, "="))
    print("="*80 + "\n")

def print_step(step_num, total_steps, text):
    print(f"\n[Step {step_num}/{total_steps}] {text}")
    print("-" * 80)

def clean_directories():
    """Remove processed_dataset and models directories for a fresh start"""
    dirs_to_clean = ['processed_dataset', 'models']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Removing existing {dir_name} directory...")
            try:
                shutil.rmtree(dir_name)
            except Exception as e:
                print(f"❌ Error removing {dir_name}: {str(e)}")
                return False
    return True

def run_command(command, log_file):
    try:
        # Run command and capture output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Write output to both terminal and log file
        with open(log_file, 'a') as f:
            for line in process.stdout:
                sys.stdout.write(line)  # Print to terminal
                f.write(line)          # Write to log file
                f.flush()              # Ensure immediate write to file

        # Wait for process to complete
        return_code = process.wait()
        return return_code == 0
    except Exception as e:
        print(f"\n❌ Error executing command: {str(e)}")
        return False

def main():
    start_time = time.time()
    
    print_header("Face Recognition System Training Pipeline")
    
    # Check if required directories exist
    if not os.path.exists("dataset"):
        print("❌ Error: 'dataset' directory not found!")
        print("Please make sure you have the dataset directory with student images.")
        return
    
    # Clean existing processed data and models
    print_step(1, 3, "Cleaning Previous Data")
    if not clean_directories():
        print("❌ Error: Failed to clean directories!")
        return
    print("✅ Directories cleaned successfully!")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Step 2: Preprocessing
    print_step(2, 3, "Running Preprocessing")
    print("Processing face images and creating aligned dataset...")
    if not run_command(["python", "src/preprocess.py"], log_file):
        print("\n❌ Preprocessing failed! Check the log file for details:")
        print(f"   {os.path.abspath(log_file)}")
        return
    
    # Verify processed dataset
    if not os.path.exists("processed_dataset"):
        print("❌ Error: Preprocessing did not create 'processed_dataset' directory!")
        return
    
    # Step 3: Training
    print_step(3, 3, "Running Model Training")
    print("Training the face recognition model...")
    print("This may take a while depending on your hardware...")
    if not run_command(["python", "src/train.py"], log_file):
        print("\n❌ Training failed! Check the log file for details:")
        print(f"   {os.path.abspath(log_file)}")
        return
    
    # Verify model file
    if not os.path.exists("models/best_model.pth"):
        print("❌ Error: Training did not create 'models/best_model.pth'!")
        return
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print_header("Training Pipeline Completed Successfully!")
    print(f"✅ Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"✅ Log file: {os.path.abspath(log_file)}")
    print(f"✅ Model saved: {os.path.abspath('models/best_model.pth')}")
    print("\nYou can now run the attendance system using:")
    print("python src/attendance.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training pipeline interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        sys.exit(1) 