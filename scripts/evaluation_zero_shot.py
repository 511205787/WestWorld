import os
import argparse
import subprocess
import csv

CKPTS = {"MLPEnsemble":["./Trajworld-MLPEnsemble/6lp4pek1/checkpoints/last.ckpt"], 
         "TDM":["./Trajworld-TDM/lghcs7pz/checkpoints/epoch=0-val_loss=0.94.ckpt"]}


dataset_mapping = {"walker2d":124, "hopper_gym":125, "franka":121}

class run_zero_shot:
    def __init__(self, ckpt, data_name, model_type="MLPEnsemble", csv_path="evaluation_results.csv"):
        # Get the parent directory (one level up from scripts/)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.script_dir)

        # Convert checkpoint path to absolute path if it's relative
        if not os.path.isabs(ckpt):
            self.ckpt_path = os.path.abspath(os.path.join(self.parent_dir, ckpt))
        else:
            self.ckpt_path = ckpt

        self.data_name = data_name
        assert data_name in dataset_mapping
        self.data_idx = dataset_mapping[self.data_name]
        self.model_type = model_type
        self.csv_path = os.path.join(self.parent_dir, csv_path)

    def check_result_exists(self):
        """
        Check if result already exists in CSV file.

        Returns:
            bool: True if result exists, False otherwise
        """
        if not os.path.exists(self.csv_path):
            return False

        # Extract checkpoint filename for comparison
        checkpoint_name = os.path.basename(self.ckpt_path)

        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if same model, dataset, and checkpoint
                    if (row.get('model_name') == self.model_type and
                        row.get('dataset_name') == self.data_name and
                        row.get('checkpoint') == checkpoint_name):
                        print(f"Result already exists in {self.csv_path}:")
                        print(f"  Model: {self.model_type}")
                        print(f"  Dataset: {self.data_name}")
                        print(f"  Checkpoint: {checkpoint_name}")
                        print(f"  MAE: {row.get('mae')}, MSE: {row.get('mse')}, RMSE: {row.get('rmse')}")
                        return True
        except Exception as e:
            print(f"Warning: Error reading CSV file: {e}")
            return False

        return False

    def run(self):
        if self.model_type == "MLPEnsemble":
            # Create results directory if it doesn't exist
            results_dir = os.path.join(self.parent_dir, "results")
            os.makedirs(results_dir, exist_ok=True)

            # Path to evaluation script in parent directory
            eval_script = os.path.join(self.parent_dir, "evaluation_MLPEnsemble.py")

            # Extract checkpoint name (without extension) for directory naming
            checkpoint_name = os.path.splitext(os.path.basename(self.ckpt_path))[0]

            # Set figure directory inside results folder with checkpoint name
            fig_dir = os.path.join(results_dir, self.model_type, f"figures_{self.data_name}_{checkpoint_name}")

            # Build the command with hydra overrides
            cmd = [
                "python",
                eval_script,
                f"ckpt_path='{self.ckpt_path}'",
                f"method={self.model_type}",
                f"data.filter_task_ids=[{self.data_idx}]",
                f"data.test_task_ids=[{self.data_idx}]",
                f"data.h5_dir={os.path.join(self.parent_dir, f'{self.data_name}_h5')}",
                f"data.test_h5_dir={os.path.join(self.parent_dir, f'{self.data_name}_h5')}",
                f"eval_figure_dir='{fig_dir}'",  # Save figures to results folder
                f"+eval_csv_path={self.csv_path}",  # Save CSV to results folder
                f"+dataset_name={self.data_name}",  # Pass dataset name for CSV logging (+ for new key)
                "+save_to_csv=true"  # Enable CSV saving (+ for new key)
            ]

            print(f"Running command: {' '.join(cmd)}")

            # Run the subprocess from parent directory
            result = subprocess.run(
                cmd,
                cwd=self.parent_dir,  # Set working directory to parent
                check=True,  # Raise exception if command fails
                capture_output=False  # Show output in real-time
            )

            return result.returncode

        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.parent_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Path to evaluation script in parent directory
        eval_script = os.path.join(self.parent_dir, "evaluation_TDM.py")

        # Extract checkpoint name (without extension) for directory naming
        checkpoint_name = os.path.splitext(os.path.basename(self.ckpt_path))[0]

        # Set figure directory inside results folder with checkpoint name
        fig_dir = os.path.join(results_dir, self.model_type, f"figures_{self.data_name}_{checkpoint_name}")

        # Build the command with hydra overrides
        cmd = [
            "python",
            eval_script,
            f"ckpt_path='{self.ckpt_path}'",
            f"method={self.model_type}",
            f"data.filter_task_ids=[{self.data_idx}]",
            f"data.test_task_ids=[{self.data_idx}]",
            f"data.h5_dir={os.path.join(self.parent_dir, f'{self.data_name}_h5')}",
            f"data.test_h5_dir={os.path.join(self.parent_dir, f'{self.data_name}_h5')}",
            f"eval_figure_dir='{fig_dir}'",  # Save figures to results folder
            f"+eval_csv_path={self.csv_path}",  # Save CSV to results folder
            f"+dataset_name={self.data_name}",  # Pass dataset name for CSV logging (+ for new key)
            "+save_to_csv=true"  # Enable CSV saving (+ for new key)
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Run the subprocess from parent directory
        result = subprocess.run(
            cmd,
            cwd=self.parent_dir,  # Set working directory to parent
            check=True,  # Raise exception if command fails
            capture_output=False  # Show output in real-time
        )

        return result.returncode



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot evaluation")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--data", type=str, help="Dataset name")
    parser.add_argument("--model", type=str, default="MLPEnsemble", help="Model type")
    parser.add_argument("--csv", type=str, default=None, help="CSV file path for results (default: results/{model}/evaluation_results.csv)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if result exists")

    args = parser.parse_args()

    # Set default CSV path with model subdirectory if not specified
    if args.csv is None:
        args.csv = f"results/{args.model}/evaluation_results.csv"

    for ckpt in CKPTS[args.model]:
        # Check if checkpoint exists
        ckpt_abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ckpt))
        if not os.path.exists(ckpt_abs_path):
            print(f"\n{'='*60}")
            print(f"ERROR: Checkpoint file does not exist!")
            print(f"  Path: {ckpt}")
            print(f"  Absolute path: {ckpt_abs_path}")
            print(f"{'='*60}\n")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs_path}")

        for data in dataset_mapping:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(ckpt)} on {data}")
            print(f"{'='*60}")

            # Create evaluator
            evaluator = run_zero_shot(
                ckpt=ckpt,
                data_name=data,
                model_type=args.model,
                csv_path=args.csv
            )

            # Check if result already exists
            if not args.force and evaluator.check_result_exists():
                print("Skipping evaluation (result already exists).")
                print("Use --force to re-run evaluation anyway.\n")
                continue  # Skip to next iteration instead of exiting

            # Run evaluation
            try:
                evaluator.run()
                print("Evaluation completed successfully!\n")
            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed with error code {e.returncode}\n")
                # Continue to next instead of exiting
                continue
            except Exception as e:
                print(f"Error: {e}\n")
                # Continue to next instead of exiting
                continue
        break