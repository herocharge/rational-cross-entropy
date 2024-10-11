import subprocess
import os

class Problem:
    def __init__(self, desc, sample_in_file, sample_out_file) -> None:
        self.desc = desc
        self.sample_in_file = sample_in_file
        self.sample_out_file = sample_out_file
        with open(sample_in_file, 'r') as f:
            self.sample_in = f.read().strip()
        with open(sample_out_file, 'r') as f:
            self.sample_out = f.read().strip()
        self.custom_test_in_files = []
        self.custom_test_out_files = []
        self.solutions = []

    def add_solution(self, solution: str):
        self.solutions.append(solution)

    def run_cpp_solution(self, code: str, filename: str, input_file: str, timeout: int = 5) -> str:
        filename = filename + ".cpp"
        # save code to file
        with open(filename, "w") as f:
            f.write(code)

        # Compile the C++ code
        compile_command = f"g++ -std=c++17 {filename} -o temp_executable"
        compile_result = subprocess.run(compile_command, shell=True, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            print(f"Compilation error: {compile_result.stderr}")
            return ""
        
        # Run the compiled executable with timeout
        run_command = f"./temp_executable < \"{input_file}\""
        try:
            run_result = subprocess.run(run_command, shell=True, capture_output=True, text=True, timeout=timeout)
            output = run_result.stdout.strip()
        except subprocess.TimeoutExpired:
            output = "Timeout"
        
        # Clean up
        os.remove("temp_executable")
        
        return output

    def run_py_solution(self, code: str, filename: str, input_file: str, timeout: int = 5) -> str:
        filename = filename + ".py"
        # save code to file (python)
        with open(filename, "w") as f:
            f.write(code)
        
        # Run the Python code with timeout
        run_command = f"python {filename} < {input_file}"
        try:
            run_result = subprocess.run(run_command, shell=True, capture_output=True, text=True, timeout=timeout)
            output = run_result.stdout.strip()
        except subprocess.TimeoutExpired:
            output = "Timeout"
        
        return output

    def test_solution(self, output: str, expected_output: str):
        if output == "Timeout":
            return 0, [("Timeout", "Expected output", "Timeout")]
        
        output_lines = output.split('\n')
        expected_lines = expected_output.split('\n')
        correct_count = 0
        wrong_cases = []
        for i, (expected, actual) in enumerate(zip(expected_lines, output_lines)):
            if expected == actual:
                correct_count += 1
            else:
                wrong_cases.append((f"Case #{i+1}", expected, actual))
        score = correct_count / len(expected_lines)
        return score, wrong_cases

    def test_code(self, code: str, filename: str = "temp", lang: str = "cpp") -> float:
        '''
        return [0, 1] based on number of correct answers
        '''
        if lang == "cpp":
            run_solution = self.run_cpp_solution
        else:
            run_solution = self.run_py_solution

        # Test sample input
        sample_output = run_solution(code, filename, self.sample_in_file)
        with open(self.sample_out_file, 'r') as f:
            expected_sample_output = f.read().strip()
        sample_score, sample_wrong_cases = self.test_solution(sample_output, expected_sample_output)

        # Test custom inputs
        custom_scores = []
        failed_testcases = sample_wrong_cases

        for in_file, out_file in zip(self.custom_test_in_files, self.custom_test_out_files):
            custom_output = run_solution(code, filename, in_file)
            with open(out_file, 'r') as f:
                expected_custom_output = f.read().strip()
            custom_score, custom_wrong_cases = self.test_solution(custom_output, expected_custom_output)
            custom_scores.append(custom_score)
            failed_testcases.extend(custom_wrong_cases)

        # Calculate overall score
        total_score = (sample_score + sum(custom_scores)) / (1 + len(custom_scores))
        
        return total_score, failed_testcases