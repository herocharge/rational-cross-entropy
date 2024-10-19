import dspy
from dspy import InputField, OutputField, Signature
def extract_code(response: str) -> str:
    # extract all text between <code> and </code> tags
    # there might be multiple code blocks, return the first one
    print(response)
    import re
    code_blocks = re.findall(r'<code>(.*?)</code>', response, re.DOTALL)
    out =  code_blocks[-1] if code_blocks else ""
    if out == "":
        code_blocks = re.findall(r'```cpp(.*?)```', response, re.DOTALL)
        out = code_blocks[-1] if code_blocks else ""
    if out == "":
        code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
        out = code_blocks[-1] if code_blocks else ""
    return out

class ExpandDescSignature(Signature):
    """You are an expert in problem description analysis. Your task is to expand the problem description to include all the constraints and conditions.
        For every sentence in the provided description, write a note in paranthesis describing what you can infer and what might be some common misinterpretations that can happen.
        Also turn any latex math into normal natural language text so that it can be easier to reason about.
    """
    desc: str = InputField(format=str)
    expanded_desc: str = OutputField(format=str)

class ExpandDesc(dspy.Module):
    def __init__(self):
        super().__init__()
        self.expand = dspy.Predict(ExpandDescSignature)

    def forward(self, desc: str):
        return dspy.Prediction(expanded_desc=self.expand(desc=desc).expanded_desc)

class Desc2PlanSignature(Signature):
    """You are a reasoning expert. You make arguments for why a particular solution is correct VERY CAREFULLY and RIGOROUSLY. 
        Your task is take a problem description and convert it into valid plan.
        
        It should be HIGH LEVEL do not bother about the syntax of the code or implementation details.
        Note:
        * The plan should be in natural language so that further reasoning can be done on it.
        * We also provide you additional guidelines that you need to keep in mind while writing the plan.
        * Based on the guidelines, decide if the solution should fundamentally change.
        * The problems are often straight forward and are not meant to be tricky, always prioritize O(1) solutions, then O(N), then O(NlogN), then O(N^2) if necessary.
        * FOR EVERY STEP OF THE PLAN, CITE THE PART OF THE PROBLEM DESCRIPTION THAT JUSTIFIES IT. (eg: Step n: <step-description> <problem-description-part-that-justifies-it>)
        * Do not just randomly apply well known techniques, think step by step.
    """
    problem_description: str = InputField(format=str)
    guidelines: str = InputField(format=str)
    solution: str = OutputField(
        format=str, desc="A plan for how we should go about solving this problem."
    )
    explanation: str = OutputField(
        format=str, desc="An explanation of the solution and how it corresponds to the sample input/output."
    )
    time_complexity: str = OutputField(
        format=str, desc="Give the time complexity of the solution in O(.) notation and see if it is efficient."
    )
    plan: str = OutputField(format=str)

class Pseudo2GuidelineGenerator(dspy.Module):
    def __init__(self, desc: str):
        super().__init__()
        # Initialize variables
        self.desc = desc

        # Initialize layers
        self.generate_NL = dspy.ChainOfThought("plan -> natural_language_of_problem_plan_is_solving", n=4)
        self.generate_mistakes = dspy.ChainOfThought("natural_language_of_problem_plan_is_solving, problem_description -> mistakes_to_avoid")
        self.key_sentences = dspy.ChainOfThought("problem_description -> key_sentences")
        self.time_complexity = dspy.ChainOfThought("natural_language_of_problem_plan_is_solving, problem_description -> time_complexity")
        self.is_time_efficient = dspy.ChainOfThought("time_complexity -> is_time_efficient")
        self.alternative_solutions = dspy.ChainOfThought("key_sentences, problem_description, is_time_efficient -> alternative_solutions")
        self.generate_guideline = dspy.ChainOfThought("key_sentences, mistakes_to_avoid, alternative_solutions, time_complexity -> guidelines")
    
    def forward(self, plan):
        natural_language_of_problem = self.generate_NL(plan=plan)
        mistakes_to_avoid = self.generate_mistakes(natural_language_of_problem_plan_is_solving=natural_language_of_problem, problem_description=self.desc)
        key_sentences = self.key_sentences(problem_description=self.desc)
        time_complexity = self.time_complexity(natural_language_of_problem_plan_is_solving=natural_language_of_problem, problem_description=self.desc)
        is_time_efficient = self.is_time_efficient(time_complexity=time_complexity)
        alternative_solutions = self.alternative_solutions(key_sentences=key_sentences, problem_description=self.desc, is_time_efficient=is_time_efficient)
        guidelines = self.generate_guideline(key_sentences=key_sentences, mistakes_to_avoid=mistakes_to_avoid, alternative_solutions=alternative_solutions, time_complexity=time_complexity)
        return dspy.Prediction(guidelines=guidelines)


class IsTimeEfficientSignature(Signature):
    """You are an expert in time complexity analysis. Your task is to analyze the time complexity of a given solution and determine if it is efficient.
        It is only efficient if it is doable in 1 second.
        Some rules of thumb:
        If N = 1e6 or 1,000,000, then O(NlogN) is 1 second, O(N^2) is 1,000,000 seconds, O(N^3) is 1e18 seconds.
        If N = 1e8 or 100,000,000, then O(N) is 1 second, O(N^2) is 1e16 seconds, O(N^3) is 1e24 seconds.
        If N = 1e3 or 1,000, then O(N^3) is 1 second.
        If N = 1e5 or 100,000, then O(N^2) is 1 second.
        There might be more than one variable, in that case you should multiply the constraints.
        Take these into consideration and the problem statement to determine if the solution is efficient.
        Severely discourage solutions that are more than O(N)
        EXPONENTIAL AND FACTORIAL ARE VERY VERY VERY BAD!!! CHange approach immediately.
    """
    time_complexity: str = InputField(format=str)
    problem_description: str = InputField(format=str)
    is_time_efficient: str = OutputField(format=str)

class IsTimeEfficient(dspy.Module):
    def __init__(self):
        super().__init__()
        self.is_time_efficient = dspy.Predict(IsTimeEfficientSignature)
    
    def forward(self, time_complexity: str, problem_description: str):
        return dspy.Prediction(is_time_efficient=self.is_time_efficient(time_complexity=time_complexity, problem_description=problem_description))

class Plan2TimeComplexityGuidelineGenerator(dspy.Module):
    def __init__(self, desc: str):
        super().__init__()
        self.desc = desc
        self.time_complexity = dspy.ChainOfThought("natural_language_of_problem_plan_is_solving, problem_description -> time_complexity")
        self.is_time_efficient = IsTimeEfficient()
        self.generate_time_complexity_guideline = dspy.ChainOfThought("plan, problem_description, is_time_efficient -> time_complexity_guideline")  

    def forward(self, plan):
        time_complexity = self.time_complexity(natural_language_of_problem_plan_is_solving=plan, problem_description=self.desc)
        is_time_efficient = self.is_time_efficient(time_complexity=time_complexity, problem_description=self.desc)
        return dspy.Prediction(
            time_complexity_guideline=self.generate_time_complexity_guideline(
                plan=plan, 
                problem_description=self.desc, 
                is_time_efficient=is_time_efficient
            )
        )

class UpdatePlan(dspy.Module):
    def __init__(self, desc: str):
        super().__init__()
        self.desc = desc
        self.is_statement_crucial = dspy.ChainOfThought("statement, problem_description -> is_statement_crucial")
        self.generate_plan = dspy.ChainOfThought("is_statement_crucial, statement, plan -> improved_plan")

    def forward(self, statement: str, plan: str):
        is_statement_crucial = self.is_statement_crucial(statement=statement, problem_description=self.desc)
        return dspy.Prediction(improved_plan=self.generate_plan(is_statement_crucial=is_statement_crucial, statement=statement, plan=plan))

class Plan2AlternativeSolutionsGenerator(dspy.Module):
    def __init__(self, desc: str):
        super().__init__()
        self.desc = desc
        self.key_sentences = dspy.ChainOfThought("problem_description -> key_sentences")
        self.generate_alternative_solutions = dspy.ChainOfThought("key_sentences, problem_description, previous_guidelines -> alternative_solutions")
        self.time_complexity = dspy.ChainOfThought("alternative_solutions -> time_complexity")
        self.are_alternatives_time_efficient = IsTimeEfficient()
        self.generate_time_efficient_alternatives = dspy.ChainOfThought("are_alternatives_time_efficient, alternative_solutions -> time_efficient_alternatives")
    def forward(self, plan: str, previous_guidelines: str):
        key_sentences = self.key_sentences(problem_description=self.desc)
        alternative_solutions = self.generate_alternative_solutions(key_sentences=key_sentences, problem_description=self.desc, previous_guidelines=previous_guidelines)
        alternative_time_complexity = self.time_complexity(alternative_solutions=alternative_solutions)
        are_alternatives_time_efficient = self.are_alternatives_time_efficient(time_complexity=alternative_time_complexity, problem_description=self.desc)
        return dspy.Prediction(
            # time_efficient_alternatives=self.generate_time_efficient_alternatives(are_alternatives_time_efficient=are_alternatives_time_efficient, alternative_solutions=alternative_solutions)
            alternative_solutions=alternative_solutions
        )

class Plan2MistakesGenerator(dspy.Module):
    def __init__(self, desc: str):
        super().__init__()
        self.desc = desc
        self.generate_NL = dspy.ChainOfThought("plan -> natural_language_of_problem_plan_is_solving")
        self.generate_mistakes = dspy.ChainOfThought("natural_language_of_problem_plan_is_solving, problem_description, previous_guidelines -> mistakes")
        self.correction_guideline = dspy.ChainOfThought("mistakes, previous_guidelines -> correction_guideline")
    def forward(self, plan: str, previous_guidelines: str):
        natural_language_of_problem = self.generate_NL(plan=plan)
        mistakes = self.generate_mistakes(natural_language_of_problem_plan_is_solving=natural_language_of_problem, problem_description=self.desc, previous_guidelines=previous_guidelines)
        return dspy.Prediction(mistakes=mistakes, correction_guideline=self.correction_guideline(mistakes=mistakes, previous_guidelines=previous_guidelines))

class Desc2InvariantsSignature(Signature):
    """To find invariants and monovariants in Codeforces problems:

        Invariants:

        An invariant is a property that remains unchanged through operations.
        Steps: Identify key operations, focus on what stays constant (e.g., parity, sums), and test small cases.
        Common Examples: Array sums, number of inversions in a sequence, or connected components in a graph.
        Monovariants:

        A monovariant is a property that consistently increases or decreases through operations.
        Steps: Identify a measure that moves in one direction and has clear bounds. Ensure it doesn't go backward.
        Common Examples: Degree of sortedness, number of misplaced elements, or traversal depth in a graph.
        Problem Types:

        Sorting problems, game theory, graph manipulation, and traversal are common areas where these concepts apply.
        Practice:

        Review editorials and solve classic problems (e.g., Nim game) to get better at identifying invariants and monovariants.
        Simulate step-by-step on small inputs and hypothesize the invariant or monovariant to prove it holds throughout.

        Given a problem description, identify the invariants and monovariants.
    """
    problem_description: str = InputField(format=str)
    invariants: str = OutputField(format=str)
    monovariants: str = OutputField(format=str)

class Desc2Invariants(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_invariants = dspy.Predict(Desc2InvariantsSignature)
    
    def forward(self, problem_description: str):
        result = self.generate_invariants(problem_description=problem_description)
        return dspy.Prediction(invariants=result.invariants, monovariants=result.monovariants)

class Plan2InvariantsGenerator(dspy.Module):
    def __init__(self, desc: str):
        super().__init__()
        self.desc = desc
        self.desc2invariants = Desc2Invariants()
        self.generate_invariants = dspy.ChainOfThought("plan, problem_description, invariants, monovariants -> invariants_and_monovariants_guideline")

    def forward(self, plan: str):
        result = self.desc2invariants(self.desc)
        return dspy.Prediction(
            invariants_and_monovariants_guideline=self.generate_invariants(
                plan=plan, 
                problem_description=self.desc, 
                invariants=result.invariants, 
                monovariants=result.monovariants
            )
        )

class SummarizeGuideline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("guidelines -> summarized_guidelines") 

    def forward(self, guidelines):
        return dspy.Prediction(summarized_guidelines=self.summarize(guidelines=guidelines))

class Desc2PlanGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_plan = dspy.Predict(Desc2PlanSignature)

    def forward(self, problem_description, guidelines):
        plan = (
            self.generate_plan(
                problem_description=problem_description,
                guidelines=guidelines
            ).plan
        )

        return dspy.Prediction(plan=plan)

class Plan2PseudoSignature(Signature):
    """You are an expert programmer. Your task it to take the plan and problem_description and convert it into pseudo code.
    The pseudo code should contain only the following coding constructs.
    * for
    * if else
    * functions
    * input
    * output
    * assignment
    * sort
    * search
    * math operations

    You need to carefully construct the pseudo code step by step according to the plan. Insert comments to guide the coder.
    """
    problem_description: str = InputField(format=str)
    plan: str = InputField(format=str)
    pseudo_code: str = OutputField(format=str)
    
class Plan2PseudoCodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_pseudo_code = dspy.Predict(Plan2PseudoSignature)

    def forward(self, plan: str, problem_description: str):
        return dspy.Prediction(pseudo_code=self.generate_pseudo_code(problem_description=problem_description, plan=plan).pseudo_code)

class Reason2CodeSignature(Signature):
    """You are an expert coder. Your task is take pseudo code and input output format and convert it into valid c++ code.
    
    Note:
    * USE VERBOSE VARIABLE NAMES.
    * Use time efficient functions and data structures.
    * use #include<bits/stdc++.h>
    * use #define ll long long
    * use ll instead of int
    * Surround the code with <code> tags only.
        For example:
    <code>
    ...
    </code>
    """

    pseudo_code: str = InputField(format=str)
    time_complexity: str = OutputField(
        format=str, desc="Give the time complexity of the solution in O(.) notation and see if it is efficient."
    )
    cpp_program: str = OutputField(format=str)


class Reason2CodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_code = dspy.Predict(Reason2CodeSignature)

    def forward(self, pseudo_code, input_ouput_format: str):
        cpp_code = extract_code(
            self.generate_code(
                pseudo_code=pseudo_code,
                input_ouput_format=input_ouput_format
            ).cpp_program
        )

        return dspy.Prediction(cpp_program=cpp_code)