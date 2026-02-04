"""
MIT-Style Recursive Language Model for Pharmaceutical Labels
Based on: https://arxiv.org/abs/2512.24601
Authors: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)

Key Innovation: Treats the label text as an external environment variable
that the LLM can programmatically inspect, search, and recursively process.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from io import StringIO
import sys


@dataclass
class RLMTrace:
    """Trace of RLM execution for debugging and visualization"""
    step: int
    action: str
    code: str
    result: Any
    sub_calls: int = 0


@dataclass
class DrugInfo:
    """Structured drug information"""
    drug_name: Optional[str] = None
    indications: List[str] = None
    usage: Optional[str] = None
    dosage: Optional[str] = None
    warnings: List[str] = None
    active_ingredients: List[str] = None
    
    def __post_init__(self):
        if self.indications is None:
            self.indications = []
        if self.warnings is None:
            self.warnings = []
        if self.active_ingredients is None:
            self.active_ingredients = []


class REPLEnvironment:
    """
    Python REPL environment for RLM
    Stores context and provides helper functions
    """
    
    def __init__(self, context: str):
        self.context = context
        self.variables = {}
        self.execution_trace = []
        
    def execute(self, code: str) -> Any:
        """
        Execute Python code in the REPL environment
        The context is available as 'context' variable
        """
        # Create safe execution environment
        safe_globals = {
            'context': self.context,
            're': re,
            'len': len,
            'str': str,
            'list': list,
            'dict': dict,
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'print': print,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'min': min,
                'max': max,
            }
        }
        
        # Add stored variables
        safe_globals.update(self.variables)
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Execute code
            exec(code, safe_globals, self.variables)
            
            # Get printed output
            output = captured_output.getvalue()
            
            # Try to get return value from 'result' variable
            result = self.variables.get('result', output)
            
            return result
            
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
    
    def peek(self, n: int = 1000) -> str:
        """Helper: Preview first n characters"""
        return self.context[:n]
    
    def grep(self, pattern: str) -> List[str]:
        """Helper: Find lines matching regex pattern"""
        matches = []
        for line in self.context.split('\n'):
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(line)
        return matches
    
    def extract_section(self, section_name: str) -> str:
        """Helper: Extract a labeled section from the text"""
        # Common pharmaceutical label sections
        patterns = {
            'indications': r'(?:INDICATIONS?|USES?|USAGE?|THERAPEUTIC INDICATIONS?)[\s\S]*?(?=\n[A-Z\s]{10,}|\Z)',
            'dosage': r'(?:DOSAGE|ADMINISTRATION)[\s\S]*?(?=\n[A-Z\s]{10,}|\Z)',
            'warnings': r'(?:WARNINGS?|CONTRAINDICATIONS?|PRECAUTIONS?)[\s\S]*?(?=\n[A-Z\s]{10,}|\Z)',
            'description': r'(?:DESCRIPTION|DRUG|MEDICATION)[\s\S]*?(?=\n[A-Z\s]{10,}|\Z)',
        }
        
        pattern = patterns.get(section_name.lower())
        if pattern:
            match = re.search(pattern, self.context, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""


class MITStyleRLM:
    """
    MIT-Style Recursive Language Model
    Implements the RLM pattern: LLM + REPL + Recursive Calls
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.trace = []
        self.call_count = 0
        
    def _llm_call(self, prompt: str, system: str = None, temp: float = 0.3) -> str:
        """
        Base LLM call (sub-LLM in RLM terminology)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temp,
            "max_tokens": 2000
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        self.call_count += 1
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"LLM call failed: {response.status_code}")
    
    def rlm_completion(self, query: str, context: str) -> str:
        """
        Main RLM completion method
        Follows MIT's RLM pattern:
        1. Load context into REPL as variable
        2. Root LLM writes code to inspect and decompose context
        3. Root LLM makes recursive sub-LLM calls on selected portions
        4. Aggregate results and return answer
        """
        print("\n" + "="*80)
        print("🔄 MIT-STYLE RECURSIVE LANGUAGE MODEL")
        print("="*80 + "\n")
        
        # Initialize REPL environment with context
        env = REPLEnvironment(context)
        
        # System prompt for Root LLM (this is the key innovation)
        root_system = """You are a Recursive Language Model (RLM) operating in a Python REPL environment.

The user's input context is stored in a variable called 'context' (a string).

You can interact with this context by writing Python code to:
1. PEEK: Print context[:1000] to preview the beginning
2. GREP: Use re.search() or re.findall() to find patterns
3. SLICE: Access context[start:end] to get specific portions
4. RECURSIVE CALL: Use llm_call(snippet) to invoke a sub-LLM on selected text

AVAILABLE FUNCTIONS:
- llm_call(text, task) -> calls a sub-LLM to process 'text' for 'task'
- env.peek(n) -> preview first n chars
- env.grep(pattern) -> find lines matching regex
- env.extract_section(name) -> extract common sections

Your code should:
1. First inspect/understand the context structure
2. Identify relevant portions for the query
3. Make focused sub-LLM calls on those portions
4. Aggregate results
5. Store final answer in 'result' variable

Write Python code only. Be efficient - don't process irrelevant portions."""

        # Create the root LLM query
        root_query = f"""Query: {query}

The context variable contains pharmaceutical label text.

Write Python code to:
1. Inspect the context structure
2. Find relevant sections (indications, dosage, warnings, etc.)
3. Make sub-LLM calls to extract structured information
4. Aggregate results into 'result' variable

Remember: The context is a string variable. You can slice, search, and process it programmatically."""

        print("📋 Step 1: Root LLM planning inspection strategy...")
        
        # Root LLM generates code
        code = self._llm_call(root_query, system=root_system, temp=0.3)
        
        print("📝 Generated code:")
        print("-" * 80)
        print(code)
        print("-" * 80 + "\n")
        
        # Clean code (remove markdown formatting)
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Inject llm_call function into environment
        def llm_call(text: str, task: str = "extract information") -> str:
            """Sub-LLM call function available in REPL"""
            print(f"  🔄 Sub-LLM call: {task[:50]}...")
            prompt = f"Task: {task}\n\nText:\n{text}\n\nProvide a clear, structured answer:"
            return self._llm_call(prompt, temp=0.2)
        
        env.variables['llm_call'] = llm_call
        env.variables['env'] = env
        
        print("⚙️  Step 2: Executing code in REPL environment...")
        
        # Execute the code
        result = env.execute(code)
        
        print(f"\n✅ Execution complete! Made {self.call_count} LLM calls\n")
        
        # Get final result
        final_result = env.variables.get('result', result)
        
        return str(final_result)
    
    def extract_drug_info(self, context: str) -> DrugInfo:
        """
        High-level method: Extract structured drug information using RLM
        """
        print("\n🏥 PHARMACEUTICAL LABEL EXTRACTION (MIT RLM)")
        print("="*80 + "\n")
        
        # Query for extraction
        query = """Extract the following information from this pharmaceutical label:
1. Drug name
2. Active ingredients
3. Indications and uses
4. Dosage and administration
5. Warnings and contraindications

Return as structured data."""
        
        # Use RLM to process
        result = self.rlm_completion(query, context)
        
        # Parse result into DrugInfo
        # The RLM should have returned structured text
        drug_info = self._parse_result_to_druginfo(result, context)
        
        return drug_info
    
    def extract_indications_only(self, context: str) -> str:
        """
        Specialized RLM method: Extract ONLY the INDICATIONS AND USAGE section
        from pharmaceutical labels using focused RLM approach.
        
        This method is optimized for FDA label formats and doesn't apply OCR cleaning
        to work with raw OCR outputs.
        
        Args:
            context: Full pharmaceutical label text (raw OCR output)
            
        Returns:
            Extracted INDICATIONS AND USAGE section text
        """
        print("\n🎯 FOCUSED EXTRACTION: INDICATIONS AND USAGE")
        print("="*80 + "\n")
        
        # Import improved extraction patterns
        from improved_extraction_methods import extract_indications_and_usage_fda
        
        # Update REPL environment with FDA-specific helper
        query = """Extract the INDICATIONS AND USAGE section from this pharmaceutical label.

TASK:
1. First, try to locate the section using patterns:
   - Look for '---INDICATIONS AND USAGE---' (FDA highlights format)
   - Look for '1 INDICATIONS AND USAGE' (numbered section format)
   - Look for 'INDICATIONS AND USAGE' header
   
2. The section typically ends when you encounter:
   - Another major section like 'DOSAGE AND ADMINISTRATION'
   - 'FULL PRESCRIBING INFORMATION'
   - Next numbered section (e.g., '2 DOSAGE')

3. Use regex or string slicing to extract the section

4. Store the result in 'result' variable

AVAILABLE HELPERS:
- env.grep(pattern) - find lines matching pattern
- re.search(pattern, context, flags) - regex search
- context[start:end] - slice the context string

Write Python code to extract this section. Be precise with your extraction."""
        
        # Create enhanced system prompt for this specific task
        enhanced_system = """You are a Recursive Language Model (RLM) specializing in pharmaceutical label extraction.

The context variable contains a pharmaceutical label (raw OCR output).

You can interact with it using Python code:
1. PEEK: context[:1000] to preview
2. GREP: env.grep(r'INDICATIONS') to find section markers
3. REGEX: Use re.search() with FDA-specific patterns
4. SLICE: context[start:end] to extract ranges

FDA Label Patterns:
- Highlights format: '---INDICATIONS AND USAGE---'
- Numbered format: '1 INDICATIONS AND USAGE' or '1.0 INDICATIONS'
- Section ends at: 'DOSAGE AND ADMINISTRATION', 'FULL PRESCRIBING', or next number

Your code should:
1. Find the start of INDICATIONS section
2. Find the end boundary
3. Extract the text between start and end
4. Store in 'result' variable

Write efficient Python code only."""
        
        # Use RLM with enhanced system prompt
        old_system = None
        result = self.rlm_completion(query, context)
        
        print("\n" + "="*80)
        print("📋 EXTRACTED INDICATIONS AND USAGE SECTION")
        print("="*80)
        
        # Also try the regex method as fallback
        try:
            regex_result = extract_indications_and_usage_fda(context)
            if regex_result and len(str(result).strip()) < 100:
                print("\n⚠️  RLM extraction was short, using regex fallback")
                result = regex_result
        except Exception as e:
            print(f"\n⚠️  Regex fallback failed: {e}")
        
        return str(result)
    
    def _parse_result_to_druginfo(self, result: str, context: str) -> DrugInfo:
        """Parse RLM result into structured DrugInfo"""
        
        # The result might be a string or dict-like structure
        # Use another LLM call to structure it
        structure_prompt = f"""Convert this extracted information into JSON format:

{result}

Return a JSON object with keys: drug_name, active_ingredients (list), indications (list), 
usage (string), dosage (string), warnings (list)."""

        structured = self._llm_call(structure_prompt, temp=0.1)
        
        # Parse JSON
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', structured, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return DrugInfo(
                    drug_name=data.get('drug_name'),
                    active_ingredients=data.get('active_ingredients', []),
                    indications=data.get('indications', []),
                    usage=data.get('usage'),
                    dosage=data.get('dosage'),
                    warnings=data.get('warnings', [])
                )
        except:
            pass
        
        # Fallback: basic parsing
        return DrugInfo(
            drug_name="Unable to extract",
            indications=[result]
        )


class PharmaceuticalRLM:
    """
    Complete pipeline with MIT-style RLM
    """
    
    def __init__(self, api_key: str):
        self.rlm = MITStyleRLM(api_key)
    
    def process_label(self, text: str, query: Optional[str] = None) -> DrugInfo:
        """Process pharmaceutical label using RLM"""
        
        if query is None:
            # Default: extract all drug info
            return self.rlm.extract_drug_info(text)
        else:
            # Custom query
            result = self.rlm.rlm_completion(query, text)
            print("\n" + "="*80)
            print("📊 RESULT")
            print("="*80)
            print(result)
            return None
    
    def extract_indications(self, text: str) -> str:
        """Extract INDICATIONS AND USAGE section using specialized RLM method"""
        return self.rlm.extract_indications_only(text)
    
    def format_output(self, info: DrugInfo) -> str:
        """Pretty print drug info"""
        lines = ["\n" + "="*80, "📊 EXTRACTED DRUG INFORMATION", "="*80 + "\n"]
        
        if info.drug_name:
            lines.append(f"💊 DRUG NAME: {info.drug_name}\n")
        
        if info.active_ingredients:
            lines.append("🧪 ACTIVE INGREDIENTS:")
            for ing in info.active_ingredients:
                lines.append(f"   • {ing}")
            lines.append("")
        
        if info.indications:
            lines.append("⚕️  INDICATIONS:")
            for i, ind in enumerate(info.indications, 1):
                lines.append(f"   {i}. {ind}")
            lines.append("")
        
        if info.usage:
            lines.append(f"💊 USAGE:\n   {info.usage}\n")
        
        if info.dosage:
            lines.append(f"📏 DOSAGE:\n   {info.dosage}\n")
        
        if info.warnings:
            lines.append("⚠️  WARNINGS:")
            for warn in info.warnings:
                lines.append(f"   ⚠ {warn}")
            lines.append("")
        
        lines.append("="*80)
        return "\n".join(lines)


def main():
    """Example usage"""
    
    DEEPSEEK_API_KEY = "sk-beb19132cd414a7281a6589a3a4def64"
    
    if not DEEPSEEK_API_KEY:
        print("⚠️  Set DEEPSEEK_API_KEY environment variable")
        #ask user to set the key
        Ask = input("Do you want to set it now? (y/n): ")
        if Ask.lower() == 'y':
            key = input("Enter your Deepseek API Key: ")
            os.environ["DEEPSEEK_API_KEY"] = key
            DEEPSEEK_API_KEY = key
        else:
            return
    
    # Sample pharmaceutical label
    sample_label = """
COSENTYX (secukinumab) injection, for subcutaneous use

DESCRIPTION
COSENTYX is a human IgG1/κ monoclonal antibody that selectively binds to interleukin-17A (IL-17A).
Each 1 mL of solution contains 150 mg of secukinumab.

INDICATIONS AND USAGE
Plaque Psoriasis
COSENTYX is indicated for the treatment of moderate to severe plaque psoriasis in adult patients 
who are candidates for systemic therapy or phototherapy.

Psoriatic Arthritis
COSENTYX is indicated for the treatment of active psoriatic arthritis in adult patients.

Ankylosing Spondylitis
COSENTYX is indicated for the treatment of active ankylosing spondylitis in adult patients.

Non-radiographic Axial Spondyloarthritis
COSENTYX is indicated for the treatment of active non-radiographic axial spondyloarthritis in 
adult patients with objective signs of inflammation.

DOSAGE AND ADMINISTRATION
Plaque Psoriasis: 300 mg by subcutaneous injection at Weeks 0, 1, 2, 3, and 4 followed by 
300 mg every 4 weeks. Each 300 mg dose is given as two subcutaneous injections of 150 mg.

Psoriatic Arthritis: 150 mg by subcutaneous injection at Weeks 0, 1, 2, 3, and 4 followed by 
150 mg every 4 weeks.

WARNINGS AND PRECAUTIONS
Infections: COSENTYX may increase the risk of infections. Serious infections have occurred.
Evaluate for tuberculosis infection prior to initiating treatment.

Pre-existing Inflammatory Bowel Disease: Caution should be used when prescribing COSENTYX 
to patients with inflammatory bowel disease. Crohn's disease and ulcerative colitis have occurred.

Hypersensitivity Reactions: Anaphylaxis and cases of urticaria occurred. If a serious allergic 
reaction occurs, discontinue COSENTYX immediately.
    """
    
    # Initialize RLM
    processor = PharmaceuticalRLM(DEEPSEEK_API_KEY)
    
    # Example 1: Specialized INDICATIONS extraction (NEW!)
    print("\n" + "="*80)
    print("EXAMPLE 1: Extract INDICATIONS AND USAGE Only (Specialized Method)")
    print("="*80)
    indications = processor.extract_indications(sample_label)
    print("\n" + "="*80)
    print("✅ EXTRACTED SECTION")
    print("="*80)
    # Print first 500 chars
    print(indications[:500] if len(indications) > 500 else indications)
    if len(indications) > 500:
        print(f"\n... (Total: {len(indications)} characters)")
    
    # Example 2: Full extraction
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Full Drug Information Extraction")
    print("="*80)
    drug_info = processor.process_label(sample_label)
    print(processor.format_output(drug_info))
    
    # Example 3: Custom query
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Custom Query")
    print("="*80)
    processor.process_label(
        sample_label,
        query="What are the dosing schedules for different indications?"
    )


if __name__ == "__main__":
    main()
