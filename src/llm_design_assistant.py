"""
LLM-Powered Design Assistant
Converts natural language prompts into car design parameters and execution plans
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

DB_PATH = "sqlite:///data/metadata.db"

class DesignAssistant:
    """LLM-powered assistant for car design optimization"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the design assistant
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        elif not OPENAI_AVAILABLE:
            print("⚠️  OpenAI package not installed. Install with: pip install openai")
        elif not self.api_key:
            print("⚠️  OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    def parse_design_prompt(self, prompt: str) -> Dict:
        """
        Parse natural language prompt to extract design requirements
        
        Args:
            prompt: Natural language description of desired car design
            
        Returns:
            Dictionary with extracted parameters and plan
        """
        if not self.client:
            return self._fallback_parse(prompt)
        
        system_prompt = """You are a car design assistant. Extract design parameters from user prompts.
        
Available parameters:
- length (m): 3.5-5.5 (compact to full-size)
- width (m): 1.6-2.0 (narrow to wide)
- height (m): 1.4-1.8 (low to tall)
- drag_coefficient (Cd): 0.20-0.35 (lower is better, typical: 0.25-0.30)
- wheelbase (m): 2.1-3.3 (distance between axles)
- roof_angle (degrees): -30 to 30 (roof slope)

Return JSON with:
{
    "parameters": {
        "length": value or null,
        "width": value or null,
        "height": value or null,
        "drag_coefficient": value or null,
        "wheelbase": value or null,
        "roof_angle": value or null
    },
    "goals": ["list", "of", "design", "goals"],
    "constraints": ["list", "of", "constraints"]
}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return self._fallback_parse(prompt)
    
    def _fallback_parse(self, prompt: str) -> Dict:
        """Fallback parser using regex when LLM is unavailable"""
        prompt_lower = prompt.lower()
        
        # Extract numeric values with units
        length_match = re.search(r'length[:\s]+([\d.]+)\s*m', prompt_lower)
        width_match = re.search(r'width[:\s]+([\d.]+)\s*m', prompt_lower)
        height_match = re.search(r'height[:\s]+([\d.]+)\s*m', prompt_lower)
        cd_match = re.search(r'(?:drag|cd|coefficient)[:\s]+([\d.]+)', prompt_lower)
        wheelbase_match = re.search(r'wheelbase[:\s]+([\d.]+)\s*m', prompt_lower)
        angle_match = re.search(r'(?:roof|angle)[:\s]+([-\d.]+)', prompt_lower)
        
        # Extract goals
        goals = []
        if 'aerodynamic' in prompt_lower or 'low drag' in prompt_lower:
            goals.append("minimize_drag")
        if 'efficient' in prompt_lower or 'fuel' in prompt_lower:
            goals.append("maximize_efficiency")
        if 'sport' in prompt_lower or 'fast' in prompt_lower:
            goals.append("sporty_design")
        if 'family' in prompt_lower or 'comfort' in prompt_lower:
            goals.append("family_car")
        
        return {
            "parameters": {
                "length": float(length_match.group(1)) if length_match else None,
                "width": float(width_match.group(1)) if width_match else None,
                "height": float(height_match.group(1)) if height_match else None,
                "drag_coefficient": float(cd_match.group(1)) if cd_match else None,
                "wheelbase": float(wheelbase_match.group(1)) if wheelbase_match else None,
                "roof_angle": float(angle_match.group(1)) if angle_match else None
            },
            "goals": goals,
            "constraints": []
        }
    
    def generate_plan(self, parsed_requirements: Dict, current_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate an execution plan based on parsed requirements
        
        Args:
            parsed_requirements: Output from parse_design_prompt
            current_data: Current dataset for reference
            
        Returns:
            Dictionary with execution plan
        """
        if not self.client:
            return self._fallback_plan(parsed_requirements, current_data)
        
        # Get data insights if available
        data_context = ""
        if current_data is not None and not current_data.empty:
            avg_stress = current_data['max_stress'].mean()
            optimal_designs = current_data.nsmallest(10, 'max_stress')
            data_context = f"""
Current dataset insights:
- Average stress: {avg_stress:,.0f} Pa
- Optimal designs have:
  * Length: {optimal_designs['length'].mean():.2f}m
  * Width: {optimal_designs['width'].mean():.2f}m
  * Height: {optimal_designs['height'].mean():.2f}m
  * Drag Coefficient: {optimal_designs['drag_coefficient'].mean():.3f}
"""
        
        system_prompt = """You are a car design optimization assistant. Create a step-by-step plan to achieve the design goals.

Generate a JSON plan with:
{
    "steps": [
        {
            "step_number": 1,
            "action": "set_parameter",
            "parameter": "length",
            "value": 4.5,
            "reason": "why this value"
        },
        {
            "step_number": 2,
            "action": "optimize",
            "target": "minimize_stress",
            "method": "iterative_refinement"
        }
    ],
    "expected_outcome": "description of expected result"
}

Available actions:
- set_parameter: Set a specific parameter value
- optimize: Optimize for a goal (minimize_stress, minimize_drag, etc.)
- analyze: Analyze current design
- compare: Compare with dataset"""
        
        user_prompt = f"""
Design Requirements:
{json.dumps(parsed_requirements, indent=2)}

{data_context}

Create an execution plan to achieve these goals."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
        except Exception as e:
            print(f"Error generating plan: {e}")
            return self._fallback_plan(parsed_requirements, current_data)
    
    def _fallback_plan(self, parsed_requirements: Dict, current_data: Optional[pd.DataFrame] = None) -> Dict:
        """Fallback plan generation"""
        steps = []
        params = parsed_requirements.get("parameters", {})
        
        # Set explicit parameters
        for param, value in params.items():
            if value is not None:
                steps.append({
                    "step_number": len(steps) + 1,
                    "action": "set_parameter",
                    "parameter": param,
                    "value": value,
                    "reason": f"User specified {param} = {value}"
                })
        
        # Optimize based on goals
        goals = parsed_requirements.get("goals", [])
        if "minimize_drag" in goals:
            steps.append({
                "step_number": len(steps) + 1,
                "action": "optimize",
                "target": "minimize_drag",
                "method": "lower_drag_coefficient"
            })
        
        if "minimize_stress" in goals or not goals:
            steps.append({
                "step_number": len(steps) + 1,
                "action": "optimize",
                "target": "minimize_stress",
                "method": "use_optimal_from_data"
            })
        
        return {
            "steps": steps,
            "expected_outcome": "Design optimized based on requirements"
        }
    
    def get_optimal_parameters(self, goal: str = "minimize_stress", data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get optimal parameters from dataset based on goal
        
        Args:
            goal: Optimization goal (minimize_stress, minimize_drag, etc.)
            data: Dataset to analyze (loads from DB if None)
            
        Returns:
            Dictionary with optimal parameter values
        """
        if data is None:
            try:
                engine = create_engine(DB_PATH)
                data = pd.read_sql('simulations', engine)
            except:
                return {}
        
        if data.empty:
            return {}
        
        if goal == "minimize_stress":
            optimal = data.nsmallest(10, 'max_stress')
        elif goal == "minimize_drag":
            optimal = data.nsmallest(10, 'drag_coefficient')
        else:
            optimal = data.nsmallest(10, 'max_stress')
        
        return {
            "length": float(optimal['length'].mean()),
            "width": float(optimal['width'].mean()),
            "height": float(optimal['height'].mean()),
            "drag_coefficient": float(optimal['drag_coefficient'].mean()) if 'drag_coefficient' in optimal.columns else 0.26,
            "wheelbase": float(optimal['wheelbase'].mean()) if 'wheelbase' in optimal.columns else 2.8,
            "roof_angle": float(optimal['roof_angle'].mean()) if 'roof_angle' in optimal.columns else 0.0
        }
    
    def execute_plan(self, plan: Dict, current_params: Dict) -> Dict:
        """
        Execute a plan to generate final parameters
        
        Args:
            plan: Execution plan from generate_plan
            current_params: Current parameter values
            
        Returns:
            Final parameter values after plan execution
        """
        final_params = current_params.copy()
        steps = plan.get("steps", [])
        
        for step in steps:
            action = step.get("action")
            
            if action == "set_parameter":
                param = step.get("parameter")
                value = step.get("value")
                if param and value is not None:
                    final_params[param] = value
            
            elif action == "optimize":
                target = step.get("target")
                # Get optimal values from data
                optimal = self.get_optimal_parameters(target)
                # Merge with current params (only update if not explicitly set)
                for key, value in optimal.items():
                    if key not in final_params or final_params[key] is None:
                        final_params[key] = value
        
        return final_params

