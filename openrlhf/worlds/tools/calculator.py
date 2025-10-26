from typing import Any, Dict, Tuple
from marti.worlds.tools.base import BaseToolExecutor

class CalculatorToolExecutor(BaseToolExecutor):
    """Calculator tool executor."""
    
    def get_name(self) -> str:
        return "calculator"
    
    async def execute(self, parameters: Dict[str, Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Execute calculation."""
        expression = parameters.get("expression", "")
        
        try:
            # Safe evaluation
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
            return f"Result: {result}", {"success": True, "result": result}
        except Exception as e:
            return f"Calculation error: {str(e)}", {"success": False, "error": str(e)}