import json
import json5
import re
from typing import List, Dict, Any, Optional

class ToolParser:
    """Parse the response text containing tool calls"""
    
    # Error type constant
    ERROR_NAMES = {
        'EXTRACTION_FAILED': '<error>',
        'EMPTY_CALL': '<empty>',
        'PARSE_ERROR': '<parse_error>'
    }
    
    ERROR_MESSAGES = {
        'EXTRACTION_FAILED': '# Extract the tool name failed',
        'EMPTY_CALL': '# Empty tool call content',
        'PARSE_ERROR': '# Failed to parse tool call JSON',
        'MISSING_FIELDS': '# Missing required fields (name/arguments)',
        'INVALID_FORMAT': '# Invalid tool call format'
    }
    
    def parse_tools(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the tool calls in the response
        
        Args:
            response: Response text containing tool calls
            
        Returns:
            List[Dict[str, str]]: Parsed tool call list
        """
        if not response or not isinstance(response, str):
            return self._create_error_tool('EXTRACTION_FAILED')
        
        # Check if there is a tool call tag
        if '<tool_call>' not in response:
            return [response] if response.strip() else self._create_error_tool('EXTRACTION_FAILED')
        
        # Check if the tags appear in pairs
        if response.count('<tool_call>') != response.count('</tool_call>'):
            return self._create_error_tool('EXTRACTION_FAILED')
        
        return self._extract_tool_calls(response)
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, str]]:
        """Extract tool call"""
        parsed_tools = []
        
        # Use regular expression extraction tool to call block
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if not matches:
            return self._create_error_tool('EXTRACTION_FAILED')
        
        for match in matches:
            tool_call = self._parse_single_tool_call(match.strip())
            parsed_tools.append(tool_call)
        
        return parsed_tools if parsed_tools else self._create_error_tool('EXTRACTION_FAILED')
    
    def _parse_single_tool_call(self, content: str) -> Dict[str, str]:
        """Analyze a single tool call"""
        if not content:
            return self._create_single_error_tool('EMPTY_CALL')
        
        try:
            # Attempt to parse JSON
            parsed_json = json5.loads(content)
            
            # Verify required fields
            if not isinstance(parsed_json, dict):
                return self._create_single_error_tool('PARSE_ERROR')
            
            if 'name' not in parsed_json or 'arguments' not in parsed_json:
                return self._create_single_error_tool('PARSE_ERROR', 'MISSING_FIELDS')
            
            # Verify field type
            if not isinstance(parsed_json['name'], str):
                return self._create_single_error_tool('PARSE_ERROR', 'INVALID_FORMAT')
            
            return {
                "name": parsed_json['name'],
                "args": json.dumps(parsed_json['arguments'], ensure_ascii=False, indent=2)
            }
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return self._create_single_error_tool('PARSE_ERROR')
    
    def _create_error_tool(self, error_type: str, message_key: Optional[str] = None) -> List[Dict[str, str]]:
        """Create error tool call list"""
        return [self._create_single_error_tool(error_type, message_key)]
    
    def _create_single_error_tool(self, error_type: str, message_key: Optional[str] = None) -> Dict[str, str]:
        """Create a single error tool call"""
        return {
            "name": self.ERROR_NAMES[error_type],
            "args": self.ERROR_MESSAGES[message_key or error_type]
        }
