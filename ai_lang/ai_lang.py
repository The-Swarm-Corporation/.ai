from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

import os
import asyncio
from swarms import Agent
from swarm_models import OpenAIChat
from dotenv import load_dotenv
import json
import subprocess
import tempfile


class NaturalAIExecutor:
    """NaturalAI Executor for processing and executing natural language tasks"""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.output_dir = Path("natural_ai_output")
        self.output_dir.mkdir(exist_ok=True)
        self.agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        """Initialize the primary agent with specific instructions"""
        system_prompt = """You are an advanced AI executor that helps users create and run various types of content and code. 
        Your tasks include:
        
        1. Understanding the user's natural language request
        2. Creating appropriate files and content
        3. Executing code when necessary
        4. Providing clear feedback about actions taken
        
        For code tasks:
        - Create working, well-documented code
        - Include error handling
        - Follow best practices
        - Only output the code and nothing else. No markdown nothing, just output the code as it is.
        
        For content tasks:
        - Create high-quality, appropriate content
        - Format according to user specifications
        - Save in the requested format
        
        Always ensure outputs are practical and executable.
        """

        model = OpenAIChat(
            openai_api_key=self.api_key,
            model_name="gpt-4",
            temperature=0.7,
        )

        return Agent(
            agent_name="NaturalAI-Executor",
            system_prompt=system_prompt,
            llm=model,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path="natural_ai_executor.json",
            user_name="natural_ai",
            retry_attempts=2,
            context_length=200000,
            return_step_meta=False,
            output_type="string",
            streaming_on=False,
        )

    def _parse_request(self, content: str) -> Dict[str, Any]:
        """Parse the natural language request to determine action and requirements"""

        # Create a structured prompt for the agent to analyze the request
        analysis_prompt = f"""Analyze the following request and provide a structured response:
        
        Request: {content}
        
        Provide response in the following JSON format:
        {{
            "task_type": "code|content|data|script",
            "output_type": "file|execution|both",
            "file_type": "py|txt|json|csv|etc",
            "execution_required": true|false,
            "expected_output": "description of expected output",
            "additional_requirements": ["requirement1", "requirement2"]
        }}
        """

        response = self.agent.run(analysis_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "task_type": "content",
                "output_type": "file",
                "file_type": "txt",
                "execution_required": False,
                "expected_output": "text content",
                "additional_requirements": [],
            }

    def _create_content(
        self, request: str, task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create the requested content based on task information"""

        creation_prompt = f"""Create the content for the following request:
        
        Request: {request}
        Task Type: {task_info['task_type']}
        Expected Output: {task_info['expected_output']}
        Additional Requirements: {', '.join(task_info['additional_requirements'])}
        
        Provide the content in a format ready to be saved or executed.
        """

        content = self.agent.run(creation_prompt)

        # Generate unique filename
        timestamp = asyncio.get_event_loop().time()
        filename = f"{task_info['task_type']}_{timestamp}.{task_info['file_type']}"
        filepath = self.output_dir / filename

        # Save content to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "filename": filename,
            "filepath": str(filepath),
            "content": content,
            "task_info": task_info,
        }

    def _execute_code(
        self, content_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code if required and return results"""
        if not content_info["task_info"]["execution_required"]:
            return {"execution_required": False}

        result = {
            "execution_required": True,
            "success": False,
            "output": "",
            "error": None,
        }

        if content_info["task_info"]["file_type"] == "py":
            try:
                # Create a temporary directory for execution
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = Path(temp_dir) / "temp_script.py"
                    with open(temp_file, "w") as f:
                        f.write(content_info["content"])

                    # Execute the Python script
                    process = subprocess.run(
                        ["python", str(temp_file)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    result.update(
                        {
                            "success": process.returncode == 0,
                            "output": process.stdout,
                            "error": (
                                process.stderr
                                if process.returncode != 0
                                else None
                            ),
                        }
                    )
            except Exception as e:
                result["error"] = str(e)

        return result

    async def process_file(
        self, file_path: str
    ) -> List[Dict[str, Any]]:
        """Process a .ai file containing multiple requests"""
        results = []

        try:
            with open(file_path, "r") as f:
                requests = [
                    req.strip()
                    for req in f.read().split("\n\n")
                    if req.strip()
                ]

            for request in requests:
                result = await self.process_request(request)
                results.append(result)

        except Exception as e:
            results.append(
                {
                    "error": f"Error processing file: {str(e)}",
                    "request": file_path,
                }
            )

        return results

    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process a single natural language request"""
        try:
            # Parse the request
            task_info = self._parse_request(request)

            # Create content
            content_info = self._create_content(request, task_info)

            # Execute if required
            execution_result = self._execute_code(content_info)

            return {
                "request": request,
                "content_info": content_info,
                "execution_result": execution_result,
                "success": True,
            }

        except Exception as e:
            return {
                "request": request,
                "error": str(e),
                "success": False,
            }


async def process_ai_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Process a .ai file containing natural language requests.

    Args:
        filepath: Path to the .ai file to process

    Returns:
        List of results from processing each request in the file

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file extension is not .ai
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix != ".ai":
        raise ValueError(
            f"File must have .ai extension, got: {filepath.suffix}"
        )

    logger.info(f"Processing AI file: {filepath}")

    executor = NaturalAIExecutor()
    results = await executor.process_file(filepath)

    for result in results:
        if result["success"]:
            logger.info(
                f"Successfully processed request: {result['request']}"
            )
            logger.info(
                f"Created file: {result['content_info']['filepath']}"
            )

            if result.get("execution_result", {}).get(
                "execution_required"
            ):
                logger.info("Execution output:")
                logger.info(
                    result["execution_result"].get(
                        "output", "No output"
                    )
                )

                if result["execution_result"].get("error"):
                    logger.error(
                        f"Execution error: {result['execution_result']['error']}"
                    )
        else:
            logger.error(
                f"Error processing request: {result.get('error')}"
            )

    return results


def process_ai_file_sync(filepath: str) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for process_ai_file.

    Args:
        filepath: String path to the .ai file to process

    Returns:
        List of results from processing each request in the file
    """
    import asyncio

    path = Path(filepath)
    return asyncio.run(process_ai_file(path))
