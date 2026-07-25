"""
Prompt generation and management for BenchLab.

Generates standardized prompts for different workload types
with configurable length and complexity.
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchlab.config import WorkloadLength, WorkloadType


@dataclass
class Prompt:
    """A single benchmark prompt."""
    id: str
    text: str
    expected_length: int
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSet:
    """A collection of prompts for a workload."""
    workload_type: str
    length: str
    prompts: list[Prompt] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Any:
        return iter(self.prompts)

    def __getitem__(self, index: int) -> Prompt:
        return self.prompts[index]


class PromptGenerator:
    """Generate benchmark prompts for various workload types."""

    # Code prompts
    CODING_PROMPTS: list[str] = [
        "Implement a function that {task} in {language}. Include error handling and type hints.",
        "Write a {language} class that implements {interface}. Include unit tests.",
        "Create a {language} script that reads {data_source}, processes it, and outputs {format}.",
        "Implement a {pattern} design pattern in {language} for {use_case}.",
        "Write a REST API endpoint in {language} that {operation} with authentication.",
        "Create a database migration script in {language} for {schema_change}.",
        "Implement a {algorithm} algorithm in {language} with complexity analysis.",
        "Write a concurrent {language} program that {concurrency_task}.",
        "Build a CLI tool in {language} that {cli_task} with argument parsing.",
        "Implement a {language} middleware that {middleware_task}.",
    ]

    CODING_TASKS: list[str] = [
        "parse and validate JSON input",
        "connect to a PostgreSQL database",
        "implement a thread-safe cache",
        "handle file uploads efficiently",
        "serialize and deserialize binary data",
        "implement rate limiting",
        "handle concurrent connections",
        "parse and transform CSV data",
        "implement a connection pool",
        "handle graceful shutdown",
        "implement a retry mechanism",
        "parse command-line arguments",
        "implement a simple router",
        "handle WebSocket connections",
        "implement a message queue",
    ]

    CODING_LANGUAGES: list[str] = ["Python", "Rust", "Go", "TypeScript", "C++"]

    # Reasoning prompts
    REASONING_PROMPTS: list[str] = [
        "Solve this logic puzzle: {puzzle}",
        "Analyze the following argument and identify any logical fallacies: {argument}",
        "Given the premises: {premises}, what conclusion can be drawn?",
        "Compare and contrast {concept1} and {concept2} in terms of {criteria}.",
        "Explain the reasoning behind {reasoning_question}.",
        "Solve this mathematical proof: {proof}",
        "What are the implications of {implication_scenario}?",
        "Analyze the trade-offs between {tradeoff1} and {tradeoff2}.",
        "Evaluate the validity of the following claim: {claim}",
        "Deduce the answer from these clues: {clues}",
    ]

    # Creative writing prompts
    CREATIVE_PROMPTS: list[str] = [
        "Write a {genre} story about {theme} that is approximately {word_count} words.",
        "Compose a {format} about {subject} in the style of {author_style}.",
        "Write a dialogue between {character1} and {character2} about {topic}.",
        "Create a {format} describing {scene} with vivid sensory details.",
        "Write a {genre} narrative from the perspective of {perspective}.",
        "Compose a poem about {poem_subject} using {poetic_form} structure.",
        "Write a short essay analyzing {essay_topic}.",
        "Create a character description for {character} in a {genre} setting.",
        "Write a scene where {scene_description}.",
        "Draft a letter from {sender} to {recipient} about {letter_topic}.",
    ]

    # Math prompts
    MATH_PROMPTS: list[str] = [
        "Calculate: {math_problem}",
        "Solve the following equation: {equation}",
        "Compute the {math_operation} of the following numbers: {numbers}",
        "Find the {math_result} of: {math_expression}",
        "Evaluate: {math_expression}",
        "Solve this {math_type} problem: {math_problem_detail}",
        "Calculate the {calculation_type} for: {calculation_input}",
        "Determine the {determination_type}: {determination_input}",
        "Compute the integral: {integral}",
        "Solve the system of equations: {system}",
    ]

    # RAG prompts
    RAG_PROMPTS: list[str] = [
        "Based on the provided context, answer: {rag_question}",
        "Summarize the key points from the context regarding {rag_topic}.",
        "Extract all relevant information about {rag_entity} from the context.",
        "What does the context say about {rag_relation}?",
        "Compare the information in the context with the question: {rag_question}",
        "Identify any contradictions in the context regarding {rag_topic}.",
        "What is the main argument presented in the context about {rag_topic}?",
        "List all facts mentioned in the context about {rag_entity}.",
        "Infer the author's stance on {rag_topic} based on the context.",
        "Synthesize information from the context to answer: {rag_question}",
    ]

    # Long context prompts
    LONG_CONTEXT_PROMPTS: list[str] = [
        "Read the following document and answer questions about it.",
        "Summarize the key points from this lengthy document.",
        "Extract all named entities from this text.",
        "Identify the main topics discussed in this document.",
        "Find all references to specific dates and events.",
        "Analyze the structure and flow of arguments in this text.",
        "Extract all numerical data and statistics mentioned.",
        "Identify the relationships between different concepts.",
        "Summarize each section of this document.",
        "Create a table of contents based on the document structure.",
    ]

    # Conversation prompts
    CONVERSATION_PROMPTS: list[str] = [
        "Have a natural conversation about {topic}.",
        "Roleplay as a {role} discussing {topic} with a customer.",
        "Simulate a technical interview for a {role} position.",
        "Engage in a debate about {topic}.",
        "Provide customer support for {scenario}.",
        "Conduct a job interview for the position of {role}.",
        "Simulate a therapy session discussing {topic}.",
        "Have a philosophical discussion about {topic}.",
        "Roleplay a negotiation about {scenario}.",
        "Simulate a team standup meeting discussing {topic}.",
    ]

    # Tool calling prompts
    TOOL_CALLING_PROMPTS: list[str] = [
        "Find the current weather in {location} and convert the temperature to Celsius.",
        "Search for information about {topic} and summarize the top 3 results.",
        "Calculate the distance between {location1} and {location2}.",
        "Translate the following text from English to {language}: {text}",
        "Convert {amount} {currency1} to {currency2}.",
        "Find the population of {city} and its growth rate over the last decade.",
        "Look up the ingredients for {recipe} and create a shopping list.",
        "Calculate the best route between {location1}, {location2}, and {location3}.",
        "Set a reminder for {time} about {task}.",
        "Search for recent news about {topic} and provide a summary.",
    ]

    # JSON generation prompts
    JSON_PROMPTS: list[str] = [
        "Generate a valid JSON object representing a user profile with the following fields: {fields}",
        "Create a JSON array of {count} items, each representing a {item_type} with properties: {properties}",
        "Generate a JSON response for an API that returns {response_type} data.",
        "Create a JSON configuration file for a {config_type} with settings: {settings}",
        "Generate a JSON representation of a {entity} with nested {nested_type} objects.",
        "Create a JSON schema for validating {schema_type} data.",
        "Generate a JSON document representing a {document_type} with sections: {sections}",
        "Create a JSON object representing a database record for a {table} table.",
        "Generate a JSON response mimicking a {api_type} API response.",
        "Create a JSON configuration for a {service_type} service.",
    ]

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate_prompts(
        self,
        workload_type: WorkloadType,
        length: WorkloadLength,
        num_prompts: int = 100,
        template: str | None = None,
    ) -> PromptSet:
        """Generate a set of prompts for the given workload."""
        if template:
            return self._generate_from_template(template, num_prompts, workload_type, length)

        generator_method = getattr(self, f"_generate_{workload_type.value}", None)
        if generator_method:
            return generator_method(length, num_prompts)

        return PromptSet(
            workload_type=workload_type.value,
            length=length.value,
            prompts=[],
        )

    def _generate_coding(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            task = self._rng.choice(self.CODING_TASKS)
            lang = self._rng.choice(self.CODING_LANGUAGES)
            template = self._rng.choice(self.CODING_PROMPTS)
            text = template.format(task=task, language=lang)

            expected_lengths = {
                WorkloadLength.SHORT: 500,
                WorkloadLength.MEDIUM: 1500,
                WorkloadLength.LONG: 5000,
                WorkloadLength.CONTEXT_128K: 50000,
                WorkloadLength.CONTEXT_200K: 80000,
            }

            prompts.append(Prompt(
                id=f"coding-{i+1}",
                text=text,
                expected_length=expected_lengths.get(length, 1500),
                category="coding",
                metadata={"language": lang, "task": task},
            ))

        return PromptSet(
            workload_type=WorkloadType.CODING.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_reasoning(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            template = self._rng.choice(self.REASONING_PROMPTS)
            text = template.format(
                puzzle="A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have?",
                argument="All cats are animals. Some animals are pets. Therefore, all cats are pets.",
                premises="All humans are mortal. Socrates is a human.",
                concept1="classical conditioning",
                concept2="operant conditioning",
                criteria="effectiveness and limitations",
                reasoning_question="Why do we perceive time differently during exciting events?",
                proof="Prove that the square root of 2 is irrational.",
                implication_scenario="complete reliance on autonomous vehicles",
                tradeoff1="security",
                tradeoff2="convenience",
                claim="Social media has a net negative effect on society.",
                clues="Alice is taller than Bob. Bob is taller than Charlie. Charlie is taller than Dave.",
            )

            prompts.append(Prompt(
                id=f"reasoning-{i+1}",
                text=text,
                expected_length=1000,
                category="reasoning",
            ))

        return PromptSet(
            workload_type=WorkloadType.REASONING.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_creative_writing(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        genres = ["science fiction", "fantasy", "mystery", "horror", "romance", "thriller"]
        formats = ["short story", "poem", "essay", "monologue", "dialogue"]
        for i in range(num):
            template = self._rng.choice(self.CREATIVE_PROMPTS)
            text = template.format(
                genre=self._rng.choice(genres),
                theme=self._rng.choice(["love", "loss", "discovery", "betrayal", "redemption", "identity"]),
                word_count=self._rng.choice(["200", "500", "1000", "2000"]),
                format=self._rng.choice(formats),
                subject=self._rng.choice(["the ocean", "a forgotten city", "the stars", "memory", "time"]),
                author_style=self._rng.choice(["Hemingway", "Austen", "Tolkien", "Orwell", "Bradbury"]),
                character1=self._rng.choice(["a detective", "a scientist", "a traveler"]),
                character2=self._rng.choice(["a stranger", "an old friend", "a mentor"]),
                topic=self._rng.choice(["the meaning of life", "artificial intelligence", "climate change"]),
                scene=self._rng.choice(["a storm at sea", "a quiet morning", "a bustling market"]),
                perspective=self._rng.choice(["an AI", "a child", "an elderly person", "an animal"]),
                poem_subject=self._rng.choice(["autumn", "the sea", "solitude", "hope"]),
                poetic_form=self._rng.choice(["haiku", "sonnet", "free verse", "ballad"]),
                essay_topic=self._rng.choice(["the nature of consciousness", "the impact of technology"]),
                character=self._rng.choice(["a rogue AI", "a wandering knight", "a time traveler"]),
                scene_description=self._rng.choice(["two strangers meet in an elevator", "a secret is revealed"]),
                sender=self._rng.choice(["a soldier", "a scientist", "a traveler"]),
                recipient=self._rng.choice(["their family", "a colleague", "a stranger"]),
                letter_topic=self._rng.choice(["a difficult decision", "a message of hope"]),
            )

            prompts.append(Prompt(
                id=f"creative-{i+1}",
                text=text,
                expected_length=1000,
                category="creative_writing",
            ))

        return PromptSet(
            workload_type=WorkloadType.CREATIVE_WRITING.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_math(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            template = self._rng.choice(self.MATH_PROMPTS)
            text = template.format(
                math_problem="What is the sum of all prime numbers between 1 and 100?",
                equation="x^2 + 5x + 6 = 0",
                math_operation="GCD and LCM",
                numbers="252 and 105",
                math_result="derivative",
                math_expression="d/dx (3x^3 + 2x^2 - 5x + 7)",
                math_type="calculus",
                math_problem_detail="Find the area under the curve y = x^2 from x=0 to x=5.",
                calculation_type="compound interest",
                calculation_input="$10,000 at 5% annual rate for 10 years",
                determination_type="limit",
                determination_input="lim(x→0) sin(x)/x",
                integral="∫(2x + 3)dx from 0 to 5",
                system="2x + y = 5, x - y = 1",
            )

            prompts.append(Prompt(
                id=f"math-{i+1}",
                text=text,
                expected_length=500,
                category="math",
            ))

        return PromptSet(
            workload_type=WorkloadType.MATH.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_rag(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            template = self._rng.choice(self.RAG_PROMPTS)
            text = template.format(
                rag_question="How does quantum computing work?",
                rag_topic="machine learning",
                rag_entity="Artificial Intelligence",
                rag_relation="between AI and healthcare",
            )

            prompts.append(Prompt(
                id=f"rag-{i+1}",
                text=text,
                expected_length=500,
                category="rag",
            ))

        return PromptSet(
            workload_type=WorkloadType.RAG.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_long_context(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            template = self._rng.choice(self.LONG_CONTEXT_PROMPTS)
            text = template.format()

            expected_lengths = {
                WorkloadLength.SHORT: 1000,
                WorkloadLength.MEDIUM: 5000,
                WorkloadLength.LONG: 20000,
                WorkloadLength.CONTEXT_128K: 100000,
                WorkloadLength.CONTEXT_200K: 160000,
            }

            prompts.append(Prompt(
                id=f"long_context-{i+1}",
                text=text,
                expected_length=expected_lengths.get(length, 5000),
                category="long_context",
            ))

        return PromptSet(
            workload_type=WorkloadType.LONG_CONTEXT.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_conversation(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        topics = [
            "artificial intelligence",
            "climate change",
            "space exploration",
            "quantum computing",
            "biotechnology",
            "education reform",
            "mental health",
            "cryptocurrency",
        ]
        roles = ["teacher", "student", "consultant", "patient", "manager"]
        scenarios = [
            "a product return",
            "a technical support issue",
            "a billing dispute",
            "a service complaint",
        ]
        for i in range(num):
            template = self._rng.choice(self.CONVERSATION_PROMPTS)
            text = template.format(
                topic=self._rng.choice(topics),
                role=self._rng.choice(roles),
                scenario=self._rng.choice(scenarios),
            )

            prompts.append(Prompt(
                id=f"conversation-{i+1}",
                text=text,
                expected_length=500,
                category="conversation",
            ))

        return PromptSet(
            workload_type=WorkloadType.CONVERSATION.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_tool_calling(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            template = self._rng.choice(self.TOOL_CALLING_PROMPTS)
            text = template.format(
                location=self._rng.choice(["New York", "London", "Tokyo", "Paris", "Sydney"]),
                topic=self._rng.choice(["AI", "climate change", "space exploration"]),
                location1=self._rng.choice(["New York", "London", "Tokyo"]),
                location2=self._rng.choice(["London", "Tokyo", "Paris"]),
                location3=self._rng.choice(["Paris", "Sydney", "Tokyo"]),
                language=self._rng.choice(["Spanish", "French", "German", "Japanese"]),
                text=self._rng.choice(["Hello world", "Good morning", "Thank you"]),
                amount=self._rng.choice(["100", "500", "1000"]),
                currency1=self._rng.choice(["USD", "EUR", "GBP"]),
                currency2=self._rng.choice(["EUR", "GBP", "JPY"]),
                city=self._rng.choice(["New York", "London", "Tokyo"]),
                recipe=self._rng.choice(["pasta carbonara", "chicken tikka masala", "sushi"]),
                time=self._rng.choice(["tomorrow at 9am", "next Monday", "in 30 minutes"]),
                task=self._rng.choice(["review the report", "prepare the presentation"]),
            )

            prompts.append(Prompt(
                id=f"tool_calling-{i+1}",
                text=text,
                expected_length=500,
                category="tool_calling",
            ))

        return PromptSet(
            workload_type=WorkloadType.TOOL_CALLING.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_json_generation(self, length: WorkloadLength, num: int) -> PromptSet:
        prompts = []
        for i in range(num):
            template = self._rng.choice(self.JSON_PROMPTS)
            text = template.format(
                fields="name, email, age, address, preferences",
                count=str(self._rng.randint(5, 50)),
                item_type=self._rng.choice(["product", "user", "order", "review"]),
                properties=self._rng.choice(["id, name, price", "user_id, rating, comment"]),
                response_type=self._rng.choice(["user", "product", "order"]),
                config_type=self._rng.choice(["web server", "database", "microservice"]),
                settings=self._rng.choice(["host, port, workers", "connection string, pool size"]),
                entity=self._rng.choice(["order", "user", "product"]),
                nested_type=self._rng.choice(["address", "items", "metadata"]),
                schema_type=self._rng.choice(["user profile", "order", "product catalog"]),
                document_type=self._rng.choice(["report", "article", "manual"]),
                sections=self._rng.choice(["intro, body, conclusion", "abstract, methods, results"]),
                table=self._rng.choice(["users", "products", "orders"]),
                api_type=self._rng.choice(["REST", "GraphQL", "gRPC"]),
                service_type=self._rng.choice(["web", "database", "cache"]),
            )

            prompts.append(Prompt(
                id=f"json-{i+1}",
                text=text,
                expected_length=500,
                category="json_generation",
            ))

        return PromptSet(
            workload_type=WorkloadType.JSON_GENERATION.value,
            length=length.value,
            prompts=prompts,
        )

    def _generate_from_template(
        self,
        template: str,
        num: int,
        workload_type: WorkloadType,
        length: WorkloadLength,
    ) -> PromptSet:
        """Generate prompts from a custom template."""
        prompts = []
        for i in range(num):
            prompts.append(Prompt(
                id=f"custom-{i+1}",
                text=template,
                expected_length=1000,
                category="custom",
            ))

        return PromptSet(
            workload_type=workload_type.value,
            length=length.value,
            prompts=prompts,
        )

    def save_prompt_set(self, prompt_set: PromptSet, output_dir: str | Path) -> Path:
        """Save a prompt set to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{prompt_set.workload_type}_{prompt_set.length}.json"
        filepath = output_dir / filename

        data = {
            "workload_type": prompt_set.workload_type,
            "length": prompt_set.length,
            "metadata": prompt_set.metadata,
            "prompts": [
                {
                    "id": p.id,
                    "text": p.text,
                    "expected_length": p.expected_length,
                    "category": p.category,
                    "metadata": p.metadata,
                }
                for p in prompt_set.prompts
            ],
        }

        filepath.write_text(json.dumps(data, indent=2))
        return filepath

    def load_prompt_set(self, filepath: str | Path) -> PromptSet:
        """Load a prompt set from a JSON file."""
        filepath = Path(filepath)
        data = json.loads(filepath.read_text())

        prompts = [
            Prompt(
                id=p["id"],
                text=p["text"],
                expected_length=p["expected_length"],
                category=p["category"],
                metadata=p.get("metadata", {}),
            )
            for p in data["prompts"]
        ]

        return PromptSet(
            workload_type=data["workload_type"],
            length=data["length"],
            prompts=prompts,
            metadata=data.get("metadata", {}),
        )
