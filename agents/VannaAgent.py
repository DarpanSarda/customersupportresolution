"""Vanna Agent for text-to-SQL query generation and execution."""

import logging
from typing import Optional, Dict, Any, List
from core.BaseAgent import BaseAgent, Patch, AgentExecutionContext

logger = logging.getLogger(__name__)


class VannaAgent(BaseAgent):
    """
    Text-to-SQL agent using Vanna AI.

    Converts natural language to SQL and executes it.
    """

    agent_name = "VannaAgent"
    allowed_section = "vanna"

    def __init__(self, config: Dict[str, Any], prompt: str):
        super().__init__(config, prompt)
        self._vanna = None
        self._conn = None
        self._vanna_config = config.get("vanna_config", {})
        self._db_type = self._vanna_config.get("db_type", "sqlite")

    def _initialize(self):
        """Initialize Vanna and DB connection."""
        if self._vanna:
            return

        from vanna.vannibase import VannaBase

        llm_client = self._config.get("llm_client")

        # Simple Vanna wrapper
        class SimpleVanna(VannaBase):
            def __init__(self, llm_client, dialect="postgres"):
                self.llm_client = llm_client
                self.dialect = dialect

            def system_prompt(self, msg):
                return msg

            def generate_sql(self, question, **kwargs):
                schema = kwargs.get("ddl", "")
                prompt = f"""Convert to SQL (dialect: {self.dialect}):

{schema}

Question: {question}

Return only SQL."""

                resp = self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500
                )
                return resp.strip().removeprefix("```sql").removeprefix("```").strip()

            def run_sql(self, sql):
                # Handled by agent
                pass

        self._vanna = SimpleVanna(
            llm_client=llm_client,
            dialect=self._vanna_config.get("dialect", "postgres")
        )

        # Connect to DB
        conn_str = self._vanna_config.get("connection_string")
        db_type = self._db_type

        if db_type == "sqlite":
            import sqlite3
            self._conn = sqlite3.connect(conn_str)
            self._conn.row_factory = sqlite3.Row
        elif db_type == "postgres":
            import psycopg2
            self._conn = psycopg2.connect(conn_str)
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        self._initialize()

        question = state.get("conversation", {}).get("latest_message", "")
        if not question:
            return Patch(target_section="vanna", updates={"status": "error", "error": "No question"})

        # Generate SQL
        schema = self._vanna_config.get("schema", "")
        sql = self._vanna.generate_sql(question, ddl=schema)

        # Execute
        cursor = self._conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        # Convert to list of dicts
        if self._db_type == "sqlite":
            results = [dict(row) for row in rows]
        else:
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

        cursor.close()

        return Patch(target_section="vanna", updates={
            "status": "success",
            "question": question,
            "sql": sql,
            "results": results,
            "count": len(results)
        })
