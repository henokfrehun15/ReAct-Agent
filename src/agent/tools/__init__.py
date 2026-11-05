"""Tool package: importing this registers all built-in tools in the registry."""
from agent.tools import calculator  # noqa: F401  (registers the @tool decorator)
from agent.tools import clock  # noqa: F401
from agent.tools import notes  # noqa: F401
from agent.tools import registry  # noqa: F401